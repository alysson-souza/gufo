// Microbenchmarks for Flint's decode-critical and prefill kernels.
//
// Weights and KV caches are replicated past the 32 MiB MALL and rotated per
// call so every timed call streams from DRAM; routed experts draw fresh
// random routes each call. Reported GB/s counts the weight (or KV) bytes a
// call must read once; compare with tools/bench/gfx1151_peak.hip DRAM read.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "tests/models/flint/ops_test_support.hpp"

namespace {

using namespace gufo::models::flint;
using test::DeviceBuffer;

constexpr std::size_t kRotateBytes = std::size_t{768} << 20;

/// Mean milliseconds per call over `iters` calls after warm-up.
double TimeMs(const std::function<void(int)>& call, int iters = 40) {
  for (int i = 0; i < 4; ++i) {
    call(i);
  }
  test::CheckHip(hipDeviceSynchronize(), "warmup");
  hipEvent_t start;
  hipEvent_t stop;
  test::CheckHip(hipEventCreate(&start), "event");
  test::CheckHip(hipEventCreate(&stop), "event");
  test::CheckHip(hipEventRecord(start, nullptr), "record");
  for (int i = 0; i < iters; ++i) {
    call(i + 4);
  }
  test::CheckHip(hipEventRecord(stop, nullptr), "record");
  test::CheckHip(hipEventSynchronize(stop), "sync");
  float ms = 0.0f;
  test::CheckHip(hipEventElapsedTime(&ms, start, stop), "elapsed");
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  return ms / iters;
}

/// `copies` replicas of one random weight tensor.
struct Replicas {
  std::size_t bytes = 0;
  int copies = 0;
  DeviceBuffer<std::uint8_t> data;

  Replicas(rocm::WeightType type, std::size_t rows, int cols,
           std::mt19937& rng) {
    const std::vector<std::uint8_t> host =
        test::RandomWeights(type, rows, cols, rng);
    bytes = host.size();
    copies = static_cast<int>(std::max<std::size_t>(2, kRotateBytes / bytes));
    data.Resize(bytes * copies);
    for (int c = 0; c < copies; ++c) {
      test::CheckHip(hipMemcpy(data.get() + c * bytes, host.data(), bytes,
                               hipMemcpyHostToDevice),
                     "replicate");
    }
  }
  const void* At(int i) const { return data.get() + (i % copies) * bytes; }
};

void BenchProject(rocm::WeightType type, int m, int k, int rows,
                  std::mt19937& rng) {
  const Replicas w(type, m, k, rng);
  const std::vector<float> x =
      test::RandomFloats(std::size_t(rows) * k, -1.0f, 1.0f, rng);
  DeviceBuffer<float> dx{std::span<const float>(x)};
  DeviceBuffer<float> dy(std::size_t(rows) * m);
  const std::size_t sb = rocm::ProjectionScratchBytes(rows, k);
  DeviceBuffer<std::uint8_t> scratch(sb);
  const double ms = TimeMs([&](int i) {
    test::CheckHip(rocm::Project({w.At(i), type, m, k}, dx.get(), dy.get(),
                                 rows, scratch.get(), sb, nullptr),
                   "Project");
  });
  const double flops = 2.0 * m * k * rows;
  std::printf(
      "project %-6s %6dx%-5d rows=%-5d %8.1f us  %6.1f GB/s  %6.2f TFLOP/s\n",
      test::Name(type), m, k, rows, ms * 1e3, w.bytes / ms / 1e6,
      flops / ms / 1e9);
}

void BenchRouted(rocm::WeightType gu, rocm::WeightType dn, int tokens,
                 std::mt19937& rng) {
  constexpr int kExperts = 256;
  constexpr int kUsed = 6;
  const Replicas gate(gu, std::size_t(kExperts) * 512, 2048, rng);
  const Replicas up(gu, std::size_t(kExperts) * 512, 2048, rng);
  const Replicas down(dn, std::size_t(kExperts) * 2048, 512, rng);
  constexpr int kRoutes = 64;
  std::vector<std::int32_t> ids(std::size_t(kRoutes) * tokens * kUsed);
  std::vector<std::int32_t> all(kExperts);
  std::iota(all.begin(), all.end(), 0);
  for (std::size_t t = 0; t < std::size_t(kRoutes) * tokens; ++t) {
    std::shuffle(all.begin(), all.end(), rng);
    std::copy_n(all.begin(), kUsed, ids.begin() + t * kUsed);
  }
  DeviceBuffer<std::int32_t> dids{std::span<const std::int32_t>(ids)};
  const std::vector<float> x =
      test::RandomFloats(std::size_t(tokens) * 2048, -1.0f, 1.0f, rng);
  DeviceBuffer<float> dx{std::span<const float>(x)};
  DeviceBuffer<float> out(std::size_t(tokens) * kUsed * 2048);
  rocm::RoutedExperts e{{gate.At(0), gu, 512, 2048},
                        {up.At(0), gu, 512, 2048},
                        {down.At(0), dn, 2048, 512},
                        kExperts};
  const std::size_t sb = rocm::RoutedScratchBytes(e, tokens, kUsed);
  DeviceBuffer<std::uint8_t> scratch(sb);
  const double ms = TimeMs([&](int i) {
    e.gate.data = gate.At(i);
    e.up.data = up.At(i);
    e.down.data = down.At(i);
    test::CheckHip(
        rocm::RoutedForward(
            e, dx.get(), dids.get() + std::size_t(i % kRoutes) * tokens * kUsed,
            tokens, kUsed, out.get(), scratch.get(), sb, nullptr),
        "RoutedForward");
  });
  // Weight bytes of the distinct experts one call touches (6 per token for
  // a single token).
  const double per_expert =
      2.0 * gate.bytes / kExperts + double(down.bytes) / kExperts;
  const double touched = std::min<double>(kExperts, double(tokens) * kUsed);
  std::printf(
      "routed  %-6s/%-6s tokens=%-5d %8.1f us  %6.1f GB/s (%.0f experts)  "
      "%8.1f tok/s\n",
      test::Name(gu), test::Name(dn), tokens, ms * 1e3,
      per_expert * touched / ms / 1e6, touched, tokens / ms * 1e3);
}

void BenchDecodeAttention(int position, int window) {
  const int capacity = window > 0 ? 600 : position + 1;
  const std::size_t cache_bytes =
      std::size_t(capacity) * rocm::kKvHeads * rocm::kHeadDim * 2 * 2;
  const int copies =
      static_cast<int>(std::max<std::size_t>(2, kRotateBytes / cache_bytes));
  DeviceBuffer<std::uint16_t> kv(cache_bytes / 2 * copies);
  kv.Fill(0x22);
  std::vector<rocm::KvSequence> seqs;
  for (int c = 0; c < copies; ++c) {
    auto* base = reinterpret_cast<__half*>(kv.get()) + cache_bytes / 2 * c;
    seqs.push_back({base, base + cache_bytes / 4, capacity});
  }
  DeviceBuffer<rocm::KvSequence> dseqs{std::span<const rocm::KvSequence>(seqs)};
  std::vector<std::int32_t> row_seq(copies);
  std::iota(row_seq.begin(), row_seq.end(), 0);
  DeviceBuffer<std::int32_t> drs{std::span<const std::int32_t>(row_seq)};
  DeviceBuffer<std::int32_t> dpos{
      std::span<const std::int32_t>(std::vector<std::int32_t>{position})};
  std::mt19937 rng(1);
  const std::vector<float> q =
      test::RandomFloats(rocm::kQHeads * rocm::kHeadDim, -1.0f, 1.0f, rng);
  DeviceBuffer<float> dq{std::span<const float>(q)};
  DeviceBuffer<float> sinks(rocm::kQHeads);
  sinks.Fill(0);
  DeviceBuffer<float> out(q.size());
  const std::size_t sb = rocm::AttentionDecodeScratchBytes(1);
  DeviceBuffer<std::uint8_t> scratch(sb);
  const double ms = TimeMs([&](int i) {
    test::CheckHip(
        rocm::AttentionDecode(dq.get(), dpos.get(), drs.get() + i % copies,
                              dseqs.get(), sinks.get(), window, out.get(), 1,
                              scratch.get(), sb, nullptr),
        "AttentionDecode");
  });
  const int span = window > 0 ? std::min(position + 1, window) : position + 1;
  const double bytes = double(span) * rocm::kKvHeads * rocm::kHeadDim * 2 * 2;
  std::printf("decode attention %-7s context=%-7d %8.1f us  %6.1f GB/s\n",
              window > 0 ? "sliding" : "full", position + 1, ms * 1e3,
              bytes / ms / 1e6);
}

void BenchPrefillAttention(int first, int rows, int window) {
  const int capacity =
      window > 0 ? rocm::SlidingRingCapacity(rows) : first + rows;
  DeviceBuffer<std::uint16_t> k(std::size_t(capacity) * rocm::kKvHeads *
                                rocm::kHeadDim);
  DeviceBuffer<std::uint16_t> v(k.size());
  k.Fill(0x22);
  v.Fill(0x22);
  std::mt19937 rng(2);
  const std::vector<float> q = test::RandomFloats(
      std::size_t(rows) * rocm::kQHeads * rocm::kHeadDim, -1.0f, 1.0f, rng);
  DeviceBuffer<float> dq{std::span<const float>(q)};
  DeviceBuffer<float> sinks(rocm::kQHeads);
  sinks.Fill(0);
  DeviceBuffer<float> out(q.size());
  const rocm::KvSequence seq{reinterpret_cast<__half*>(k.get()),
                             reinterpret_cast<__half*>(v.get()), capacity};
  const double ms = TimeMs(
      [&](int) {
        test::CheckHip(rocm::AttentionPrefill(dq.get(), first, seq, sinks.get(),
                                              window, out.get(), rows, nullptr),
                       "AttentionPrefill");
      },
      10);
  double pairs = 0.0;
  for (int r = 0; r < rows; ++r) {
    const int p = first + r;
    pairs += window > 0 ? std::min(p + 1, window) : p + 1;
  }
  const double flops = pairs * rocm::kQHeads * rocm::kHeadDim * 4.0;
  std::printf(
      "prefill attention %-7s first=%-6d rows=%-5d %9.1f us  %6.2f TFLOP/s\n",
      window > 0 ? "sliding" : "full", first, rows, ms * 1e3, flops / ms / 1e9);
}

void BenchRouter(int tokens) {
  std::mt19937 rng(3);
  DeviceBuffer<std::uint16_t> w(std::size_t(256) * 2048);
  w.Fill(0x3c);
  const std::vector<float> x =
      test::RandomFloats(std::size_t(tokens) * 2048, -1.0f, 1.0f, rng);
  DeviceBuffer<float> dx{std::span<const float>(x)};
  DeviceBuffer<std::int32_t> ids(std::size_t(tokens) * 6);
  DeviceBuffer<float> weights(std::size_t(tokens) * 6);
  const std::size_t sb = rocm::RouterScratchBytes(tokens, 256);
  DeviceBuffer<std::uint8_t> scratch(sb);
  const double ms = TimeMs([&](int) {
    test::CheckHip(
        rocm::RouterTopK(w.get(), dx.get(), nullptr, 1.0f, tokens, 2048, 256, 6,
                         ids.get(), weights.get(), scratch.get(), sb, nullptr),
        "RouterTopK");
  });
  std::printf("router tokens=%-5d %8.1f us\n", tokens, ms * 1e3);
}

}  // namespace

int main() {
  try {
    std::mt19937 rng(5);
    std::puts("== one-token projections (decode)");
    const std::pair<int, int> shapes[] = {
        {4096, 2048}, {512, 2048},  {2048, 4096}, {1024, 2048},
        {2048, 1024}, {6144, 2048}, {2048, 6144}};
    for (const auto [m, k] : shapes) {
      BenchProject(rocm::WeightType::kQ8_0, m, k, 1, rng);
    }
    BenchProject(rocm::WeightType::kQ8_0, 150144, 2048, 1, rng);
    for (rocm::WeightType type :
         {rocm::WeightType::kQ4_K, rocm::WeightType::kQ5_K,
          rocm::WeightType::kQ6_K, rocm::WeightType::kIQ4_XS,
          rocm::WeightType::kIQ3_S}) {
      BenchProject(type, 4096, 2048, 1, rng);
      BenchProject(type, 1024, 2048, 1, rng);
      BenchProject(type, 2048, 1024, 1, rng);
    }
    std::puts("== routed experts");
    const std::pair<rocm::WeightType, rocm::WeightType> combos[] = {
        {rocm::WeightType::kIQ4_XS, rocm::WeightType::kQ5_K},
        {rocm::WeightType::kQ5_K, rocm::WeightType::kQ5_K},
        {rocm::WeightType::kIQ4_XS, rocm::WeightType::kIQ4_XS},
        {rocm::WeightType::kQ4_K, rocm::WeightType::kQ6_K},
        {rocm::WeightType::kIQ3_S, rocm::WeightType::kQ8_0}};
    for (const auto [gu, dn] : combos) {
      BenchRouted(gu, dn, 1, rng);
    }
    BenchRouted(rocm::WeightType::kIQ4_XS, rocm::WeightType::kQ5_K, 4, rng);
    BenchRouted(rocm::WeightType::kIQ4_XS, rocm::WeightType::kQ5_K, 512, rng);
    BenchRouted(rocm::WeightType::kIQ4_XS, rocm::WeightType::kQ5_K, 2048, rng);
    std::puts("== router");
    BenchRouter(1);
    BenchRouter(2048);
    std::puts("== decode attention");
    for (int context : {4096, 32768, 131072}) {
      BenchDecodeAttention(context - 1, 0);
    }
    BenchDecodeAttention(4095, rocm::kSlidingWindow);
    std::puts("== prefill");
    BenchProject(rocm::WeightType::kQ8_0, 4096, 2048, 2048, rng);
    BenchProject(rocm::WeightType::kQ8_0, 2048, 4096, 2048, rng);
    BenchProject(rocm::WeightType::kQ5_K, 4096, 2048, 2048, rng);
    BenchProject(rocm::WeightType::kQ8_0, 4096, 2048, 512, rng);
    BenchPrefillAttention(0, 2048, 0);
    BenchPrefillAttention(0, 2048, rocm::kSlidingWindow);
    BenchPrefillAttention(30720, 2048, 0);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "flint_kernels_bench: %s\n", e.what());
    return 1;
  }
}
