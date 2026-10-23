// Every decode operator's output row must be bitwise independent of how
// many rows (1..kVectorRows) share the launch: batched speculative or
// multi-sequence decode then replays single-row decode token for token.
// Each operator runs once over n rows and once per row alone; every row of
// the batch must equal its single-row result byte for byte.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "tests/models/flint/ops_test_support.hpp"

namespace {

using namespace gufo::models::flint;
using test::Checker;
using test::DeviceBuffer;

constexpr int kMaxRows = rocm::kVectorRows;

/// Runs `op(rows, offset_row)` for the batch and every single row, where
/// the operator reads input rows [offset_row, offset_row + rows) and writes
/// `out_floats` floats per row into `out`; compares the rows bitwise.
void CompareRows(const std::string& label, int out_floats,
                 const std::function<void(int rows, int first, float* out)>& op,
                 Checker& check) {
  for (int n = 2; n <= kMaxRows; ++n) {
    DeviceBuffer<float> batch(std::size_t(n) * out_floats);
    DeviceBuffer<float> single(out_floats);
    op(n, 0, batch.get());
    test::CheckHip(hipDeviceSynchronize(), label);
    const std::vector<float> b = batch.Download();
    int bad = 0;
    for (int r = 0; r < n; ++r) {
      op(1, r, single.get());
      test::CheckHip(hipDeviceSynchronize(), label);
      const std::vector<float> s = single.Download();
      bad += std::memcmp(s.data(), &b[std::size_t(r) * out_floats],
                         sizeof(float) * out_floats) != 0;
    }
    check.Expect(bad == 0, label + " rows=" + std::to_string(n) +
                               " differing rows=" + std::to_string(bad));
  }
}

void CheckProjections(std::mt19937& rng, Checker& check) {
  constexpr int k = 2048;
  const std::vector<float> x =
      test::RandomFloats(std::size_t(kMaxRows) * k, -2.0f, 2.0f, rng);
  DeviceBuffer<float> dx{std::span<const float>(x)};
  for (rocm::WeightType type : test::kAllTypes) {
    test::QuantTensor w(type, 1, 1000, k, rng);
    const std::size_t sb = rocm::ProjectionScratchBytes(kMaxRows, k);
    DeviceBuffer<std::uint8_t> scratch(sb);
    CompareRows(
        std::string("Project ") + test::Name(type), w.m,
        [&](int rows, int first, float* out) {
          test::CheckHip(
              rocm::Project(w.View(), dx.get() + std::size_t(first) * k, out,
                            rows, scratch.get(), sb, nullptr),
              "Project");
        },
        check);
  }
  // The vocabulary head's shape.
  test::QuantTensor head(rocm::WeightType::kQ8_0, 1, 150144, k, rng);
  const std::size_t sb = rocm::ProjectionScratchBytes(kMaxRows, k);
  DeviceBuffer<std::uint8_t> scratch(sb);
  CompareRows(
      "Project head", head.m,
      [&](int rows, int first, float* out) {
        test::CheckHip(
            rocm::Project(head.View(), dx.get() + std::size_t(first) * k, out,
                          rows, scratch.get(), sb, nullptr),
            "Project");
      },
      check);
}

/// Q8_1 rows of x produced by RmsNorm, one buffer row per input row.
struct NormedRows {
  DeviceBuffer<float> out;
  DeviceBuffer<std::uint8_t> q8;
  std::size_t row_bytes;
};

NormedRows Normalize(const DeviceBuffer<float>& x, const DeviceBuffer<float>& g,
                     int k) {
  NormedRows n{
      DeviceBuffer<float>(std::size_t(kMaxRows) * k),
      DeviceBuffer<std::uint8_t>(rocm::VectorActivationBytes(kMaxRows, k)),
      rocm::VectorActivationBytes(1, k)};
  test::CheckHip(rocm::RmsNorm(x.get(), g.get(), n.out.get(), n.q8.get(),
                               kMaxRows, k, 1e-6f, nullptr),
                 "RmsNorm(q8)");
  return n;
}

void CheckSwiGlu(std::mt19937& rng, Checker& check) {
  constexpr int k = 2048;
  constexpr int inner = 1024;
  const std::vector<float> x =
      test::RandomFloats(std::size_t(kMaxRows) * k, -2.0f, 2.0f, rng);
  const std::vector<float> g = test::RandomFloats(k, 0.5f, 1.5f, rng);
  DeviceBuffer<float> dx{std::span<const float>(x)};
  DeviceBuffer<float> dg{std::span<const float>(g)};
  const NormedRows n = Normalize(dx, dg, k);
  const std::size_t sb = rocm::SwiGluScratchBytes(kMaxRows, k, inner);
  DeviceBuffer<std::uint8_t> scratch(sb);
  for (const auto [gt, ut] :
       {std::pair{rocm::WeightType::kQ8_0, rocm::WeightType::kQ8_0},
        std::pair{rocm::WeightType::kQ5_K, rocm::WeightType::kQ5_K},
        std::pair{rocm::WeightType::kIQ4_XS, rocm::WeightType::kIQ4_XS},
        std::pair{rocm::WeightType::kQ4_K, rocm::WeightType::kQ6_K}}) {
    test::QuantTensor gate(gt, 1, inner, k, rng);
    test::QuantTensor up(ut, 1, inner, k, rng);
    const std::string name = std::string(test::Name(gt)) + "/" + test::Name(ut);
    CompareRows(
        "ProjectSwiGlu " + name, inner,
        [&](int rows, int first, float* out) {
          test::CheckHip(
              rocm::ProjectSwiGlu(gate.View(), up.View(),
                                  n.out.get() + std::size_t(first) * k, out,
                                  rows, scratch.get(), sb, nullptr),
              "ProjectSwiGlu");
        },
        check);
    CompareRows(
        "ProjectSwiGluQuantized " + name, inner,
        [&](int rows, int first, float* out) {
          test::CheckHip(
              rocm::ProjectSwiGluQuantized(
                  gate.View(), up.View(), n.q8.get() + first * n.row_bytes, out,
                  rows, scratch.get(), sb, nullptr),
              "ProjectSwiGluQuantized");
        },
        check);
  }
}

void CheckRouted(std::mt19937& rng, Checker& check) {
  constexpr int kExperts = 32;
  constexpr int kUsed = 6;
  constexpr int k = 2048;
  const std::vector<float> x =
      test::RandomFloats(std::size_t(kMaxRows) * k, -2.0f, 2.0f, rng);
  const std::vector<float> g = test::RandomFloats(k, 0.5f, 1.5f, rng);
  DeviceBuffer<float> dx{std::span<const float>(x)};
  DeviceBuffer<float> dg{std::span<const float>(g)};
  const NormedRows n = Normalize(dx, dg, k);
  std::vector<std::int32_t> ids(std::size_t(kMaxRows) * kUsed);
  std::vector<std::int32_t> all(kExperts);
  std::iota(all.begin(), all.end(), 0);
  for (int t = 0; t < kMaxRows; ++t) {
    std::shuffle(all.begin(), all.end(), rng);
    std::copy_n(all.begin(), kUsed, ids.begin() + t * kUsed);
  }
  DeviceBuffer<std::int32_t> did{std::span<const std::int32_t>(ids)};
  for (const auto [gu, dn] :
       {std::pair{rocm::WeightType::kIQ4_XS, rocm::WeightType::kQ5_K},
        std::pair{rocm::WeightType::kQ5_K, rocm::WeightType::kQ5_K},
        std::pair{rocm::WeightType::kQ4_K, rocm::WeightType::kQ6_K},
        std::pair{rocm::WeightType::kIQ3_S, rocm::WeightType::kQ8_0},
        std::pair{rocm::WeightType::kQ8_0, rocm::WeightType::kIQ4_XS}}) {
    test::QuantTensor gate(gu, kExperts, 512, k, rng);
    test::QuantTensor up(gu, kExperts, 512, k, rng);
    test::QuantTensor down(dn, kExperts, k, 512, rng);
    const rocm::RoutedExperts e{gate.View(), up.View(), down.View(), kExperts};
    const std::size_t sb = rocm::RoutedScratchBytes(e, kMaxRows, kUsed);
    DeviceBuffer<std::uint8_t> scratch(sb);
    const std::string name = std::string(test::Name(gu)) + "/" + test::Name(dn);
    CompareRows(
        "RoutedForward " + name, kUsed * k,
        [&](int rows, int first, float* out) {
          test::CheckHip(
              rocm::RoutedForward(e, n.out.get() + std::size_t(first) * k,
                                  did.get() + first * kUsed, rows, kUsed, out,
                                  scratch.get(), sb, nullptr),
              "RoutedForward");
        },
        check);
    CompareRows(
        "RoutedForwardQuantized " + name, kUsed * k,
        [&](int rows, int first, float* out) {
          test::CheckHip(rocm::RoutedForwardQuantized(
                             e, n.q8.get() + first * n.row_bytes,
                             did.get() + first * kUsed, rows, kUsed, out,
                             scratch.get(), sb, nullptr),
                         "RoutedForwardQuantized");
        },
        check);
  }
}

void CheckSmallOps(std::mt19937& rng, Checker& check) {
  constexpr int k = 2048;
  const std::vector<float> x =
      test::RandomFloats(std::size_t(kMaxRows) * k, -2.0f, 2.0f, rng);
  const std::vector<float> d =
      test::RandomFloats(std::size_t(kMaxRows) * k, -2.0f, 2.0f, rng);
  const std::vector<float> g = test::RandomFloats(k, 0.5f, 1.5f, rng);
  DeviceBuffer<float> dx{std::span<const float>(x)};
  DeviceBuffer<float> dd{std::span<const float>(d)};
  DeviceBuffer<float> dg{std::span<const float>(g)};
  CompareRows(
      "RmsNorm", k,
      [&](int rows, int first, float* out) {
        test::CheckHip(rocm::RmsNorm(dx.get() + std::size_t(first) * k,
                                     dg.get(), out, rows, k, 1e-6f, nullptr),
                       "RmsNorm");
      },
      check);
  CompareRows(
      "AddRmsNorm", 2 * k,
      [&](int rows, int first, float* out) {
        // out holds [rows][k] normalized then [rows][k] residual.
        DeviceBuffer<float> residual(std::size_t(rows) * k);
        test::CheckHip(
            hipMemcpy(residual.get(), dx.get() + std::size_t(first) * k,
                      residual.bytes(), hipMemcpyDeviceToDevice),
            "copy");
        DeviceBuffer<float> normed(std::size_t(rows) * k);
        DeviceBuffer<std::uint8_t> q8(rocm::VectorActivationBytes(rows, k));
        test::CheckHip(
            rocm::AddRmsNorm(residual.get(), dd.get() + std::size_t(first) * k,
                             dg.get(), normed.get(), q8.get(), rows, k, 1e-6f,
                             nullptr),
            "AddRmsNorm");
        // Interleave per row so CompareRows sees [rows][2k].
        for (int r = 0; r < rows; ++r) {
          test::CheckHip(hipMemcpy(out + std::size_t(r) * 2 * k,
                                   normed.get() + std::size_t(r) * k,
                                   k * sizeof(float), hipMemcpyDeviceToDevice),
                         "copy");
          test::CheckHip(hipMemcpy(out + std::size_t(r) * 2 * k + k,
                                   residual.get() + std::size_t(r) * k,
                                   k * sizeof(float), hipMemcpyDeviceToDevice),
                         "copy");
        }
      },
      check);
  // Router.
  std::vector<std::uint16_t> w(std::size_t(256) * k);
  for (auto& v : w) {
    v = static_cast<std::uint16_t>(0x3c00 + rng() % 0x0800);
  }
  DeviceBuffer<std::uint16_t> dw{std::span<const std::uint16_t>(w)};
  DeviceBuffer<std::uint8_t> rs(rocm::RouterScratchBytes(kMaxRows, 256));
  CompareRows(
      "RouterTopK", 12,
      [&](int rows, int first, float* out) {
        DeviceBuffer<std::int32_t> ids(std::size_t(rows) * 6);
        DeviceBuffer<float> weights(std::size_t(rows) * 6);
        test::CheckHip(
            rocm::RouterTopK(dw.get(), dx.get() + std::size_t(first) * k,
                             nullptr, 1.0f, rows, k, 256, 6, ids.get(),
                             weights.get(), rs.get(), rs.bytes(), nullptr),
            "RouterTopK");
        for (int r = 0; r < rows; ++r) {
          test::CheckHip(hipMemcpy(out + r * 12, ids.get() + r * 6, 24,
                                   hipMemcpyDeviceToDevice),
                         "copy");
          test::CheckHip(hipMemcpy(out + r * 12 + 6, weights.get() + r * 6, 24,
                                   hipMemcpyDeviceToDevice),
                         "copy");
        }
      },
      check);
  // MoE combine.
  const std::vector<float> e =
      test::RandomFloats(std::size_t(kMaxRows) * 6 * k, -2.0f, 2.0f, rng);
  const std::vector<float> wts =
      test::RandomFloats(std::size_t(kMaxRows) * 6, 0.0f, 0.4f, rng);
  DeviceBuffer<float> de{std::span<const float>(e)};
  DeviceBuffer<float> dwt{std::span<const float>(wts)};
  CompareRows(
      "MoeCombine", k,
      [&](int rows, int first, float* out) {
        test::CheckHip(rocm::MoeCombine(de.get() + std::size_t(first) * 6 * k,
                                        dwt.get() + first * 6,
                                        dd.get() + std::size_t(first) * k,
                                        dx.get() + std::size_t(first) * k, out,
                                        rows, 6, k, nullptr),
                       "MoeCombine");
      },
      check);
  // Embedding.
  const std::vector<std::uint8_t> table =
      test::RandomWeights(rocm::WeightType::kQ8_0, 100, k, rng);
  DeviceBuffer<std::uint8_t> dt{std::span<const std::uint8_t>(table)};
  const std::vector<std::int32_t> toks = {5, 99, 0, 17, 5, 42, 63, 8};
  DeviceBuffer<std::int32_t> dtok{std::span<const std::int32_t>(toks)};
  CompareRows(
      "EmbedQ8_0", k,
      [&](int rows, int first, float* out) {
        test::CheckHip(rocm::EmbedQ8_0(dt.get(), dtok.get() + first, rows, k,
                                       out, nullptr),
                       "EmbedQ8_0");
      },
      check);
}

void CheckAttention(std::mt19937& rng, Checker& check) {
  constexpr int kWidth = 5120;  // fused q|k|v rows
  constexpr int kCap = 4096;
  const std::vector<float> qkv =
      test::RandomFloats(std::size_t(kMaxRows) * kWidth, -2.0f, 2.0f, rng);
  const std::vector<float> norms = test::RandomFloats(256, 0.5f, 1.5f, rng);
  const std::vector<float> sinks = test::RandomFloats(32, -1.0f, 1.0f, rng);
  DeviceBuffer<float> dqkv{std::span<const float>(qkv)};
  DeviceBuffer<float> dn{std::span<const float>(norms)};
  DeviceBuffer<float> ds{std::span<const float>(sinks)};
  // Each row is its own sequence at its own position; caches hold random
  // FP16 history.
  std::vector<std::uint16_t> kv(std::size_t(kMaxRows) * 2 * 4 * kCap * 128);
  for (auto& v : kv) {
    v = test::FloatToHalfBits(
        std::uniform_real_distribution<float>(-1, 1)(rng));
  }
  DeviceBuffer<std::uint16_t> dkv{std::span<const std::uint16_t>(kv)};
  std::vector<rocm::KvSequence> seqs;
  for (int r = 0; r < kMaxRows; ++r) {
    auto* base = reinterpret_cast<__half*>(dkv.get()) +
                 std::size_t(r) * 2 * 4 * kCap * 128;
    seqs.push_back({base, base + std::size_t(4) * kCap * 128, kCap});
  }
  DeviceBuffer<rocm::KvSequence> dseqs{std::span<const rocm::KvSequence>(seqs)};
  const std::vector<std::int32_t> positions = {7, 4095, 513, 1000,
                                               3, 2048, 600, 77};
  std::vector<std::int32_t> row_seq(kMaxRows);
  std::iota(row_seq.begin(), row_seq.end(), 0);
  DeviceBuffer<std::int32_t> dpos{std::span<const std::int32_t>(positions)};
  DeviceBuffer<std::int32_t> drs{std::span<const std::int32_t>(row_seq)};
  const std::size_t ab = rocm::AttentionDecodeScratchBytes(kMaxRows);
  DeviceBuffer<std::uint8_t> scratch(ab);
  for (bool rope : {true, false}) {
    CompareRows(
        std::string("PrepareAttention rope=") + (rope ? "1" : "0"), 4096,
        [&](int rows, int first, float* out) {
          test::CheckHip(rocm::PrepareAttention(
                             dqkv.get() + std::size_t(first) * kWidth, kWidth,
                             out, dn.get(), dn.get() + 128, dpos.get() + first,
                             drs.get() + first, dseqs.get(), rows, rope,
                             10000.0f, 1e-6f, nullptr),
                         "PrepareAttention");
        },
        check);
  }
  // q rows from the last PrepareAttention batch.
  const std::vector<float> q =
      test::RandomFloats(std::size_t(kMaxRows) * 4096, -3.0f, 3.0f, rng);
  DeviceBuffer<float> dq{std::span<const float>(q)};
  for (int window : {0, rocm::kSlidingWindow}) {
    CompareRows(
        "AttentionDecode window=" + std::to_string(window), 4096 + 128 * 9,
        [&](int rows, int first, float* out) {
          DeviceBuffer<std::uint8_t> q8(
              rocm::VectorActivationBytes(rows, 4096));
          DeviceBuffer<float> o(std::size_t(rows) * 4096);
          test::CheckHip(
              rocm::AttentionDecode(dq.get() + std::size_t(first) * 4096,
                                    dpos.get() + first, drs.get() + first,
                                    dseqs.get(), ds.get(), window, o.get(),
                                    q8.get(), rows, scratch.get(), ab, nullptr),
              "AttentionDecode");
          // Row layout: 4096 floats then the row's 128 Q8_1 blocks
          // (4608 bytes = 1152 floats).
          for (int r = 0; r < rows; ++r) {
            float* dst = out + std::size_t(r) * (4096 + 128 * 9);
            test::CheckHip(
                hipMemcpy(dst, o.get() + std::size_t(r) * 4096,
                          4096 * sizeof(float), hipMemcpyDeviceToDevice),
                "copy");
            test::CheckHip(
                hipMemcpy(dst + 4096, q8.get() + std::size_t(r) * 128 * 36,
                          128 * 36, hipMemcpyDeviceToDevice),
                "copy");
          }
        },
        check);
  }
}

}  // namespace

int main() {
  try {
    Checker check;
    std::mt19937 rng(8);
    CheckSmallOps(rng, check);
    CheckAttention(rng, check);
    CheckProjections(rng, check);
    CheckSwiGlu(rng, check);
    CheckRouted(rng, check);
    return check.Finish("flint.batch_invariance_ops");
  } catch (const std::exception& e) {
    std::cerr << "flint.batch_invariance_ops: " << e.what() << '\n';
    return 1;
  }
}
