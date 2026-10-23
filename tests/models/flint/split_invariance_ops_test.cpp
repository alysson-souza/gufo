// Prefill operators must give every row bitwise the same result whether the
// rows run in one call or are split into chunks at any point (1, 7, 8, 9,
// 37, n - 1): continuing a prompt then reproduces a single pass exactly.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include "tests/models/flint/ops_test_support.hpp"

namespace {

using namespace gufo::models::flint;
using test::Checker;
using test::DeviceBuffer;

constexpr int kRows = 100;

/// Runs `op(first_row, rows, out)` once over all rows and then split at
/// each point; every split must reproduce the single pass byte for byte.
/// The op writes `out_floats` floats per row into out (its rows only).
void CompareSplits(
    const std::string& label, int out_floats,
    const std::function<void(int first, int rows, float* out)>& op,
    Checker& check) {
  const std::size_t n = std::size_t(kRows) * out_floats;
  DeviceBuffer<float> whole(n);
  op(0, kRows, whole.get());
  test::CheckHip(hipDeviceSynchronize(), label);
  const std::vector<float> ref = whole.Download();
  for (int split : {1, 7, 8, 9, 37, kRows - 1}) {
    DeviceBuffer<float> parts(n);
    op(0, split, parts.get());
    op(split, kRows - split, parts.get() + std::size_t(split) * out_floats);
    test::CheckHip(hipDeviceSynchronize(), label);
    const std::vector<float> got = parts.Download();
    int bad = 0;
    for (int r = 0; r < kRows; ++r) {
      bad += std::memcmp(&got[std::size_t(r) * out_floats],
                         &ref[std::size_t(r) * out_floats],
                         sizeof(float) * out_floats) != 0;
    }
    check.Expect(bad == 0, label + " split=" + std::to_string(split) +
                               " differing rows=" + std::to_string(bad));
  }
}

void CheckProjections(std::mt19937& rng, Checker& check) {
  constexpr int k = 2048;
  const std::vector<float> x =
      test::RandomFloats(std::size_t(kRows) * k, -2.0f, 2.0f, rng);
  DeviceBuffer<float> dx{std::span<const float>(x)};
  for (rocm::WeightType type : test::kAllTypes) {
    test::QuantTensor w(type, 1, 520, k, rng);
    const std::size_t sb = rocm::MatrixProjectionScratchBytes(kRows, k);
    DeviceBuffer<std::uint8_t> scratch(sb);
    const std::string name = std::string("ProjectMatrix ") + test::Name(type);
    CompareSplits(
        name, w.m,
        [&](int first, int rows, float* out) {
          test::CheckHip(
              rocm::ProjectMatrix(w.View(), dx.get() + std::size_t(first) * k,
                                  out, rows, scratch.get(), sb, nullptr),
              "ProjectMatrix");
        },
        check);
    // Above kVectorRows the matrix tier is what Project runs.
    DeviceBuffer<float> a(std::size_t(kRows) * w.m);
    DeviceBuffer<float> b(a.size());
    test::CheckHip(rocm::Project(w.View(), dx.get(), a.get(), kRows,
                                 scratch.get(), sb, nullptr),
                   "Project");
    test::CheckHip(rocm::ProjectMatrix(w.View(), dx.get(), b.get(), kRows,
                                       scratch.get(), sb, nullptr),
                   "ProjectMatrix");
    check.Expect(a.Download() == b.Download(), name + " != Project");
  }
  for (const auto [gt, ut] :
       {std::pair{rocm::WeightType::kQ8_0, rocm::WeightType::kQ8_0},
        std::pair{rocm::WeightType::kQ5_K, rocm::WeightType::kQ6_K}}) {
    test::QuantTensor gate(gt, 1, 1024, k, rng);
    test::QuantTensor up(ut, 1, 1024, k, rng);
    const std::size_t sb = rocm::MatrixSwiGluScratchBytes(kRows, k, 1024);
    DeviceBuffer<std::uint8_t> scratch(sb);
    CompareSplits(
        std::string("ProjectSwiGluMatrix ") + test::Name(gt) + "/" +
            test::Name(ut),
        1024,
        [&](int first, int rows, float* out) {
          test::CheckHip(
              rocm::ProjectSwiGluMatrix(gate.View(), up.View(),
                                        dx.get() + std::size_t(first) * k, out,
                                        rows, scratch.get(), sb, nullptr),
              "ProjectSwiGluMatrix");
        },
        check);
  }
}

void CheckRouted(std::mt19937& rng, Checker& check) {
  constexpr int kExperts = 32;
  constexpr int kUsed = 6;
  constexpr int k = 2048;
  const std::vector<float> x =
      test::RandomFloats(std::size_t(kRows) * k, -2.0f, 2.0f, rng);
  DeviceBuffer<float> dx{std::span<const float>(x)};
  std::vector<std::int32_t> ids(std::size_t(kRows) * kUsed);
  std::vector<std::int32_t> all(kExperts);
  std::iota(all.begin(), all.end(), 0);
  for (int t = 0; t < kRows; ++t) {
    std::shuffle(all.begin(), all.end(), rng);
    std::copy_n(all.begin(), kUsed, ids.begin() + t * kUsed);
  }
  DeviceBuffer<std::int32_t> did{std::span<const std::int32_t>(ids)};
  for (const auto [gu, dn] :
       {std::pair{rocm::WeightType::kIQ4_XS, rocm::WeightType::kQ5_K},
        std::pair{rocm::WeightType::kQ4_K, rocm::WeightType::kQ6_K},
        std::pair{rocm::WeightType::kIQ3_S, rocm::WeightType::kQ8_0}}) {
    test::QuantTensor gate(gu, kExperts, 512, k, rng);
    test::QuantTensor up(gu, kExperts, 512, k, rng);
    test::QuantTensor down(dn, kExperts, k, 512, rng);
    const rocm::RoutedExperts e{gate.View(), up.View(), down.View(), kExperts};
    const std::size_t sb = rocm::MatrixRoutedScratchBytes(e, kRows, kUsed);
    DeviceBuffer<std::uint8_t> scratch(sb);
    CompareSplits(
        std::string("RoutedForwardMatrix ") + test::Name(gu) + "/" +
            test::Name(dn),
        kUsed * k,
        [&](int first, int rows, float* out) {
          test::CheckHip(
              rocm::RoutedForwardMatrix(e, dx.get() + std::size_t(first) * k,
                                        did.get() + first * kUsed, rows, kUsed,
                                        out, scratch.get(), sb, nullptr),
              "RoutedForwardMatrix");
        },
        check);
  }
}

void CheckAttention(std::mt19937& rng, Checker& check) {
  const std::vector<float> q =
      test::RandomFloats(std::size_t(kRows) * 4096, -3.0f, 3.0f, rng);
  const std::vector<float> sinks = test::RandomFloats(32, -1.0f, 1.0f, rng);
  DeviceBuffer<float> dq{std::span<const float>(q)};
  DeviceBuffer<float> ds{std::span<const float>(sinks)};
  for (const auto [first, capacity, window] :
       {std::tuple{0, 4096, 0}, std::tuple{50, 4096, 0},
        std::tuple{1003, 4096, 0}, std::tuple{50, 4096, rocm::kSlidingWindow},
        std::tuple{2000, rocm::SlidingRingCapacity(kRows),
                   rocm::kSlidingWindow}}) {
    // A cache whose slots hold the latest positions <= first + kRows - 1.
    const std::size_t n = std::size_t(4) * capacity * 128;
    std::vector<std::uint16_t> k(n);
    std::vector<std::uint16_t> v(n);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
      k[i] = test::FloatToHalfBits(dist(rng));
      v[i] = test::FloatToHalfBits(dist(rng));
    }
    DeviceBuffer<std::uint16_t> dk{std::span<const std::uint16_t>(k)};
    DeviceBuffer<std::uint16_t> dv{std::span<const std::uint16_t>(v)};
    const rocm::KvSequence seq{reinterpret_cast<__half*>(dk.get()),
                               reinterpret_cast<__half*>(dv.get()), capacity};
    CompareSplits(
        "AttentionPrefill first=" + std::to_string(first) +
            " window=" + std::to_string(window),
        4096,
        [&](int row, int rows, float* out) {
          test::CheckHip(rocm::AttentionPrefill(
                             dq.get() + std::size_t(row) * 4096, first + row,
                             seq, ds.get(), window, out, rows, nullptr),
                         "AttentionPrefill");
        },
        check);
  }
}

}  // namespace

int main() {
  try {
    Checker check;
    std::mt19937 rng(37);
    CheckAttention(rng, check);
    CheckProjections(rng, check);
    CheckRouted(rng, check);
    return check.Finish("flint.split_invariance_ops");
  } catch (const std::exception& e) {
    std::cerr << "flint.split_invariance_ops: " << e.what() << '\n';
    return 1;
  }
}
