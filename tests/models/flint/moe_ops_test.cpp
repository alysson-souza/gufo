// Routed experts and the MoE combine against core::quant decoding with
// double accumulation.
//
// Tolerances. Each routed projection stage gets Q8_1-exact activations, so
// it is held to the dense projection bound (QuantTensor::Tolerance: 1e-4 of
// sum|w x|, plus the FP16 sub-block scales of Q4_K/Q5_K MMQ tiles). The
// complete expert (also on exact inputs) propagates the gate/up bounds eg,
// eu through h = silu(g) u (|silu'| < 1.1) and re-quantizes h to Q8_1 before
// the down projection: each value moves by at most 0.562 d_b (half a code
// plus the FP16 rounding of d_b = amax_b / 127 times 127 codes), weighted by
// the part of the down row that multiplies codes. The combine has a fully
// specified FP32 order and must match its host emulation bit for bit.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "tests/models/flint/ops_test_support.hpp"

namespace {

using namespace gufo::models::flint;
using test::Checker;
using test::DeviceBuffer;
using test::QuantTensor;

constexpr int kUsed = 6;
constexpr int kHidden = 2048;
constexpr int kInner = 512;

enum class Routing { kRandom, kShared, kDisjoint };

const char* Name(Routing r) {
  switch (r) {
    case Routing::kRandom:
      return "random";
    case Routing::kShared:
      return "shared";
    case Routing::kDisjoint:
      return "disjoint";
  }
  return "?";
}

std::vector<std::int32_t> MakeIds(Routing routing, int tokens, int experts,
                                  std::mt19937& rng) {
  std::vector<std::int32_t> ids(std::size_t(tokens) * kUsed);
  std::vector<std::int32_t> all(experts);
  std::iota(all.begin(), all.end(), 0);
  std::vector<std::int32_t> shared = all;
  std::shuffle(shared.begin(), shared.end(), rng);
  for (int t = 0; t < tokens; ++t) {
    std::vector<std::int32_t> pick;
    switch (routing) {
      case Routing::kRandom:
        pick = all;
        std::shuffle(pick.begin(), pick.end(), rng);
        break;
      case Routing::kShared:
        pick = shared;
        break;
      case Routing::kDisjoint:
        for (int j = 0; j < kUsed; ++j) {
          pick.push_back((t * kUsed + j) % experts);
        }
        break;
    }
    std::copy_n(pick.begin(), kUsed, ids.begin() + std::size_t(t) * kUsed);
  }
  return ids;
}

std::string Label(const QuantTensor& w, int tokens, Routing r,
                  const char* what) {
  std::ostringstream os;
  os << test::Name(w.type) << ' ' << w.m << 'x' << w.k << " E=" << w.groups
     << " tokens=" << tokens << ' ' << Name(r) << ' ' << what;
  return os.str();
}

std::vector<float> RunRoutedProject(QuantTensor& w, const std::vector<float>& x,
                                    bool per_slot,
                                    const std::vector<std::int32_t>& ids,
                                    int tokens) {
  DeviceBuffer<float> dx{std::span<const float>(x)};
  DeviceBuffer<std::int32_t> did{std::span<const std::int32_t>(ids)};
  DeviceBuffer<float> out(std::size_t(tokens) * kUsed * w.m);
  out.Fill(0xff);
  const std::size_t bytes = rocm::RoutedProjectScratchBytes(
      w.View(), w.groups, tokens, kUsed, per_slot);
  DeviceBuffer<std::uint8_t> scratch(bytes);
  test::CheckHip(rocm::RoutedProject(w.View(), w.groups, dx.get(), per_slot,
                                     did.get(), tokens, kUsed, out.get(),
                                     scratch.get(), bytes, nullptr),
                 "RoutedProject");
  test::CheckHip(hipDeviceSynchronize(), "sync");
  return out.Download();
}

/// One routed projection stage with exact activations.
void CheckStage(QuantTensor& w, bool per_slot, Routing routing, int tokens,
                std::mt19937& rng, Checker& check) {
  const std::vector<std::int32_t> ids = MakeIds(routing, tokens, w.groups, rng);
  const std::vector<float> x = test::ExactActivations(
      std::size_t(tokens) * (per_slot ? kUsed : 1), w.k, rng);
  const std::vector<float> y = RunRoutedProject(w, x, per_slot, ids, tokens);
  int bad = 0;
  for (int t = 0; t < tokens; ++t) {
    for (int j = 0; j < kUsed; ++j) {
      const int e = ids[std::size_t(t) * kUsed + j];
      const float* xr =
          &x[(per_slot ? std::size_t(t) * kUsed + j : std::size_t(t)) * w.k];
      for (int o = 0; o < w.m; ++o) {
        const test::Dot ref = test::DotRow(w.Row(e, o), xr, w.k);
        const double got = y[(std::size_t(t) * kUsed + j) * w.m + o];
        const double tol =
            w.Tolerance(e, o, xr, tokens > rocm::kVectorRows, ref);
        if (!(std::fabs(got - ref.value) <= tol) && bad++ < 3) {
          std::ostringstream os;
          os << Label(w, tokens, routing, "stage") << " t=" << t << " j=" << j
             << " e=" << e << " o=" << o << " got=" << got
             << " ref=" << ref.value << " tol=" << tol;
          check.Expect(false, os.str());
        }
      }
    }
  }
  check.Expect(bad == 0, Label(w, tokens, routing,
                               per_slot ? "down stage" : "gate stage") +
                             " mismatches=" + std::to_string(bad));
}

double Silu(double g) {
  return g / (1.0 + std::exp(-g));
}

/// The complete expert against the double pipeline.
/// `lift` scales the (still Q8_1-exact) activations by 2^lift; lift 5 makes
/// silu(g) * u values reach ~1e5 and their 32-value block sums pass FP16
/// range (as real massive activations do), which the down projection's
/// quantized block sums must survive. (Q8_1 itself needs |value| < 8.3e6:
/// its FP16 scale is amax / 127.)
void CheckForward(QuantTensor& gate, QuantTensor& up, QuantTensor& down,
                  Routing routing, int tokens, std::mt19937& rng,
                  Checker& check, int lift = 0) {
  const int experts = gate.groups;
  const std::vector<std::int32_t> ids = MakeIds(routing, tokens, experts, rng);
  std::vector<float> x = test::ExactActivations(tokens, kHidden, rng);
  for (float& v : x) {
    v = std::ldexp(v, lift);
  }
  const rocm::RoutedExperts e{gate.View(), up.View(), down.View(), experts};
  DeviceBuffer<float> dx{std::span<const float>(x)};
  DeviceBuffer<std::int32_t> did{std::span<const std::int32_t>(ids)};
  DeviceBuffer<float> out(std::size_t(tokens) * kUsed * kHidden);
  out.Fill(0xff);
  const std::size_t bytes = rocm::RoutedScratchBytes(e, tokens, kUsed);
  DeviceBuffer<std::uint8_t> scratch(bytes);
  test::CheckHip(rocm::RoutedForward(e, dx.get(), did.get(), tokens, kUsed,
                                     out.get(), scratch.get(), bytes, nullptr),
                 "RoutedForward");
  test::CheckHip(hipDeviceSynchronize(), "sync");
  const std::vector<float> y = out.Download();
  const bool mmq = tokens > rocm::kVectorRows;
  double max_block_sum = 0.0;
  int bad = 0;
  std::vector<float> h(kInner);
  std::vector<double> hd(kInner);
  std::vector<double> dh(kInner);
  std::vector<double> requant(kInner);
  for (int t = 0; t < tokens; ++t) {
    const float* xr = &x[std::size_t(t) * kHidden];
    for (int j = 0; j < kUsed; ++j) {
      const int ex = ids[std::size_t(t) * kUsed + j];
      for (int i = 0; i < kInner; ++i) {
        const test::Dot g = test::DotRow(gate.Row(ex, i), xr, kHidden);
        const test::Dot u = test::DotRow(up.Row(ex, i), xr, kHidden);
        const double eg = gate.Tolerance(ex, i, xr, mmq, g);
        const double eu = up.Tolerance(ex, i, xr, mmq, u);
        hd[i] = Silu(g.value) * u.value;
        h[i] = static_cast<float>(hd[i]);
        dh[i] = 1.1 * (std::fabs(u.value) + eu) * eg +
                (std::fabs(Silu(g.value)) + 1.1 * eg) * eu;
      }
      for (int b = 0; b < kInner / 32; ++b) {
        double amax = 0.0;
        double block_sum = 0.0;
        for (int i = b * 32; i < b * 32 + 32; ++i) {
          amax = std::max(amax, std::fabs(hd[i]) + dh[i]);
          block_sum += hd[i];
        }
        max_block_sum = std::max(max_block_sum, std::fabs(block_sum));
        for (int i = b * 32; i < b * 32 + 32; ++i) {
          requant[i] = 0.562 * amax / 127.0 + dh[i];
        }
      }
      for (int o = 0; o < kHidden; ++o) {
        const float* wr = down.Row(ex, o);
        const float* ws = down.ScaleRow(ex, o);
        double ref = 0.0;
        double tol = 0.0;
        for (int i = 0; i < kInner; ++i) {
          ref += static_cast<double>(wr[i]) * hd[i];
          tol += std::fabs(double(ws[i])) * requant[i] +
                 std::fabs(double(wr[i]) - ws[i]) * dh[i];
        }
        tol += down.Tolerance(ex, o, h.data(), mmq,
                              test::DotRow(wr, h.data(), kInner));
        const double got = y[(std::size_t(t) * kUsed + j) * kHidden + o];
        if (!(std::fabs(got - ref) <= tol) && bad++ < 3) {
          std::ostringstream os;
          os << Label(gate, tokens, routing, "forward") << " t=" << t
             << " j=" << j << " o=" << o << " got=" << got << " ref=" << ref
             << " tol=" << tol;
          check.Expect(false, os.str());
        }
      }
    }
  }
  if (lift > 0) {
    check.Expect(
        max_block_sum > 65504.0,
        Label(gate, tokens, routing, "lifted sums stay in FP16 range"));
  }
  check.Expect(bad == 0, Label(gate, tokens, routing, "forward") +
                             std::string(" lift=") + std::to_string(lift) +
                             " mismatches=" + std::to_string(bad));
}

/// RoutedForwardQuantized on RmsNorm's Q8_1 rows equals RoutedForward on its
/// float rows bit for bit.
void CheckForwardQuantized(QuantTensor& gate, QuantTensor& up,
                           QuantTensor& down, std::mt19937& rng,
                           Checker& check) {
  const rocm::RoutedExperts e{gate.View(), up.View(), down.View(), gate.groups};
  for (int tokens : {1, 3, 8}) {
    const std::vector<std::int32_t> ids =
        MakeIds(Routing::kRandom, tokens, gate.groups, rng);
    const std::vector<float> x =
        test::RandomFloats(std::size_t(tokens) * kHidden, -2.0f, 2.0f, rng);
    const std::vector<float> w = test::RandomFloats(kHidden, 0.5f, 1.5f, rng);
    DeviceBuffer<float> dx{std::span<const float>(x)};
    DeviceBuffer<float> dw{std::span<const float>(w)};
    DeviceBuffer<float> xn(x.size());
    DeviceBuffer<std::uint8_t> q8(rocm::VectorActivationBytes(tokens, kHidden));
    test::CheckHip(rocm::RmsNorm(dx.get(), dw.get(), xn.get(), q8.get(), tokens,
                                 kHidden, 1e-6f, nullptr),
                   "RmsNorm(q8)");
    DeviceBuffer<std::int32_t> did{std::span<const std::int32_t>(ids)};
    DeviceBuffer<float> a(std::size_t(tokens) * kUsed * kHidden);
    DeviceBuffer<float> b(a.size());
    const std::size_t sb = rocm::RoutedScratchBytes(e, tokens, kUsed);
    DeviceBuffer<std::uint8_t> scratch(sb);
    test::CheckHip(rocm::RoutedForward(e, xn.get(), did.get(), tokens, kUsed,
                                       a.get(), scratch.get(), sb, nullptr),
                   "RoutedForward");
    test::CheckHip(
        rocm::RoutedForwardQuantized(e, q8.get(), did.get(), tokens, kUsed,
                                     b.get(), scratch.get(), sb, nullptr),
        "RoutedForwardQuantized");
    check.Expect(a.Download() == b.Download(),
                 Label(gate, tokens, Routing::kRandom, "forward quantized"));
  }
}

/// MoeCombine against its documented FP32 order (exact) and double.
void CheckCombine(std::mt19937& rng, Checker& check) {
  for (int tokens : {1, 3, 9, 130}) {
    for (int variant = 0; variant < 4; ++variant) {
      const bool with_shared = (variant & 1) != 0;
      const bool with_residual = (variant & 2) != 0;
      const std::size_t n = std::size_t(tokens) * kHidden;
      const std::vector<float> experts =
          test::RandomFloats(n * kUsed, -3.0f, 3.0f, rng);
      const std::vector<float> weights =
          test::RandomFloats(std::size_t(tokens) * kUsed, 0.0f, 0.4f, rng);
      const std::vector<float> shared = test::RandomFloats(n, -2.0f, 2.0f, rng);
      const std::vector<float> residual =
          test::RandomFloats(n, -5.0f, 5.0f, rng);
      DeviceBuffer<float> de{std::span<const float>(experts)};
      DeviceBuffer<float> dw{std::span<const float>(weights)};
      DeviceBuffer<float> ds{std::span<const float>(shared)};
      DeviceBuffer<float> dr{std::span<const float>(residual)};
      DeviceBuffer<float> dout(n);
      // In place over the residual, as the engine uses it.
      float* out = with_residual ? dr.get() : dout.get();
      test::CheckHip(
          rocm::MoeCombine(de.get(), dw.get(), with_shared ? ds.get() : nullptr,
                           with_residual ? dr.get() : nullptr, out, tokens,
                           kUsed, kHidden, nullptr),
          "MoeCombine");
      test::CheckHip(hipDeviceSynchronize(), "sync");
      const std::vector<float> got =
          with_residual ? dr.Download() : dout.Download();
      int exact_bad = 0;
      int tol_bad = 0;
      for (int t = 0; t < tokens; ++t) {
        for (int d = 0; d < kHidden; ++d) {
          const std::size_t i = std::size_t(t) * kHidden + d;
          float acc = 0.0f;
          double ref = 0.0;
          double mag = 0.0;
          for (int j = 0; j < kUsed; ++j) {
            const float w = weights[std::size_t(t) * kUsed + j];
            const float v = experts[(std::size_t(t) * kUsed + j) * kHidden + d];
            acc = std::fmaf(w, v, acc);
            ref += double(w) * v;
            mag += std::fabs(double(w) * v);
          }
          if (with_shared) {
            acc = shared[i] + acc;
            ref += shared[i];
            mag += std::fabs(shared[i]);
          }
          if (with_residual) {
            acc = residual[i] + acc;
            ref += residual[i];
            mag += std::fabs(residual[i]);
          }
          exact_bad += got[i] != acc;
          tol_bad += !(std::fabs(got[i] - ref) <= 1e-6 * mag);
        }
      }
      std::ostringstream os;
      os << "combine tokens=" << tokens << " shared=" << with_shared
         << " residual=" << with_residual << " exact_bad=" << exact_bad
         << " tol_bad=" << tol_bad;
      check.Expect(exact_bad == 0 && tol_bad == 0, os.str());
    }
  }
}

}  // namespace

int main() {
  try {
    Checker check;
    std::mt19937 rng(7031);
    constexpr int kExperts = 16;
    // Every weight type in both roles: gate/up (2048 -> 512) and down
    // (512 -> 2048).
    const std::pair<rocm::WeightType, rocm::WeightType> combos[] = {
        {rocm::WeightType::kIQ4_XS, rocm::WeightType::kQ5_K},
        {rocm::WeightType::kQ4_K, rocm::WeightType::kQ6_K},
        {rocm::WeightType::kIQ3_S, rocm::WeightType::kQ4_K},
        {rocm::WeightType::kQ5_K, rocm::WeightType::kQ8_0},
        {rocm::WeightType::kQ8_0, rocm::WeightType::kIQ4_XS},
        {rocm::WeightType::kQ6_K, rocm::WeightType::kIQ3_S},
    };
    bool first = true;
    for (const auto [gu, dn] : combos) {
      QuantTensor gate(gu, kExperts, kInner, kHidden, rng);
      QuantTensor up(gu, kExperts, kInner, kHidden, rng);
      QuantTensor down(dn, kExperts, kHidden, kInner, rng);
      std::vector<int> stage_tokens = {1, 2, 3, 8, 9, 33};
      if (first) {
        stage_tokens.insert(stage_tokens.end(), {4, 5, 130});
      }
      for (int tokens : stage_tokens) {
        CheckStage(gate, false, Routing::kRandom, tokens, rng, check);
        CheckStage(down, true, Routing::kRandom, tokens, rng, check);
      }
      for (int tokens : {3, 33}) {
        CheckStage(gate, false, Routing::kShared, tokens, rng, check);
        CheckStage(down, true, Routing::kShared, tokens, rng, check);
      }
      for (int tokens : {2, 9}) {
        CheckStage(gate, false, Routing::kDisjoint, tokens, rng, check);
        CheckStage(down, true, Routing::kDisjoint, tokens, rng, check);
      }
      for (int tokens : {1, 5, 9, 33}) {
        CheckForward(gate, up, down, Routing::kRandom, tokens, rng, check);
      }
      CheckForward(gate, up, down, Routing::kShared, 12, rng, check);
      CheckForwardQuantized(gate, up, down, rng, check);
      for (int tokens : {1, 9, 33}) {
        CheckForward(gate, up, down, Routing::kRandom, tokens, rng, check, 5);
      }
      first = false;
    }
    // All 256 experts addressable: high expert offsets in both paths.
    {
      QuantTensor gate(rocm::WeightType::kIQ4_XS, 256, kInner, kHidden, rng);
      QuantTensor down(rocm::WeightType::kQ5_K, 256, kHidden, kInner, rng);
      for (int tokens : {1, 2, 9}) {
        CheckStage(gate, false, Routing::kRandom, tokens, rng, check);
        CheckStage(down, true, Routing::kRandom, tokens, rng, check);
      }
    }
    CheckCombine(rng, check);
    return check.Finish("flint.moe_ops");
  } catch (const std::exception& e) {
    std::cerr << "flint.moe_ops: " << e.what() << '\n';
    return 1;
  }
}
