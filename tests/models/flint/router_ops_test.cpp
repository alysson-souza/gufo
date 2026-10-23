// Sigmoid top-k router against a double oracle.
//
// Tolerances. Scores come from an FP32 dot product of BF16 weights with FP32
// inputs (2048 terms, one tree) and an accurate sigmoid: |score - exact| <=
// 1e-6 * (sum|w x| + 1) bounds the accumulation (sigmoid' <= 1/4). The
// selection is checked exactly on the device's own scores, so near-ties
// cannot flake: ids must be the documented top-k of score + bias with ties
// to the lower index, and weights the documented FP32 renormalization
// (emulated on the host, relative 1e-6 against double).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

#include "tests/models/flint/ops_test_support.hpp"

namespace {

using namespace gufo::models::flint;
using test::Checker;
using test::DeviceBuffer;

constexpr int kExperts = 256;
constexpr int kUsed = 6;
constexpr int kHidden = 2048;

std::uint16_t ToBf16(float v) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);  // round to nearest even
  return static_cast<std::uint16_t>(bits >> 16);
}

float FromBf16(std::uint16_t b) {
  const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
  float v = 0.0f;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

struct Result {
  std::vector<float> scores;
  std::vector<std::int32_t> ids;
  std::vector<float> weights;
};

Result Run(const std::vector<std::uint16_t>& w, const std::vector<float>& x,
           const std::vector<float>* bias, float scale, int tokens) {
  DeviceBuffer<std::uint16_t> dw{std::span<const std::uint16_t>(w)};
  DeviceBuffer<float> dx{std::span<const float>(x)};
  DeviceBuffer<float> db;
  if (bias != nullptr) {
    db.Resize(bias->size());
    db.Upload(*bias);
  }
  DeviceBuffer<std::int32_t> ids(std::size_t(tokens) * kUsed);
  DeviceBuffer<float> weights(std::size_t(tokens) * kUsed);
  const std::size_t bytes = rocm::RouterScratchBytes(tokens, kExperts);
  DeviceBuffer<float> scratch(bytes / sizeof(float));
  test::CheckHip(
      rocm::RouterTopK(dw.get(), dx.get(), bias != nullptr ? db.get() : nullptr,
                       scale, tokens, kHidden, kExperts, kUsed, ids.get(),
                       weights.get(), scratch.get(), bytes, nullptr),
      "RouterTopK");
  test::CheckHip(hipDeviceSynchronize(), "sync");
  return {scratch.Download(std::size_t(tokens) * kExperts), ids.Download(),
          weights.Download()};
}

/// Checks scores against double and the selection/weights against the
/// contract applied to the device scores.
void Verify(const std::vector<std::uint16_t>& w, const std::vector<float>& x,
            const std::vector<float>* bias, float scale, int tokens,
            const Result& got, const std::string& label, Checker& check) {
  int score_bad = 0;
  int select_bad = 0;
  int weight_bad = 0;
  for (int t = 0; t < tokens; ++t) {
    const float* xr = &x[std::size_t(t) * kHidden];
    const float* s = &got.scores[std::size_t(t) * kExperts];
    for (int e = 0; e < kExperts; ++e) {
      double logit = 0.0;
      double mag = 0.0;
      for (int k = 0; k < kHidden; ++k) {
        const double p =
            double(FromBf16(w[std::size_t(e) * kHidden + k])) * xr[k];
        logit += p;
        mag += std::fabs(p);
      }
      const double ref = 1.0 / (1.0 + std::exp(-logit));
      if (std::isfinite(logit)) {
        score_bad += !(std::fabs(s[e] - ref) <= 1e-6 * (mag + 1.0));
      }
    }
    // Expected selection from the device scores.
    std::vector<std::pair<float, int>> keys;
    for (int e = 0; e < kExperts; ++e) {
      float key = bias != nullptr ? s[e] + (*bias)[e] : s[e];
      if (std::isnan(key)) {
        key = -std::numeric_limits<float>::infinity();
      }
      keys.emplace_back(key, e);
    }
    std::stable_sort(
        keys.begin(), keys.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    float sum = 0.0f;
    double dsum = 0.0;
    for (int j = 0; j < kUsed; ++j) {
      const int expect = keys[j].second;
      select_bad += got.ids[std::size_t(t) * kUsed + j] != expect;
      sum += s[expect];
      dsum += s[expect];
    }
    for (int j = 0; j < kUsed; ++j) {
      const int e = got.ids[std::size_t(t) * kUsed + j];
      if (e < 0 || e >= kExperts) {
        ++select_bad;
        continue;
      }
      const float expect = s[e] / (sum + 1e-20f) * scale;
      const double dexpect = s[e] / (dsum + 1e-20) * scale;
      const float value = got.weights[std::size_t(t) * kUsed + j];
      if (!std::isfinite(expect)) {
        continue;  // a NaN row has no meaningful weights
      }
      weight_bad += value != expect || !(std::fabs(value - dexpect) <=
                                         1e-6 * std::fabs(dexpect) + 1e-30);
    }
  }
  std::ostringstream os;
  os << label << " score_bad=" << score_bad << " select_bad=" << select_bad
     << " weight_bad=" << weight_bad;
  check.Expect(score_bad == 0 && select_bad == 0 && weight_bad == 0, os.str());
}

}  // namespace

int main() {
  try {
    Checker check;
    std::mt19937 rng(424242);
    std::vector<std::uint16_t> w(std::size_t(kExperts) * kHidden);
    std::normal_distribution<float> normal(0.0f, 0.02f);
    for (auto& v : w) {
      v = ToBf16(normal(rng));
    }
    // Ordinary routing over both launch shapes (<= 8 and > 8 tokens).
    for (int tokens : {1, 2, 3, 8, 9, 33, 515}) {
      const std::vector<float> x =
          test::RandomFloats(std::size_t(tokens) * kHidden, -2.0f, 2.0f, rng);
      const Result r = Run(w, x, nullptr, 1.0f, tokens);
      Verify(w, x, nullptr, 1.0f, tokens, r,
             "plain tokens=" + std::to_string(tokens), check);
    }
    // Scores do not depend on the batch: 9 tokens vs their first 8.
    {
      const std::vector<float> x =
          test::RandomFloats(std::size_t(9) * kHidden, -2.0f, 2.0f, rng);
      const Result nine = Run(w, x, nullptr, 1.0f, 9);
      const Result eight = Run(w, x, nullptr, 1.0f, 8);
      check.Expect(
          std::equal(eight.scores.begin(), eight.scores.end(),
                     nine.scores.begin()) &&
              std::equal(eight.ids.begin(), eight.ids.end(), nine.ids.begin()),
          "batch independence 8 vs 9 tokens");
    }
    // Exact ties: experts 200, 7, 100, 3 share one row, which dominates;
    // the order must be 3, 7, 100, 200, then the next best.
    {
      std::vector<std::uint16_t> tied = w;
      const std::vector<float> x =
          test::RandomFloats(std::size_t(3) * kHidden, -1.0f, 1.0f, rng);
      for (int k = 0; k < kHidden; ++k) {
        const std::uint16_t v = ToBf16(x[k] > 0.0f ? 0.05f : -0.05f);
        for (int e : {200, 7, 100, 3}) {
          tied[std::size_t(e) * kHidden + k] = v;
        }
      }
      const Result r = Run(tied, x, nullptr, 2.5f, 3);
      Verify(tied, x, nullptr, 2.5f, 3, r, "ties", check);
      check.Expect(
          r.ids[0] == 3 && r.ids[1] == 7 && r.ids[2] == 100 && r.ids[3] == 200,
          "tie order 3,7,100,200");
      // Fully saturated scores: every expert ties at 1.0 for a huge input,
      // so the lowest six indices win.
      std::vector<float> huge(kHidden, 0.0f);
      for (int k = 0; k < kHidden; ++k) {
        huge[k] = 1e30f;
      }
      std::vector<std::uint16_t> positive(w.size(), ToBf16(0.01f));
      const Result s = Run(positive, huge, nullptr, 1.0f, 1);
      check.Expect(s.ids == std::vector<std::int32_t>({0, 1, 2, 3, 4, 5}),
                   "saturated ties pick 0..5");
    }
    // Bias changes the selection but not the weights' scores.
    {
      std::vector<float> bias(kExperts, 0.0f);
      for (int e : {250, 17, 128}) {
        bias[e] = 2.0f;
      }
      bias[5] = -3.0f;
      for (int tokens : {1, 9}) {
        const std::vector<float> x =
            test::RandomFloats(std::size_t(tokens) * kHidden, -2.0f, 2.0f, rng);
        const Result r = Run(w, x, &bias, 1.0f, tokens);
        Verify(w, x, &bias, 1.0f, tokens, r,
               "bias tokens=" + std::to_string(tokens), check);
        std::vector<std::int32_t> top(r.ids.begin(), r.ids.begin() + 3);
        std::sort(top.begin(), top.end());
        check.Expect(top == std::vector<std::int32_t>({17, 128, 250}) &&
                         std::find(r.ids.begin(), r.ids.begin() + kUsed, 5) ==
                             r.ids.begin() + kUsed,
                     "bias selects 17, 128, 250 first and never 5");
      }
    }
    // A NaN input still yields six distinct valid experts.
    {
      std::vector<float> x =
          test::RandomFloats(std::size_t(2) * kHidden, -1.0f, 1.0f, rng);
      x[5] = std::numeric_limits<float>::quiet_NaN();
      const Result r = Run(w, x, nullptr, 1.0f, 2);
      std::vector<std::int32_t> first(r.ids.begin(), r.ids.begin() + kUsed);
      std::sort(first.begin(), first.end());
      check.Expect(first == std::vector<std::int32_t>({0, 1, 2, 3, 4, 5}),
                   "NaN scores rank as -inf: lowest indices");
      Verify(w, x, nullptr, 1.0f, 2, r, "nan row", check);
    }
    return check.Finish("flint.router_ops");
  } catch (const std::exception& e) {
    std::cerr << "flint.router_ops: " << e.what() << '\n';
    return 1;
  }
}
