// Embedding, RMSNorm (plain and residual-fused) and SwiGLU against double
// oracles.
//
// Tolerances. Q8_0 embedding is fp16(d) * int8, exact in FP32: compared bit
// for bit with core::quant. RMSNorm accumulates at most 2048 squares through
// FMA chains and a fixed tree (relative error < 2e-6), then one sqrt, one
// division and two products: 1e-5 relative per element. The fused residual
// sum is a single FP32 addition and must match exactly. SwiGLU uses the
// accurate expf and one division and product: 4e-6 relative.
#include <cmath>
#include <cstdint>
#include <random>
#include <sstream>
#include <vector>

#include "tests/models/flint/ops_test_support.hpp"

namespace {

using namespace gufo::models::flint;
using test::Checker;
using test::DeviceBuffer;

void CheckEmbed(std::mt19937& rng, Checker& check) {
  constexpr int kVocab = 3001;
  constexpr int kHidden = 2048;
  const std::vector<std::uint8_t> table =
      test::RandomWeights(rocm::WeightType::kQ8_0, kVocab, kHidden, rng);
  const std::vector<float> ref =
      test::Dequantize(rocm::WeightType::kQ8_0, table.data(), kVocab, kHidden);
  DeviceBuffer<std::uint8_t> dt{std::span<const std::uint8_t>(table)};
  for (int count : {1, 7, 64}) {
    std::vector<std::int32_t> tokens(count);
    std::uniform_int_distribution<int> pick(0, kVocab - 1);
    for (auto& t : tokens) {
      t = pick(rng);
    }
    tokens[0] = kVocab - 1;  // the last row
    DeviceBuffer<std::int32_t> dtok{std::span<const std::int32_t>(tokens)};
    DeviceBuffer<float> out(std::size_t(count) * kHidden);
    test::CheckHip(rocm::EmbedQ8_0(dt.get(), dtok.get(), count, kHidden,
                                   out.get(), nullptr),
                   "EmbedQ8_0");
    const std::vector<float> got = out.Download();
    int bad = 0;
    for (int i = 0; i < count; ++i) {
      for (int d = 0; d < kHidden; ++d) {
        bad += got[std::size_t(i) * kHidden + d] !=
               ref[std::size_t(tokens[i]) * kHidden + d];
      }
    }
    check.Expect(bad == 0, "embed count=" + std::to_string(count));
  }
}

void CheckRmsNorm(std::mt19937& rng, Checker& check) {
  constexpr float kEps = 1e-6f;
  for (int dim : {2048, 128, 1000}) {
    for (int rows : {1, 3, 9, 33}) {
      for (float magnitude : {1e-3f, 1.0f, 1e3f}) {
        const std::size_t n = std::size_t(rows) * dim;
        const std::vector<float> x =
            test::RandomFloats(n, -magnitude, magnitude, rng);
        const std::vector<float> delta =
            test::RandomFloats(n, -magnitude, magnitude, rng);
        const std::vector<float> w = test::RandomFloats(dim, 0.5f, 1.5f, rng);
        DeviceBuffer<float> dx{std::span<const float>(x)};
        DeviceBuffer<float> dd{std::span<const float>(delta)};
        DeviceBuffer<float> dw{std::span<const float>(w)};
        DeviceBuffer<float> out(n);
        test::CheckHip(rocm::RmsNorm(dx.get(), dw.get(), out.get(), rows, dim,
                                     kEps, nullptr),
                       "RmsNorm");
        const std::vector<float> plain = out.Download();
        test::CheckHip(rocm::AddRmsNorm(dx.get(), dd.get(), dw.get(), out.get(),
                                        rows, dim, kEps, nullptr),
                       "AddRmsNorm");
        const std::vector<float> fused = out.Download();
        const std::vector<float> summed = dx.Download();
        int bad_plain = 0;
        int bad_fused = 0;
        int bad_sum = 0;
        for (int r = 0; r < rows; ++r) {
          for (int pass = 0; pass < 2; ++pass) {
            std::vector<double> v(dim);
            double ss = 0.0;
            for (int d = 0; d < dim; ++d) {
              const std::size_t i = std::size_t(r) * dim + d;
              const float sum = x[i] + delta[i];
              if (pass == 1) {
                bad_sum += summed[i] != sum;
              }
              v[d] = pass == 0 ? x[i] : sum;
              ss += v[d] * v[d];
            }
            const double scale = 1.0 / std::sqrt(ss / dim + kEps);
            for (int d = 0; d < dim; ++d) {
              const double ref = v[d] * scale * w[d];
              const float got =
                  (pass == 0 ? plain : fused)[std::size_t(r) * dim + d];
              (pass == 0 ? bad_plain : bad_fused) +=
                  !(std::fabs(got - ref) <= 1e-5 * std::fabs(ref) + 1e-30);
            }
          }
        }
        std::ostringstream os;
        os << "rmsnorm dim=" << dim << " rows=" << rows << " mag=" << magnitude
           << " plain=" << bad_plain << " fused=" << bad_fused
           << " sum=" << bad_sum;
        check.Expect(bad_plain == 0 && bad_fused == 0 && bad_sum == 0,
                     os.str());
      }
    }
  }
}

void CheckSwiGlu(std::mt19937& rng, Checker& check) {
  for (std::size_t count : {std::size_t{1}, std::size_t{255}, std::size_t{257},
                            std::size_t{100003}}) {
    std::vector<float> g = test::RandomFloats(count, -12.0f, 12.0f, rng);
    const std::vector<float> u = test::RandomFloats(count, -4.0f, 4.0f, rng);
    g[0] = count > 1 ? -100.0f : 90.0f;  // exp overflow / underflow edges
    DeviceBuffer<float> dg{std::span<const float>(g)};
    DeviceBuffer<float> du{std::span<const float>(u)};
    DeviceBuffer<float> out(count);
    test::CheckHip(rocm::SwiGlu(dg.get(), du.get(), out.get(), count, nullptr),
                   "SwiGlu");
    const std::vector<float> got = out.Download();
    int bad = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const double ref = g[i] / (1.0 + std::exp(-double(g[i]))) * u[i];
      bad += !(std::isfinite(got[i]) &&
               std::fabs(got[i] - ref) <= 4e-6 * std::fabs(ref) + 1e-37);
    }
    check.Expect(bad == 0, "swiglu count=" + std::to_string(count) +
                               " bad=" + std::to_string(bad));
  }
}

}  // namespace

int main() {
  try {
    Checker check;
    std::mt19937 rng(99);
    CheckEmbed(rng, check);
    CheckRmsNorm(rng, check);
    CheckSwiGlu(rng, check);
    return check.Finish("flint.elementwise_ops");
  } catch (const std::exception& e) {
    std::cerr << "flint.elementwise_ops: " << e.what() << '\n';
    return 1;
  }
}
