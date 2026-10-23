// Dense quantized projections against the core::quant CPU decoder with
// double accumulation.
//
// Tolerances. With ExactActivations the Q8_1 activation encoding is lossless,
// so the device result differs from the exact product only by FP32 rounding
// of per-block scale products and partial sums (tens of roundings at 2^-24);
// 1e-4 of sum|w x| leaves two orders of margin while any decode or indexing
// error moves a result by at least one weight-activation product of a
// sub-block (>= 1e-2 relative). Above eight rows, llama.cpp's MMQ tiles hold
// Q4_K/Q5_K sub-block scale d*sc and min dmin*m as FP16 (relative 2^-11), so
// those rows also allow 2^-11 * sum(|scale part * x| + |min part * x|), with
// the parts decoded separately (min or scale zeroed).
// Arbitrary activations add Q8_1 rounding of
// at most half a code (amax / 254) per value; its relative RMS effect on a
// random dot product is ~0.3%, checked against 1%.
#include <algorithm>
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

/// A dense matrix: one group of the shared test tensor.
class Matrix : public test::QuantTensor {
public:
  Matrix(rocm::WeightType t, int rows, int cols, std::mt19937& rng)
      : QuantTensor(t, 1, rows, cols, rng) {}
  double Tolerance(int o, const float* x, int rows, const test::Dot& exact) {
    return QuantTensor::Tolerance(0, o, x, rows > rocm::kVectorRows, exact);
  }
  const float* Row(int o) { return QuantTensor::Row(0, o); }
};

std::string Label(Matrix& w, int rows, const char* what) {
  std::ostringstream os;
  os << test::Name(w.type) << ' ' << w.m << 'x' << w.k << " rows=" << rows
     << ' ' << what;
  return os.str();
}

std::vector<float> RunProject(Matrix& w, const std::vector<float>& x,
                              int rows) {
  DeviceBuffer<float> dx{std::span<const float>(x)};
  DeviceBuffer<float> dy(static_cast<std::size_t>(rows) * w.m);
  dy.Fill(0xff);  // NaN: every output must be written
  const std::size_t scratch_bytes = rocm::ProjectionScratchBytes(rows, w.k);
  DeviceBuffer<std::uint8_t> scratch(scratch_bytes);
  test::CheckHip(rocm::Project(w.View(), dx.get(), dy.get(), rows,
                               scratch.get(), scratch_bytes, nullptr),
                 "Project");
  test::CheckHip(hipDeviceSynchronize(), "sync");
  return dy.Download();
}

void CheckExact(Matrix& w, int rows, std::mt19937& rng, Checker& check) {
  const std::vector<float> x = test::ExactActivations(rows, w.k, rng);
  const std::vector<float> y = RunProject(w, x, rows);
  int bad = 0;
  for (int r = 0; r < rows; ++r) {
    for (int o = 0; o < w.m; ++o) {
      const test::Dot ref =
          test::DotRow(w.Row(o), &x[std::size_t(r) * w.k], w.k);
      const double got = y[std::size_t(r) * w.m + o];
      const double tol = w.Tolerance(o, &x[std::size_t(r) * w.k], rows, ref);
      if (!std::isfinite(got) || std::fabs(got - ref.value) > tol) {
        if (bad++ < 3) {
          std::ostringstream os;
          os << Label(w, rows, "exact") << " r=" << r << " o=" << o
             << " got=" << got << " ref=" << ref.value << " tol=" << tol;
          check.Expect(false, os.str());
        }
      }
    }
  }
  check.Expect(bad == 0,
               Label(w, rows, "exact mismatches=") + std::to_string(bad));
}

void CheckArbitrary(Matrix& w, int rows, std::mt19937& rng, Checker& check) {
  const std::vector<float> x =
      test::RandomFloats(std::size_t(rows) * w.k, -1.0f, 1.0f, rng);
  const std::vector<float> y = RunProject(w, x, rows);
  double err2 = 0.0;
  double ref2 = 0.0;
  bool finite = true;
  for (int r = 0; r < rows; ++r) {
    for (int o = 0; o < w.m; ++o) {
      const test::Dot ref =
          test::DotRow(w.Row(o), &x[std::size_t(r) * w.k], w.k);
      const double got = y[std::size_t(r) * w.m + o];
      finite = finite && std::isfinite(got);
      err2 += (got - ref.value) * (got - ref.value);
      ref2 += ref.value * ref.value;
    }
  }
  const double rel = std::sqrt(err2 / ref2);
  check.Expect(finite && rel < 0.01,
               Label(w, rows, "arbitrary rel_rms=") + std::to_string(rel));
}

void CheckDeterministic(Matrix& w, int rows, std::mt19937& rng,
                        Checker& check) {
  const std::vector<float> x =
      test::RandomFloats(std::size_t(rows) * w.k, -1.0f, 1.0f, rng);
  const std::vector<float> a = RunProject(w, x, rows);
  const std::vector<float> b = RunProject(w, x, rows);
  check.Expect(a == b, Label(w, rows, "repeat not bitwise identical"));
}

/// q/k/v-style shared input: every target must match its own oracle and
/// the standalone projection bit for bit.
void CheckShared(std::mt19937& rng, Checker& check) {
  constexpr int k = 2048;
  Matrix q(rocm::WeightType::kQ5_K, 4096, k, rng);
  Matrix kk(rocm::WeightType::kQ8_0, 512, k, rng);
  Matrix v(rocm::WeightType::kQ6_K, 512, k, rng);
  for (int rows : {1, 5, 33}) {
    const std::vector<float> x = test::ExactActivations(rows, k, rng);
    DeviceBuffer<float> dx{std::span<const float>(x)};
    std::vector<DeviceBuffer<float>> outs;
    std::vector<rocm::ProjectionTarget> targets;
    for (Matrix* w : {&q, &kk, &v}) {
      outs.emplace_back(std::size_t(rows) * w->m);
      targets.push_back({w->View(), outs.back().get()});
    }
    const std::size_t bytes = rocm::ProjectionScratchBytes(rows, k);
    DeviceBuffer<std::uint8_t> scratch(bytes);
    test::CheckHip(rocm::ProjectShared(targets, dx.get(), rows, k,
                                       scratch.get(), bytes, nullptr),
                   "ProjectShared");
    test::CheckHip(hipDeviceSynchronize(), "sync");
    int index = 0;
    for (Matrix* w : {&q, &kk, &v}) {
      const std::vector<float> got = outs[index++].Download();
      const std::vector<float> alone = RunProject(*w, x, rows);
      check.Expect(got == alone, Label(*w, rows, "shared != standalone"));
      int bad = 0;
      for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < w->m; ++o) {
          const test::Dot ref =
              test::DotRow(w->Row(o), &x[std::size_t(r) * k], k);
          bad += std::fabs(got[std::size_t(r) * w->m + o] - ref.value) >
                 w->Tolerance(o, &x[std::size_t(r) * k], rows, ref);
        }
      }
      check.Expect(bad == 0, Label(*w, rows, "shared exact"));
    }
  }
}

double Silu(double g) {
  return g / (1.0 + std::exp(-g));
}

/// silu(gate) * up. The fused one-row kernel and the staged path share the
/// projection error bound; propagated through the product:
/// |d(silu(g) u)| <= 1.1 |u| eg + |silu(g)| eu (|silu'| < 1.1).
void CheckSwiGlu(std::mt19937& rng, Checker& check) {
  for (const auto [gt, ut] :
       {std::pair{rocm::WeightType::kQ4_K, rocm::WeightType::kQ4_K},
        std::pair{rocm::WeightType::kIQ4_XS, rocm::WeightType::kIQ4_XS},
        std::pair{rocm::WeightType::kQ5_K, rocm::WeightType::kQ6_K}}) {
    Matrix gate(gt, 1024, 2048, rng);
    Matrix up(ut, 1024, 2048, rng);
    for (int rows : {1, 2, 9, 33}) {
      const std::vector<float> x = test::ExactActivations(rows, 2048, rng);
      DeviceBuffer<float> dx{std::span<const float>(x)};
      DeviceBuffer<float> dy(std::size_t(rows) * 1024);
      dy.Fill(0xff);
      const std::size_t bytes = rocm::SwiGluScratchBytes(rows, 2048, 1024);
      DeviceBuffer<std::uint8_t> scratch(bytes);
      test::CheckHip(
          rocm::ProjectSwiGlu(gate.View(), up.View(), dx.get(), dy.get(), rows,
                              scratch.get(), bytes, nullptr),
          "ProjectSwiGlu");
      test::CheckHip(hipDeviceSynchronize(), "sync");
      const std::vector<float> y = dy.Download();
      int bad = 0;
      for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < 1024; ++o) {
          const float* xr = &x[std::size_t(r) * 2048];
          const test::Dot g = test::DotRow(gate.Row(o), xr, 2048);
          const test::Dot u = test::DotRow(up.Row(o), xr, 2048);
          const double eg = gate.Tolerance(o, xr, rows, g);
          const double eu = up.Tolerance(o, xr, rows, u);
          const double ref = Silu(g.value) * u.value;
          const double tol = 1.1 * (std::fabs(u.value) + eu) * eg +
                             (std::fabs(Silu(g.value)) + 1.1 * eg) * eu + 1e-7;
          const double got = y[std::size_t(r) * 1024 + o];
          bad += !(std::fabs(got - ref) <= tol);
        }
      }
      check.Expect(bad == 0, Label(gate, rows, "swiglu mismatches=") +
                                 std::to_string(bad));
    }
  }
}

/// Producer-quantized rows (RmsNorm/AddRmsNorm with out_q8) must give the
/// same bytes the float-input projections quantize internally: every
/// pre-quantized projection is compared bit for bit with its float twin.
void CheckQuantizedInputs(std::mt19937& rng, Checker& check) {
  constexpr int k = 2048;
  std::vector<Matrix> mats;
  for (rocm::WeightType type : test::kAllTypes) {
    mats.emplace_back(type, 520, k, rng);
  }
  Matrix gate(rocm::WeightType::kQ8_0, 1024, k, rng);
  Matrix up(rocm::WeightType::kQ8_0, 1024, k, rng);
  Matrix up5(rocm::WeightType::kQ5_K, 1024, k, rng);
  for (int rows : {1, 3, 8}) {
    for (bool add : {false, true}) {
      const std::vector<float> x =
          test::RandomFloats(std::size_t(rows) * k, -3.0f, 3.0f, rng);
      const std::vector<float> d =
          test::RandomFloats(std::size_t(rows) * k, -3.0f, 3.0f, rng);
      const std::vector<float> g = test::RandomFloats(k, 0.5f, 1.5f, rng);
      DeviceBuffer<float> dx{std::span<const float>(x)};
      DeviceBuffer<float> dd{std::span<const float>(d)};
      DeviceBuffer<float> dg{std::span<const float>(g)};
      DeviceBuffer<float> plain(std::size_t(rows) * k);
      DeviceBuffer<float> normed(std::size_t(rows) * k);
      DeviceBuffer<std::uint8_t> q8(rocm::VectorActivationBytes(rows, k));
      if (add) {
        DeviceBuffer<float> residual{std::span<const float>(x)};
        test::CheckHip(rocm::AddRmsNorm(residual.get(), dd.get(), dg.get(),
                                        plain.get(), rows, k, 1e-6f, nullptr),
                       "AddRmsNorm");
        test::CheckHip(
            rocm::AddRmsNorm(dx.get(), dd.get(), dg.get(), normed.get(),
                             q8.get(), rows, k, 1e-6f, nullptr),
            "AddRmsNorm(q8)");
        check.Expect(residual.Download() == dx.Download(),
                     "AddRmsNorm(q8) residual differs");
      } else {
        test::CheckHip(rocm::RmsNorm(dx.get(), dg.get(), plain.get(), rows, k,
                                     1e-6f, nullptr),
                       "RmsNorm");
        test::CheckHip(rocm::RmsNorm(dx.get(), dg.get(), normed.get(), q8.get(),
                                     rows, k, 1e-6f, nullptr),
                       "RmsNorm(q8)");
      }
      const std::vector<float> xn = normed.Download();
      check.Expect(
          xn == plain.Download(),
          "norm with q8 changes the float output rows=" + std::to_string(rows));
      const std::size_t sb = rocm::SwiGluScratchBytes(rows, k, 1024);
      DeviceBuffer<std::uint8_t> scratch(sb);
      for (Matrix& w : mats) {
        DeviceBuffer<float> a(std::size_t(rows) * w.m);
        DeviceBuffer<float> b(std::size_t(rows) * w.m);
        test::CheckHip(rocm::Project(w.View(), normed.get(), a.get(), rows,
                                     scratch.get(), sb, nullptr),
                       "Project");
        test::CheckHip(
            rocm::ProjectQuantized(w.View(), q8.get(), b.get(), rows, nullptr),
            "ProjectQuantized");
        check.Expect(a.Download() == b.Download(),
                     Label(w, rows,
                           add ? "quantized after AddRmsNorm"
                               : "quantized after RmsNorm"));
      }
      for (Matrix* u : {&up, &up5}) {
        DeviceBuffer<float> a(std::size_t(rows) * 1024);
        DeviceBuffer<float> b(std::size_t(rows) * 1024);
        test::CheckHip(
            rocm::ProjectSwiGlu(gate.View(), u->View(), normed.get(), a.get(),
                                rows, scratch.get(), sb, nullptr),
            "ProjectSwiGlu");
        test::CheckHip(rocm::ProjectSwiGluQuantized(gate.View(), u->View(),
                                                    q8.get(), b.get(), rows,
                                                    scratch.get(), sb, nullptr),
                       "ProjectSwiGluQuantized");
        check.Expect(a.Download() == b.Download(),
                     Label(*u, rows, "swiglu quantized"));
      }
    }
  }
}

}  // namespace

int main() {
  try {
    Checker check;
    std::mt19937 rng(20261022);
    // Awkward output count (not a tile multiple) over every batch class.
    for (rocm::WeightType type : test::kAllTypes) {
      Matrix w(type, 1000, 2048, rng);
      for (int rows : {1, 2, 3, 8, 9, 32, 33}) {
        CheckExact(w, rows, rng, check);
      }
      CheckArbitrary(w, 1, rng, check);
      CheckArbitrary(w, 9, rng, check);
      CheckDeterministic(w, 1, rng, check);
      CheckDeterministic(w, 33, rng, check);
      Matrix wide(type, 264, 2048, rng);
      for (int rows : {512, 515}) {
        CheckExact(wide, rows, rng, check);
      }
    }
    // Flint's projection shapes.
    const std::pair<int, int> shapes[] = {
        {4096, 2048}, {512, 2048},  {2048, 4096}, {6144, 2048},
        {2048, 6144}, {1024, 2048}, {2048, 1024}, {2048, 512}};
    for (rocm::WeightType type : test::kAllTypes) {
      for (const auto [m, k] : shapes) {
        Matrix w(type, m, k, rng);
        CheckExact(w, 1, rng, check);
        CheckExact(w, 9, rng, check);
      }
    }
    // The Q8_0 vocabulary head.
    {
      Matrix head(rocm::WeightType::kQ8_0, 150144, 2048, rng);
      CheckExact(head, 1, rng, check);
      CheckExact(head, 2, rng, check);
    }
    CheckShared(rng, check);
    CheckSwiGlu(rng, check);
    CheckQuantizedInputs(rng, check);
    return check.Finish("flint.projection_ops");
  } catch (const std::exception& e) {
    std::cerr << "flint.projection_ops: " << e.what() << '\n';
    return 1;
  }
}
