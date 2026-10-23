#include "src/models/flint/cpu_ops.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::flint::cpu {

void DequantizeRow(const TensorRef& t, std::uint64_t expert, std::uint64_t row,
                   float* out) {
  const std::uint8_t* src = t.Row(expert, row);
  const std::size_t k = t.cols;
  switch (t.type) {
    case core::GgmlType::kF32:
      std::memcpy(out, src, k * sizeof(float));
      return;
    case core::GgmlType::kF16:
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + 2 * i, 2);
        out[i] = quant::Fp16ToFloat(h);
      }
      return;
    case core::GgmlType::kBF16:
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + 2 * i, 2);
        const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
        std::memcpy(out + i, &bits, 4);
      }
      return;
    case core::GgmlType::kQ8_0:
      quant::DequantizeQ8_0(src, out, k);
      return;
    case core::GgmlType::kQ3_K:
      quant::DequantizeQ3_K(src, out, k);
      return;
    case core::GgmlType::kQ4_K:
      quant::DequantizeQ4_K(src, out, k);
      return;
    case core::GgmlType::kQ5_K:
      quant::DequantizeQ5_K(src, out, k);
      return;
    case core::GgmlType::kQ6_K:
      quant::DequantizeQ6_K(src, out, k);
      return;
    case core::GgmlType::kIQ4_NL:
      quant::DequantizeIQ4_NL(src, out, k);
      return;
    case core::GgmlType::kIQ4_XS:
      quant::DequantizeIQ4_XS(src, out, k);
      return;
    case core::GgmlType::kIQ3_S:
      quant::DequantizeIQ3_S(src, out, k);
      return;
    default:
      // ModelWeights::Bind admits only the formats above.
      std::fprintf(stderr, "flint reference: no decoder for %s (%s)\n",
                   std::string(t.name).c_str(),
                   std::string(core::ToString(t.type)).c_str());
      std::abort();
  }
}

void MatMul(const TensorRef& t, std::uint64_t expert, const float* x,
            std::size_t x_stride, std::size_t count, float* y,
            std::size_t y_stride) {
  const auto rows = static_cast<std::int64_t>(t.rows);
  const std::size_t cols = t.cols;
#pragma omp parallel
  {
    std::vector<float> w(cols);
#pragma omp for schedule(static)
    for (std::int64_t r = 0; r < rows; ++r) {
      DequantizeRow(t, expert, static_cast<std::uint64_t>(r), w.data());
      for (std::size_t n = 0; n < count; ++n) {
        const float* xn = x + n * x_stride;
        double acc = 0.0;
        for (std::size_t i = 0; i < cols; ++i) {
          acc += static_cast<double>(w[i]) * xn[i];
        }
        y[n * y_stride + static_cast<std::size_t>(r)] = static_cast<float>(acc);
      }
    }
  }
}

void MatVec(const TensorRef& t, std::uint64_t expert, std::span<const float> x,
            std::span<float> y) {
  MatMul(t, expert, x.data(), x.size(), 1, y.data(), y.size());
}

void RmsNorm(std::span<float> x, const float* w, float eps) {
  double ss = 0.0;
  for (const float v : x) {
    ss += static_cast<double>(v) * v;
  }
  const double scale =
      1.0 / std::sqrt(ss / static_cast<double>(x.size()) + eps);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<float>(x[i] * scale * w[i]);
  }
}

float Sigmoid(float x) noexcept {
  return 1.0F / (1.0F + std::exp(-x));
}

float Silu(float x) noexcept {
  return x * Sigmoid(x);
}

void RopeNeox(float* x, std::uint32_t heads, std::uint32_t head_dim,
              std::uint32_t pos, float theta) {
  const std::uint32_t half = head_dim / 2;
  for (std::uint32_t i = 0; i < half; ++i) {
    const double freq =
        std::pow(static_cast<double>(theta),
                 -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
    const double angle = static_cast<double>(pos) * freq;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    for (std::uint32_t h = 0; h < heads; ++h) {
      float* v = x + static_cast<std::size_t>(h) * head_dim;
      const double a = v[i];
      const double b = v[i + half];
      v[i] = static_cast<float>(a * c - b * s);
      v[i + half] = static_cast<float>(a * s + b * c);
    }
  }
}

}  // namespace gufo::models::flint::cpu
