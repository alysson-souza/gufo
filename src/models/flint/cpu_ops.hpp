#ifndef GUFO_MODELS_FLINT_CPU_OPS_HPP_
#define GUFO_MODELS_FLINT_CPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/flint/weights.hpp"

/// Scalar CPU operators for the Flint reference model: every weight row is
/// decoded on the fly through `core::quant` and reduced in double precision.
/// Correctness oracle only; nothing here is tuned.
namespace gufo::models::flint::cpu {

/// Decodes row `row` of expert `expert` of `t` into `out` (t.cols floats).
void DequantizeRow(const TensorRef& t, std::uint64_t expert, std::uint64_t row,
                   float* out);

/// Y[n][r] = sum_c W[expert][r][c] * X[n][c] for `count` input rows: X rows
/// are t.cols floats `x_stride` apart, Y rows t.rows floats `y_stride` apart.
/// Weight rows run in parallel (OpenMP) and are decoded once for all inputs;
/// every output accumulates sequentially over c, so a result does not depend
/// on `count`, the batch it is part of, or the thread count.
void MatMul(const TensorRef& t, std::uint64_t expert, const float* x,
            std::size_t x_stride, std::size_t count, float* y,
            std::size_t y_stride);

/// MatMul of one input row: y = W[expert] x.
void MatVec(const TensorRef& t, std::uint64_t expert, std::span<const float> x,
            std::span<float> y);

/// x <- x / sqrt(mean(x^2) + eps) * w, with `w` holding x.size() floats.
void RmsNorm(std::span<float> x, const float* w, float eps);

[[nodiscard]] float Sigmoid(float x) noexcept;
[[nodiscard]] float Silu(float x) noexcept;

/// NeoX rotary embedding over the full `head_dim` of each of `heads`
/// contiguous heads: element i pairs with i + head_dim/2 and rotates by
/// pos * theta^(-2i/head_dim).
void RopeNeox(float* x, std::uint32_t heads, std::uint32_t head_dim,
              std::uint32_t pos, float theta);

}  // namespace gufo::models::flint::cpu

#endif  // GUFO_MODELS_FLINT_CPU_OPS_HPP_
