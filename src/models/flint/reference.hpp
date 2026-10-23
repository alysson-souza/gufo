#ifndef GUFO_MODELS_FLINT_REFERENCE_HPP_
#define GUFO_MODELS_FLINT_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/flint/weights.hpp"

namespace gufo::models::flint {

/// FP32 oracle of the Flint graph. Every weight row is decoded on the fly and
/// reduced in double precision; keys and values stay FP32 for every
/// position. A token's results are identical whether it is evaluated by Step
/// or inside any Forward chunk, and do not depend on the OpenMP thread count.
///
/// Per layer: RMSNorm -> q/k/v projections -> per-head q/k RMSNorm -> NeoX
/// RoPE on layers the file marks -> GQA attention (sliding window or full,
/// scale 1/sqrt(head_dim), the head's learned sink joins the softmax
/// denominator) -> output projection + residual -> RMSNorm -> dense SwiGLU, or
/// sigmoid-routed top-k SwiGLU experts (renormalized, scaled) plus the
/// ungated shared expert -> residual. Then the final RMSNorm and LM head.
class ReferenceModel {
public:
  /// Observes one token's intermediate row. `layer` is -1 for rows outside
  /// the blocks. Names follow llama.cpp's graph callbacks (`inp_embd`,
  /// `attn_norm`, `Qcur`, `Kcur`, `Vcur`, `kqv_out`, `attn_out`, `ffn_inp`,
  /// `ffn_norm`, `ffn_moe_out`, `ffn_shexp`, `ffn_out`, `l_out`,
  /// `result_norm`); `Qcur` and `Kcur` are taken after normalization and
  /// RoPE, `ffn_moe_out` excludes the shared expert.
  using Tap =
      std::function<void(std::string_view name, std::int32_t layer,
                         std::uint32_t position, std::span<const float> row)>;

  /// `weights` must outlive the model. Allocates FP32 K/V for `max_context`
  /// positions in every layer.
  ReferenceModel(const ModelWeights& weights, std::uint32_t max_context);

  /// Consumes `token` at Position() and writes the next-token logits
  /// (vocab_size floats). Fails without side effects when the context is
  /// full, the token is out of range or `logits` has the wrong size.
  bool Step(std::uint32_t token, std::span<float> logits,
            std::string* error_msg = nullptr);

  /// Consumes `tokens` at Position() onwards, teacher-forced, and writes one
  /// row of vocab_size logits per token. Each weight row (and each routed
  /// expert) is decoded once per call; K-quant decoding dominates the cost,
  /// so long probes should pass hundreds of tokens per call.
  bool Forward(std::span<const std::uint32_t> tokens, std::span<float> logits,
               std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t Position() const { return position_; }
  void Reset() { position_ = 0; }
  void SetTap(Tap tap) { tap_ = std::move(tap); }

private:
  void Emit(std::string_view name, std::int32_t layer,
            std::span<const float> rows, std::size_t width) const;
  void Attention(std::uint32_t il, std::vector<float>& x, std::size_t count);
  void FeedForward(std::uint32_t il, std::vector<float>& x, std::size_t count);

  const ModelWeights& w_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  /// Per layer [max_context][kv_heads * head_dim].
  std::vector<std::vector<float>> keys_;
  std::vector<std::vector<float>> values_;
  Tap tap_;
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_REFERENCE_HPP_
