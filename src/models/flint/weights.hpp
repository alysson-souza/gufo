#ifndef GUFO_MODELS_FLINT_WEIGHTS_HPP_
#define GUFO_MODELS_FLINT_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/flint/config.hpp"

namespace gufo::models::flint {

/// Non-owning view of one GGUF tensor inside the mapped artifact. Shapes follow
/// the GGUF convention: `cols` (ne0) is the contiguous reduction dimension,
/// `rows` (ne1) the output dimension and `experts` (ne2) stacks whole matrices.
/// Vectors (norms, sinks) have `rows == 1`.
struct TensorRef {
  const void* data = nullptr;
  core::GgmlType type{};
  std::string_view name;
  std::uint64_t cols = 0, rows = 0, experts = 1;
  /// Encoded bytes of one row, one stacked expert matrix and the whole tensor.
  std::size_t row_bytes = 0, expert_bytes = 0, size_bytes = 0;
  /// Byte offset of the payload inside its mapped region (the file for a
  /// single-file artifact).
  std::size_t file_offset = 0;

  explicit operator bool() const { return data != nullptr; }

  /// First byte of row `row` of expert `expert`.
  [[nodiscard]] const std::uint8_t* Row(std::uint64_t expert,
                                        std::uint64_t row) const noexcept {
    return static_cast<const std::uint8_t*>(data) + expert * expert_bytes +
           row * row_bytes;
  }
};

/// One decoder block. Dense layers (`Config::IsDense`) bind `gate/up/down`;
/// MoE layers bind the router, the stacked experts and the shared expert.
struct LayerWeights {
  TensorRef attn_norm;  ///< F32 [hidden]
  TensorRef q;          ///< [hidden -> heads * head_dim]
  TensorRef k;          ///< [hidden -> kv_heads * head_dim]
  TensorRef v;          ///< [hidden -> kv_heads * head_dim]
  TensorRef o;          ///< [heads * head_dim -> hidden]
  TensorRef q_norm;     ///< F32 [head_dim], shared by every query head
  TensorRef k_norm;     ///< F32 [head_dim], shared by every key head
  TensorRef sinks;      ///< F32 [heads], one softmax-denominator logit per head
  TensorRef ffn_norm;   ///< F32 [hidden]

  TensorRef gate;  ///< dense [hidden -> dense_ffn]
  TensorRef up;    ///< dense [hidden -> dense_ffn]
  TensorRef down;  ///< dense [dense_ffn -> hidden]

  TensorRef router;       ///< BF16/F32 [hidden -> experts]
  TensorRef router_bias;  ///< F32 [experts] selection-only bias; optional
  TensorRef gate_exps;    ///< [hidden -> expert_ffn] x experts
  TensorRef up_exps;      ///< [hidden -> expert_ffn] x experts
  TensorRef down_exps;    ///< [expert_ffn -> hidden] x experts
  TensorRef gate_shexp;   ///< [hidden -> shared_ffn]
  TensorRef up_shexp;     ///< [hidden -> shared_ffn]
  TensorRef down_shexp;   ///< [shared_ffn -> hidden]
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;   ///< [hidden -> vocab], read one row per token
  TensorRef output_norm;  ///< F32 [hidden]
  TensorRef output;       ///< [hidden -> vocab] (untied)
  std::vector<LayerWeights> layers;

  /// Parses the config and binds every tensor with its exact shape and an
  /// accepted storage type. Payloads stay mapped and untouched; the result
  /// borrows from `gguf`, which must outlive it.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_WEIGHTS_HPP_
