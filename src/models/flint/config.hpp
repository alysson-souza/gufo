#ifndef GUFO_MODELS_FLINT_CONFIG_HPP_
#define GUFO_MODELS_FLINT_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::flint {

/// Architecture parameters of a `flint` GGUF artifact (Recursive Flint).
/// Every value is read from the file. The checks bound what this runtime
/// implements: GQA attention with per-head QK RMSNorm and learned sinks,
/// per-layer sliding-window or full attention, NeoX RoPE on the layers the
/// file marks, leading dense SwiGLU layers, then sigmoid-routed SwiGLU experts
/// with renormalized weights and one ungated shared expert.
struct Config {
  enum class AttentionKind : std::uint8_t { kFull, kSliding };

  std::uint32_t num_layers{0};
  std::uint32_t hidden_size{0};
  std::uint32_t vocab_size{0};
  std::uint32_t context_length{0};
  float rms_eps{0.0F};

  std::uint32_t num_heads{0};
  std::uint32_t num_kv_heads{0};
  std::uint32_t head_dim{0};
  float rope_theta{0.0F};
  /// Tokens visible to a sliding-window query, the current token included.
  std::uint32_t sliding_window{0};

  std::uint32_t dense_ffn_dim{0};
  std::uint32_t leading_dense_layers{0};
  std::uint32_t num_experts{0};
  std::uint32_t experts_per_token{0};
  std::uint32_t expert_ffn_dim{0};
  std::uint32_t shared_expert_ffn_dim{0};
  /// Multiplier applied to the renormalized routing weights.
  float expert_weights_scale{1.0F};

  std::vector<AttentionKind> attention;
  std::vector<bool> rope;

  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);

  [[nodiscard]] bool IsSliding(std::uint32_t layer) const {
    return attention[layer] == AttentionKind::kSliding;
  }
  [[nodiscard]] bool UsesRope(std::uint32_t layer) const { return rope[layer]; }
  [[nodiscard]] bool IsDense(std::uint32_t layer) const {
    return layer < leading_dense_layers;
  }
  [[nodiscard]] std::uint32_t QueryWidth() const {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t KvWidth() const {
    return num_kv_heads * head_dim;
  }
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_CONFIG_HPP_
