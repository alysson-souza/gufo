#include "src/models/flint/config.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>
#include <variant>

namespace gufo::models::flint {
namespace {

constexpr std::string_view kArchitecture = "flint";
// llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID.
constexpr std::uint32_t kSigmoidGating = 2;

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  static std::string Key(std::string_view suffix) {
    return std::string(kArchitecture) + "." + std::string(suffix);
  }

  std::uint32_t U32(std::string_view suffix) {
    const auto key = Key(suffix);
    const auto value = gguf.GetMetadataUint32(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF integer " + key);
      return 0;
    }
    return *value;
  }

  float F32(std::string_view suffix, std::optional<float> fallback = {}) {
    const auto key = Key(suffix);
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value.has_value() && fallback.has_value() &&
        gguf.FindMetadata(key) == nullptr) {
      return *fallback;
    }
    if (!value.has_value() || !std::isfinite(*value)) {
      Fail("missing or invalid GGUF float " + key);
      return 0.0F;
    }
    return *value;
  }

  bool Bool(std::string_view suffix) {
    const auto key = Key(suffix);
    const auto value = gguf.GetMetadataBool(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF boolean " + key);
      return false;
    }
    return *value;
  }

  /// GGUF boolean arrays are widened to unsigned integers by the reader.
  std::vector<bool> BoolArray(std::string_view suffix, std::uint32_t count) {
    const auto key = Key(suffix);
    const auto* meta = gguf.FindMetadata(key);
    const auto* values =
        meta == nullptr ? nullptr
                        : std::get_if<std::vector<std::uint64_t>>(&meta->value);
    if (values == nullptr || meta->type != core::GgufValueType::kArray ||
        values->size() != count) {
      Fail("GGUF array " + key + " must hold one boolean per layer");
      return {};
    }
    std::vector<bool> out;
    out.reserve(count);
    for (const auto value : *values) {
      out.push_back(value != 0);
    }
    return out;
  }
};

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not flint");
    return std::nullopt;
  }

  Config c;
  c.num_layers = r.U32("block_count");
  c.hidden_size = r.U32("embedding_length");
  c.context_length = r.U32("context_length");
  c.rms_eps = r.F32("attention.layer_norm_rms_epsilon");
  c.num_heads = r.U32("attention.head_count");
  c.num_kv_heads = r.U32("attention.head_count_kv");
  c.head_dim = r.U32("attention.key_length");
  if (r.U32("attention.value_length") != c.head_dim)
    r.Fail("attention value length must match the key length");
  c.rope_theta = r.F32("rope.freq_base");
  c.sliding_window = r.U32("attention.sliding_window");
  c.dense_ffn_dim = r.U32("feed_forward_length");
  c.leading_dense_layers = r.U32("leading_dense_block_count");
  c.num_experts = r.U32("expert_count");
  c.experts_per_token = r.U32("expert_used_count");
  c.expert_ffn_dim = r.U32("expert_feed_forward_length");
  c.shared_expert_ffn_dim = r.U32("expert_shared_feed_forward_length");
  c.expert_weights_scale = r.F32("expert_weights_scale", 1.0F);
  if (r.U32("expert_gating_func") != kSigmoidGating)
    r.Fail("flint expert gating must be sigmoid");
  if (!r.Bool("expert_weights_norm"))
    r.Fail("flint routing weights must be renormalized");
  if (!r.ok)
    return std::nullopt;

  if (c.num_layers == 0 || c.hidden_size == 0 || c.context_length == 0 ||
      c.num_heads == 0 || c.num_kv_heads == 0 ||
      c.num_heads % c.num_kv_heads != 0 || c.head_dim == 0 ||
      c.head_dim % 2 != 0 || c.sliding_window == 0 || c.rms_eps <= 0.0F ||
      c.rope_theta <= 0.0F) {
    r.Fail("invalid flint attention geometry");
    return std::nullopt;
  }
  if (c.leading_dense_layers > c.num_layers || c.dense_ffn_dim == 0 ||
      c.num_experts == 0 || c.experts_per_token == 0 ||
      c.experts_per_token > c.num_experts || c.expert_ffn_dim == 0 ||
      c.shared_expert_ffn_dim == 0 || c.expert_weights_scale <= 0.0F) {
    r.Fail("invalid flint feed-forward geometry");
    return std::nullopt;
  }

  const auto sliding =
      r.BoolArray("attention.sliding_window_pattern", c.num_layers);
  c.rope = r.BoolArray("attention.rope_pattern", c.num_layers);
  if (!r.ok)
    return std::nullopt;
  c.attention.reserve(c.num_layers);
  for (const bool is_sliding : sliding) {
    c.attention.push_back(is_sliding ? AttentionKind::kSliding
                                     : AttentionKind::kFull);
  }

  const auto* embedding = gguf.FindTensor("token_embd.weight");
  if (embedding == nullptr || embedding->dimensions.size() != 2 ||
      embedding->dimensions[0] != c.hidden_size ||
      embedding->dimensions[1] == 0 ||
      embedding->dimensions[1] > std::numeric_limits<std::uint32_t>::max()) {
    r.Fail("token_embd.weight must be [embedding_length, vocab]");
    return std::nullopt;
  }
  c.vocab_size = static_cast<std::uint32_t>(embedding->dimensions[1]);
  return c;
}

}  // namespace gufo::models::flint
