#include "src/models/flint/weights.hpp"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::flint {
namespace {

using core::GgmlType;

/// Matrix storage formats with a CPU decoder in `core::quant`.
constexpr std::initializer_list<GgmlType> kMatrixTypes = {
    GgmlType::kF32,    GgmlType::kF16,    GgmlType::kBF16, GgmlType::kQ8_0,
    GgmlType::kQ3_K,   GgmlType::kQ4_K,   GgmlType::kQ5_K, GgmlType::kQ6_K,
    GgmlType::kIQ4_NL, GgmlType::kIQ4_XS, GgmlType::kIQ3_S};
constexpr std::initializer_list<GgmlType> kRouterTypes = {
    GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32};
constexpr std::initializer_list<GgmlType> kVectorTypes = {GgmlType::kF32};

std::string Shape(std::uint64_t cols, std::uint64_t rows,
                  std::uint64_t experts) {
  return "[" + std::to_string(cols) + ", " + std::to_string(rows) + ", " +
         std::to_string(experts) + "]";
}

struct Binder {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Binds `name` with the exact shape and one of `types`. A missing optional
  /// tensor yields an empty reference.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts, std::initializer_list<GgmlType> types,
                bool required = true) {
    TensorRef t;
    if (!ok) {
      return t;
    }
    const auto* info = gguf.FindTensor(name);
    if (info == nullptr) {
      if (required) {
        Fail("flint: missing tensor " + name);
      }
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_cols = !d.empty() ? d[0] : 1;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_experts = d.size() > 2 ? d[2] : 1;
    bool trailing_ones = true;
    for (std::size_t i = 3; i < d.size(); ++i) {
      trailing_ones = trailing_ones && d[i] == 1;
    }
    if (got_cols != cols || got_rows != rows || got_experts != experts ||
        !trailing_ones) {
      Fail("flint: tensor " + name + " has shape " +
           Shape(got_cols, got_rows, got_experts) + ", expected " +
           Shape(cols, rows, experts));
      return t;
    }
    bool accepted = false;
    for (const GgmlType type : types) {
      accepted = accepted || info->type == type;
    }
    if (!accepted) {
      std::string allowed;
      for (const GgmlType type : types) {
        allowed +=
            (allowed.empty() ? "" : ", ") + std::string(core::ToString(type));
      }
      Fail("flint: tensor " + name + " has unsupported type " +
           std::string(core::ToString(info->type)) + " (accepted: " + allowed +
           ")");
      return t;
    }
    const std::size_t row_bytes = quant::EncodedSizeBytes(info->type, cols);
    if (row_bytes == 0) {
      Fail("flint: tensor " + name + " row of " + std::to_string(cols) +
           " elements is not block aligned for " +
           std::string(core::ToString(info->type)));
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.name = info->name;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.row_bytes = row_bytes;
    t.expert_bytes = row_bytes * rows;
    t.size_bytes = t.expert_bytes * experts;
    if (info->size_bytes != 0 && info->size_bytes != t.size_bytes) {
      Fail("flint: tensor " + name + " stores " +
           std::to_string(info->size_bytes) + " bytes, geometry needs " +
           std::to_string(t.size_bytes));
      return TensorRef{};
    }
    // A payload running past its mapped region is caught here instead of as
    // a fault in a kernel or the reference.
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool inside = false;
    for (const auto& region : gguf.GetMappedRegions()) {
      const auto base = reinterpret_cast<std::uintptr_t>(region.data);
      if (address >= base && address < base + region.size) {
        t.file_offset = address - base;
        inside = t.size_bytes <= region.size - t.file_offset;
        break;
      }
    }
    if (!inside) {
      Fail("flint: tensor " + name + " is truncated");
      return TensorRef{};
    }
    return t;
  }

  LayerWeights Layer(const Config& c, std::uint32_t il) {
    LayerWeights l;
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = c.hidden_size;
    l.attn_norm = Get(p + "attn_norm.weight", hidden, 1, 1, kVectorTypes);
    l.q = Get(p + "attn_q.weight", hidden, c.QueryWidth(), 1, kMatrixTypes);
    l.k = Get(p + "attn_k.weight", hidden, c.KvWidth(), 1, kMatrixTypes);
    l.v = Get(p + "attn_v.weight", hidden, c.KvWidth(), 1, kMatrixTypes);
    l.o =
        Get(p + "attn_output.weight", c.QueryWidth(), hidden, 1, kMatrixTypes);
    l.q_norm = Get(p + "attn_q_norm.weight", c.head_dim, 1, 1, kVectorTypes);
    l.k_norm = Get(p + "attn_k_norm.weight", c.head_dim, 1, 1, kVectorTypes);
    l.sinks = Get(p + "attn_sinks.weight", c.num_heads, 1, 1, kVectorTypes);
    l.ffn_norm = Get(p + "ffn_norm.weight", hidden, 1, 1, kVectorTypes);
    if (c.IsDense(il)) {
      l.gate =
          Get(p + "ffn_gate.weight", hidden, c.dense_ffn_dim, 1, kMatrixTypes);
      l.up = Get(p + "ffn_up.weight", hidden, c.dense_ffn_dim, 1, kMatrixTypes);
      l.down =
          Get(p + "ffn_down.weight", c.dense_ffn_dim, hidden, 1, kMatrixTypes);
      return l;
    }
    l.router =
        Get(p + "ffn_gate_inp.weight", hidden, c.num_experts, 1, kRouterTypes);
    l.router_bias = Get(p + "ffn_exp_probs_b.bias", c.num_experts, 1, 1,
                        kVectorTypes, /*required=*/false);
    l.gate_exps = Get(p + "ffn_gate_exps.weight", hidden, c.expert_ffn_dim,
                      c.num_experts, kMatrixTypes);
    l.up_exps = Get(p + "ffn_up_exps.weight", hidden, c.expert_ffn_dim,
                    c.num_experts, kMatrixTypes);
    l.down_exps = Get(p + "ffn_down_exps.weight", c.expert_ffn_dim, hidden,
                      c.num_experts, kMatrixTypes);
    l.gate_shexp = Get(p + "ffn_gate_shexp.weight", hidden,
                       c.shared_expert_ffn_dim, 1, kMatrixTypes);
    l.up_shexp = Get(p + "ffn_up_shexp.weight", hidden, c.shared_expert_ffn_dim,
                     1, kMatrixTypes);
    l.down_shexp = Get(p + "ffn_down_shexp.weight", c.shared_expert_ffn_dim,
                       hidden, 1, kMatrixTypes);
    return l;
  }
};

}  // namespace

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& gguf,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(gguf, error_msg);
  if (!config) {
    return std::nullopt;
  }
  ModelWeights w;
  w.config = std::move(*config);
  const Config& c = w.config;
  Binder b{gguf, error_msg};
  w.token_embd =
      b.Get("token_embd.weight", c.hidden_size, c.vocab_size, 1, kMatrixTypes);
  w.output_norm =
      b.Get("output_norm.weight", c.hidden_size, 1, 1, kVectorTypes);
  w.output =
      b.Get("output.weight", c.hidden_size, c.vocab_size, 1, kMatrixTypes);
  w.layers.reserve(c.num_layers);
  for (std::uint32_t il = 0; il < c.num_layers && b.ok; ++il) {
    w.layers.push_back(b.Layer(c, il));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

}  // namespace gufo::models::flint
