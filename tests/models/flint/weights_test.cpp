// Binds the real Flint GGUF named by GUFO_FLINT_MODEL and checks every bound
// tensor against the file: shape, type, payload address, byte geometry, the
// dense/MoE split and the attention patterns. Exits 77 (skip) without it.

#include "src/models/flint/weights.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace flint = gufo::models::flint;
using gufo::core::GgmlType;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

/// Block geometry from the GGML format definitions, kept independent of
/// core::quant: {elements per block, bytes per block}.
std::pair<std::uint64_t, std::uint64_t> Block(GgmlType type) {
  switch (type) {
    case GgmlType::kF32:
      return {1, 4};
    case GgmlType::kBF16:
    case GgmlType::kF16:
      return {1, 2};
    case GgmlType::kQ8_0:
      return {32, 34};
    case GgmlType::kQ4_K:
      return {256, 144};
    case GgmlType::kQ5_K:
      return {256, 176};
    case GgmlType::kQ6_K:
      return {256, 210};
    case GgmlType::kIQ4_XS:
      return {256, 136};
    case GgmlType::kIQ3_S:
      return {256, 110};
    default:
      throw std::runtime_error("unexpected tensor type in the Flint artifact");
  }
}

struct Checker {
  const gufo::core::GgufReader& gguf;
  std::set<std::string> bound;

  void Tensor(const flint::TensorRef& t, const std::string& name,
              std::uint64_t cols, std::uint64_t rows, std::uint64_t experts) {
    Require(static_cast<bool>(t), name + " is not bound");
    Require(t.name == name, "bound " + std::string(t.name) + " for " + name);
    const auto* info = gguf.FindTensor(name);
    Require(info != nullptr && info->data == t.data && info->type == t.type,
            name + " does not alias the file tensor");
    Require(t.cols == cols && t.rows == rows && t.experts == experts,
            name + " has the wrong shape");
    const auto [block, bytes] = Block(t.type);
    Require(cols % block == 0, name + " row is not block aligned");
    Require(t.row_bytes == cols / block * bytes, name + " row bytes");
    Require(t.expert_bytes == t.row_bytes * rows, name + " expert bytes");
    Require(t.size_bytes == t.expert_bytes * experts &&
                t.size_bytes == info->size_bytes,
            name + " size bytes");
    Require(t.Row(experts - 1, rows - 1) + t.row_bytes ==
                static_cast<const std::uint8_t*>(t.data) + t.size_bytes,
            name + " last row does not end the payload");
    Require(static_cast<const std::uint8_t*>(gguf.GetData()) + t.file_offset ==
                t.data,
            name + " file offset");
    bound.insert(name);
  }
};

void CheckModel(const gufo::core::GgufReader& gguf) {
  std::string error;
  const auto weights = flint::ModelWeights::Bind(gguf, &error);
  Require(weights.has_value(), "bind failed: " + error);
  const flint::Config& c = weights->config;
  Require(c.num_layers == 49 && c.hidden_size == 2048 &&
              c.vocab_size == 150144 && c.num_heads == 32 &&
              c.num_kv_heads == 4 && c.head_dim == 128 &&
              c.leading_dense_layers == 2 && c.num_experts == 256 &&
              c.experts_per_token == 6 && c.expert_ffn_dim == 512 &&
              c.shared_expert_ffn_dim == 1024 && c.dense_ffn_dim == 6144 &&
              c.sliding_window == 513,
          "unexpected Flint geometry");
  Require(weights->layers.size() == c.num_layers, "layer count");

  const auto* sliding =
      gguf.FindMetadata("flint.attention.sliding_window_pattern");
  const auto* rope = gguf.FindMetadata("flint.attention.rope_pattern");
  Require(sliding != nullptr && rope != nullptr, "missing layer patterns");
  // Bool arrays are stored as unsigned values by the reader.
  const auto& sliding_values =
      std::get<std::vector<std::uint64_t>>(sliding->value);
  const auto& rope_values = std::get<std::vector<std::uint64_t>>(rope->value);
  std::uint32_t sliding_layers = 0;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    Require(c.IsSliding(il) == (sliding_values.at(il) != 0),
            "sliding pattern of layer " + std::to_string(il));
    Require(c.UsesRope(il) == (rope_values.at(il) != 0),
            "rope pattern of layer " + std::to_string(il));
    sliding_layers += c.IsSliding(il) ? 1 : 0;
  }
  Require(sliding_layers == 35, "expected 35 sliding-window layers");

  Checker check{gguf, {}};
  check.Tensor(weights->token_embd, "token_embd.weight", 2048, 150144, 1);
  check.Tensor(weights->output_norm, "output_norm.weight", 2048, 1, 1);
  check.Tensor(weights->output, "output.weight", 2048, 150144, 1);
  Require(weights->token_embd.type == GgmlType::kQ8_0 &&
              weights->output.type == GgmlType::kQ8_0,
          "embedding and LM head are Q8_0 in UD-Q4_K_XL");
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const flint::LayerWeights& l = weights->layers[il];
    const std::string p = "blk." + std::to_string(il) + ".";
    check.Tensor(l.attn_norm, p + "attn_norm.weight", 2048, 1, 1);
    check.Tensor(l.q, p + "attn_q.weight", 2048, 4096, 1);
    check.Tensor(l.k, p + "attn_k.weight", 2048, 512, 1);
    check.Tensor(l.v, p + "attn_v.weight", 2048, 512, 1);
    check.Tensor(l.o, p + "attn_output.weight", 4096, 2048, 1);
    check.Tensor(l.q_norm, p + "attn_q_norm.weight", 128, 1, 1);
    check.Tensor(l.k_norm, p + "attn_k_norm.weight", 128, 1, 1);
    check.Tensor(l.sinks, p + "attn_sinks.weight", 32, 1, 1);
    check.Tensor(l.ffn_norm, p + "ffn_norm.weight", 2048, 1, 1);
    for (const auto* norm :
         {&l.attn_norm, &l.q_norm, &l.k_norm, &l.sinks, &l.ffn_norm}) {
      Require(norm->type == GgmlType::kF32, p + " vectors are F32");
    }
    Require(!l.router_bias, p + " has no router bias in this artifact");
    if (il < 2) {
      check.Tensor(l.gate, p + "ffn_gate.weight", 2048, 6144, 1);
      check.Tensor(l.up, p + "ffn_up.weight", 2048, 6144, 1);
      check.Tensor(l.down, p + "ffn_down.weight", 6144, 2048, 1);
      Require(!l.router && !l.gate_exps && !l.gate_shexp,
              p + " dense layer bound MoE tensors");
      continue;
    }
    Require(!l.gate && !l.up && !l.down, p + " MoE layer bound dense FFN");
    check.Tensor(l.router, p + "ffn_gate_inp.weight", 2048, 256, 1);
    Require(l.router.type == GgmlType::kBF16, p + " router is BF16");
    check.Tensor(l.gate_exps, p + "ffn_gate_exps.weight", 2048, 512, 256);
    check.Tensor(l.up_exps, p + "ffn_up_exps.weight", 2048, 512, 256);
    check.Tensor(l.down_exps, p + "ffn_down_exps.weight", 512, 2048, 256);
    check.Tensor(l.gate_shexp, p + "ffn_gate_shexp.weight", 2048, 1024, 1);
    check.Tensor(l.up_shexp, p + "ffn_up_shexp.weight", 2048, 1024, 1);
    check.Tensor(l.down_shexp, p + "ffn_down_shexp.weight", 1024, 2048, 1);
  }

  // Nothing in the file may be silently ignored by the binding.
  for (const auto& info : gguf.GetTensors()) {
    Require(check.bound.count(std::string(info.name)) == 1,
            "file tensor " + std::string(info.name) + " is not bound");
  }
  Require(check.bound.size() == gguf.GetTensorCount(), "bound tensor count");
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_FLINT_MODEL");
  if (path == nullptr || *path == '\0') {
    std::cout << "flint weights: GUFO_FLINT_MODEL not set, skipping\n";
    return 77;
  }
  try {
    std::string error;
    const auto gguf = gufo::core::GgufReader::OpenFile(path, &error);
    Require(gguf != nullptr, "cannot open " + std::string(path) + ": " + error);
    CheckModel(*gguf);
  } catch (const std::exception& e) {
    std::cerr << "flint weights: " << e.what() << "\n";
    return 1;
  }
  std::cout << "flint weights: OK\n";
  return 0;
}
