#include "src/models/flint/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "src/models/flint/cpu_ops.hpp"

namespace gufo::models::flint {
namespace {

const float* F32(const TensorRef& t) {
  return static_cast<const float*>(t.data);
}

/// RMSNorm of each `width`-wide row of `x`.
void NormRows(std::vector<float>& x, std::size_t width, const float* gamma,
              float eps) {
  const auto count = static_cast<std::int64_t>(x.size() / width);
#pragma omp parallel for schedule(static)
  for (std::int64_t n = 0; n < count; ++n) {
    cpu::RmsNorm(std::span<float>(x.data() + n * width, width), gamma, eps);
  }
}

/// down(silu(gate(x)) * up(x)) for `count` rows of `x` (gate.cols wide),
/// written to `out` rows of down.rows floats.
void SwiGlu(const TensorRef& gate, const TensorRef& up, const TensorRef& down,
            std::uint64_t expert, const float* x, std::size_t count,
            float* out) {
  const std::size_t ff = gate.rows;
  std::vector<float> g(count * ff);
  std::vector<float> u(count * ff);
  cpu::MatMul(gate, expert, x, gate.cols, count, g.data(), ff);
  cpu::MatMul(up, expert, x, up.cols, count, u.data(), ff);
  for (std::size_t i = 0; i < g.size(); ++i) {
    g[i] = cpu::Silu(g[i]) * u[i];
  }
  cpu::MatMul(down, expert, g.data(), ff, count, out, down.rows);
}

}  // namespace

ReferenceModel::ReferenceModel(const ModelWeights& weights,
                               std::uint32_t max_context)
    : w_(weights), max_context_(max_context) {
  const std::size_t per_layer =
      static_cast<std::size_t>(max_context) * weights.config.KvWidth();
  keys_.assign(weights.config.num_layers, std::vector<float>(per_layer));
  values_.assign(weights.config.num_layers, std::vector<float>(per_layer));
}

void ReferenceModel::Emit(std::string_view name, std::int32_t layer,
                          std::span<const float> rows,
                          std::size_t width) const {
  if (!tap_) {
    return;
  }
  for (std::size_t n = 0; n * width < rows.size(); ++n) {
    tap_(name, layer, position_ + static_cast<std::uint32_t>(n),
         rows.subspan(n * width, width));
  }
}

void ReferenceModel::Attention(std::uint32_t il, std::vector<float>& x,
                               std::size_t count) {
  const Config& c = w_.config;
  const LayerWeights& l = w_.layers[il];
  const std::size_t hidden = c.hidden_size;
  const std::uint32_t d = c.head_dim;
  const std::size_t q_width = c.QueryWidth();
  const std::size_t kv_width = c.KvWidth();
  const auto layer = static_cast<std::int32_t>(il);

  std::vector<float> h = x;
  NormRows(h, hidden, F32(l.attn_norm), c.rms_eps);
  Emit("attn_norm", layer, h, hidden);

  std::vector<float> q(count * q_width);
  float* k = keys_[il].data() + position_ * kv_width;
  float* v = values_[il].data() + position_ * kv_width;
  cpu::MatMul(l.q, 0, h.data(), hidden, count, q.data(), q_width);
  cpu::MatMul(l.k, 0, h.data(), hidden, count, k, kv_width);
  cpu::MatMul(l.v, 0, h.data(), hidden, count, v, kv_width);
  for (std::size_t n = 0; n < count; ++n) {
    float* qn = q.data() + n * q_width;
    float* kn = k + n * kv_width;
    for (std::uint32_t head = 0; head < c.num_heads; ++head) {
      cpu::RmsNorm(std::span<float>(qn + head * d, d), F32(l.q_norm),
                   c.rms_eps);
    }
    for (std::uint32_t head = 0; head < c.num_kv_heads; ++head) {
      cpu::RmsNorm(std::span<float>(kn + head * d, d), F32(l.k_norm),
                   c.rms_eps);
    }
    if (c.UsesRope(il)) {
      const auto pos = position_ + static_cast<std::uint32_t>(n);
      cpu::RopeNeox(qn, c.num_heads, d, pos, c.rope_theta);
      cpu::RopeNeox(kn, c.num_kv_heads, d, pos, c.rope_theta);
    }
  }
  Emit("Qcur", layer, q, q_width);
  Emit("Kcur", layer, std::span<const float>(k, count * kv_width), kv_width);
  Emit("Vcur", layer, std::span<const float>(v, count * kv_width), kv_width);

  const std::uint32_t group = c.num_heads / c.num_kv_heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(d));
  const float* sinks = F32(l.sinks);
  const bool sliding = c.IsSliding(il);
  const auto tasks = static_cast<std::int64_t>(count * c.num_heads);
  std::vector<float> out(count * q_width);
#pragma omp parallel for schedule(dynamic)
  for (std::int64_t task = 0; task < tasks; ++task) {
    const auto n = static_cast<std::size_t>(task / c.num_heads);
    const auto head = static_cast<std::size_t>(task % c.num_heads);
    const std::uint32_t pos = position_ + static_cast<std::uint32_t>(n);
    // The window covers the current token and the sliding_window - 1 before.
    const std::uint32_t first =
        sliding && pos + 1 > c.sliding_window ? pos + 1 - c.sliding_window : 0;
    const float* qh = q.data() + n * q_width + head * d;
    const std::size_t kv_offset = head / group * d;
    std::vector<double> scores(pos + 1 - first);
    const double sink = sinks[head];
    double peak = sink;
    for (std::uint32_t j = first; j <= pos; ++j) {
      const float* kj = keys_[il].data() + j * kv_width + kv_offset;
      double dot = 0.0;
      for (std::uint32_t i = 0; i < d; ++i) {
        dot += static_cast<double>(qh[i]) * kj[i];
      }
      scores[j - first] = dot * scale;
      peak = std::max(peak, scores[j - first]);
    }
    // The sink is a logit without a value: it only enlarges the denominator.
    double denominator = std::exp(sink - peak);
    for (double& s : scores) {
      s = std::exp(s - peak);
      denominator += s;
    }
    std::vector<double> acc(d, 0.0);
    for (std::uint32_t j = first; j <= pos; ++j) {
      const float* vj = values_[il].data() + j * kv_width + kv_offset;
      const double p = scores[j - first];
      for (std::uint32_t i = 0; i < d; ++i) {
        acc[i] += p * vj[i];
      }
    }
    float* o = out.data() + n * q_width + head * d;
    for (std::uint32_t i = 0; i < d; ++i) {
      o[i] = static_cast<float>(acc[i] / denominator);
    }
  }
  Emit("kqv_out", layer, out, q_width);

  std::vector<float> attn(count * hidden);
  cpu::MatMul(l.o, 0, out.data(), q_width, count, attn.data(), hidden);
  Emit("attn_out", layer, attn, hidden);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] += attn[i];
  }
  Emit("ffn_inp", layer, x, hidden);
}

void ReferenceModel::FeedForward(std::uint32_t il, std::vector<float>& x,
                                 std::size_t count) {
  const Config& c = w_.config;
  const LayerWeights& l = w_.layers[il];
  const std::size_t hidden = c.hidden_size;
  const auto layer = static_cast<std::int32_t>(il);

  std::vector<float> h = x;
  NormRows(h, hidden, F32(l.ffn_norm), c.rms_eps);
  Emit("ffn_norm", layer, h, hidden);

  std::vector<float> ffn(count * hidden);
  if (c.IsDense(il)) {
    SwiGlu(l.gate, l.up, l.down, 0, h.data(), count, ffn.data());
  } else {
    const std::size_t experts = c.num_experts;
    const std::size_t top_k = c.experts_per_token;
    std::vector<float> scores(count * experts);
    cpu::MatMul(l.router, 0, h.data(), hidden, count, scores.data(), experts);
    for (float& s : scores) {
      s = cpu::Sigmoid(s);
    }
    // Per token: selected experts by rank and their routing weights.
    std::vector<std::uint32_t> chosen(count * top_k);
    std::vector<float> weight(count * top_k);
    std::vector<std::vector<std::size_t>> slots(experts);
    for (std::size_t n = 0; n < count; ++n) {
      const float* sn = scores.data() + n * experts;
      std::vector<float> choice(sn, sn + experts);
      if (l.router_bias) {
        for (std::size_t e = 0; e < experts; ++e) {
          choice[e] += F32(l.router_bias)[e];
        }
      }
      // Highest selection score first; equal scores keep the lower index.
      std::vector<std::uint32_t> order(experts);
      std::iota(order.begin(), order.end(), 0U);
      std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                        [&](std::uint32_t a, std::uint32_t b) {
                          return choice[a] > choice[b] ||
                                 (choice[a] == choice[b] && a < b);
                        });
      float sum = 0.0F;
      for (std::size_t r = 0; r < top_k; ++r) {
        sum += sn[order[r]];
      }
      for (std::size_t r = 0; r < top_k; ++r) {
        const std::size_t slot = n * top_k + r;
        chosen[slot] = order[r];
        weight[slot] = sn[order[r]] / (sum + 1e-20F) * c.expert_weights_scale;
        slots[order[r]].push_back(slot);
      }
    }
    // Each expert runs once over the tokens routed to it.
    std::vector<float> expert_out(count * top_k * hidden);
    std::vector<float> in;
    std::vector<float> out;
    for (std::size_t e = 0; e < experts; ++e) {
      const auto& assigned = slots[e];
      if (assigned.empty()) {
        continue;
      }
      in.resize(assigned.size() * hidden);
      out.resize(assigned.size() * hidden);
      for (std::size_t i = 0; i < assigned.size(); ++i) {
        const std::size_t n = assigned[i] / top_k;
        std::copy_n(h.data() + n * hidden, hidden, in.data() + i * hidden);
      }
      SwiGlu(l.gate_exps, l.up_exps, l.down_exps, e, in.data(), assigned.size(),
             out.data());
      for (std::size_t i = 0; i < assigned.size(); ++i) {
        std::copy_n(out.data() + i * hidden, hidden,
                    expert_out.data() + assigned[i] * hidden);
      }
    }
    std::vector<float> moe(count * hidden);
    for (std::size_t n = 0; n < count; ++n) {
      for (std::size_t i = 0; i < hidden; ++i) {
        double acc = 0.0;
        for (std::size_t r = 0; r < top_k; ++r) {
          const std::size_t slot = n * top_k + r;
          acc +=
              static_cast<double>(weight[slot]) * expert_out[slot * hidden + i];
        }
        moe[n * hidden + i] = static_cast<float>(acc);
      }
    }
    Emit("ffn_moe_out", layer, moe, hidden);
    std::vector<float> shared(count * hidden);
    SwiGlu(l.gate_shexp, l.up_shexp, l.down_shexp, 0, h.data(), count,
           shared.data());
    Emit("ffn_shexp", layer, shared, hidden);
    for (std::size_t i = 0; i < ffn.size(); ++i) {
      ffn[i] = moe[i] + shared[i];
    }
  }
  Emit("ffn_out", layer, ffn, hidden);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] += ffn[i];
  }
  Emit("l_out", layer, x, hidden);
}

bool ReferenceModel::Step(std::uint32_t token, std::span<float> logits,
                          std::string* error_msg) {
  return Forward(std::span<const std::uint32_t>(&token, 1), logits, error_msg);
}

bool ReferenceModel::Forward(std::span<const std::uint32_t> tokens,
                             std::span<float> logits, std::string* error_msg) {
  const Config& c = w_.config;
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr) {
      *error_msg = message;
    }
    return false;
  };
  const std::size_t count = tokens.size();
  if (count == 0 || count > max_context_ - position_) {
    return fail("flint reference: " + std::to_string(count) +
                " tokens do not fit the context of " +
                std::to_string(max_context_) + " at position " +
                std::to_string(position_));
  }
  for (const std::uint32_t token : tokens) {
    if (token >= c.vocab_size) {
      return fail("flint reference: token " + std::to_string(token) +
                  " is outside the vocabulary");
    }
  }
  if (logits.size() != count * c.vocab_size) {
    return fail("flint reference: logits span has " +
                std::to_string(logits.size()) + " entries, expected " +
                std::to_string(count * c.vocab_size));
  }

  const std::size_t hidden = c.hidden_size;
  std::vector<float> x(count * hidden);
  for (std::size_t n = 0; n < count; ++n) {
    cpu::DequantizeRow(w_.token_embd, 0, tokens[n], x.data() + n * hidden);
  }
  Emit("inp_embd", -1, x, hidden);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    Attention(il, x, count);
    FeedForward(il, x, count);
  }
  NormRows(x, hidden, F32(w_.output_norm), c.rms_eps);
  Emit("result_norm", -1, x, hidden);
  cpu::MatMul(w_.output, 0, x.data(), hidden, count, logits.data(),
              c.vocab_size);
  position_ += static_cast<std::uint32_t>(count);
  return true;
}

}  // namespace gufo::models::flint
