#include "src/models/flint/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <initializer_list>
#include <optional>
#include <string_view>

#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"

namespace gufo::models::flint::rocm {
namespace {

std::optional<WeightType> QuantType(const TensorRef& t) {
  return WeightTypeFromGgml(static_cast<std::uint32_t>(t.type));
}

/// Checks every tensor against the encodings the HIP kernels implement and
/// sums the device bytes the upload will request.
struct Validator {
  std::string* error;
  bool ok = true;
  std::uint64_t bytes = 0;

  void Fail(const TensorRef& t, std::string_view accepted) {
    if (ok && error != nullptr) {
      *error = "unsupported HIP tensor format " +
               std::string(core::ToString(t.type)) + " for " +
               std::string(t.name) + " (kernels accept " +
               std::string(accepted) + ")";
    }
    ok = false;
  }
  void Quant(const TensorRef& t) {
    if (!t)
      return;
    if (!QuantType(t) || t.cols % 256 != 0)
      Fail(t, "Q8_0, Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_XS with 256-multiple rows");
    bytes += t.size_bytes;
  }
  void Exact(const TensorRef& t, core::GgmlType type) {
    if (!t)
      return;
    if (t.type != type)
      Fail(t, core::ToString(type));
    bytes += t.size_bytes;
  }
};

struct Uploader {
  hip::WeightUpload& stager;
  const hip::WeightAllocator& weights;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::string* error;
  bool ok = true;

  void Fail(const std::string& message) {
    if (ok && error != nullptr)
      *error = message;
    ok = false;
  }

  /// Uploads the tensors' payloads back to back into one allocation.
  void* Copy(std::initializer_list<const TensorRef*> parts) {
    std::size_t size = 0;
    for (const TensorRef* t : parts)
      size += t->size_bytes;
    if (!ok || size == 0)
      return nullptr;
    void* ptr = nullptr;
    if (weights.Allocate(&ptr, size) != hipSuccess) {
      (void)hipGetLastError();
      Fail("device allocation failed for " +
           std::string((*parts.begin())->name) + " (" + std::to_string(size) +
           " bytes)");
      return nullptr;
    }
    allocations.push_back(ptr);
    bytes += size;
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      if (!stager.Copy(0, t->file_offset, t->size_bytes,
                       static_cast<std::uint8_t*>(ptr) + offset, error)) {
        Fail("upload failed for " + std::string(t->name) +
             (error != nullptr ? ": " + *error : std::string()));
        return nullptr;
      }
      offset += t->size_bytes;
    }
    return ptr;
  }

  const float* Vector(const TensorRef& t) {
    return t ? static_cast<const float*>(Copy({&t})) : nullptr;
  }

  QuantMatrix Matrix(const TensorRef& t) {
    if (!t)
      return {};
    return {.data = Copy({&t}),
            .type = *QuantType(t),
            .rows = static_cast<int>(t.rows),
            .cols = static_cast<int>(t.cols)};
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_)
    (void)hipFree(p);
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const ModelWeights& w,
                                                 const core::GgufReader& reader,
                                                 std::string* error_msg) {
  const Config& c = w.config;
  if (c.hidden_size != kHidden || c.num_heads != kQHeads ||
      c.num_kv_heads != kKvHeads || c.head_dim != kHeadDim ||
      c.sliding_window != kSlidingWindow) {
    if (error_msg != nullptr)
      *error_msg =
          "Flint HIP kernels support hidden 2048, 32/4 heads of 128 "
          "and a 513-token window only";
    return nullptr;
  }
  if (reader.GetMappedRegions().size() != 1) {
    if (error_msg != nullptr)
      *error_msg = "Flint expects a single-file GGUF";
    return nullptr;
  }
  Validator check{error_msg};
  check.Exact(w.token_embd, core::GgmlType::kQ8_0);
  check.Exact(w.output_norm, core::GgmlType::kF32);
  check.Quant(w.output);
  for (const LayerWeights& l : w.layers) {
    for (const TensorRef* t : {&l.attn_norm, &l.q_norm, &l.k_norm, &l.sinks,
                               &l.ffn_norm, &l.router_bias})
      check.Exact(*t, core::GgmlType::kF32);
    for (const TensorRef* t :
         {&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down, &l.gate_exps,
          &l.up_exps, &l.down_exps, &l.gate_shexp, &l.up_shexp, &l.down_shexp})
      check.Quant(*t);
    check.Exact(l.router, core::GgmlType::kBF16);
    if (check.ok && (l.q.type != l.k.type || l.q.type != l.v.type)) {
      if (error_msg != nullptr)
        *error_msg = "q, k and v of " + std::string(l.q.name) +
                     " use different encodings";
      return nullptr;
    }
  }
  if (!check.ok)
    return nullptr;

  auto stager = hip::WeightUpload::Create(reader.GetMappedRegions(), error_msg);
  if (!stager)
    return nullptr;
  const hip::WeightAllocator allocator(check.bytes);
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = c;
  Uploader up{*stager, allocator, m->allocations_, m->bytes_, error_msg};
  m->token_embd_ = up.Copy({&w.token_embd});
  m->output_norm_ = up.Vector(w.output_norm);
  m->output_ = up.Matrix(w.output);
  m->layers_.reserve(w.layers.size());
  for (std::uint32_t il = 0; il < w.layers.size() && up.ok; ++il) {
    const LayerWeights& l = w.layers[il];
    DeviceLayer d;
    d.sliding = c.IsSliding(il);
    d.rope = c.UsesRope(il);
    d.dense = c.IsDense(il);
    d.attn_norm = up.Vector(l.attn_norm);
    d.q_norm = up.Vector(l.q_norm);
    d.k_norm = up.Vector(l.k_norm);
    d.sinks = up.Vector(l.sinks);
    d.ffn_norm = up.Vector(l.ffn_norm);
    d.qkv = {.data = up.Copy({&l.q, &l.k, &l.v}),
             .type = *QuantType(l.q),
             .rows = static_cast<int>(l.q.rows + l.k.rows + l.v.rows),
             .cols = static_cast<int>(l.q.cols)};
    d.o = up.Matrix(l.o);
    if (d.dense) {
      d.gate = up.Matrix(l.gate);
      d.up = up.Matrix(l.up);
      d.down = up.Matrix(l.down);
    } else {
      d.router = static_cast<const std::uint16_t*>(up.Copy({&l.router}));
      d.router_bias = up.Vector(l.router_bias);
      d.experts = {.gate = up.Matrix(l.gate_exps),
                   .up = up.Matrix(l.up_exps),
                   .down = up.Matrix(l.down_exps),
                   .experts = static_cast<int>(l.gate_exps.experts)};
      d.shared_gate = up.Matrix(l.gate_shexp);
      d.shared_up = up.Matrix(l.up_shexp);
      d.shared_down = up.Matrix(l.down_shexp);
    }
    m->layers_.push_back(d);
  }
  if (!up.ok) {
    (void)stager->Finish(nullptr);
    return nullptr;
  }
  if (!stager->Finish(error_msg))
    return nullptr;
  const hipError_t status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr)
      *error_msg =
          "weight upload failed: " + std::string(hipGetErrorString(status));
    return nullptr;
  }
  return m;
}

}  // namespace gufo::models::flint::rocm
