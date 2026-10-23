#include "src/models/flint/kernels/rocm/executor.hpp"

#include <algorithm>
#include <cstring>

#include "src/core/hip/managed_alloc.hpp"

namespace gufo::models::flint::rocm {
namespace {

constexpr int kQkvWidth = (kQHeads + 2 * kKvHeads) * kHeadDim;
constexpr std::size_t kAlign = 256;
/// Bytes of one position of one layer's K (or V): every KV head's row.
constexpr std::size_t kSlotBytes = std::size_t{kKvHeads} * kHeadDim * 2;

std::size_t AlignUp(std::size_t n) {
  return (n + kAlign - 1) / kAlign * kAlign;
}

bool Fail(std::string* error_msg, std::string_view what, hipError_t status) {
  if (error_msg != nullptr) {
    *error_msg = std::string(what) + ": " + hipGetErrorString(status);
  }
  return false;
}

/// Sequential carve-out of one allocation; with `base` null it only counts.
struct Carver {
  std::uint8_t* base;
  std::size_t used = 0;
  template<typename T>
  T* Take(std::size_t count) {
    T* p = base == nullptr ? nullptr : reinterpret_cast<T*>(base + used);
    used += AlignUp(count * sizeof(T));
    return p;
  }
};

}  // namespace

#define FLINT_TRY(call)                          \
  do {                                           \
    if (const hipError_t flint_status_ = (call); \
        flint_status_ != hipSuccess) {           \
      return flint_status_;                      \
    }                                            \
  } while (false)

// ---------------------------------------------------------------------------
// SessionCache

std::size_t SessionCache::Bytes(const Config& config, std::uint32_t context,
                                std::uint32_t ring) {
  std::size_t bytes = AlignUp(config.num_layers * sizeof(KvSequence));
  for (std::uint32_t il = 0; il < config.num_layers; ++il) {
    const std::uint32_t slots =
        config.IsSliding(il) ? std::min(context, ring) : context;
    bytes += 2 * AlignUp(slots * kSlotBytes);
  }
  return bytes;
}

std::unique_ptr<SessionCache> SessionCache::Create(const Config& config,
                                                   std::uint32_t context,
                                                   std::uint32_t ring,
                                                   std::string* error_msg) {
  std::unique_ptr<SessionCache> cache(new SessionCache());
  cache->context_ = context;
  cache->bytes_ = Bytes(config, context, ring);
  hipError_t status = hip::AllocateDevice(&cache->storage_, cache->bytes_);
  if (status != hipSuccess) {
    (void)hipGetLastError();
    cache->storage_ = nullptr;
    Fail(error_msg, "session cache allocation", status);
    return nullptr;
  }
  auto* base = static_cast<std::uint8_t*>(cache->storage_);
  cache->device_seqs_ = reinterpret_cast<KvSequence*>(base);
  std::size_t offset = AlignUp(config.num_layers * sizeof(KvSequence));
  cache->seqs_.resize(config.num_layers);
  for (std::uint32_t il = 0; il < config.num_layers; ++il) {
    const std::uint32_t slots =
        config.IsSliding(il) ? std::min(context, ring) : context;
    const std::size_t bytes = AlignUp(slots * kSlotBytes);
    cache->seqs_[il] = {.k = reinterpret_cast<__half*>(base + offset),
                        .v = reinterpret_cast<__half*>(base + offset + bytes),
                        .capacity = static_cast<std::int32_t>(slots)};
    offset += 2 * bytes;
  }
  status = hipMemset(cache->storage_, 0, cache->bytes_);
  if (status == hipSuccess) {
    status = hipMemcpy(cache->device_seqs_, cache->seqs_.data(),
                       cache->seqs_.size() * sizeof(KvSequence),
                       hipMemcpyHostToDevice);
  }
  if (status != hipSuccess) {
    Fail(error_msg, "session cache initialization", status);
    return nullptr;
  }
  return cache;
}

SessionCache::~SessionCache() {
  if (storage_ != nullptr)
    (void)hipFree(storage_);
}

// ---------------------------------------------------------------------------
// Executor

/// Where one pass reads its per-row inputs and its sequences' caches.
struct Executor::Rows {
  const std::int32_t* positions;
  const std::int32_t* row_seq;  ///< null for one sequence
  const KvSequence* seqs;       ///< layer il uses seqs + il * seq_stride
  int seq_stride;
  /// Prefill only: the sequence and its first position.
  const SessionCache* cache;
  int first_position;
};

Executor::Executor(const DeviceModel& model) : model_(model) {}

Executor::~Executor() {
  if (stream_ != nullptr)
    (void)hipStreamSynchronize(stream_);
  for (hipGraphExec_t graph : graphs_) {
    if (graph != nullptr)
      (void)hipGraphExecDestroy(graph);
  }
  for (hipEvent_t event : layer_done_) {
    if (event != nullptr)
      (void)hipEventDestroy(event);
  }
  if (arena_ != nullptr)
    (void)hipFree(arena_);
  if (params_host_ != nullptr)
    (void)hipHostFree(params_host_);
  if (stream_ != nullptr)
    (void)hipStreamDestroy(stream_);
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           int prefill_rows, int max_batch,
                                           std::string* error_msg) {
  if (prefill_rows < 1 || max_batch < 1 || max_batch > kVectorRows) {
    if (error_msg != nullptr)
      *error_msg = "executor needs positive prefill rows and 1 to " +
                   std::to_string(kVectorRows) + " decode rows";
    return nullptr;
  }
  std::unique_ptr<Executor> e(new Executor(model));
  e->prefill_rows_ = prefill_rows;
  e->max_batch_ = max_batch;
  e->graphs_.assign(static_cast<std::size_t>(max_batch) + 1, nullptr);
  const Config& c = model.config();
  const int rows = std::max(prefill_rows, max_batch);
  const int used = static_cast<int>(c.experts_per_token);
  const int vocab = model.output().rows;

  // Operator scratch is shared by the ops of one stream, so it only has to
  // hold the largest single request: decode runs the vector tier for 1 to
  // max_batch rows, prefill the matrix tier for 1 to prefill_rows rows.
  const int hidden = static_cast<int>(c.hidden_size);
  const int dense = static_cast<int>(c.dense_ffn_dim);
  const int shared = static_cast<int>(c.shared_expert_ffn_dim);
  const int query = static_cast<int>(c.QueryWidth());
  std::size_t scratch = AttentionDecodeScratchBytes(max_batch);
  for (int n = 1; n <= max_batch; ++n) {
    for (const int cols : {hidden, query, dense, shared})
      scratch = std::max(scratch, ProjectionScratchBytes(n, cols));
    scratch =
        std::max({scratch, SwiGluScratchBytes(n, hidden, dense),
                  SwiGluScratchBytes(n, hidden, shared),
                  RouterScratchBytes(n, static_cast<int>(c.num_experts))});
    for (const DeviceLayer& l : model.layers()) {
      if (!l.dense)
        scratch = std::max(scratch, RoutedScratchBytes(l.experts, n, used));
    }
  }
  for (int n = 1; n <= prefill_rows; ++n) {
    for (const int cols : {hidden, query, dense, shared})
      scratch = std::max(scratch, MatrixProjectionScratchBytes(n, cols));
    scratch =
        std::max({scratch, MatrixSwiGluScratchBytes(n, hidden, dense),
                  MatrixSwiGluScratchBytes(n, hidden, shared),
                  RouterScratchBytes(n, static_cast<int>(c.num_experts))});
    for (const DeviceLayer& l : model.layers()) {
      if (!l.dense)
        scratch =
            std::max(scratch, MatrixRoutedScratchBytes(l.experts, n, used));
    }
  }

  // Per-call parameters head the arena; the pinned host staging copy uses
  // the same layout, so one copy moves them all.
  const auto params = [&](Carver& a, Params& p) {
    p.tokens = a.Take<std::int32_t>(rows);
    p.positions = a.Take<std::int32_t>(rows);
    p.row_seq = a.Take<std::int32_t>(max_batch);
    p.seqs =
        a.Take<KvSequence>(static_cast<std::size_t>(c.num_layers) * max_batch);
  };
  Carver measure_params{nullptr};
  params(measure_params, e->host_);
  const auto carve = [&](Carver& a) {
    params(a, e->device_);
    const auto n = static_cast<std::size_t>(rows);
    e->residual_ = a.Take<float>(n * kHidden);
    e->xn_ = a.Take<float>(n * kHidden);
    e->qkv_ = a.Take<float>(n * kQkvWidth);
    e->q_ = a.Take<float>(n * kQHeads * kHeadDim);
    e->attn_ = a.Take<float>(n * kQHeads * kHeadDim);
    e->delta_ = a.Take<float>(n * kHidden);
    e->hidden_ =
        a.Take<float>(n * std::max(c.dense_ffn_dim, c.shared_expert_ffn_dim));
    e->shared_ = a.Take<float>(n * kHidden);
    e->expert_out_ = a.Take<float>(n * used * kHidden);
    e->ids_ = a.Take<std::int32_t>(n * used);
    e->weights_ = a.Take<float>(n * used);
    e->logits_ = a.Take<float>(static_cast<std::size_t>(max_batch) * vocab);
    e->q8_ = a.Take<std::uint8_t>(
        VectorActivationBytes(max_batch, static_cast<int>(c.QueryWidth())));
    e->scratch_ = a.Take<std::uint8_t>(scratch);
    e->scratch_size_ = scratch;
  };
  Carver measure{nullptr};
  carve(measure);
  e->arena_bytes_ = measure.used;
  hipError_t status = hip::AllocateDevice(&e->arena_, e->arena_bytes_);
  if (status != hipSuccess) {
    (void)hipGetLastError();
    e->arena_ = nullptr;
    Fail(error_msg, "executor scratch allocation", status);
    return nullptr;
  }
  Carver arena{static_cast<std::uint8_t*>(e->arena_)};
  carve(arena);

  e->params_bytes_ = measure_params.used;
  status =
      hipHostMalloc(&e->params_host_, e->params_bytes_, hipHostMallocDefault);
  if (status != hipSuccess) {
    (void)hipGetLastError();
    e->params_host_ = nullptr;
    Fail(error_msg, "executor parameter allocation", status);
    return nullptr;
  }
  Carver host{static_cast<std::uint8_t*>(e->params_host_)};
  params(host, e->host_);
  for (int i = 0; i < max_batch; ++i)
    e->host_.row_seq[i] = i;

  status = hipStreamCreateWithFlags(&e->stream_, hipStreamNonBlocking);
  for (hipEvent_t& event : e->layer_done_) {
    if (status == hipSuccess)
      status = hipEventCreateWithFlags(&event, hipEventDisableTiming);
  }
  if (status != hipSuccess) {
    Fail(error_msg, "executor stream creation", status);
    return nullptr;
  }
  return e;
}

hipError_t Executor::EnqueueLayer(std::uint32_t il, int rows, const Rows& in) {
  const DeviceLayer& l = model_.layers()[il];
  const Config& c = model_.config();
  const float eps = c.rms_eps;
  // Decode rows (<= kVectorRows) quantize each projection input once, where
  // it is produced, and every consumer reads that Q8_1 copy; the results are
  // bit-identical to quantizing inside each projection.
  const bool decode = in.cache == nullptr;
  const auto add_norm = [&](const float* w) {
    return decode ? AddRmsNorm(residual_, delta_, w, xn_, q8_, rows, kHidden,
                               eps, stream_)
                  : AddRmsNorm(residual_, delta_, w, xn_, rows, kHidden, eps,
                               stream_);
  };
  if (il == 0) {
    FLINT_TRY(decode ? RmsNorm(residual_, l.attn_norm, xn_, q8_, rows, kHidden,
                               eps, stream_)
                     : RmsNorm(residual_, l.attn_norm, xn_, rows, kHidden, eps,
                               stream_));
  } else {
    FLINT_TRY(add_norm(l.attn_norm));
  }
  // Prefill runs the matrix tier for every row count, so a row's result does
  // not depend on how a prompt was split into chunks.
  const auto project = [&](const QuantMatrix& w, const float* x, float* y) {
    return decode
               ? Project(w, x, y, rows, scratch_, scratch_size_, stream_)
               : ProjectMatrix(w, x, y, rows, scratch_, scratch_size_, stream_);
  };
  FLINT_TRY(decode ? ProjectQuantized(l.qkv, q8_, qkv_, rows, stream_)
                   : project(l.qkv, xn_, qkv_));
  const KvSequence* seqs =
      in.seqs + static_cast<std::ptrdiff_t>(il) * in.seq_stride;
  FLINT_TRY(PrepareAttention(qkv_, kQkvWidth, q_, l.q_norm, l.k_norm,
                             in.positions, in.row_seq, seqs, rows, l.rope,
                             c.rope_theta, eps, stream_));
  const int window = l.sliding ? kSlidingWindow : 0;
  if (decode) {
    FLINT_TRY(AttentionDecode(q_, in.positions, in.row_seq, seqs, l.sinks,
                              window, attn_, q8_, rows, scratch_, scratch_size_,
                              stream_));
    FLINT_TRY(ProjectQuantized(l.o, q8_, delta_, rows, stream_));
  } else {
    FLINT_TRY(AttentionPrefill(q_, in.first_position, in.cache->seqs()[il],
                               l.sinks, window, attn_, rows, stream_));
    FLINT_TRY(project(l.o, attn_, delta_));
  }
  FLINT_TRY(add_norm(l.ffn_norm));
  const auto swiglu = [&](const QuantMatrix& gate, const QuantMatrix& up) {
    return decode ? ProjectSwiGluQuantized(gate, up, q8_, hidden_, rows,
                                           scratch_, scratch_size_, stream_)
                  : ProjectSwiGluMatrix(gate, up, xn_, hidden_, rows, scratch_,
                                        scratch_size_, stream_);
  };
  if (l.dense) {
    FLINT_TRY(swiglu(l.gate, l.up));
    return project(l.down, hidden_, delta_);
  }
  const int used = static_cast<int>(c.experts_per_token);
  FLINT_TRY(RouterTopK(l.router, xn_, l.router_bias, c.expert_weights_scale,
                       rows, kHidden, static_cast<int>(c.num_experts), used,
                       ids_, weights_, scratch_, scratch_size_, stream_));
  FLINT_TRY(swiglu(l.shared_gate, l.shared_up));
  FLINT_TRY(project(l.shared_down, hidden_, shared_));
  // The shared down projection quantized its own input, so q8_ still holds
  // the ffn_norm rows.
  FLINT_TRY(decode ? RoutedForwardQuantized(l.experts, q8_, ids_, rows, used,
                                            expert_out_, scratch_,
                                            scratch_size_, stream_)
                   : RoutedForwardMatrix(l.experts, xn_, ids_, rows, used,
                                         expert_out_, scratch_, scratch_size_,
                                         stream_));
  // delta = shared + routed; the next norm adds it to the residual, the same
  // single rounding as adding into the residual here.
  return MoeCombine(expert_out_, weights_, shared_, nullptr, delta_, rows, used,
                    kHidden, stream_);
}

Executor::Status Executor::Prefill(SessionCache& cache,
                                   std::span<const std::uint32_t> tokens,
                                   std::uint32_t first_position, float* logits,
                                   const std::function<bool()>& cancelled,
                                   std::string* error_msg, float* all_logits) {
  const int rows = static_cast<int>(tokens.size());
  if (rows < 1 || rows > prefill_rows_ ||
      first_position + tokens.size() > cache.context()) {
    if (error_msg != nullptr)
      *error_msg = "prefill chunk exceeds the executor or session bounds";
    return Status::kFailed;
  }
  const auto failed = [&](std::string_view what, hipError_t status) {
    (void)hipStreamSynchronize(stream_);
    Fail(error_msg, what, status);
    return Status::kFailed;
  };
  for (int i = 0; i < rows; ++i) {
    host_.tokens[i] = static_cast<std::int32_t>(tokens[i]);
    host_.positions[i] = static_cast<std::int32_t>(first_position + i);
  }
  hipError_t status =
      hipMemcpyAsync(device_.tokens, host_.tokens,
                     reinterpret_cast<std::uint8_t*>(host_.row_seq) -
                         reinterpret_cast<std::uint8_t*>(host_.tokens),
                     hipMemcpyHostToDevice, stream_);
  if (status == hipSuccess) {
    status = EmbedQ8_0(model_.token_embd(), device_.tokens, rows, kHidden,
                       residual_, stream_);
  }
  if (status != hipSuccess)
    return failed("prefill embedding", status);
  const Rows in{.positions = device_.positions,
                .row_seq = nullptr,
                .seqs = cache.device_seqs(),
                .seq_stride = 1,
                .cache = &cache,
                .first_position = static_cast<int>(first_position)};
  const auto layers = static_cast<std::uint32_t>(model_.layers().size());
  for (std::uint32_t il = 0; il < layers; ++il) {
    status = EnqueueLayer(il, rows, in);
    if (status != hipSuccess)
      return failed("prefill layer", status);
    if (tap_) {
      status = hipStreamSynchronize(stream_);
      if (status != hipSuccess)
        return failed("prefill tap", status);
      tap_("kqv_out", il, attn_, kQHeads * kHeadDim);
      tap_("ffn_inp", il, residual_, kHidden);
      tap_("ffn_norm", il, xn_, kHidden);
      tap_("ffn_out", il, delta_, kHidden);
    }
    if (!cancelled)
      continue;
    // Keep one layer queued ahead while polling, so the device never idles
    // on the check.
    status = hipEventRecord(layer_done_[il & 1], stream_);
    if (status == hipSuccess && il > 0)
      status = hipEventSynchronize(layer_done_[(il - 1) & 1]);
    if (status != hipSuccess)
      return failed("prefill progress", status);
    if (cancelled()) {
      (void)hipStreamSynchronize(stream_);
      return Status::kCancelled;
    }
  }
  const Config& c = model_.config();
  const std::size_t vocab = static_cast<std::size_t>(model_.output().rows);
  const std::size_t last = static_cast<std::size_t>(rows - 1) * kHidden;
  if (all_logits != nullptr) {
    status = AddRmsNorm(residual_, delta_, model_.output_norm(), xn_, rows,
                        kHidden, c.rms_eps, stream_);
    for (int r = 0; r < rows && status == hipSuccess; ++r) {
      status =
          Project(model_.output(), xn_ + static_cast<std::size_t>(r) * kHidden,
                  logits_, 1, scratch_, scratch_size_, stream_);
      if (status == hipSuccess) {
        status = hipMemcpyAsync(all_logits + r * vocab, logits_,
                                vocab * sizeof(float), hipMemcpyDeviceToHost,
                                stream_);
      }
      if (status == hipSuccess)
        status = hipStreamSynchronize(stream_);
    }
  } else {
    status = AddRmsNorm(residual_ + last, delta_ + last, model_.output_norm(),
                        xn_, 1, kHidden, c.rms_eps, stream_);
    if (status == hipSuccess) {
      status = Project(model_.output(), xn_, logits_, 1, scratch_,
                       scratch_size_, stream_);
    }
  }
  if (status == hipSuccess) {
    status = hipMemcpyAsync(logits, logits_, vocab * sizeof(float),
                            hipMemcpyDeviceToHost, stream_);
  }
  if (status == hipSuccess)
    status = hipStreamSynchronize(stream_);
  if (status != hipSuccess)
    return failed("prefill head", status);
  return Status::kOk;
}

hipError_t Executor::EnqueueDecode(int rows) {
  FLINT_TRY(EmbedQ8_0(model_.token_embd(), device_.tokens, rows, kHidden,
                      residual_, stream_));
  const Rows in{.positions = device_.positions,
                .row_seq = device_.row_seq,
                .seqs = device_.seqs,
                .seq_stride = max_batch_,
                .cache = nullptr,
                .first_position = 0};
  const auto layers = static_cast<std::uint32_t>(model_.layers().size());
  for (std::uint32_t il = 0; il < layers; ++il)
    FLINT_TRY(EnqueueLayer(il, rows, in));
  FLINT_TRY(AddRmsNorm(residual_, delta_, model_.output_norm(), xn_, rows,
                       kHidden, model_.config().rms_eps, stream_));
  return Project(model_.output(), xn_, logits_, rows, scratch_, scratch_size_,
                 stream_);
}

bool Executor::Decode(std::span<SessionCache* const> caches,
                      std::span<const std::uint32_t> tokens,
                      std::span<const std::uint32_t> positions,
                      std::span<float* const> logits, std::string* error_msg) {
  const int rows = static_cast<int>(caches.size());
  if (rows < 1 || rows > max_batch_ || tokens.size() != caches.size() ||
      positions.size() != caches.size() || logits.size() != caches.size()) {
    if (error_msg != nullptr)
      *error_msg = "decode batch exceeds the executor bounds";
    return false;
  }
  const std::size_t layers = model_.layers().size();
  for (int i = 0; i < rows; ++i) {
    if (positions[i] >= caches[i]->context()) {
      if (error_msg != nullptr)
        *error_msg = "decode position exceeds the session context";
      return false;
    }
    host_.tokens[i] = static_cast<std::int32_t>(tokens[i]);
    host_.positions[i] = static_cast<std::int32_t>(positions[i]);
    const auto seqs = caches[i]->seqs();
    for (std::size_t il = 0; il < layers; ++il)
      host_.seqs[il * max_batch_ + i] = seqs[il];
  }
  const auto failed = [&](std::string_view what, hipError_t status) {
    (void)hipStreamSynchronize(stream_);
    return Fail(error_msg, what, status);
  };
  hipError_t status =
      hipMemcpyAsync(device_.tokens, host_.tokens, params_bytes_,
                     hipMemcpyHostToDevice, stream_);
  if (status != hipSuccess)
    return failed("decode parameters", status);
  hipGraphExec_t& graph = graphs_[rows];
  if (graph == nullptr) {
    // Thread-local capture: snapshot copies on other threads stay legal.
    status = hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
    if (status != hipSuccess)
      return failed("decode capture", status);
    const hipError_t enqueued = EnqueueDecode(rows);
    hipGraph_t captured = nullptr;
    status = hipStreamEndCapture(stream_, &captured);
    if (enqueued != hipSuccess || status != hipSuccess) {
      if (captured != nullptr)
        (void)hipGraphDestroy(captured);
      return failed("decode capture",
                    enqueued != hipSuccess ? enqueued : status);
    }
    status = hipGraphInstantiate(&graph, captured, nullptr, nullptr, 0);
    (void)hipGraphDestroy(captured);
    if (status != hipSuccess) {
      graph = nullptr;
      return failed("decode graph instantiation", status);
    }
  }
  status = hipGraphLaunch(graph, stream_);
  const std::size_t vocab = static_cast<std::size_t>(model_.output().rows);
  for (int i = 0; i < rows && status == hipSuccess; ++i) {
    status =
        hipMemcpyAsync(logits[i], logits_ + i * vocab, vocab * sizeof(float),
                       hipMemcpyDeviceToHost, stream_);
  }
  if (status == hipSuccess)
    status = hipStreamSynchronize(stream_);
  if (status != hipSuccess)
    return failed("decode", status);
  return true;
}

#undef FLINT_TRY

}  // namespace gufo::models::flint::rocm
