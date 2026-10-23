#include "src/models/flint/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <mutex>
#include <optional>
#include <type_traits>
#include <unordered_set>

#include "src/core/gguf_reader.hpp"
#include "src/models/flint/chat_template.hpp"
#include "src/models/flint/kernels/rocm/device_model.hpp"
#include "src/models/flint/kernels/rocm/executor.hpp"
#include "src/models/flint/kernels/rocm/kernels.hpp"
#include "src/models/flint/weights.hpp"

namespace gufo::models::flint {
namespace {

/// New tokens one Sync call evaluates in a single pass. Sliding-window rings
/// hold the window plus this many slots. gfx1151, release: 2048 against 1024
/// gives pp2048 2600 vs 2207 t/s and pp2048 @ d16384 1677 vs 1552 t/s.
constexpr std::uint32_t kPrefillChunk = 2048;
/// EvaluateBatch width.
constexpr int kMaxBatch = 8;
/// Previous keys a sliding-window query reads besides its own.
constexpr std::uint32_t kSlidingHistory = rocm::kSlidingWindow - 1;
/// Bytes of one position of one layer's K (or V) across the KV heads.
constexpr std::size_t kSlotBytes =
    std::size_t{rocm::kKvHeads} * rocm::kHeadDim * sizeof(std::uint16_t);

constexpr std::array<char, 8> kSnapshotMagic{'F', 'L', 'I', 'N',
                                             'T', 'S', 'E', 'S'};

/// Leads every serialized snapshot. Logits follow, then per layer in order
/// K then V as [kv_head][rows][head_dim] FP16, rows ascending by position:
/// [0, position) for full-attention layers, the last
/// min(position, 512) positions for sliding layers.
struct SnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint64_t fingerprint;
  std::uint64_t total_bytes;
  std::uint32_t position;
  std::uint32_t vocab;
  std::uint32_t layers;
  std::uint32_t sliding_rows;
};
static_assert(std::is_trivially_copyable_v<SnapshotHeader>);

bool Fail(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr)
    *error_msg = message;
  return false;
}

bool FailHip(std::string* error_msg, std::string_view what, hipError_t status) {
  return Fail(error_msg, std::string(what) + ": " + hipGetErrorString(status));
}

/// FNV-1a over the configuration and the artifact's tensor layout (types,
/// offsets, sizes): cached K/V rows are only valid for the same weights.
class Fingerprint {
public:
  template<typename T>
  void Add(const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
      hash_ ^= bytes[i];
      hash_ *= 0x100000001b3ULL;
    }
  }
  void Add(const TensorRef& t) {
    Add(static_cast<std::uint32_t>(t.type));
    Add(std::uint64_t{t.file_offset});
    Add(std::uint64_t{t.size_bytes});
  }
  [[nodiscard]] std::uint64_t value() const noexcept { return hash_; }

private:
  std::uint64_t hash_ = 0xcbf29ce484222325ULL;
};

std::uint64_t ArtifactFingerprint(const ModelWeights& w) {
  const Config& c = w.config;
  Fingerprint f;
  for (const std::uint32_t v :
       {c.num_layers, c.hidden_size, c.vocab_size, c.num_heads, c.num_kv_heads,
        c.head_dim, c.sliding_window, c.dense_ffn_dim, c.leading_dense_layers,
        c.num_experts, c.experts_per_token, c.expert_ffn_dim,
        c.shared_expert_ffn_dim})
    f.Add(v);
  f.Add(std::bit_cast<std::uint32_t>(c.rms_eps));
  f.Add(std::bit_cast<std::uint32_t>(c.rope_theta));
  f.Add(std::bit_cast<std::uint32_t>(c.expert_weights_scale));
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    f.Add(static_cast<std::uint8_t>(c.attention[il]));
    f.Add(static_cast<std::uint8_t>(c.rope[il]));
  }
  f.Add(w.token_embd);
  f.Add(w.output_norm);
  f.Add(w.output);
  for (const LayerWeights& l : w.layers) {
    for (const TensorRef* t :
         {&l.attn_norm, &l.q,           &l.k,         &l.v,
          &l.o,         &l.q_norm,      &l.k_norm,    &l.sinks,
          &l.ffn_norm,  &l.gate,        &l.up,        &l.down,
          &l.router,    &l.router_bias, &l.gate_exps, &l.up_exps,
          &l.down_exps, &l.gate_shexp,  &l.up_shexp,  &l.down_shexp})
      f.Add(*t);
  }
  return f.value();
}

/// Pinned host buffer for one session's frontier logits.
struct PinnedLogits {
  float* data = nullptr;
  ~PinnedLogits() {
    if (data != nullptr)
      (void)hipHostFree(data);
  }
};

}  // namespace

/// Everything a loaded model owns. Members are destroyed bottom-up: the
/// executor before the weights it reads, the weights before the mapping.
class DeviceModel {
public:
  ~DeviceModel() {
    if (snapshot_stream != nullptr)
      (void)hipStreamSynchronize(snapshot_stream);
    for (hipEvent_t event : snapshot_ready) {
      if (event != nullptr)
        (void)hipEventDestroy(event);
    }
    if (snapshot_stream != nullptr)
      (void)hipStreamDestroy(snapshot_stream);
    if (snapshot_staging != nullptr)
      (void)hipHostFree(snapshot_staging);
  }

  std::unique_ptr<core::GgufReader> reader;
  Config config;
  std::optional<Tokenizer> tokenizer;
  std::string name;
  std::uint64_t fingerprint = 0;
  /// Sliding-window ring slots per session (before clamping to context).
  std::uint32_t ring = 0;
  std::unique_ptr<rocm::DeviceModel> weights;
  std::unique_ptr<rocm::Executor> executor;

  /// Snapshot downloads run off the scheduler thread while other sessions
  /// execute: one at a time, on a stream of their own, through
  /// double-buffered pinned staging (pageable copies and per-call stream
  /// creation queued behind decode work and took ~100 ms).
  std::mutex snapshot_mutex;
  hipStream_t snapshot_stream = nullptr;
  std::array<hipEvent_t, 2> snapshot_ready{};
  std::uint8_t* snapshot_staging = nullptr;  ///< 2 x kStagingBytes

  [[nodiscard]] std::uint32_t vocab() const noexcept {
    return static_cast<std::uint32_t>(weights->output().rows);
  }
};

class Session::Impl {
public:
  DeviceModel* device = nullptr;
  std::unique_ptr<rocm::SessionCache> cache;
  PinnedLogits logits;
  std::uint32_t position = 0;
  CancellationCheck cancelled;

  [[nodiscard]] std::uint32_t SlidingRows() const noexcept {
    return std::min(position, kSlidingHistory);
  }

  [[nodiscard]] std::uint64_t PayloadBytes(std::uint32_t at) const noexcept {
    const Config& c = device->config;
    std::uint64_t bytes =
        sizeof(SnapshotHeader) + std::uint64_t{device->vocab()} * sizeof(float);
    const std::uint64_t sliding = std::min(at, kSlidingHistory);
    for (std::uint32_t il = 0; il < c.num_layers; ++il)
      bytes += 2 * (c.IsSliding(il) ? sliding : at) * kSlotBytes;
    return bytes;
  }

  [[nodiscard]] bool ValidTokens(std::span<const std::uint32_t> tokens,
                                 std::string* error_msg) const {
    const std::uint32_t vocab = device->vocab();
    for (const std::uint32_t token : tokens) {
      if (token >= vocab)
        return Fail(error_msg, "token id " + std::to_string(token) +
                                   " is outside the vocabulary");
    }
    return true;
  }
};

namespace {

/// One contiguous run of ring slots and where it sits in the snapshot rows.
struct SlotRun {
  std::uint32_t slot;
  std::uint32_t row;
  std::uint32_t count;
};

/// The slots holding positions [first, first + count) of a layer whose
/// arrays have `capacity` slots: one run, or two when the ring wraps.
std::array<SlotRun, 2> SlotRuns(std::uint32_t first, std::uint32_t count,
                                std::uint32_t capacity) {
  const std::uint32_t slot = first % capacity;
  const std::uint32_t head = std::min(count, capacity - slot);
  return {SlotRun{slot, 0, head}, SlotRun{0, head, count - head}};
}

constexpr std::size_t kRowBytes = kSlotBytes / rocm::kKvHeads;
/// Bytes of one of the two pinned snapshot staging buffers.
constexpr std::size_t kStagingBytes = std::size_t{16} << 20;

/// Calls fn(device, host, bytes) for every contiguous device run of one
/// session's snapshot rows, in snapshot byte order: per layer, K then V,
/// [kv_head][rows] with rows ascending by position (one or two ring runs).
template<typename Fn>
void ForEachKvRun(const Config& config, const rocm::SessionCache& cache,
                  std::uint32_t position, std::uint8_t* bytes, Fn&& fn) {
  const std::uint32_t sliding = std::min(position, kSlidingHistory);
  for (std::uint32_t il = 0; il < config.num_layers; ++il) {
    const rocm::KvSequence& seq = cache.seqs()[il];
    const auto capacity = static_cast<std::uint32_t>(seq.capacity);
    const std::uint32_t rows = config.IsSliding(il) ? sliding : position;
    const auto runs = SlotRuns(position - rows, rows, capacity);
    for (void* device :
         {static_cast<void*>(seq.k), static_cast<void*>(seq.v)}) {
      auto* base = static_cast<std::uint8_t*>(device);
      for (std::uint32_t h = 0; h < rocm::kKvHeads; ++h) {
        for (const SlotRun& run : runs) {
          if (run.count != 0) {
            fn(base + (std::size_t{h} * capacity + run.slot) * kRowBytes,
               bytes + (std::size_t{h} * rows + run.row) * kRowBytes,
               std::size_t{run.count} * kRowBytes);
          }
        }
      }
      bytes += std::size_t{rocm::kKvHeads} * rows * kRowBytes;
    }
  }
}

/// Restores snapshot rows into a session's cache on `stream`.
hipError_t UploadKvRows(const Config& config, const rocm::SessionCache& cache,
                        std::uint32_t position, const std::uint8_t* bytes,
                        hipStream_t stream) {
  hipError_t status = hipSuccess;
  ForEachKvRun(config, cache, position, const_cast<std::uint8_t*>(bytes),
               [&](std::uint8_t* device, std::uint8_t* host, std::size_t size) {
                 if (status == hipSuccess) {
                   status = hipMemcpyAsync(device, host, size,
                                           hipMemcpyHostToDevice, stream);
                 }
               });
  return status;
}

/// Downloads a session's snapshot rows into `bytes` through the model's two
/// pinned staging buffers: while one buffer's DMA runs, the other's rows
/// are copied out on the host. The caller holds snapshot_mutex.
hipError_t DownloadKvRows(DeviceModel& d, const rocm::SessionCache& cache,
                          std::uint32_t position, std::uint8_t* bytes) {
  struct Piece {
    const std::uint8_t* device;
    std::uint8_t* host;
    std::size_t size;
  };
  std::vector<Piece> pieces;
  ForEachKvRun(d.config, cache, position, bytes,
               [&](std::uint8_t* device, std::uint8_t* host, std::size_t size) {
                 for (std::size_t at = 0; at < size; at += kStagingBytes) {
                   pieces.push_back({device + at, host + at,
                                     std::min(kStagingBytes, size - at)});
                 }
               });
  hipError_t status = hipSuccess;
  std::array<std::size_t, 2> first{};
  std::array<std::size_t, 2> last{};
  std::size_t next = 0;
  for (int slot = 0;
       status == hipSuccess &&
       (next < pieces.size() || first[0] != last[0] || first[1] != last[1]);
       slot ^= 1) {
    std::uint8_t* staging = d.snapshot_staging + slot * kStagingBytes;
    if (first[slot] != last[slot]) {
      status = hipEventSynchronize(d.snapshot_ready[slot]);
      std::size_t offset = 0;
      for (std::size_t i = first[slot]; status == hipSuccess && i < last[slot];
           ++i) {
        std::memcpy(pieces[i].host, staging + offset, pieces[i].size);
        offset += pieces[i].size;
      }
      first[slot] = last[slot];
    }
    if (next == pieces.size())
      continue;
    first[slot] = next;
    std::size_t used = 0;
    while (status == hipSuccess && next < pieces.size() &&
           used + pieces[next].size <= kStagingBytes) {
      status =
          hipMemcpyAsync(staging + used, pieces[next].device, pieces[next].size,
                         hipMemcpyDeviceToHost, d.snapshot_stream);
      used += pieces[next++].size;
    }
    last[slot] = next;
    if (status == hipSuccess)
      status = hipEventRecord(d.snapshot_ready[slot], d.snapshot_stream);
  }
  // Staging is reused by the next snapshot: never leave copies in flight.
  const hipError_t drained = hipStreamSynchronize(d.snapshot_stream);
  return status == hipSuccess ? drained : status;
}

}  // namespace

// ---------------------------------------------------------------------------
// Model

Model::Model() = default;
Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  m->options_ = options;
  auto device = std::make_unique<DeviceModel>();
  device->reader = core::GgufReader::OpenFile(model_path, error_msg);
  if (!device->reader)
    return nullptr;
  auto weights = ModelWeights::Bind(*device->reader, error_msg);
  if (!weights)
    return nullptr;
  device->config = weights->config;
  if (options.max_context == 0 ||
      options.max_context > device->config.context_length) {
    Fail(error_msg, "context exceeds the model's " +
                        std::to_string(device->config.context_length) +
                        " tokens");
    return nullptr;
  }
  device->tokenizer = Tokenizer::FromGguf(*device->reader, error_msg);
  if (!device->tokenizer)
    return nullptr;
  if (device->tokenizer->VocabSize() != weights->output.rows) {
    Fail(error_msg, "tokenizer vocabulary (" +
                        std::to_string(device->tokenizer->VocabSize()) +
                        ") differs from the output head (" +
                        std::to_string(weights->output.rows) + ")");
    return nullptr;
  }
  device->name = std::string(
      device->reader->GetMetadataString("general.name").value_or("Flint"));
  device->fingerprint = ArtifactFingerprint(*weights);
  device->weights =
      rocm::DeviceModel::Upload(*weights, *device->reader, error_msg);
  if (!device->weights)
    return nullptr;
  const auto chunk = std::min(kPrefillChunk, options.max_context);
  device->ring = static_cast<std::uint32_t>(rocm::SlidingRingCapacity(chunk));
  device->executor = rocm::Executor::Create(
      *device->weights, static_cast<int>(chunk), kMaxBatch, error_msg);
  if (!device->executor)
    return nullptr;
  hipError_t status =
      hipStreamCreateWithFlags(&device->snapshot_stream, hipStreamNonBlocking);
  for (hipEvent_t& event : device->snapshot_ready) {
    if (status == hipSuccess)
      status = hipEventCreateWithFlags(&event, hipEventDisableTiming);
  }
  if (status == hipSuccess) {
    status = hipHostMalloc(reinterpret_cast<void**>(&device->snapshot_staging),
                           2 * kStagingBytes, hipHostMallocDefault);
  }
  if (status != hipSuccess) {
    (void)hipGetLastError();
    FailHip(error_msg, "snapshot staging allocation", status);
    return nullptr;
  }
  m->device_ = std::move(device);
  return m;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    Fail(error_msg, "session context must be between 1 and " +
                        std::to_string(options_.max_context));
    return nullptr;
  }
  auto impl = std::make_unique<Session::Impl>();
  impl->device = device_.get();
  impl->cache = rocm::SessionCache::Create(device_->config, max_context,
                                           device_->ring, error_msg);
  if (!impl->cache)
    return nullptr;
  const hipError_t status = hipHostMalloc(
      reinterpret_cast<void**>(&impl->logits.data),
      std::size_t{device_->vocab()} * sizeof(float), hipHostMallocDefault);
  if (status != hipSuccess) {
    impl->logits.data = nullptr;
    (void)hipGetLastError();
    FailHip(error_msg, "session logits allocation", status);
    return nullptr;
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(impl)));
}

bool Model::EvaluateBatch(std::span<const SessionBatchItem> items,
                          std::string* error_msg, bool* interrupted) const {
  if (interrupted != nullptr)
    *interrupted = false;
  if (items.empty() || items.size() > static_cast<std::size_t>(kMaxBatch))
    return Fail(error_msg, "batch must hold 1 to " + std::to_string(kMaxBatch) +
                               " sessions");
  std::array<rocm::SessionCache*, kMaxBatch> caches{};
  std::array<std::uint32_t, kMaxBatch> tokens{};
  std::array<std::uint32_t, kMaxBatch> positions{};
  std::array<float*, kMaxBatch> logits{};
  for (std::size_t i = 0; i < items.size(); ++i) {
    const Session* session = items[i].session;
    if (session == nullptr || session->model_.get() != this)
      return Fail(error_msg, "batch item is not a session of this model");
    const Session::Impl& s = *session->impl_;
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == session)
        return Fail(error_msg, "a session appears twice in one batch");
    }
    if (s.position == 0)
      return Fail(error_msg, "batched decode needs a prefilled session");
    if (s.position >= s.cache->context())
      return Fail(error_msg, "session context is full");
    if (!s.ValidTokens(std::span(&items[i].token, 1), error_msg))
      return false;
    caches[i] = s.cache.get();
    tokens[i] = items[i].token;
    positions[i] = s.position;
    logits[i] = s.logits.data;
  }
  for (const SessionBatchItem& item : items) {
    if (item.session->impl_->cancelled && item.session->impl_->cancelled()) {
      if (interrupted != nullptr)
        *interrupted = true;
      return Fail(error_msg, "batched decode cancelled");
    }
  }
  const std::size_t n = items.size();
  if (!device_->executor->Decode(std::span(caches.data(), n),
                                 std::span(tokens.data(), n),
                                 std::span(positions.data(), n),
                                 std::span(logits.data(), n), error_msg)) {
    for (const SessionBatchItem& item : items)
      item.session->Invalidate();
    return false;
  }
  for (const SessionBatchItem& item : items)
    ++item.session->impl_->position;
  return true;
}

std::size_t Model::MaxBatch() const noexcept {
  return kMaxBatch;
}

const Config& Model::config() const noexcept {
  return device_->config;
}

const Tokenizer& Model::tokenizer() const noexcept {
  return *device_->tokenizer;
}

std::vector<std::uint32_t> Model::EncodeChat(
    std::span<const tokenization::ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools,
    bool add_generation_prompt) const {
  return device_->tokenizer->Encode(
      RenderChat(messages, tools, add_generation_prompt), true);
}

std::string Model::ModelName() const {
  return device_->name;
}

std::uint32_t Model::MaxContext() const noexcept {
  return options_.max_context;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->weights->resident_bytes() +
         device_->executor->scratch_bytes();
}

std::size_t Model::SessionBytes(std::uint32_t max_context) const {
  return rocm::SessionCache::Bytes(device_->config, max_context, device_->ring);
}

// ---------------------------------------------------------------------------
// Session

Session::Session(std::shared_ptr<Model> model, std::unique_ptr<Impl> impl)
    : model_(std::move(model)), impl_(std::move(impl)) {}

Session::~Session() = default;

bool Session::Sync(std::span<const std::uint32_t> prompt,
                   std::string* error_msg) {
  Impl& s = *impl_;
  if (prompt.size() < s.position)
    return Fail(error_msg, "prompt is shorter than the held prefix");
  if (prompt.size() > s.cache->context())
    return Fail(error_msg, "prompt exceeds the session context");
  const auto fresh = prompt.subspan(s.position);
  if (fresh.empty()) {
    return s.position > 0 ? true : Fail(error_msg, "prompt is empty");
  }
  if (fresh.size() > PrefillCapacity())
    return Fail(error_msg, "prompt adds more tokens than PrefillCapacity()");
  if (!s.ValidTokens(fresh, error_msg))
    return false;
  if (s.cancelled && s.cancelled())
    return Fail(error_msg, "prefill cancelled");
  switch (s.device->executor->Prefill(*s.cache, fresh, s.position,
                                      s.logits.data, s.cancelled, error_msg)) {
    case rocm::Executor::Status::kOk:
      s.position = static_cast<std::uint32_t>(prompt.size());
      return true;
    case rocm::Executor::Status::kCancelled:
      return Fail(error_msg, "prefill cancelled");
    case rocm::Executor::Status::kFailed:
      break;
  }
  Invalidate();
  return false;
}

bool Session::Evaluate(std::uint32_t token, std::string* error_msg) {
  const SessionBatchItem item{.session = this, .token = token};
  Impl& s = *impl_;
  if (s.position == 0)
    return Fail(error_msg, "Evaluate needs a prefilled session; use Sync");
  if (s.position >= s.cache->context())
    return Fail(error_msg, "session context is full");
  if (!s.ValidTokens(std::span(&item.token, 1), error_msg))
    return false;
  rocm::SessionCache* cache = s.cache.get();
  float* logits = s.logits.data;
  if (!s.device->executor->Decode(std::span(&cache, 1), std::span(&token, 1),
                                  std::span(&s.position, 1),
                                  std::span(&logits, 1), error_msg)) {
    Invalidate();
    return false;
  }
  ++s.position;
  return true;
}

std::span<const float> Session::Logits() const noexcept {
  if (impl_->position == 0)
    return {};
  return {impl_->logits.data, impl_->device->vocab()};
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  const Impl& s = *impl_;
  if (s.position == 0) {
    Fail(error_msg, "an empty session has no snapshot");
    return nullptr;
  }
  const DeviceModel& d = *s.device;
  const std::size_t size = s.PayloadBytes(s.position);
  // Every byte is written below, so skip zeroing up to a gigabyte.
  auto bytes = std::make_unique_for_overwrite<std::uint8_t[]>(size);
  const SnapshotHeader header{.magic = kSnapshotMagic,
                              .version = SessionSnapshot::kPayloadVersion,
                              .header_bytes = sizeof(SnapshotHeader),
                              .fingerprint = d.fingerprint,
                              .total_bytes = size,
                              .position = s.position,
                              .vocab = d.vocab(),
                              .layers = d.config.num_layers,
                              .sliding_rows = s.SlidingRows()};
  std::memcpy(bytes.get(), &header, sizeof(header));
  std::memcpy(bytes.get() + sizeof(header), s.logits.data,
              std::size_t{d.vocab()} * sizeof(float));
  hipError_t status;
  {
    std::lock_guard lock(s.device->snapshot_mutex);
    status = DownloadKvRows(
        *s.device, *s.cache, s.position,
        bytes.get() + sizeof(header) + std::size_t{d.vocab()} * sizeof(float));
  }
  if (status != hipSuccess) {
    FailHip(error_msg, "snapshot copy", status);
    return nullptr;
  }
  return std::unique_ptr<SessionSnapshot>(
      new SessionSnapshot(std::move(bytes), size));
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  return RestoreSnapshot(snapshot.bytes(), error_msg);
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  Impl& s = *impl_;
  const DeviceModel& d = *s.device;
  SnapshotHeader header{};
  if (payload.size() < sizeof(header))
    return Fail(error_msg, "snapshot is truncated");
  std::memcpy(&header, payload.data(), sizeof(header));
  if (header.magic != kSnapshotMagic)
    return Fail(error_msg, "not a Flint session snapshot");
  if (header.version != SessionSnapshot::kPayloadVersion ||
      header.header_bytes != sizeof(header))
    return Fail(error_msg, "unsupported Flint snapshot version " +
                               std::to_string(header.version));
  if (header.fingerprint != d.fingerprint || header.vocab != d.vocab() ||
      header.layers != d.config.num_layers)
    return Fail(error_msg, "snapshot belongs to a different model");
  if (header.position == 0 || header.position > s.cache->context())
    return Fail(error_msg, "snapshot position " +
                               std::to_string(header.position) +
                               " does not fit this session's context");
  if (header.sliding_rows != std::min(header.position, kSlidingHistory) ||
      header.total_bytes != payload.size() ||
      payload.size() != s.PayloadBytes(header.position))
    return Fail(error_msg, "snapshot size does not match its header");
  const std::uint8_t* logits = payload.data() + sizeof(header);
  hipStream_t stream = d.executor->stream();
  hipError_t status =
      UploadKvRows(d.config, *s.cache, header.position,
                   logits + std::size_t{d.vocab()} * sizeof(float), stream);
  const hipError_t synced = hipStreamSynchronize(stream);
  if (status == hipSuccess)
    status = synced;
  if (status != hipSuccess) {
    Invalidate();
    return FailHip(error_msg, "snapshot restore", status);
  }
  std::memcpy(s.logits.data, logits, std::size_t{d.vocab()} * sizeof(float));
  s.position = header.position;
  return true;
}

void Session::SetCancellationCheck(CancellationCheck is_cancelled) {
  impl_->cancelled = std::move(is_cancelled);
}

void Session::Invalidate() noexcept {
  impl_->position = 0;
}

std::uint32_t Session::Position() const noexcept {
  return impl_->position;
}

std::uint32_t Session::ContextSize() const noexcept {
  return impl_->cache->context();
}

std::uint32_t Session::PrefillCapacity() const noexcept {
  return static_cast<std::uint32_t>(impl_->device->executor->prefill_rows());
}

std::uint64_t Session::PayloadBytes() const noexcept {
  return impl_->position == 0 ? 0 : impl_->PayloadBytes(impl_->position);
}

// ---------------------------------------------------------------------------
// SessionSnapshot

SessionSnapshot::SessionSnapshot(std::unique_ptr<std::uint8_t[]> bytes,
                                 std::size_t size)
    : bytes_(std::move(bytes)), size_(size) {}

SessionSnapshot::~SessionSnapshot() = default;

std::uint32_t SessionSnapshot::Position() const noexcept {
  SnapshotHeader header{};
  std::memcpy(&header, bytes_.get(), sizeof(header));
  return header.position;
}

std::uint64_t SessionSnapshot::SizeBytes() const noexcept {
  return size_;
}

std::span<const std::uint8_t> SessionSnapshot::bytes() const noexcept {
  return {bytes_.get(), size_};
}

}  // namespace gufo::models::flint
