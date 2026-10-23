#ifndef GUFO_MODELS_FLINT_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_FLINT_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/flint/kernels/rocm/device_model.hpp"
#include "src/models/flint/kernels/rocm/kernels.hpp"

namespace gufo::models::flint::rocm {

/// One session's attention cache: per layer FP16 K and V in the kernels'
/// head-major layout. Full-attention layers hold `context` slots (slot ==
/// position); sliding layers hold a ring of `ring` slots, at least
/// SlidingRingCapacity(widest prefill chunk), so a chunk never overwrites
/// keys that it, or the prefix before it, still reads. A cancelled chunk only
/// overwrites slots of positions older than any later chunk reads, so the
/// prefix it started from stays intact.
class SessionCache {
public:
  ~SessionCache();
  SessionCache(const SessionCache&) = delete;
  SessionCache& operator=(const SessionCache&) = delete;

  /// Device bytes Create allocates.
  [[nodiscard]] static std::size_t Bytes(const Config& config,
                                         std::uint32_t context,
                                         std::uint32_t ring);
  /// Allocates and zeroes the cache (zeroed slots keep masked reads finite).
  [[nodiscard]] static std::unique_ptr<SessionCache> Create(
      const Config& config, std::uint32_t context, std::uint32_t ring,
      std::string* error_msg);

  [[nodiscard]] std::uint32_t context() const noexcept { return context_; }
  /// Per-layer K/V arrays (host copy of device_seqs()).
  [[nodiscard]] std::span<const KvSequence> seqs() const noexcept {
    return seqs_;
  }
  /// The same array in device memory, one entry per layer.
  [[nodiscard]] const KvSequence* device_seqs() const noexcept {
    return device_seqs_;
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  SessionCache() = default;

  std::uint32_t context_ = 0;
  std::vector<KvSequence> seqs_;
  void* storage_ = nullptr;
  KvSequence* device_seqs_ = nullptr;
  std::size_t bytes_ = 0;
};

/// Runs Flint forward passes on one stream with model-wide scratch sized at
/// creation. Calls are serialized by the caller.
class Executor {
public:
  enum class Status { kOk, kCancelled, kFailed };

  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  /// `prefill_rows` bounds one Prefill call; `max_batch` one Decode call.
  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, int prefill_rows, int max_batch,
      std::string* error_msg);

  /// Device bytes of the model-wide scratch.
  [[nodiscard]] std::size_t scratch_bytes() const noexcept {
    return arena_bytes_;
  }
  [[nodiscard]] int prefill_rows() const noexcept { return prefill_rows_; }
  [[nodiscard]] int max_batch() const noexcept { return max_batch_; }
  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }

  /// Diagnostics: after each Prefill layer, called with that layer's device
  /// rows [rows][width] once the stream has drained, named as the reference
  /// taps: "kqv_out" (attention output), "ffn_inp" (residual entering the
  /// FFN), "ffn_norm" (router input) and "ffn_out" (FFN output before the
  /// residual add, so l_out = ffn_inp + ffn_out).
  using Tap = std::function<void(std::string_view name, std::uint32_t layer,
                                 const float* rows, int width)>;
  void SetTap(Tap tap) { tap_ = std::move(tap); }

  /// Evaluates `tokens` at positions first_position.. of one sequence,
  /// writing their K/V rows into `cache`, and leaves the last token's logits
  /// in `logits` (host, vocab floats). `cancelled` (may be empty) is polled
  /// between layers; a cancelled pass returns kCancelled having written only
  /// cache slots at or past first_position (and sliding slots older than
  /// the window), and does not touch `logits`. With `all_logits` (host,
  /// tokens.size() x vocab) every position's logits are written there too,
  /// each by the same single-row head the last token uses.
  [[nodiscard]] Status Prefill(SessionCache& cache,
                               std::span<const std::uint32_t> tokens,
                               std::uint32_t first_position, float* logits,
                               const std::function<bool()>& cancelled,
                               std::string* error_msg,
                               float* all_logits = nullptr);

  /// One token for each of up to max_batch independent sequences; row i
  /// evaluates tokens[i] at positions[i] of caches[i] and writes its logits
  /// to logits[i] (host, vocab floats). Replays one captured HIP graph per
  /// batch width.
  [[nodiscard]] bool Decode(std::span<SessionCache* const> caches,
                            std::span<const std::uint32_t> tokens,
                            std::span<const std::uint32_t> positions,
                            std::span<float* const> logits,
                            std::string* error_msg);

private:
  struct Rows;

  explicit Executor(const DeviceModel& model);

  [[nodiscard]] hipError_t EnqueueLayer(std::uint32_t layer, int rows,
                                        const Rows& in);
  [[nodiscard]] hipError_t EnqueueDecode(int rows);

  const DeviceModel& model_;
  int prefill_rows_ = 0;
  int max_batch_ = 0;
  hipStream_t stream_ = nullptr;
  std::array<hipEvent_t, 2> layer_done_{};

  void* arena_ = nullptr;
  std::size_t arena_bytes_ = 0;
  void* scratch_ = nullptr;
  std::size_t scratch_size_ = 0;
  float* residual_ = nullptr;
  float* xn_ = nullptr;
  float* qkv_ = nullptr;
  float* q_ = nullptr;
  float* attn_ = nullptr;
  float* delta_ = nullptr;
  float* hidden_ = nullptr;
  float* shared_ = nullptr;
  float* expert_out_ = nullptr;
  std::int32_t* ids_ = nullptr;
  float* weights_ = nullptr;
  float* logits_ = nullptr;

  /// Q8_1 copy of the current decode projection input.
  void* q8_ = nullptr;
  /// Per-call inputs, staged in pinned host memory and copied to the device
  /// buffer the kernels (and captured graphs) read.
  struct Params {
    std::int32_t* tokens;
    std::int32_t* positions;
    std::int32_t* row_seq;
    KvSequence* seqs;  ///< [layers][max_batch]
  };
  void* params_host_ = nullptr;
  std::size_t params_bytes_ = 0;
  Params host_{};
  Params device_{};
  std::vector<hipGraphExec_t> graphs_;
  Tap tap_;
};

}  // namespace gufo::models::flint::rocm

#endif  // GUFO_MODELS_FLINT_KERNELS_ROCM_EXECUTOR_HPP_
