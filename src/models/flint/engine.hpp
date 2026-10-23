#ifndef GUFO_MODELS_FLINT_ENGINE_HPP_
#define GUFO_MODELS_FLINT_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/flint/config.hpp"
#include "src/models/flint/tokenizer.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::flint {

struct ModelOptions {
  /// Largest context any session may request; bounds per-session KV storage.
  std::uint32_t max_context = 4096;
};

class Session;
class SessionSnapshot;
class DeviceModel;

/// One decode step of an independent session in a batched evaluation.
struct SessionBatchItem {
  Session* session = nullptr;
  std::uint32_t token = 0;
};

/// GPU operations, including session creation/destruction, share scratch
/// storage and must be serialized by the caller. The server scheduler owns
/// that serialization and advances concurrent requests through EvaluateBatch.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;

  /// Loads a `flint` GGUF: binds and uploads the weights and the tokenizer.
  /// The caller validates the chat template (ValidateGgufTemplate) first.
  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  /// Advances every item's session by one token and leaves each session's
  /// frontier logits ready. Results equal per-session Evaluate calls. On
  /// cancellation `interrupted` is true and no session has been mutated.
  [[nodiscard]] bool EvaluateBatch(std::span<const SessionBatchItem> items,
                                   std::string* error_msg = nullptr,
                                   bool* interrupted = nullptr) const;

  /// Largest batch EvaluateBatch accepts.
  [[nodiscard]] std::size_t MaxBatch() const noexcept;

  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const Tokenizer& tokenizer() const noexcept;
  /// `tokenizer().Encode(RenderChat(...), true)`.
  [[nodiscard]] std::vector<std::uint32_t> EncodeChat(
      std::span<const tokenization::ChatMessage> messages,
      std::span<const tokenization::ChatTool> tools,
      bool add_generation_prompt = true) const;
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept;
  /// Device bytes held by weights and model-wide scratch.
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Device bytes one session of `max_context` tokens allocates.
  [[nodiscard]] std::size_t SessionBytes(std::uint32_t max_context) const;

private:
  Model();

  std::unique_ptr<DeviceModel> device_;
  ModelOptions options_;

  friend class Session;
};

class Session final {
public:
  using CancellationCheck = std::function<bool()>;

  ~Session();

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) = delete;
  Session& operator=(Session&&) = delete;

  /// Extends the held prefix to `prompt`: `prompt` must start with the
  /// Position() tokens already evaluated and add at most PrefillCapacity()
  /// new ones. Leaves the frontier logits of the last token ready. A
  /// cancelled call returns false with the position and logits unchanged;
  /// the next call on the session must then be Sync, RestoreSnapshot or
  /// Invalidate.
  ///
  /// Rows Evaluate wrote are first evaluated again through prefill, so the
  /// extended prefix holds exactly what a cold Sync of `prompt` would. That
  /// work, about one prefilled token per decoded token, comes on top of the
  /// new tokens.
  [[nodiscard]] bool Sync(std::span<const std::uint32_t> prompt,
                          std::string* error_msg = nullptr);
  /// Evaluates one token after the held prefix through the decode kernels.
  [[nodiscard]] bool Evaluate(std::uint32_t token,
                              std::string* error_msg = nullptr);
  /// Host copy of the frontier logits (VocabSize() floats).
  [[nodiscard]] std::span<const float> Logits() const noexcept;

  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr) const;
  /// Restores a snapshot taken from any session of the same model whose
  /// context does not exceed this session's.
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  /// Restores serialized snapshot bytes (SessionSnapshot::bytes()).
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);

  void SetCancellationCheck(CancellationCheck is_cancelled);
  /// Drops the held prefix; the next Sync starts from position zero.
  void Invalidate() noexcept;

  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t ContextSize() const noexcept;
  /// Largest number of new tokens one Sync call accepts.
  [[nodiscard]] std::uint32_t PrefillCapacity() const noexcept;
  /// Bytes a snapshot of the current prefix occupies.
  [[nodiscard]] std::uint64_t PayloadBytes() const noexcept;

private:
  class Impl;
  Session(std::shared_ptr<Model> model, std::unique_ptr<Impl> impl);

  std::shared_ptr<Model> model_;
  std::unique_ptr<Impl> impl_;

  friend class Model;
};

/// Host copy of a session prefix: position, logits and the KV rows it needs
/// (every full-attention row, the last window of sliding-window rows). A
/// prefix that ends in decoded tokens also carries their token ids and the
/// window before them, so a later Sync can evaluate them again exactly. The
/// byte layout is versioned and self-describing so it can be persisted.
class SessionSnapshot final {
public:
  /// Version of the serialized layout; any layout change increments it.
  static constexpr std::uint32_t kPayloadVersion = 3;

  ~SessionSnapshot();

  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;
  SessionSnapshot(SessionSnapshot&&) = delete;
  SessionSnapshot& operator=(SessionSnapshot&&) = delete;

  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint64_t SizeBytes() const noexcept;
  [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept;

private:
  /// Takes `size` bytes the caller fills (left uninitialized, not zeroed).
  SessionSnapshot(std::unique_ptr<std::uint8_t[]> bytes, std::size_t size);

  std::unique_ptr<std::uint8_t[]> bytes_;
  std::size_t size_ = 0;

  friend class Session;
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_ENGINE_HPP_
