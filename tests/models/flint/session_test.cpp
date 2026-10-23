// Model-gated Flint session and serving replay checks. Greedy outputs must
// not depend on how a prefix was produced: prefill chunking, RAM or
// serialized snapshot restores, batched decode, cancellation and the disk
// cache all have to replay the same tokens as cold execution.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <latch>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"
#include "src/core/sampling.hpp"
#include "src/models/flint/engine.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace {

namespace flint = gufo::models::flint;
namespace server = gufo::server;
namespace sampling = gufo::sampling;
using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;
using Tokens = std::vector<std::uint32_t>;

constexpr std::uint32_t kContext = 4096;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void RequireExact(std::span<const float> expected,
                  std::span<const float> actual, const std::string& message) {
  Require(expected.size() == actual.size() &&
              std::memcmp(expected.data(), actual.data(),
                          expected.size_bytes()) == 0,
          message);
}

std::string Join(std::span<const std::uint32_t> tokens) {
  std::string text;
  for (const auto token : tokens)
    text += std::to_string(token) + ' ';
  return text;
}

void RequireTokens(std::span<const std::uint32_t> expected,
                   std::span<const std::uint32_t> actual,
                   const std::string& message) {
  if (!std::ranges::equal(expected, actual)) {
    throw std::runtime_error(message + "\n  expected: " + Join(expected) +
                             "\n  actual:   " + Join(actual));
  }
}

std::uint32_t Argmax(std::span<const float> logits) {
  Require(!logits.empty(), "session has no frontier logits");
  return static_cast<std::uint32_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

std::unique_ptr<flint::Session> NewSession(
    const std::shared_ptr<flint::Model>& model) {
  std::string error;
  auto session = model->CreateSession(kContext, &error);
  Require(session != nullptr, "CreateSession: " + error);
  return session;
}

// Extends the session to `prompt` in steps of at most `chunk` new tokens,
// zero meaning the engine's own capacity.
void Prefill(flint::Session& session, std::span<const std::uint32_t> prompt,
             std::size_t chunk = 0) {
  std::string error;
  while (session.Position() < prompt.size()) {
    std::size_t step = session.PrefillCapacity();
    Require(step > 0, "session has no prefill capacity");
    if (chunk != 0)
      step = std::min(step, chunk);
    const auto next = std::min(
        prompt.size(), static_cast<std::size_t>(session.Position()) + step);
    Require(session.Sync(prompt.first(next), &error), "Sync: " + error);
  }
}

// Greedy continuation that stops before an end-of-generation token and
// leaves every emitted token evaluated.
Tokens Greedy(const flint::Model& model, flint::Session& session,
              std::size_t count) {
  Tokens tokens;
  std::string error;
  while (tokens.size() < count) {
    const auto token = Argmax(session.Logits());
    if (model.tokenizer().IsEndOfGeneration(token))
      break;
    tokens.push_back(token);
    Require(session.Evaluate(token, &error), "Evaluate: " + error);
  }
  return tokens;
}

Tokens Direct(const std::shared_ptr<flint::Model>& model,
              std::span<const std::uint32_t> prompt, std::size_t count) {
  auto session = NewSession(model);
  Prefill(*session, prompt);
  return Greedy(*model, *session, count);
}

// Longer than two sliding windows (513), so restored and chunked prefixes
// cross the ring wrap of the sliding-window layers.
Tokens LongPrompt(const flint::Model& model, std::size_t length) {
  const auto pattern = model.tokenizer().Encode(
      "Virtual memory maps pages to physical storage. A train travels sixty "
      "kilometers per hour. Keep Unicode intact: Caf\u00e9, "
      "\u65e5\u672c\u8a9e. "
      "Continue red, green, blue, red, green, blue. ",
      false);
  Require(!pattern.empty(), "empty prompt pattern");
  Tokens tokens(length);
  for (std::size_t i = 0; i < length; ++i)
    tokens[i] = pattern[i % pattern.size()];
  return tokens;
}

// Serving splits prompts wherever a cache boundary, the decode-active budget
// or a prefix snapshot falls, often one token before the end. Replay of a
// cached or concurrently scheduled request is exact only when every split
// leaves bitwise the same frontier as one pass.
void CheckPrefillChunks(const std::shared_ptr<flint::Model>& model) {
  for (const std::size_t length :
       {std::size_t{21}, std::size_t{200}, std::size_t{1300}}) {
    const auto prompt = LongPrompt(*model, length);
    auto fresh = NewSession(model);
    Prefill(*fresh, prompt);
    const std::vector<float> frontier(fresh->Logits().begin(),
                                      fresh->Logits().end());
    const auto reference = Greedy(*model, *fresh, 16);
    Require(reference.size() == 16, "fixture continuation stopped early");
    const auto check = [&](const std::string& label, flint::Session& split) {
      float max_diff = 0.0F;
      for (std::size_t i = 0; i < frontier.size(); ++i)
        max_diff =
            std::max(max_diff, std::abs(frontier[i] - split.Logits()[i]));
      std::cout << "prefill length=" << length << ' ' << label
                << " frontier_max_diff=" << max_diff << '\n'
                << std::flush;
      RequireExact(frontier, split.Logits(),
                   "split prefill changed frontier logits: length=" +
                       std::to_string(length) + ' ' + label);
      RequireTokens(reference, Greedy(*model, split, 16),
                    "split prefill changed greedy replay: length=" +
                        std::to_string(length) + ' ' + label);
    };
    for (const std::size_t boundary :
         {std::size_t{1}, length / 2, length - 1}) {
      auto split = NewSession(model);
      Prefill(*split, std::span(prompt).first(boundary));
      Prefill(*split, prompt);
      check("boundary=" + std::to_string(boundary), *split);
    }
    for (const std::size_t chunk : {37U, 512U, 513U, 514U}) {
      if (chunk >= length)
        continue;
      auto split = NewSession(model);
      Prefill(*split, prompt, chunk);
      check("chunk=" + std::to_string(chunk), *split);
    }
  }
  std::cout << "prefill chunks: frontier logits and greedy replay exact\n"
            << std::flush;
}

void CheckSnapshots(const std::shared_ptr<flint::Model>& model) {
  std::string error;
  const auto prompt = LongPrompt(*model, 1100);
  auto source = NewSession(model);
  Prefill(*source, prompt);
  const std::vector<float> frontier(source->Logits().begin(),
                                    source->Logits().end());
  const auto payload = source->PayloadBytes();
  auto snapshot = source->SaveSnapshot(&error);
  Require(snapshot != nullptr, "SaveSnapshot: " + error);
  Require(snapshot->Position() == prompt.size() &&
              snapshot->SizeBytes() == snapshot->bytes().size() &&
              snapshot->SizeBytes() == payload,
          "snapshot size or position differs from the session's report");
  const auto reference = Greedy(*model, *source, 16);
  Require(reference.size() == 16, "fixture continuation stopped early");

  // Mid-generation boundary: decode state must resume as well as prefill.
  auto middle = NewSession(model);
  Prefill(*middle, prompt);
  const auto head = Greedy(*model, *middle, 8);
  auto mid_snapshot = middle->SaveSnapshot(&error);
  Require(mid_snapshot != nullptr, "SaveSnapshot: " + error);

  for (const bool serialized : {false, true}) {
    const std::string mode = serialized ? "serialized" : "RAM";
    // Restoring into the already-advanced source rewinds it in place.
    const auto fresh = NewSession(model);
    for (flint::Session* target : {fresh.get(), source.get()}) {
      const bool ok = serialized
                          ? target->RestoreSnapshot(snapshot->bytes(), &error)
                          : target->RestoreSnapshot(*snapshot, &error);
      Require(ok, mode + " restore: " + error);
      Require(target->Position() == prompt.size(), mode + " restore position");
      RequireExact(frontier, target->Logits(),
                   mode + " restore changed frontier logits");
      RequireTokens(reference, Greedy(*model, *target, 16),
                    mode + " restore changed greedy replay");
    }
    auto resumed = NewSession(model);
    Require(serialized ? resumed->RestoreSnapshot(mid_snapshot->bytes(), &error)
                       : resumed->RestoreSnapshot(*mid_snapshot, &error),
            mode + " mid restore: " + error);
    auto tail = head;
    const auto rest = Greedy(*model, *resumed, 8);
    tail.insert(tail.end(), rest.begin(), rest.end());
    RequireTokens(reference, tail,
                  mode + " mid-generation restore changed greedy replay");
  }
  std::cout << "snapshots: RAM and serialized restore exact\n" << std::flush;
}

void CheckBatchedDecode(const std::shared_ptr<flint::Model>& model) {
  std::string error;
  const std::size_t widest = std::min<std::size_t>(4, model->MaxBatch());
  Require(widest >= 2, "engine does not batch decode");
  // Different lengths, one past two sliding windows, so rows do not share
  // positions or ring offsets.
  std::vector<Tokens> prompts;
  for (std::size_t i = 0; i < widest; ++i) {
    auto prompt = model->tokenizer().Encode(
        "Batch request " + std::to_string(i) +
            ": continue the list red, blue, blue, red,",
        false);
    if (i == 1) {
      const auto tail = LongPrompt(*model, 1200);
      prompt.insert(prompt.begin(), tail.begin(), tail.end());
    }
    prompts.push_back(std::move(prompt));
  }
  std::vector<Tokens> serial;
  std::vector<std::unique_ptr<flint::SessionSnapshot>> snapshots;
  for (const auto& prompt : prompts) {
    auto session = NewSession(model);
    Prefill(*session, prompt);
    snapshots.push_back(session->SaveSnapshot(&error));
    Require(snapshots.back() != nullptr, "SaveSnapshot: " + error);
    serial.push_back(Greedy(*model, *session, 12));
  }
  for (std::size_t width = 2; width <= widest; ++width) {
    std::vector<std::unique_ptr<flint::Session>> sessions;
    std::vector<Tokens> batched(width);
    for (std::size_t i = 0; i < width; ++i) {
      sessions.push_back(NewSession(model));
      Require(sessions.back()->RestoreSnapshot(*snapshots[i], &error),
              "restore: " + error);
    }
    for (std::size_t step = 0; step < 12; ++step) {
      std::vector<flint::SessionBatchItem> items;
      for (std::size_t i = 0; i < width; ++i) {
        if (batched[i].size() == serial[i].size())
          continue;
        const auto token = Argmax(sessions[i]->Logits());
        batched[i].push_back(token);
        items.push_back({.session = sessions[i].get(), .token = token});
      }
      if (items.size() == 1) {
        Require(items.front().session->Evaluate(items.front().token, &error),
                "Evaluate: " + error);
      } else if (!items.empty()) {
        Require(model->EvaluateBatch(items, &error), "EvaluateBatch: " + error);
      }
    }
    for (std::size_t i = 0; i < width; ++i) {
      RequireTokens(serial[i], batched[i],
                    "batched decode changed greedy replay: width=" +
                        std::to_string(width) + " row=" + std::to_string(i));
    }
  }
  std::cout << "batched decode: widths 2.." << widest << " match serial\n"
            << std::flush;
}

std::vector<ChatMessage> Chat(std::string_view user) {
  return {{ChatRole::kSystem, "Answer in one short sentence."},
          {ChatRole::kUser, std::string(user)}};
}

void CheckServing(const std::shared_ptr<flint::Model>& model) {
  server::InferenceBackend backend;
  std::string error;
  Require(backend.load(model, &error, kContext, 4), error);
  Require(backend.model_id() == model->ModelName(), "HTTP model identifier");
  Require(
      backend.sampling_defaults().model == sampling::TextModelPreset::kFlint,
      "Flint sampling preset");

  const std::string raw = "The capital of France is";
  const auto direct_raw =
      Direct(model, model->tokenizer().Encode(raw, false), 6);
  const auto http_raw = backend.complete(raw, 6, sampling::SamplingConfig{});
  RequireTokens(direct_raw, http_raw.tokens, "raw direct/HTTP parity");

  const auto messages = Chat("Name one primary color.");
  const auto direct_chat = Direct(model, model->EncodeChat(messages, {}), 8);
  const auto first = backend.chat(messages, 8, sampling::SamplingConfig{});
  RequireTokens(direct_chat, first.tokens, "chat direct/HTTP parity");
  const auto repeated = backend.chat(messages, 8, sampling::SamplingConfig{});
  Require(repeated.cache_hit && repeated.prefill_tokens == 0,
          "repeated chat restores the complete prompt");
  RequireTokens(direct_chat, repeated.tokens,
                "cached chat differs from cold execution");

  // Two branches from the retained root restore it concurrently and prefill
  // only their suffixes.
  auto branch = [&](std::string_view user) {
    auto history = messages;
    history.emplace_back(ChatRole::kAssistant, first.text);
    history.emplace_back(ChatRole::kUser, std::string(user));
    return history;
  };
  const auto continued = branch("Name a different primary color.");
  const auto forked = branch("Name one warm color.");
  const auto direct_continued =
      Direct(model, model->EncodeChat(continued, {}), 8);
  const auto direct_forked = Direct(model, model->EncodeChat(forked, {}), 8);
  server::ChatRequest continued_request(continued);
  continued_request.client_id = "flint-branch-a";
  server::ChatRequest forked_request(forked);
  forked_request.client_id = "flint-branch-b";
  auto pending_continued = backend.start_chat(
      continued_request, 8, sampling::SamplingConfig{}, {}, true);
  auto pending_forked = backend.start_chat(
      forked_request, 8, sampling::SamplingConfig{}, {}, true);
  const auto http_continued = pending_continued->Wait();
  const auto http_forked = pending_forked->Wait();
  for (const auto* result : {&http_continued, &http_forked}) {
    Require(result->cache_hit && result->cached_prompt_tokens > 0 &&
                result->cached_prompt_tokens < result->prompt_tokens,
            "branches reuse the root and prefill their suffixes");
  }
  RequireTokens(direct_continued, http_continued.tokens,
                "cached continuation differs from cold prefill");
  RequireTokens(direct_forked, http_forked.tokens,
                "forked continuation differs from cold prefill");

  // Two to four concurrent requests decode together and match isolation.
  std::vector<std::vector<ChatMessage>> concurrent;
  for (const auto* topic : {"one, two, three,", "red, green, blue,",
                            "Monday, Tuesday,", "alpha, beta, gamma,"}) {
    concurrent.push_back(
        Chat(std::string("Continue with four short items: ") + topic));
  }
  std::vector<Tokens> isolated;
  for (const auto& history : concurrent)
    isolated.push_back(Direct(model, model->EncodeChat(history, {}), 12));
  for (std::size_t width = 2; width <= concurrent.size(); ++width) {
    std::vector<std::shared_ptr<server::InferenceBackend::GenerationRequest>>
        pending;
    for (std::size_t i = 0; i < width; ++i) {
      server::ChatRequest request(concurrent[i]);
      request.client_id = "flint-concurrent-" + std::to_string(i);
      request.cache_prompt = false;
      pending.push_back(backend.start_chat(
          request, 12, sampling::SamplingConfig{}, {}, true));
      Require(pending.back() != nullptr, "concurrent request admitted");
    }
    std::size_t batched = 0;
    for (std::size_t i = 0; i < width; ++i) {
      const auto result = pending[i]->Wait();
      RequireTokens(isolated[i], result.tokens,
                    "concurrent request differs from isolated execution: "
                    "width=" +
                        std::to_string(width) + " row=" + std::to_string(i));
      batched += result.physical_execution_width > 1;
    }
    Require(batched > 0, "concurrent requests did not decode in a batch");
  }

  // Cancellation during a multi-chunk prefill leaves the state reusable.
  const auto long_prompt = model->tokenizer().Decode(LongPrompt(*model, 3000));
  const auto long_tokens = model->tokenizer().Encode(long_prompt, false);
  std::size_t checks = 0;
  const auto cancelled =
      backend.complete(long_prompt, 4, sampling::SamplingConfig{},
                       [&checks] { return ++checks > 3; });
  Require(cancelled.cancelled && cancelled.completion_tokens == 0,
          "cancellation interrupts prefill before any output");
  RequireTokens(direct_raw,
                backend.complete(raw, 6, sampling::SamplingConfig{}).tokens,
                "state reuse after cancelled prefill");
  RequireTokens(
      Direct(model, long_tokens, 6),
      backend.complete(long_prompt, 6, sampling::SamplingConfig{}).tokens,
      "cancelled prompt replays exactly when resubmitted");
  std::cout << "serving: chat cache, branches, concurrency and cancellation "
               "exact\n"
            << std::flush;
}

// Synthetic grammar pieces force the turn boundary: any ordinary first token
// completes the empty object, then only end-of-generation tokens may follow.
// Token IDs, EOG recognition, execution and checkpoints stay real.
void CheckEndOfGeneration(const std::shared_ptr<flint::Model>& model) {
  server::InferenceBackend backend;
  std::string error;
  Require(backend.load(model, &error, kContext, 2), error);
  const auto& tokenizer = model->tokenizer();
  auto constraint = std::make_shared<sampling::TokenConstraint>();
  constraint->grammar = sampling::JsonConstraint::Compile(
      gufo::json::parse(
          R"({"type":"object","properties":{},"additionalProperties":false})"),
      true);
  constraint->vocabulary = std::make_shared<sampling::ConstraintVocabulary>(
      tokenizer.VocabSize(), [&](std::uint32_t token) {
        const bool stop = tokenizer.IsEndOfGeneration(token);
        return sampling::ConstraintVocabulary::Piece{
            .text = stop ? "" : "{}",
            .stop = stop,
        };
      });
  sampling::SamplingConfig config;
  config.constraint = std::move(constraint);
  const std::string prompt = "EOG fixture: continue red, blue,";
  const auto first = backend.complete(prompt, 8, config);
  Require(
      first.tokens.size() == 1 &&
          !tokenizer.IsEndOfGeneration(first.tokens.front()) &&
          first.finish_reason == server::InferenceBackend::FinishReason::kStop,
      "end-of-generation token did not stop the request");
  const auto replay = backend.complete(prompt, 8, config);
  Require(replay.tokens == first.tokens && replay.prefill_tokens == 0,
          "EOG checkpoint did not replay without prefill");
  const auto ignored =
      backend.start_complete(prompt, 8, config, {}, false, true)->Wait();
  Require(ignored.tokens.size() == 8 &&
              std::all_of(ignored.tokens.begin() + 1, ignored.tokens.end(),
                          [&](auto token) {
                            return tokenizer.IsEndOfGeneration(token);
                          }),
          "ignore_eos request lost its token budget");
  std::cout << "serving: end-of-generation stop and ignore_eos exact\n"
            << std::flush;
}

// Flint opens <think> itself and has no reasoning switch, so thinking
// controls leave the turn unchanged. A budget counts from the generated
// <think>; a zero budget closes it at once.
void CheckReasoningControls(const std::shared_ptr<flint::Model>& model) {
  server::InferenceBackend backend;
  std::string error;
  Require(backend.load(model, &error, kContext, 1), error);
  const auto think = model->tokenizer().FindSpecial("<think>");
  const auto think_end = model->tokenizer().FindSpecial("</think>");
  Require(think && think_end, "Flint reasoning tags");
  const std::vector<ChatMessage> messages{
      {ChatRole::kUser, "What is 17*23? Think briefly."}};

  server::ChatRequest automatic(messages);
  automatic.cache_prompt = false;
  const auto reasoned =
      backend.chat(automatic, 64, sampling::SamplingConfig{}, {}, {});
  Require(!reasoned.tokens.empty() && reasoned.tokens.front() == *think,
          "the fixture's default turn opens reasoning (model behaviour)");
  const auto end = std::ranges::find(reasoned.tokens, *think_end);
  Require(reasoned.reasoning_tokens ==
              static_cast<std::size_t>(end - reasoned.tokens.begin()) - 1,
          "usage counts the reasoning the model opened, without the opener");
  for (const bool enabled : {false, true}) {
    auto controlled = automatic;
    controlled.reasoning.enabled = enabled;
    if (enabled)
      controlled.reasoning.effort = gufo::ReasoningEffort::kHigh;
    Require(backend.initial_output_state(controlled) ==
                server::InferenceBackend::InitialOutputState::kAuto,
            "thinking controls leave a Flint turn to the model");
    RequireTokens(
        reasoned.tokens,
        backend.chat(controlled, 64, sampling::SamplingConfig{}, {}, {}).tokens,
        std::string("thinking ") + (enabled ? "effort" : "disable") +
            " changed a Flint turn");
  }

  for (const std::size_t budget : {std::size_t{0}, std::size_t{2}}) {
    auto budgeted = automatic;
    budgeted.reasoning.budget_tokens = budget;
    const auto capped =
        backend.chat(budgeted, 16, sampling::SamplingConfig{}, {}, {});
    Require(
        capped.tokens.size() > budget + 2 && capped.tokens[0] == *think &&
            capped.tokens[budget + 1] == *think_end &&
            capped.reasoning_tokens == budget &&
            std::ranges::equal(std::span(capped.tokens).subspan(1, budget),
                               std::span(reasoned.tokens).subspan(1, budget)),
        "a budget of " + std::to_string(budget) +
            " counts from the generated <think> and forces its end");
  }
  std::cout << "serving: reasoning controls and auto budget exact\n"
            << std::flush;
}

void CheckDiskCache(const std::shared_ptr<flint::Model>& model) {
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("flint-session-cache-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } cleanup{directory};
  server::TextDiskCacheConfig disk{
      .directory = directory,
      .model_artifact_fingerprint = std::string(64, 'a'),
      .draft_model_artifact_fingerprint = {},
  };
  const auto messages = Chat("Name one primary color.");
  const auto expected = Direct(model, model->EncodeChat(messages, {}), 8);
  for (int run = 0; run < 3; ++run) {
    // A different artifact identity must not restore the stored state.
    if (run == 2)
      disk.model_artifact_fingerprint = std::string(64, 'b');
    server::InferenceBackend backend;
    std::string error;
    Require(backend.load(model, &error, kContext, 1, {}, {}, {}, disk), error);
    const auto result = backend.chat(messages, 8, sampling::SamplingConfig{});
    Require(result.cache_disk_hit == (run == 1),
            "disk reuse requires the matching artifact identity: run " +
                std::to_string(run));
    RequireTokens(
        expected, result.tokens,
        "disk restore changed greedy replay: run " + std::to_string(run));
  }
  std::cout << "disk cache: restart restore exact, identity enforced\n"
            << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
  const char* model_path = std::getenv("GUFO_FLINT_MODEL");
  if (model_path == nullptr || *model_path == '\0') {
    std::cout << "SKIP: GUFO_FLINT_MODEL is not set\n";
    return 77;
  }
  const std::string_view only = argc > 1 ? argv[1] : "";
  try {
    std::string error;
    auto model =
        flint::Model::Load(model_path, {.max_context = kContext}, &error);
    Require(model != nullptr, "Model::Load: " + error);
    if (only.empty() || only == "--engine-only") {
      CheckPrefillChunks(model);
      CheckSnapshots(model);
      CheckBatchedDecode(model);
    }
    if (only.empty() || only == "--serving-only") {
      CheckServing(model);
      CheckEndOfGeneration(model);
      CheckReasoningControls(model);
      CheckDiskCache(model);
    }
    std::cout << "Flint session test passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}
