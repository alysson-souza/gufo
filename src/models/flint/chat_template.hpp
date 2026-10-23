#ifndef GUFO_MODELS_FLINT_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_FLINT_CHAT_TEMPLATE_HPP_

#include <span>
#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::flint {

/// Renders a conversation exactly as the pinned Flint chat template does under
/// HF transformers' Jinja environment (trim_blocks, lstrip_blocks). Encode the
/// result with special-token parsing to match `apply_chat_template` + encode.
///
/// - Tool definitions come from `ChatTool::definition_json` (its "function"
///   member when present, otherwise the whole object), else from name,
///   description and parameters; they are written as Jinja's
///   `tojson(ensure_ascii=False)` writes them, in a system turn placed after
///   the leading system messages (or at the end when no other message follows).
/// - System, developer, user and tool content loses NUL, `<think>` and
///   `</think>`, and role markers become `&lt;|role|&gt;`, applied in the
///   template's order. A system or developer `framing_suffix` continues its
///   content. Assistant content and `thought` (rendered as
///   `<think>...</think>`) are written verbatim.
/// - Historical calls are `<tool_call>NAME<arg_key>K</arg_key><arg_value>V
///   </arg_value>...</tool_call>`; `ToolArgument::value` is written verbatim
///   (raw text for strings, the API layer's `tojson` spelling otherwise).
/// - The generation prompt is `<|assistant|>`; Flint decides on each turn
///   whether to open `<think>`, so the template has no thinking switch.
[[nodiscard]] std::string RenderChat(
    std::span<const tokenization::ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools, bool add_generation_prompt);

/// Suffix that opens a new assistant turn.
[[nodiscard]] constexpr std::string_view GenerationPrompt() noexcept {
  return "<|assistant|>";
}

/// Rejects artifacts that are not `flint` or whose embedded
/// `tokenizer.chat_template` differs from the template RenderChat implements.
[[nodiscard]] bool ValidateGgufTemplate(const core::GgufReader& reader,
                                        std::string* error_msg = nullptr);

[[nodiscard]] constexpr std::string_view ChatTemplateId() noexcept {
  return "flint-preview-compiled-v1";
}

/// SHA-256 of `chat_template.jinja` in the Flint source repository, identical
/// to the GGUF `tokenizer.chat_template` of unsloth/flint-GGUF.
[[nodiscard]] constexpr std::string_view ArtifactTemplateSha256() noexcept {
  return "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466";
}

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_CHAT_TEMPLATE_HPP_
