#ifndef GUFO_MODELS_FLINT_CONTROL_TOKENS_HPP_
#define GUFO_MODELS_FLINT_CONTROL_TOKENS_HPP_

#include <array>
#include <string_view>

namespace gufo::models::flint {

/// Flint's added-token texts (ids 150000..150018 in the GGUF vocabulary).
/// Role markers and end of text are control tokens; the think, tool and
/// argument tags are user-defined added tokens. Both kinds are split out of
/// text that is encoded with special-token parsing, as HF `tokenizers` does.
inline constexpr std::string_view kBeginOfText = "<|begin_of_text|>";
inline constexpr std::string_view kEndOfText = "<|end_of_text|>";
inline constexpr std::string_view kSystem = "<|system|>";
inline constexpr std::string_view kUser = "<|user|>";
inline constexpr std::string_view kAssistant = "<|assistant|>";
inline constexpr std::string_view kObservation = "<|observation|>";
inline constexpr std::string_view kDeveloper = "<|developer|>";
inline constexpr std::string_view kThinkOpen = "<think>";
inline constexpr std::string_view kThinkClose = "</think>";
inline constexpr std::string_view kToolsOpen = "<tools>";
inline constexpr std::string_view kToolsClose = "</tools>";
inline constexpr std::string_view kToolCallOpen = "<tool_call>";
inline constexpr std::string_view kToolCallClose = "</tool_call>";
inline constexpr std::string_view kArgKeyOpen = "<arg_key>";
inline constexpr std::string_view kArgKeyClose = "</arg_key>";
inline constexpr std::string_view kArgValueOpen = "<arg_value>";
inline constexpr std::string_view kArgValueClose = "</arg_value>";
inline constexpr std::string_view kToolResponseOpen = "<tool_response>";
inline constexpr std::string_view kToolResponseClose = "</tool_response>";

/// Markers that end an assistant turn: the GGUF end-of-generation tokens other
/// than `<|developer|>`, which the model never writes inside a reply.
inline constexpr std::array<std::string_view, 3> kTurnEndMarkers = {
    kEndOfText, kUser, kObservation};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_CONTROL_TOKENS_HPP_
