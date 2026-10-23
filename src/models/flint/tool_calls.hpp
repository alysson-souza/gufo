#ifndef GUFO_MODELS_FLINT_TOOL_CALLS_HPP_
#define GUFO_MODELS_FLINT_TOOL_CALLS_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/flint/control_tokens.hpp"

namespace gufo::models::flint {

/// The literals of a Flint tool call, as RenderChat writes historical calls:
/// `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`
/// with string values raw and other values as JSON. ScanFlintToolCall reads
/// the same bytes back, so the wire format cannot change in one direction only.
struct FlintToolSyntax {
  static constexpr std::string_view kCallOpen = kToolCallOpen;
  static constexpr std::string_view kCallClose = kToolCallClose;
  static constexpr std::string_view kKeyOpen = kArgKeyOpen;
  static constexpr std::string_view kKeyClose = kArgKeyClose;
  static constexpr std::string_view kValueOpen = kArgValueOpen;
  static constexpr std::string_view kValueClose = kArgValueClose;
};

/// One argument exactly as the model wrote it: key and value between their
/// tags, untrimmed. The API surface types the value against the tool schema.
struct FlintToolArgument {
  std::string key;
  std::string value;
};

/// One call scanned from `text[0]`. The accepted syntax is that of Flint's
/// reference parser (rec_mlx `parse_tool_call` over SGLang's call regex):
/// whitespace around the name and between argument tags is allowed, a name is
/// a run of characters other than whitespace and '<', keys end at the first
/// `</arg_key>` and values at the first `</arg_value>`.
struct FlintToolCallScan {
  /// `text` begins with `<tool_call>`.
  bool matched{false};
  /// Everything scanned is well formed. Without `complete` this means the
  /// text so far is a prefix of a well-formed call.
  bool valid{false};
  /// The name ended at an argument or the closing tag.
  bool name_complete{false};
  /// `</tool_call>` was consumed; `end` is the byte after it.
  bool complete{false};
  std::size_t end{0};
  std::string name;
  std::vector<FlintToolArgument> arguments;
  /// An argument whose value ran to the end of `text` without its closing
  /// tag. `value` holds back a trailing prefix of `</arg_value>`.
  std::optional<FlintToolArgument> unfinished;
};

/// Scans the Flint call beginning at `text[0]`, if any. Schema interpretation,
/// value typing and OpenAI shaping stay with the API surface.
[[nodiscard]] FlintToolCallScan ScanFlintToolCall(std::string_view text);

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_TOOL_CALLS_HPP_
