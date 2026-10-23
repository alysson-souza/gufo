#include "src/models/flint/chat_template.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/models/flint/control_tokens.hpp"

namespace gufo::models::flint {
namespace {

using tokenization::ChatMessage;
using tokenization::ChatRole;
using tokenization::ChatTool;

struct Replacement {
  std::string_view from;
  std::string_view to;
};

// The template's flint_text macro, in its order: each replace runs over the
// output of the previous one, so "<thi<think>nk>" still yields "<think>".
constexpr std::array<Replacement, 10> kFlintTextReplacements = {{
    {std::string_view("\0", 1), ""},
    {kThinkOpen, ""},
    {kThinkClose, ""},
    {kBeginOfText, "&lt;|begin_of_text|&gt;"},
    {kEndOfText, "&lt;|end_of_text|&gt;"},
    {kSystem, "&lt;|system|&gt;"},
    {kDeveloper, "&lt;|developer|&gt;"},
    {kUser, "&lt;|user|&gt;"},
    {kAssistant, "&lt;|assistant|&gt;"},
    {kObservation, "&lt;|observation|&gt;"},
}};

void AppendFlintText(std::string& output, std::string_view text) {
  // Every pattern starts with '<' or is NUL; most content has neither.
  if (text.find_first_of(std::string_view("<\0", 2)) ==
      std::string_view::npos) {
    output.append(text);
    return;
  }
  std::string current(text);
  std::string next;
  for (const auto& [from, to] : kFlintTextReplacements) {
    std::size_t position = current.find(from);
    if (position == std::string::npos)
      continue;
    next.clear();
    std::size_t start = 0;
    while (position != std::string::npos) {
      next.append(current, start, position - start);
      next.append(to);
      start = position + from.size();
      position = current.find(from, start);
    }
    next.append(current, start);
    std::swap(current, next);
  }
  output.append(current);
}

void AppendToolDefinition(std::string& output, const ChatTool& tool) {
  if (!tool.definition_json.empty()) {
    const json::Value definition = json::parse(tool.definition_json);
    const json::Value* function = definition.find("function");
    output.append(function != nullptr ? function->tojson()
                                      : definition.tojson());
    return;
  }
  json::Value function = json::Value::object();
  function["name"] = tool.name;
  function["description"] = tool.description;
  function["parameters"] = tool.parameters_json.empty()
                               ? json::Value::object()
                               : json::parse(tool.parameters_json);
  output.append(function.tojson());
}

void AppendToolPreamble(std::string& output, std::span<const ChatTool> tools) {
  if (tools.empty())
    return;
  output.append(kSystem);
  output.append(
      "\n# Tools\n\nYou may call one or more functions to assist with the "
      "user query.\n\nYou are provided with function signatures within ");
  output.append(kToolsOpen);
  output.append(kToolsClose);
  output.append(" XML tags:\n");
  output.append(kToolsOpen);
  // trim_blocks removes the template's newlines around each definition.
  for (const ChatTool& tool : tools)
    AppendToolDefinition(output, tool);
  output.append(kToolsClose);
  output.append(
      "\n\nFor each function call, output the function name and arguments "
      "within the following XML format:\n");
  output.append(kToolCallOpen);
  output.append("{function-name}");
  output.append(kArgKeyOpen);
  output.append("{arg-key-1}");
  output.append(kArgKeyClose);
  output.append(kArgValueOpen);
  output.append("{arg-value-1}");
  output.append(kArgValueClose);
  output.append("...");
  output.append(kToolCallClose);
  output.push_back('\n');
}

void AppendAssistant(std::string& output, const ChatMessage& message) {
  output.append(kAssistant);
  if (!message.thought.empty()) {
    output.append(kThinkOpen);
    output.append(message.thought);
    output.append(kThinkClose);
  }
  output.append(message.content);
  for (const ChatMessage::ToolCall& call : message.tool_calls) {
    output.append(kToolCallOpen);
    output.append(call.name);
    for (const ChatMessage::ToolArgument& argument : call.arguments) {
      output.append(kArgKeyOpen);
      output.append(argument.name);
      output.append(kArgKeyClose);
      output.append(kArgValueOpen);
      output.append(argument.value);
      output.append(kArgValueClose);
    }
    output.append(kToolCallClose);
  }
}

std::string Sha256(std::string_view value) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(value.data());
  return crypto::Sha256Hex({begin, value.size()});
}

void AssignError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr)
    *error_msg = std::move(message);
}

}  // namespace

std::string RenderChat(std::span<const ChatMessage> messages,
                       std::span<const ChatTool> tools,
                       bool add_generation_prompt) {
  std::string output;
  bool tools_pending = true;
  for (const ChatMessage& message : messages) {
    if (tools_pending && message.role != ChatRole::kSystem) {
      AppendToolPreamble(output, tools);
      tools_pending = false;
    }
    switch (message.role) {
      case ChatRole::kSystem:
      case ChatRole::kDeveloper:
        output.append(message.role == ChatRole::kSystem ? kSystem : kDeveloper);
        // Server instructions continue the message, as if it had sent them.
        if (message.framing_suffix.empty())
          AppendFlintText(output, message.content);
        else
          AppendFlintText(output, message.content + message.framing_suffix);
        break;
      case ChatRole::kUser:
        output.append(kUser);
        AppendFlintText(output, message.content);
        break;
      case ChatRole::kAssistant:
        AppendAssistant(output, message);
        break;
      case ChatRole::kTool:
        output.append(kObservation);
        output.append(kToolResponseOpen);
        AppendFlintText(output, message.content);
        output.append(kToolResponseClose);
        break;
    }
  }
  if (tools_pending)
    AppendToolPreamble(output, tools);
  if (add_generation_prompt)
    output.append(GenerationPrompt());
  return output;
}

bool ValidateGgufTemplate(const core::GgufReader& reader,
                          std::string* error_msg) {
  if (reader.GetMetadataString("general.architecture") != "flint") {
    AssignError(error_msg, "Flint artifact has unexpected architecture");
    return false;
  }
  const auto raw_template = reader.GetMetadataString("tokenizer.chat_template");
  if (!raw_template.has_value() || raw_template->empty()) {
    AssignError(error_msg,
                "Flint GGUF is missing tokenizer.chat_template metadata");
    return false;
  }
  const std::string actual_hash = Sha256(*raw_template);
  if (actual_hash != ArtifactTemplateSha256()) {
    AssignError(error_msg,
                "Flint GGUF chat template SHA-256 is not the pinned Flint "
                "Preview template: " +
                    actual_hash);
    return false;
  }
  return true;
}

}  // namespace gufo::models::flint
