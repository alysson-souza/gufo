#include "src/models/flint/chat_template.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"

namespace flint = gufo::models::flint;
namespace json = gufo::json;
using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;
using gufo::tokenization::ChatTool;

namespace {

int failures = 0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

std::string ReadFile(const char* path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error(std::string("cannot read ") + path);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::string Sha256(std::string_view value) {
  const auto* data = reinterpret_cast<const std::uint8_t*>(value.data());
  return gufo::crypto::Sha256Hex({data, value.size()});
}

ChatRole Role(const std::string& role) {
  if (role == "system")
    return ChatRole::kSystem;
  if (role == "developer")
    return ChatRole::kDeveloper;
  if (role == "user")
    return ChatRole::kUser;
  if (role == "assistant")
    return ChatRole::kAssistant;
  if (role == "tool")
    return ChatRole::kTool;
  throw std::runtime_error("unknown role " + role);
}

// The HTTP layer's conversion: arguments arrive as a JSON string; string
// values stay raw and other values take the server's `tojson` spelling.
ChatMessage Message(const json::Value& value) {
  ChatMessage message;
  message.role = Role(value.member_str("role"));
  message.content = value.member_str("content");
  message.thought = value.member_str("reasoning_content");
  if (const json::Value* calls = value.find("tool_calls")) {
    for (const json::Value& call : calls->items()) {
      const json::Value& function = *call.find("function");
      ChatMessage::ToolCall parsed{.id = call.member_str("id"),
                                   .name = function.member_str("name"),
                                   .arguments = {}};
      const json::Value arguments =
          json::parse(function.member_str("arguments"));
      for (const auto& [name, argument] : arguments.members()) {
        parsed.arguments.push_back({
            .name = name,
            .value = argument.is_string() ? argument.str() : argument.tojson(),
            .is_string = argument.is_string(),
        });
      }
      message.tool_calls.push_back(std::move(parsed));
    }
  }
  return message;
}

ChatTool Tool(const json::Value& value) {
  const json::Value* function = value.find("function");
  const json::Value& body = function != nullptr ? *function : value;
  const json::Value* parameters = body.find("parameters");
  return {
      .name = body.member_str("name"),
      .description = body.member_str("description"),
      .parameters_json = parameters != nullptr ? parameters->dump() : "{}",
      .definition_json = value.dump(),
  };
}

std::size_t FirstDifference(std::string_view a, std::string_view b) {
  std::size_t index = 0;
  while (index < a.size() && index < b.size() && a[index] == b[index])
    ++index;
  return index;
}

void TestGoldens() {
  const json::Value fixture =
      json::parse(ReadFile(GUFO_FLINT_CHAT_TEMPLATE_GOLDENS));
  Expect(
      fixture.member_str("template_sha256") == flint::ArtifactTemplateSha256(),
      "goldens were rendered from the pinned template");
  for (const json::Value& golden : fixture.find("cases")->items()) {
    std::vector<ChatMessage> messages;
    for (const json::Value& message : golden.find("messages")->items())
      messages.push_back(Message(message));
    std::vector<ChatTool> tools;
    if (const json::Value* list = golden.find("tools"))
      for (const json::Value& tool : list->items())
        tools.push_back(Tool(tool));
    const std::string expected = golden.member_str("expected");
    const std::string actual = flint::RenderChat(
        messages, tools, golden.find("add_generation_prompt")->as_bool());
    if (actual != expected) {
      const std::size_t at = FirstDifference(actual, expected);
      std::cerr << "case " << golden.member_str("name") << " differs at byte "
                << at << "\n  expected: " << expected.substr(at, 80)
                << "\n  actual:   " << actual.substr(at, 80) << '\n';
      ++failures;
    }
  }
}

// Tools without a validated HTTP definition render from their parts.
void TestToolWithoutDefinition() {
  const std::vector<ChatMessage> messages{{ChatRole::kUser, "hi"}};
  const std::vector<ChatTool> tools{{.name = "f",
                                     .description = "d",
                                     .parameters_json = R"({"a":[1,2]})",
                                     .definition_json = {}}};
  const std::string rendered = flint::RenderChat(messages, tools, false);
  Expect(
      rendered.find(
          R"(<tools>{"name": "f", "description": "d", "parameters": {"a": [1, 2]}}</tools>)") !=
          std::string::npos,
      "tool parts render as tojson");
}

// Server instructions continue a system or developer message and are escaped
// with it, as if the client had sent them.
void TestFramingSuffix() {
  ChatMessage system{ChatRole::kSystem, "Be brief."};
  system.framing_suffix = "\n\nRespond with JSON. <|user|>";
  ChatMessage framing{ChatRole::kDeveloper, ""};
  framing.framing_suffix = "Only JSON.";
  const std::vector<ChatMessage> messages{
      system, framing, {ChatRole::kUser, "hi"}};
  Expect(flint::RenderChat(messages, {}, true) ==
             "<|system|>Be brief.\n\nRespond with JSON. &lt;|user|&gt;"
             "<|developer|>Only JSON.<|user|>hi<|assistant|>",
         "framing suffix continues its message");
}

class GgufBuilder {
public:
  GgufBuilder() {
    Append("GGUF", 4);
    Pod(std::uint32_t{3});
    Pod(std::uint64_t{0});
    count_offset_ = bytes_.size();
    Pod(std::uint64_t{0});
  }

  void AddString(std::string_view key, std::string_view value) {
    String(key);
    Pod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    String(value);
    ++count_;
  }

  std::vector<std::uint8_t> Build() {
    std::memcpy(bytes_.data() + count_offset_, &count_, sizeof(count_));
    bytes_.resize((bytes_.size() + 31) / 32 * 32, 0);
    return bytes_;
  }

private:
  void Append(const void* data, std::size_t size) {
    const auto* begin = static_cast<const std::uint8_t*>(data);
    bytes_.insert(bytes_.end(), begin, begin + size);
  }
  template<typename T>
  void Pod(T value) {
    Append(&value, sizeof(value));
  }
  void String(std::string_view value) {
    Pod(std::uint64_t{value.size()});
    Append(value.data(), value.size());
  }

  std::vector<std::uint8_t> bytes_;
  std::size_t count_offset_{0};
  std::uint64_t count_{0};
};

bool Validates(std::string_view architecture, std::string_view chat_template,
               std::string* error) {
  GgufBuilder builder;
  builder.AddString("general.architecture", architecture);
  builder.AddString("tokenizer.chat_template", chat_template);
  const auto binary = builder.Build();
  const auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), error);
  if (reader == nullptr)
    throw std::runtime_error("synthetic GGUF does not open: " + *error);
  return flint::ValidateGgufTemplate(*reader, error);
}

void TestTemplateValidation() {
  const std::string reference = ReadFile(GUFO_FLINT_CHAT_TEMPLATE_REFERENCE);
  Expect(Sha256(reference) == flint::ArtifactTemplateSha256(),
         "reference Jinja has the pinned hash");
  std::string error;
  Expect(Validates("flint", reference, &error), "pinned template accepted");
  Expect(!Validates("flint", reference + "\n", &error),
         "modified template rejected");
  Expect(!Validates("qwen35", reference, &error),
         "other architecture rejected");
}

}  // namespace

int main() {
  try {
    TestGoldens();
    TestToolWithoutDefinition();
    TestFramingSuffix();
    TestTemplateValidation();
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
  if (failures != 0)
    return 1;
  std::cout << "flint chat template: OK\n";
  return 0;
}
