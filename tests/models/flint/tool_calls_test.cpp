#include "src/models/flint/tool_calls.hpp"

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/flint/chat_template.hpp"

namespace flint = gufo::models::flint;
using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;

namespace {

int failures = 0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

bool SameArguments(
    const std::vector<flint::FlintToolArgument>& actual,
    const std::vector<std::pair<std::string, std::string>>& expected) {
  if (actual.size() != expected.size())
    return false;
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (actual[i].key != expected[i].first ||
        actual[i].value != expected[i].second)
      return false;
  return true;
}

// The renderer and the scanner share one wire format: a historical call
// rendered into the prompt scans back to the same name and arguments.
void TestRenderedCallsScanBack() {
  ChatMessage assistant{ChatRole::kAssistant, ""};
  assistant.tool_calls.push_back(
      {.id = "a",
       .name = "search.web",
       .arguments = {{.name = "query", .value = "a <b> \"c\"\n</tool_call>"},
                     {.name = "limit", .value = "3", .is_string = false},
                     {.name = "filters",
                      .value = R"({"site": null, "tags": ["x"]})",
                      .is_string = false}}});
  assistant.tool_calls.push_back({.id = "b", .name = "now", .arguments = {}});
  const std::vector<ChatMessage> messages{assistant};
  const std::string rendered = flint::RenderChat(messages, {}, false);
  std::string_view rest = rendered;
  rest.remove_prefix(rest.find(flint::FlintToolSyntax::kCallOpen));
  for (const auto& call : assistant.tool_calls) {
    const auto scan = flint::ScanFlintToolCall(rest);
    Expect(scan.matched && scan.valid && scan.complete,
           "rendered call scans complete");
    Expect(scan.name == call.name, "rendered name scans back");
    std::vector<std::pair<std::string, std::string>> expected;
    for (const auto& argument : call.arguments)
      expected.emplace_back(argument.name, argument.value);
    Expect(SameArguments(scan.arguments, expected),
           "rendered arguments scan back verbatim");
    rest.remove_prefix(scan.end);
  }
  Expect(rest.empty(), "both rendered calls were consumed");
}

void TestReferenceSyntax() {
  // Whitespace around the name and between tags, as rec_mlx accepts it.
  auto scan = flint::ScanFlintToolCall(
      "<tool_call> get_weather \n<arg_key>city</arg_key> <arg_value> Oslo "
      "</arg_value>\n</tool_call>tail");
  Expect(scan.complete && scan.valid && scan.name == "get_weather" &&
             SameArguments(scan.arguments, {{"city", " Oslo "}}),
         "spaced call keeps raw values");
  Expect(scan.end == std::string_view(
                         "<tool_call> get_weather \n<arg_key>city</arg_key> "
                         "<arg_value> Oslo </arg_value>\n</tool_call>")
                         .size(),
         "end stops after the closing tag");

  scan = flint::ScanFlintToolCall("<tool_call>now</tool_call>");
  Expect(scan.complete && scan.name == "now" && scan.arguments.empty(),
         "call without arguments");

  for (const std::string_view malformed :
       {"<tool_call></tool_call>", "<tool_call>two words</tool_call>",
        "<tool_call>f<arg_value>1</arg_value></tool_call>",
        "<tool_call>f<arg_key>k</arg_key>x<arg_value>1</arg_value>"
        "</tool_call>",
        "<tool_call>f<arg_key>k</arg_key><arg_value>1</arg_value>junk"}) {
    scan = flint::ScanFlintToolCall(malformed);
    Expect(scan.matched && !scan.valid && !scan.complete,
           "malformed call is rejected");
  }
  Expect(!flint::ScanFlintToolCall("prose <tool_call>f</tool_call>").matched,
         "a scan starts at the marker");
}

// Every prefix of a well-formed call is a valid unfinished scan, and an open
// value never includes a partial closing tag.
void TestStreamingPrefixes() {
  const std::string call =
      "<tool_call>write<arg_key>path</arg_key><arg_value>a.txt</arg_value>"
      "<arg_key>body</arg_key><arg_value>x</arg_val</arg_value></tool_call>";
  for (std::size_t length = flint::FlintToolSyntax::kCallOpen.size();
       length < call.size(); ++length) {
    const auto scan = flint::ScanFlintToolCall(call.substr(0, length));
    Expect(scan.valid && !scan.complete, "prefix is a valid unfinished call");
    if (scan.unfinished) {
      const std::string& value = scan.unfinished->value;
      Expect(std::string_view("x</arg_val").starts_with(value) ||
                 std::string_view("a.txt").starts_with(value),
             "an open value is a prefix of the written value");
      Expect(!value.ends_with("</") && !value.ends_with("</arg_value"),
             "a partial closing tag is held back");
    }
  }
  const auto scan = flint::ScanFlintToolCall(call);
  Expect(scan.complete &&
             SameArguments(scan.arguments,
                           {{"path", "a.txt"}, {"body", "x</arg_val"}}),
         "the complete call keeps the value's look-alike tag");
  const auto open = flint::ScanFlintToolCall("<tool_call>wri");
  Expect(open.valid && !open.name_complete, "a name may still grow");
}

}  // namespace

int main() {
  TestRenderedCallsScanBack();
  TestReferenceSyntax();
  TestStreamingPrefixes();
  if (failures != 0)
    return 1;
  std::cout << "flint tool calls: OK\n";
  return 0;
}
