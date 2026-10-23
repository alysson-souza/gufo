#include "src/models/flint/tool_calls.hpp"

#include <algorithm>
#include <string_view>

namespace gufo::models::flint {
namespace {

constexpr std::string_view kWhitespace = " \t\n\r\f\v";

bool IsWhitespace(char character) {
  return kWhitespace.find(character) != std::string_view::npos;
}

std::size_t SkipWhitespace(std::string_view text, std::size_t position) {
  const auto next = text.find_first_not_of(kWhitespace, position);
  return next == std::string_view::npos ? text.size() : next;
}

// Bytes at the end of `text` that could begin `tag`.
std::size_t HeldTagPrefix(std::string_view text, std::string_view tag) {
  for (std::size_t length = std::min(text.size(), tag.size() - 1); length > 0;
       --length) {
    if (tag.starts_with(text.substr(text.size() - length)))
      return length;
  }
  return 0;
}

}  // namespace

FlintToolCallScan ScanFlintToolCall(std::string_view text) {
  using Syntax = FlintToolSyntax;
  FlintToolCallScan scan;
  if (!text.starts_with(Syntax::kCallOpen))
    return scan;
  scan.matched = true;
  // Every return below the name check is a well-formed prefix unless it says
  // otherwise.
  scan.valid = true;
  std::size_t position = SkipWhitespace(text, Syntax::kCallOpen.size());
  const std::size_t name_begin = position;
  while (position < text.size() && text[position] != '<' &&
         !IsWhitespace(text[position]))
    ++position;
  scan.name = std::string(text.substr(name_begin, position - name_begin));
  if (position == text.size())
    return scan;
  if (scan.name.empty()) {
    scan.valid = false;
    return scan;
  }
  scan.name_complete = true;
  while (true) {
    position = SkipWhitespace(text, position);
    const std::string_view rest = text.substr(position);
    if (rest.empty())
      return scan;
    if (rest.starts_with(Syntax::kCallClose)) {
      scan.complete = true;
      scan.end = position + Syntax::kCallClose.size();
      return scan;
    }
    if (!rest.starts_with(Syntax::kKeyOpen)) {
      scan.valid = Syntax::kCallClose.starts_with(rest) ||
                   Syntax::kKeyOpen.starts_with(rest);
      return scan;
    }
    position += Syntax::kKeyOpen.size();
    const std::size_t key_end = text.find(Syntax::kKeyClose, position);
    if (key_end == std::string_view::npos)
      return scan;
    FlintToolArgument argument{
        .key = std::string(text.substr(position, key_end - position)),
        .value = {}};
    position = SkipWhitespace(text, key_end + Syntax::kKeyClose.size());
    const std::string_view value_open = text.substr(position);
    if (!value_open.starts_with(Syntax::kValueOpen)) {
      scan.valid = Syntax::kValueOpen.starts_with(value_open);
      return scan;
    }
    position += Syntax::kValueOpen.size();
    const std::size_t value_end = text.find(Syntax::kValueClose, position);
    if (value_end == std::string_view::npos) {
      std::string_view value = text.substr(position);
      value.remove_suffix(HeldTagPrefix(value, Syntax::kValueClose));
      argument.value = std::string(value);
      scan.unfinished = std::move(argument);
      return scan;
    }
    argument.value = std::string(text.substr(position, value_end - position));
    scan.arguments.push_back(std::move(argument));
    position = value_end + Syntax::kValueClose.size();
  }
}

}  // namespace gufo::models::flint
