#include "src/models/flint/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/flint/control_tokens.hpp"

namespace flint = gufo::models::flint;
namespace json = gufo::json;

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

const json::Value& Member(const json::Value& object, const char* key) {
  const json::Value* value = object.find(key);
  if (value == nullptr)
    throw std::runtime_error(std::string("tokenizer goldens lack ") + key);
  return *value;
}

std::vector<std::uint32_t> Ids(const json::Value& array) {
  std::vector<std::uint32_t> ids;
  ids.reserve(array.size());
  for (const json::Value& id : array.items())
    ids.push_back(static_cast<std::uint32_t>(id.as_size()));
  return ids;
}

std::string Describe(const std::vector<std::uint32_t>& ids) {
  std::string out;
  for (const std::uint32_t id : ids)
    out += std::to_string(id) + ' ';
  return out;
}

void ExpectRoundTrip(const flint::Tokenizer& tokenizer, std::string_view text,
                     std::string_view name) {
  for (const bool parse_special : {true, false}) {
    const auto ids = tokenizer.Encode(text, parse_special);
    Expect(tokenizer.Decode(ids) == text,
           std::string(name) + (parse_special ? " special" : " plain") +
               ": Decode(Encode(text)) != text");
  }
}

void CheckGoldens(const flint::Tokenizer& tokenizer, const json::Value& root) {
  const json::Value& cases = Member(root, "cases");
  if (cases.empty())
    throw std::runtime_error("tokenizer goldens have no cases");
  for (const json::Value& item : cases.items()) {
    const std::string name = Member(item, "name").str();
    const std::string& text = Member(item, "text").str();
    for (const auto* mode : {"special", "plain"}) {
      const bool parse_special = std::string_view(mode) == "special";
      const auto expected = Ids(Member(item, mode));
      const auto actual = tokenizer.Encode(text, parse_special);
      if (actual != expected) {
        Expect(false, name + " " + mode + ": expected " + Describe(expected) +
                          "\n  got " + Describe(actual));
      }
    }
    Expect(Member(item, "decodes").as_bool(),
           name + ": HF decode is not the text");
    ExpectRoundTrip(tokenizer, text, name);
  }
  std::cout << "checked " << cases.size() << " golden cases (tokenizers "
            << Member(root, "tokenizers_version").str() << ")\n";
}

void CheckInvalidUtf8(const flint::Tokenizer& tokenizer) {
  const std::vector<std::string> cases = {
      "\x80",
      "a\x80z",
      "\xBF\xBF",
      "\xE4\xB8",          // truncated 3-byte sequence
      "x\xE4\xB8 y",       // truncated mid-text
      "\xF0\x9F\x98",      // truncated 4-byte sequence
      "\xC0\xAF",          // overlong '/'
      "\xE0\x80\xAF",      // overlong '/'
      "\xED\xA0\x80",      // surrogate
      "\xF4\x90\x80\x80",  // above U+10FFFF
      "\xFF",
      "\xFE\xFF hello \xFF",
      "don\xFF't \xFFHello \xC3",
      "<think>\xFF</think>\x80",
      "123\xFF"
      "456",
  };
  for (const std::string& text : cases)
    ExpectRoundTrip(tokenizer, text, "invalid utf-8");
  std::string all_bytes;
  for (int round = 0; round < 2; ++round) {
    for (int b = 0; b < 256; ++b)
      all_bytes.push_back(static_cast<char>(b));
  }
  ExpectRoundTrip(tokenizer, all_bytes, "all bytes");
}

void CheckFacts(const flint::Tokenizer& tokenizer) {
  Expect(tokenizer.VocabSize() == 150144, "vocabulary size");
  for (const std::uint32_t id : {150001U, 150003U, 150005U, 150018U})
    Expect(tokenizer.IsEndOfGeneration(id), "end-of-generation id");
  for (const std::uint32_t id : {0U, 150000U, 150002U, 150004U, 150006U})
    Expect(!tokenizer.IsEndOfGeneration(id), "not end of generation");

  Expect(tokenizer.FindSpecial(flint::kThinkOpen) == 150006U, "<think> id");
  Expect(tokenizer.FindSpecial(flint::kToolResponseClose) == 150017U,
         "</tool_response> id");
  Expect(tokenizer.FindSpecial(flint::kBeginOfText) == 150000U, "bos id");
  Expect(tokenizer.FindSpecial(flint::kDeveloper) == 150018U, "developer id");
  Expect(tokenizer.FindSpecial("<|reserved_150143|>") == 150143U,
         "last reserved id");
  Expect(!tokenizer.FindSpecial("<thin").has_value(), "partial added token");
  Expect(!tokenizer.FindSpecial("hello").has_value(), "ordinary text");
  for (std::uint32_t id = 150000; id < 150144; ++id)
    Expect(tokenizer.IsSpecial(id), "added tokens are special");
  Expect(!tokenizer.IsSpecial(0), "byte token is not special");
  Expect(!tokenizer.IsSpecial(149999), "normal token is not special");
  Expect(!tokenizer.IsSpecial(tokenizer.VocabSize()), "out of range");

  Expect(tokenizer.TokenBytes(150006) == flint::kThinkOpen, "<think> bytes");
  Expect(tokenizer.TokenBytes(tokenizer.VocabSize()).empty(),
         "out-of-range bytes");
  const auto hello = tokenizer.Encode(" hello", false);
  Expect(hello.size() == 1 && tokenizer.TokenBytes(hello[0]) == " hello",
         "' hello' is one token with its raw bytes");
  Expect(
      tokenizer.Encode("<think>", true) == std::vector<std::uint32_t>{150006},
      "<think> parses as one added token");
  const auto plain = tokenizer.Encode("<think>", false);
  Expect(plain.size() > 1 && !tokenizer.IsSpecial(plain[0]),
         "<think> is text without special parsing");
}

/// About 1 MB of the golden texts' space-separated words in pseudo-random
/// order with numbers between, so pieces meet in combinations the corpus lacks.
std::string ShuffledText(const json::Value& root) {
  std::vector<std::string_view> words;
  for (const json::Value& item : Member(root, "cases").items()) {
    const std::string_view text = Member(item, "text").str();
    for (std::size_t start = 0; start < text.size();) {
      const std::size_t end = std::min(text.find(' ', start + 1), text.size());
      words.push_back(text.substr(start, end - start));
      start = end;
    }
  }
  std::string text;
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  while (text.size() < (1U << 20U)) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    text += words[(state >> 33U) % words.size()];
    if ((state & 7U) == 0)
      text += std::to_string(state % 100000);
  }
  return text;
}

void CheckThroughput(const flint::Tokenizer& tokenizer,
                     const json::Value& root) {
  std::string corpus;
  while (corpus.size() < (1U << 20U)) {
    for (const json::Value& item : Member(root, "cases").items())
      corpus += Member(item, "text").str();
  }
  const std::string shuffled = ShuffledText(root);
  for (const auto& [label, text] :
       {std::pair<const char*, const std::string&>{"corpus", corpus},
        std::pair<const char*, const std::string&>{"shuffled", shuffled}}) {
    for (const bool parse_special : {true, false}) {
      const auto start = std::chrono::steady_clock::now();
      const auto ids = tokenizer.Encode(text, parse_special);
      const std::chrono::duration<double> seconds =
          std::chrono::steady_clock::now() - start;
      std::cout << "encode " << label << ' ' << text.size() << " bytes ("
                << (parse_special ? "special" : "plain") << "): " << ids.size()
                << " tokens in " << seconds.count() * 1e3 << " ms, "
                << text.size() / seconds.count() / 1e6 << " MB/s\n";
      Expect(seconds.count() < 5.0, "1 MB encode takes under 5 s");
      Expect(tokenizer.Decode(ids) == text, "long text round trip");
    }
  }
}

}  // namespace

int main() {
  const char* model = std::getenv("GUFO_FLINT_MODEL");
  if (model == nullptr || *model == '\0') {
    std::cout << "SKIP: GUFO_FLINT_MODEL is not set\n";
    return 77;
  }
  try {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model, &error);
    if (reader == nullptr)
      throw std::runtime_error("cannot open " + std::string(model) + ": " +
                               error);
    const auto tokenizer = flint::Tokenizer::FromGguf(*reader, &error);
    if (!tokenizer)
      throw std::runtime_error("cannot load tokenizer: " + error);
    const json::Value root =
        json::parse(ReadFile(GUFO_FLINT_TOKENIZER_GOLDENS));
    CheckGoldens(*tokenizer, root);
    CheckInvalidUtf8(*tokenizer);
    CheckFacts(*tokenizer);
    CheckThroughput(*tokenizer, root);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
