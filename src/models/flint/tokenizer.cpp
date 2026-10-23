#include "src/models/flint/tokenizer.hpp"

#include <unicode/uchar.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::flint {

namespace {

constexpr std::uint32_t kNoToken = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kTokenTypeControl = 3;
constexpr std::uint32_t kTokenTypeUserDefined = 4;

/// Code-point classes of the pre-tokenizer regexes. `kUpper` is
/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]` and `kLower` `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`;
/// `kSpace` is Oniguruma's Unicode `\s`, the White_Space property.
enum Class : std::uint8_t {
  kUpper = 1U << 0U,
  kLower = 1U << 1U,
  kLetter = 1U << 2U,
  kNumber = 1U << 3U,
  kPunctSymbol = 1U << 4U,
  kSpace = 1U << 5U,
  kNewline = 1U << 6U,
  kCjk = 1U << 7U,
};

/// Oniguruma in the pinned HF `tokenizers` classifies with Unicode 16.0, and
/// ICU may be newer: code points assigned later are unassigned (Cn) there, and
/// U+0295 is still Ll (Unicode 17 made it Lo). These are the only differences
/// from ICU 78 (Unicode 17.0) over all code points.
constexpr std::uint8_t kReferenceUnicodeMajor = 16;

std::uint32_t ReferenceCategoryMask(UChar32 value) {
  UVersionInfo age;
  u_charAge(value, age);
  if (age[0] > kReferenceUnicodeMajor ||
      (age[0] == kReferenceUnicodeMajor && (age[1] != 0 || age[2] != 0))) {
    return U_GC_CN_MASK;
  }
  if (value == 0x0295) {
    return U_GC_LL_MASK;
  }
  return static_cast<std::uint32_t>(U_GET_GC_MASK(value));
}

std::uint8_t ComputeClass(UChar32 value) {
  const std::uint32_t mask = ReferenceCategoryMask(value);
  std::uint8_t result = 0;
  if ((mask & (U_GC_LU_MASK | U_GC_LT_MASK | U_GC_LM_MASK | U_GC_LO_MASK |
               U_GC_M_MASK)) != 0) {
    result |= kUpper;
  }
  if ((mask & (U_GC_LL_MASK | U_GC_LM_MASK | U_GC_LO_MASK | U_GC_M_MASK)) !=
      0) {
    result |= kLower;
  }
  if ((mask & U_GC_L_MASK) != 0) {
    result |= kLetter;
  }
  if ((mask & U_GC_N_MASK) != 0) {
    result |= kNumber;
  }
  if ((mask & (U_GC_P_MASK | U_GC_S_MASK)) != 0) {
    result |= kPunctSymbol;
  }
  if (u_isUWhiteSpace(value) != 0) {
    result |= kSpace;
  }
  if (value == '\r' || value == '\n') {
    result |= kNewline;
  }
  if ((value >= 0x4E00 && value <= 0x9FA5) ||
      (value >= 0x3040 && value <= 0x30FF)) {
    result |= kCjk;
  }
  return result;
}

std::uint8_t ClassOf(UChar32 value) {
  static const auto kBmp = [] {
    std::vector<std::uint8_t> table(0x10000);
    for (UChar32 c = 0; c < 0x10000; ++c) {
      table[static_cast<std::size_t>(c)] = ComputeClass(c);
    }
    return table;
  }();
  return value < 0x10000 ? kBmp[static_cast<std::size_t>(value)]
                         : ComputeClass(value);
}

struct CodePoint {
  UChar32 value;
  std::uint8_t length;
  std::uint8_t cls;
};

/// Decodes the well-formed UTF-8 sequence at `offset`; any other byte is a
/// one-byte U+FFFD.
CodePoint At(std::string_view text, std::size_t offset) {
  const auto byte = [&](std::size_t index) {
    return static_cast<std::uint8_t>(text[offset + index]);
  };
  const std::uint8_t first = byte(0);
  if (first < 0x80U) {
    return {first, 1, ClassOf(first)};
  }
  const std::size_t left = text.size() - offset;
  UChar32 value = 0;
  std::uint8_t length = 0;
  std::uint8_t low = 0x80U;
  std::uint8_t high = 0xBFU;
  if (first >= 0xC2U && first <= 0xDFU) {
    length = 2;
    value = first & 0x1FU;
  } else if (first >= 0xE0U && first <= 0xEFU) {
    length = 3;
    value = first & 0x0FU;
    low = first == 0xE0U ? 0xA0U : 0x80U;
    high = first == 0xEDU ? 0x9FU : 0xBFU;
  } else if (first >= 0xF0U && first <= 0xF4U) {
    length = 4;
    value = first & 0x07U;
    low = first == 0xF0U ? 0x90U : 0x80U;
    high = first == 0xF4U ? 0x8FU : 0xBFU;
  }
  constexpr CodePoint kInvalid = {0xFFFD, 1, kPunctSymbol};
  if (length == 0 || left < length || byte(1) < low || byte(1) > high) {
    return kInvalid;
  }
  for (std::size_t index = 1; index < length; ++index) {
    const std::uint8_t next = byte(index);
    if ((next & 0xC0U) != 0x80U) {
      return kInvalid;
    }
    value = (value << 6) | (next & 0x3FU);
  }
  return {value, length, ClassOf(value)};
}

/// End of the run of code points in `cls` starting at `offset`.
std::size_t RunEnd(std::string_view text, std::size_t offset,
                   std::uint8_t cls) {
  while (offset < text.size()) {
    const CodePoint next = At(text, offset);
    if ((next.cls & cls) == 0) {
      break;
    }
    offset += next.length;
  }
  return offset;
}

/// End of `(?i:'s|'t|'re|'ve|'m|'ll|'d)` at `offset`, or `offset`.
std::size_t ContractionEnd(std::string_view text, std::size_t offset) {
  if (offset >= text.size() || text[offset] != '\'') {
    return offset;
  }
  constexpr std::array<std::string_view, 7> kSuffixes = {"s", "t",  "re", "ve",
                                                         "m", "ll", "d"};
  for (const std::string_view suffix : kSuffixes) {
    std::size_t end = offset + 1;
    bool matches = true;
    for (const char letter : suffix) {
      if (end == text.size()) {
        matches = false;
        break;
      }
      const CodePoint next = At(text, end);
      if (u_foldCase(next.value, U_FOLD_CASE_DEFAULT) != letter) {
        matches = false;
        break;
      }
      end += next.length;
    }
    if (matches) {
      return end;
    }
  }
  return offset;
}

/// End of the first alternative of the word regex that matches at `offset`
/// within `piece`, or `offset` when none does. Oniguruma tries alternatives in
/// order and backtracks within each, so the first successful alternative wins
/// even when a later one would match more.
std::size_t WordMatchEnd(std::string_view piece, std::size_t offset) {
  const CodePoint first = At(piece, offset);
  // [^\r\n\p{L}\p{N}]? is tried with the prefix first.
  std::array<std::size_t, 2> starts = {offset + first.length, offset};
  const bool prefix = (first.cls & (kNewline | kLetter | kNumber)) == 0 &&
                      offset + first.length < piece.size();
  const std::span<const std::size_t> options =
      prefix ? std::span<const std::size_t>(starts)
             : std::span<const std::size_t>(starts).subspan(1);

  // A run of [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] from each start; the last code
  // point of the run that is also lower-class is where B+ can begin once A*
  // gives back the tail.
  struct UpperRun {
    std::size_t end;
    std::size_t last_lower;
    bool lower_after;
  };
  const auto upper_run = [&](std::size_t start) {
    UpperRun run = {start, std::string_view::npos, false};
    while (run.end < piece.size()) {
      const CodePoint next = At(piece, run.end);
      if ((next.cls & kUpper) == 0) {
        run.lower_after = (next.cls & kLower) != 0;
        break;
      }
      if ((next.cls & kLower) != 0) {
        run.last_lower = run.end;
      }
      run.end += next.length;
    }
    return run;
  };
  std::array<UpperRun, 2> runs{};
  for (std::size_t i = 0; i < options.size(); ++i) {
    runs[i] = upper_run(options[i]);
  }

  // [^\r\n\p{L}\p{N}]?[A]*[B]+(?i:...)?
  for (std::size_t i = 0; i < options.size(); ++i) {
    const UpperRun& run = runs[i];
    std::size_t lower_start = std::string_view::npos;
    if (run.lower_after) {
      lower_start = run.end;
    } else if (run.last_lower != std::string_view::npos) {
      lower_start = run.last_lower;
    }
    if (lower_start != std::string_view::npos) {
      return ContractionEnd(piece, RunEnd(piece, lower_start, kLower));
    }
  }
  // [^\r\n\p{L}\p{N}]?[A]+[B]*(?i:...)?
  for (std::size_t i = 0; i < options.size(); ++i) {
    const UpperRun& run = runs[i];
    if (run.end != options[i]) {
      return ContractionEnd(piece, RunEnd(piece, run.end, kLower));
    }
  }
  // ` ?[\p{P}\p{S}]+[\r\n]*`
  std::size_t symbols = std::string_view::npos;
  if (first.value == ' ' && offset + 1 < piece.size() &&
      (At(piece, offset + 1).cls & kPunctSymbol) != 0) {
    symbols = offset + 1;
  } else if ((first.cls & kPunctSymbol) != 0) {
    symbols = offset;
  }
  if (symbols != std::string_view::npos) {
    return RunEnd(piece, RunEnd(piece, symbols, kPunctSymbol), kNewline);
  }
  if ((first.cls & kSpace) == 0) {
    return offset;
  }
  // `\s*[\r\n]+` ends after the run's last newline; `\s+(?!\S)` gives back
  // the run's last code point before a non-space; `\s+` takes the run.
  std::size_t end = offset;
  std::size_t last_start = offset;
  std::size_t newline_end = offset;
  while (end < piece.size()) {
    const CodePoint next = At(piece, end);
    if ((next.cls & kSpace) == 0) {
      break;
    }
    last_start = end;
    end += next.length;
    if ((next.cls & kNewline) != 0) {
      newline_end = end;
    }
  }
  if (newline_end != offset) {
    return newline_end;
  }
  return end < piece.size() && last_start > offset ? last_start : end;
}

constexpr std::uint64_t PairKey(std::uint32_t left, std::uint32_t right) {
  return (static_cast<std::uint64_t>(left) << 32U) | right;
}

/// GPT-2's byte-to-code-point alphabet: printable Latin-1 bytes map to
/// themselves, the rest to U+0100 onwards in byte order.
std::array<char32_t, 256> ByteAlphabet() {
  std::array<char32_t, 256> table{};
  char32_t next = 0x100;
  for (std::size_t b = 0; b < 256; ++b) {
    const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                           (b >= 0xAE && b <= 0xFF);
    table[b] = printable ? static_cast<char32_t>(b) : next++;
  }
  return table;
}

std::string EncodeUtf8(char32_t value) {
  std::string out;
  if (value < 0x80) {
    out.push_back(static_cast<char>(value));
  } else {
    out.push_back(static_cast<char>(0xC0U | (value >> 6U)));
    out.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
  }
  return out;
}

/// The raw bytes of a byte-level token string; HF's ByteLevel decoder keeps a
/// token's own bytes when it has a code point outside the alphabet.
std::string ByteLevelDecode(std::string_view token,
                            const std::array<int, 0x144>& byte_of) {
  std::string out;
  out.reserve(token.size());
  for (std::size_t offset = 0; offset < token.size();) {
    const CodePoint next = At(token, offset);
    if (next.length == 1 && static_cast<std::uint8_t>(token[offset]) >= 0x80U) {
      return std::string(token);
    }
    if (next.value >= static_cast<UChar32>(byte_of.size()) ||
        byte_of[static_cast<std::size_t>(next.value)] < 0) {
      return std::string(token);
    }
    out.push_back(
        static_cast<char>(byte_of[static_cast<std::size_t>(next.value)]));
    offset += next.length;
  }
  return out;
}

std::vector<std::uint64_t> IntegerArray(const core::GgufReader& reader,
                                        std::string_view key, bool* valid) {
  const core::GgufMetadataValue* meta = reader.FindMetadata(key);
  *valid = false;
  if (meta == nullptr) {
    return {};
  }
  if (const auto* values =
          std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
    *valid = true;
    return *values;
  }
  if (const auto* values =
          std::get_if<std::vector<std::int64_t>>(&meta->value)) {
    std::vector<std::uint64_t> out;
    out.reserve(values->size());
    for (const std::int64_t value : *values) {
      if (value < 0) {
        return {};
      }
      out.push_back(static_cast<std::uint64_t>(value));
    }
    *valid = true;
    return out;
  }
  return {};
}

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
  return false;
}

}  // namespace

/// Per-call state: the BPE work buffers and a memo of merged pieces, which
/// repeat heavily in real text. Pieces are views into the text being encoded.
struct Tokenizer::Scratch {
  struct Candidate {
    std::uint32_t rank;
    std::uint32_t index;  ///< Left symbol; also the left-to-right tie-break.
    std::uint32_t left;
    std::uint32_t right;
  };
  std::vector<std::uint32_t> symbols;
  std::vector<std::uint32_t> previous;
  std::vector<std::uint32_t> following;
  std::vector<Candidate> heap;
  std::unordered_map<std::string_view, std::pair<std::uint32_t, std::uint32_t>>
      memo;
  std::vector<std::uint32_t> memo_tokens;
};

std::optional<Tokenizer> Tokenizer::FromGguf(const core::GgufReader& reader,
                                             std::string* error_msg) {
  UVersionInfo unicode;
  u_getUnicodeVersion(unicode);
  if (unicode[0] < kReferenceUnicodeMajor) {
    Fail(error_msg, "ICU's Unicode tables are older than the tokenizer's 16.0");
    return std::nullopt;
  }
  if (reader.GetMetadataString("tokenizer.ggml.model") !=
      std::optional<std::string_view>("gpt2")) {
    Fail(error_msg, "tokenizer.ggml.model is not gpt2");
    return std::nullopt;
  }
  if (reader.GetMetadataString("tokenizer.ggml.pre") !=
      std::optional<std::string_view>("flint")) {
    Fail(error_msg, "tokenizer.ggml.pre is not flint");
    return std::nullopt;
  }
  const std::vector<std::string_view> tokens =
      reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  const std::vector<std::string_view> merges =
      reader.GetMetadataStringArray("tokenizer.ggml.merges");
  if (tokens.empty() || tokens.size() >= kNoToken || merges.empty()) {
    Fail(error_msg, "GGUF tokenizer tokens or merges are missing");
    return std::nullopt;
  }
  bool valid = false;
  const std::vector<std::uint64_t> types =
      IntegerArray(reader, "tokenizer.ggml.token_type", &valid);
  if (!valid || types.size() != tokens.size()) {
    Fail(error_msg, "tokenizer.ggml.token_type does not match the vocabulary");
    return std::nullopt;
  }
  const std::vector<std::uint64_t> eog =
      IntegerArray(reader, "tokenizer.ggml.eog_token_ids", &valid);
  if (!valid || eog.empty() || std::ranges::any_of(eog, [&](std::uint64_t id) {
        return id >= tokens.size();
      })) {
    Fail(error_msg, "tokenizer.ggml.eog_token_ids is missing or out of range");
    return std::nullopt;
  }

  Tokenizer tokenizer;
  const auto count = static_cast<std::uint32_t>(tokens.size());
  std::unordered_map<std::string_view, std::uint32_t> ids;
  ids.reserve(count);
  tokenizer.special_.assign(count, false);
  for (std::uint32_t id = 0; id < count; ++id) {
    const bool added =
        types[id] == kTokenTypeControl || types[id] == kTokenTypeUserDefined;
    if (added) {
      tokenizer.special_[id] = true;
      tokenizer.added_.emplace(std::string(tokens[id]), id);
    } else {
      ids.emplace(tokens[id], id);
    }
  }
  if (tokenizer.added_.empty()) {
    Fail(error_msg, "GGUF vocabulary has no added tokens");
    return std::nullopt;
  }

  std::array<int, 0x144> byte_of{};
  byte_of.fill(-1);
  const std::array<char32_t, 256> alphabet = ByteAlphabet();
  for (std::size_t b = 0; b < 256; ++b) {
    byte_of[alphabet[b]] = static_cast<int>(b);
    const auto found = ids.find(EncodeUtf8(alphabet[b]));
    if (found == ids.end()) {
      Fail(error_msg, "GGUF vocabulary lacks a byte-level token");
      return std::nullopt;
    }
    tokenizer.byte_tokens_[b] = found->second;
  }

  tokenizer.merges_.reserve(merges.size());
  std::string joined;
  for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
    const std::string_view merge = merges[rank];
    const std::size_t space = merge.find(' ', 1);
    if (space == std::string_view::npos) {
      Fail(error_msg, "malformed GGUF merge");
      return std::nullopt;
    }
    const std::string_view left = merge.substr(0, space);
    const std::string_view right = merge.substr(space + 1);
    joined.assign(left).append(right);
    const auto left_id = ids.find(left);
    const auto right_id = ids.find(right);
    const auto merged = ids.find(joined);
    if (left_id == ids.end() || right_id == ids.end() || merged == ids.end()) {
      Fail(error_msg, "GGUF merge names a token outside the vocabulary");
      return std::nullopt;
    }
    tokenizer.merges_[PairKey(left_id->second, right_id->second)] = {
        rank, merged->second};
  }

  tokenizer.offsets_.reserve(count + 1);
  tokenizer.offsets_.push_back(0);
  for (std::uint32_t id = 0; id < count; ++id) {
    if (tokenizer.special_[id]) {
      tokenizer.bytes_.append(tokens[id]);
    } else {
      tokenizer.bytes_.append(ByteLevelDecode(tokens[id], byte_of));
    }
    tokenizer.offsets_.push_back(
        static_cast<std::uint32_t>(tokenizer.bytes_.size()));
  }

  for (const auto& [text, id] : tokenizer.added_) {
    if (text.empty()) {
      Fail(error_msg, "GGUF vocabulary has an empty added token");
      return std::nullopt;
    }
    tokenizer.added_lengths_.push_back(text.size());
    tokenizer.added_first_byte_[static_cast<std::uint8_t>(text[0])] = true;
  }
  std::ranges::sort(tokenizer.added_lengths_, std::greater<>());
  const auto duplicate = std::ranges::unique(tokenizer.added_lengths_);
  tokenizer.added_lengths_.erase(duplicate.begin(), duplicate.end());

  tokenizer.end_of_generation_.reserve(eog.size());
  for (const std::uint64_t id : eog) {
    tokenizer.end_of_generation_.push_back(static_cast<std::uint32_t>(id));
  }
  std::ranges::sort(tokenizer.end_of_generation_);
  return tokenizer;
}

void Tokenizer::MergePiece(std::string_view piece, Scratch& scratch,
                           std::vector<std::uint32_t>& out) const {
  // Merge the lowest-ranked adjacent pair repeatedly, leftmost first on a
  // tie, as HF's BPE word does. Symbols form a linked list and candidate
  // pairs a heap, so long pieces (runs of spaces or letters) stay
  // O(n log n); stale candidates are dropped when popped.
  const auto count = static_cast<std::uint32_t>(piece.size());
  auto& symbols = scratch.symbols;
  auto& previous = scratch.previous;
  auto& following = scratch.following;
  auto& heap = scratch.heap;
  symbols.resize(count);
  previous.resize(count);
  following.resize(count);
  heap.clear();
  for (std::uint32_t i = 0; i < count; ++i) {
    symbols[i] = byte_tokens_[static_cast<std::uint8_t>(piece[i])];
    previous[i] = i == 0 ? kNoToken : i - 1;
    following[i] = i + 1 < count ? i + 1 : kNoToken;
  }
  const auto later = [](const Scratch::Candidate& a,
                        const Scratch::Candidate& b) {
    return a.rank != b.rank ? a.rank > b.rank : a.index > b.index;
  };
  const auto offer = [&](std::uint32_t index, std::uint32_t right) {
    const auto merge = merges_.find(PairKey(symbols[index], symbols[right]));
    if (merge != merges_.end()) {
      heap.push_back(
          {merge->second.rank, index, symbols[index], symbols[right]});
      std::ranges::push_heap(heap, later);
    }
  };
  for (std::uint32_t i = 0; i + 1 < count; ++i) {
    offer(i, i + 1);
  }
  while (!heap.empty()) {
    std::ranges::pop_heap(heap, later);
    const Scratch::Candidate best = heap.back();
    heap.pop_back();
    const std::uint32_t right = following[best.index];
    if (symbols[best.index] != best.left || right == kNoToken ||
        symbols[right] != best.right) {
      continue;
    }
    symbols[best.index] =
        merges_.find(PairKey(best.left, best.right))->second.token;
    symbols[right] = kNoToken;
    const std::uint32_t after = following[right];
    following[best.index] = after;
    if (after != kNoToken) {
      previous[after] = best.index;
      offer(best.index, after);
    }
    if (previous[best.index] != kNoToken) {
      offer(previous[best.index], best.index);
    }
  }
  for (std::uint32_t i = 0; i != kNoToken; i = following[i]) {
    out.push_back(symbols[i]);
  }
}

void Tokenizer::EncodePiece(std::string_view piece, Scratch& scratch,
                            std::vector<std::uint32_t>& out) const {
  if (piece.size() == 1) {
    out.push_back(byte_tokens_[static_cast<std::uint8_t>(piece[0])]);
    return;
  }
  const auto found = scratch.memo.find(piece);
  if (found != scratch.memo.end()) {
    const auto [begin, size] = found->second;
    out.insert(out.end(), scratch.memo_tokens.begin() + begin,
               scratch.memo_tokens.begin() + begin + size);
    return;
  }
  const std::size_t start = out.size();
  MergePiece(piece, scratch, out);
  const auto begin = static_cast<std::uint32_t>(scratch.memo_tokens.size());
  scratch.memo_tokens.insert(scratch.memo_tokens.end(), out.begin() + start,
                             out.end());
  scratch.memo.emplace(
      piece, std::pair(begin, static_cast<std::uint32_t>(out.size() - start)));
}

void Tokenizer::EncodeWordPieces(std::string_view piece, Scratch& scratch,
                                 std::vector<std::uint32_t>& out) const {
  // Matches of the word regex and the gaps between them are both pieces.
  std::size_t gap = 0;
  std::size_t offset = 0;
  while (offset < piece.size()) {
    const std::size_t end = WordMatchEnd(piece, offset);
    if (end == offset) {
      offset += At(piece, offset).length;
      continue;
    }
    if (gap != offset) {
      EncodePiece(piece.substr(gap, offset - gap), scratch, out);
    }
    EncodePiece(piece.substr(offset, end - offset), scratch, out);
    gap = offset = end;
  }
  if (gap != piece.size()) {
    EncodePiece(piece.substr(gap), scratch, out);
  }
}

void Tokenizer::EncodeText(std::string_view text, Scratch& scratch,
                           std::vector<std::uint32_t>& out) const {
  // `\p{N}{1,3}` splits first: each group of up to three numbers is a piece
  // the later splits cannot divide. Each run between them is split into
  // `[一-龥぀-ゟ゠-ヿ]+` runs and the gaps between, and every one of
  // those pieces is split by the word regex.
  std::size_t offset = 0;
  while (offset < text.size()) {
    CodePoint next = At(text, offset);
    if ((next.cls & kNumber) != 0) {
      std::size_t end = offset + next.length;
      for (int taken = 1; taken < 3 && end < text.size(); ++taken) {
        next = At(text, end);
        if ((next.cls & kNumber) == 0) {
          break;
        }
        end += next.length;
      }
      EncodePiece(text.substr(offset, end - offset), scratch, out);
      offset = end;
      continue;
    }
    const bool cjk = (next.cls & kCjk) != 0;
    std::size_t end = offset + next.length;
    while (end < text.size()) {
      next = At(text, end);
      if ((next.cls & kNumber) != 0 || ((next.cls & kCjk) != 0) != cjk) {
        break;
      }
      end += next.length;
    }
    EncodeWordPieces(text.substr(offset, end - offset), scratch, out);
    offset = end;
  }
}

std::vector<std::uint32_t> Tokenizer::Encode(std::string_view text,
                                             bool parse_special) const {
  std::vector<std::uint32_t> out;
  out.reserve(text.size() / 3 + 1);
  Scratch scratch;
  if (!parse_special) {
    EncodeText(text, scratch, out);
    return out;
  }
  std::size_t segment = 0;
  for (std::size_t offset = 0; offset < text.size();) {
    if (!added_first_byte_[static_cast<std::uint8_t>(text[offset])]) {
      ++offset;
      continue;
    }
    std::optional<std::uint32_t> match;
    std::size_t length = 0;
    for (const std::size_t candidate : added_lengths_) {
      if (candidate > text.size() - offset) {
        continue;
      }
      match = FindSpecial(text.substr(offset, candidate));
      if (match) {
        length = candidate;
        break;
      }
    }
    if (!match) {
      ++offset;
      continue;
    }
    EncodeText(text.substr(segment, offset - segment), scratch, out);
    out.push_back(*match);
    offset += length;
    segment = offset;
  }
  EncodeText(text.substr(segment), scratch, out);
  return out;
}

std::string Tokenizer::Decode(std::span<const std::uint32_t> tokens) const {
  std::string out;
  for (const std::uint32_t token : tokens) {
    out.append(TokenBytes(token));
  }
  return out;
}

std::string_view Tokenizer::TokenBytes(std::uint32_t token) const {
  if (token >= VocabSize()) {
    return {};
  }
  return std::string_view(bytes_).substr(offsets_[token],
                                         offsets_[token + 1] - offsets_[token]);
}

std::optional<std::uint32_t> Tokenizer::FindSpecial(
    std::string_view text) const {
  const auto found = added_.find(text);
  if (found == added_.end()) {
    return std::nullopt;
  }
  return found->second;
}

bool Tokenizer::IsSpecial(std::uint32_t token) const {
  return token < special_.size() && special_[token];
}

bool Tokenizer::IsEndOfGeneration(std::uint32_t token) const {
  return std::ranges::binary_search(end_of_generation_, token);
}

std::uint32_t Tokenizer::VocabSize() const {
  return static_cast<std::uint32_t>(special_.size());
}

}  // namespace gufo::models::flint
