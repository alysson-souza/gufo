#ifndef GUFO_MODELS_FLINT_TOKENIZER_HPP_
#define GUFO_MODELS_FLINT_TOKENIZER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::flint {

/// Flint's byte-level BPE tokenizer, loaded from GGUF metadata
/// (`tokenizer.ggml.model=gpt2`, `tokenizer.ggml.pre=flint`). Encoding matches
/// HF `tokenizers` on the model's tokenizer.json: no normalizer; a sequence
/// of three isolating splits (number runs in groups of three, kana/CJK runs,
/// then the GPT-4-style word regex within each piece); byte-level mapping;
/// BPE by merge rank.
///
/// Text need not be valid UTF-8, which HF cannot take. Each byte that does not
/// start a well-formed sequence (lone continuations, truncated or overlong
/// sequences, surrogates, values above U+10FFFF, 0xF5..0xFF) is one code point
/// that splits like U+FFFD, a symbol, and keeps its raw byte through the
/// byte-level mapping, so `Decode(Encode(text)) == text` for any bytes.
class Tokenizer {
public:
  /// Loads the vocabulary, merges, token types and end-of-generation ids.
  /// Rejects other tokenizer models or pre-tokenizers, and vocabularies HF
  /// would refuse (a merge naming a missing token).
  static std::optional<Tokenizer> FromGguf(const core::GgufReader& reader,
                                           std::string* error_msg = nullptr);

  /// Encodes text. With `parse_special`, every added token (control or
  /// user-defined) spelled in the text becomes its id, leftmost and longest
  /// first, and the text between is encoded separately; HF's
  /// `encode(text, add_special_tokens=False)`. Without it the whole text is
  /// ordinary text.
  std::vector<std::uint32_t> Encode(std::string_view text,
                                    bool parse_special) const;

  /// Concatenates the raw bytes of each token; out-of-range ids add nothing.
  std::string Decode(std::span<const std::uint32_t> tokens) const;

  /// Raw bytes of one token; an added token's text. Empty when out of range.
  std::string_view TokenBytes(std::uint32_t token) const;

  /// Id of the added token (control or user-defined) spelled exactly `text`.
  std::optional<std::uint32_t> FindSpecial(std::string_view text) const;

  /// Whether the token is an added token (GGUF control or user-defined type).
  bool IsSpecial(std::uint32_t token) const;

  /// Whether the token is in `tokenizer.ggml.eog_token_ids`.
  bool IsEndOfGeneration(std::uint32_t token) const;

  std::uint32_t VocabSize() const;

private:
  struct StringHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view text) const noexcept {
      return std::hash<std::string_view>{}(text);
    }
  };

  /// A ranked merge of a token pair and the token it produces.
  struct Merge {
    std::uint32_t rank;
    std::uint32_t token;
  };

  struct Scratch;

  Tokenizer() = default;

  void EncodeText(std::string_view text, Scratch& scratch,
                  std::vector<std::uint32_t>& out) const;
  void EncodeWordPieces(std::string_view piece, Scratch& scratch,
                        std::vector<std::uint32_t>& out) const;
  void EncodePiece(std::string_view piece, Scratch& scratch,
                   std::vector<std::uint32_t>& out) const;
  void MergePiece(std::string_view piece, Scratch& scratch,
                  std::vector<std::uint32_t>& out) const;

  /// Decoded bytes of token `i` are `bytes_[offsets_[i], offsets_[i + 1])`.
  std::string bytes_;
  std::vector<std::uint32_t> offsets_;
  /// Keyed by `(left << 32) | right`.
  std::unordered_map<std::uint64_t, Merge> merges_;
  std::array<std::uint32_t, 256> byte_tokens_{};
  std::unordered_map<std::string, std::uint32_t, StringHash, std::equal_to<>>
      added_;
  /// Distinct added-token lengths, longest first, and the bytes they start
  /// with, for the leftmost-longest scan.
  std::vector<std::size_t> added_lengths_;
  std::array<bool, 256> added_first_byte_{};
  std::vector<bool> special_;
  std::vector<std::uint32_t> end_of_generation_;
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_TOKENIZER_HPP_
