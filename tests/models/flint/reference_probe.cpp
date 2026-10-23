// Runs the Flint FP32 reference over token ids and dumps next-token logits,
// for comparison with an independent runtime (tools/flint/).
//
//   flint_reference_probe --model M.gguf (--tokens ids.txt | --prompt-ids 1,2)
//                         --dump logits.f32 [--positions all|last]
//                         [--chunk N] [--taps taps.bin [--tap-position P]]
//
// Token files hold decimal ids separated by any non-digit characters, so
// `llama-tokenize --ids` output works unchanged. --dump writes float32 rows of
// vocab_size logits: one per position (all) or only the final one (last).
// --chunk sets how many tokens share one weight decode (default 512); results
// do not depend on it. --taps writes the intermediate rows the reference
// observes (only position P with --tap-position) as records of u32 name
// length, name bytes, i32 layer, u32 position, u32 count, then `count`
// float32 values; tools/flint/compare_taps.py reads them.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/flint/reference.hpp"
#include "src/models/flint/weights.hpp"

namespace flint = gufo::models::flint;

namespace {

std::vector<std::uint32_t> ParseIds(const std::string& text) {
  std::vector<std::uint32_t> ids;
  std::uint64_t value = 0;
  bool in_number = false;
  for (const char ch : text) {
    if (ch >= '0' && ch <= '9') {
      value = value * 10 + static_cast<std::uint64_t>(ch - '0');
      in_number = true;
    } else if (in_number) {
      ids.push_back(static_cast<std::uint32_t>(value));
      value = 0;
      in_number = false;
    }
  }
  if (in_number) {
    ids.push_back(static_cast<std::uint32_t>(value));
  }
  return ids;
}

int Usage() {
  std::cerr << "usage: flint_reference_probe --model M.gguf "
               "(--tokens ids.txt | --prompt-ids 1,2,3) --dump logits.f32 "
               "[--positions all|last] [--chunk N] "
               "[--taps taps.bin [--tap-position P]]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string ids_text;
  std::string dump_path;
  std::string taps_path;
  bool all_positions = true;
  std::size_t chunk = 512;
  long tap_position = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (i + 1 >= argc) {
      return Usage();
    }
    const std::string value = argv[++i];
    if (arg == "--model") {
      model_path = value;
    } else if (arg == "--tokens") {
      std::ifstream in(value);
      if (!in) {
        std::cerr << "cannot read " << value << "\n";
        return 1;
      }
      ids_text.assign(std::istreambuf_iterator<char>(in), {});
    } else if (arg == "--prompt-ids") {
      ids_text = value;
    } else if (arg == "--dump") {
      dump_path = value;
    } else if (arg == "--taps") {
      taps_path = value;
    } else if (arg == "--tap-position") {
      tap_position = std::stol(value);
    } else if (arg == "--chunk" && std::stoul(value) > 0) {
      chunk = std::stoul(value);
    } else if (arg == "--positions" && (value == "all" || value == "last")) {
      all_positions = value == "all";
    } else {
      return Usage();
    }
  }
  const std::vector<std::uint32_t> ids = ParseIds(ids_text);
  if (model_path.empty() || dump_path.empty() || ids.empty()) {
    return Usage();
  }

  std::string error;
  const auto gguf = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!gguf) {
    std::cerr << "cannot open model: " << error << "\n";
    return 1;
  }
  const auto weights = flint::ModelWeights::Bind(*gguf, &error);
  if (!weights) {
    std::cerr << error << "\n";
    return 1;
  }
  flint::ReferenceModel model(*weights, static_cast<std::uint32_t>(ids.size()));

  std::ofstream taps;
  if (!taps_path.empty()) {
    taps.open(taps_path, std::ios::binary);
    model.SetTap([&](std::string_view name, std::int32_t layer,
                     std::uint32_t position, std::span<const float> row) {
      if (tap_position >= 0 &&
          position != static_cast<std::uint32_t>(tap_position)) {
        return;
      }
      const auto name_size = static_cast<std::uint32_t>(name.size());
      const auto count = static_cast<std::uint32_t>(row.size());
      taps.write(reinterpret_cast<const char*>(&name_size), 4);
      taps.write(name.data(), name_size);
      taps.write(reinterpret_cast<const char*>(&layer), 4);
      taps.write(reinterpret_cast<const char*>(&position), 4);
      taps.write(reinterpret_cast<const char*>(&count), 4);
      taps.write(reinterpret_cast<const char*>(row.data()),
                 static_cast<std::streamsize>(row.size_bytes()));
    });
  }

  std::ofstream dump(dump_path, std::ios::binary);
  const std::size_t vocab = weights->config.vocab_size;
  std::vector<float> logits;
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t p = 0; p < ids.size(); p += chunk) {
    const std::size_t count = std::min(chunk, ids.size() - p);
    logits.resize(count * vocab);
    if (!model.Forward(std::span(ids).subspan(p, count), logits, &error)) {
      std::cerr << "position " << p << ": " << error << "\n";
      return 1;
    }
    const std::size_t first =
        all_positions ? 0 : (p + count == ids.size() ? count - 1 : count);
    dump.write(
        reinterpret_cast<const char*>(logits.data() + first * vocab),
        static_cast<std::streamsize>((count - first) * vocab * sizeof(float)));
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::fprintf(stderr, "%zu/%zu tokens, %.3f s/token\n", p + count,
                 ids.size(), seconds / static_cast<double>(p + count));
  }
  if (!dump || (taps.is_open() && !taps)) {
    std::cerr << "write failed\n";
    return 1;
  }
  return 0;
}
