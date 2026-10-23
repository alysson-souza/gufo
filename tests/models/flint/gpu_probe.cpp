// Dumps the GPU engine's per-position logits for a token sequence, for
// comparison with the CPU reference and llama.cpp (tools/flint/).
//
// flint_gpu_probe --model M.gguf --tokens ids.txt --dump logits.f32
//                 [--chunk N] [--mode prefill|decode]
//                 [--taps taps.bin --tap-position P]
//
// prefill: chunked prefill (default chunk 2048); every position's logits come
//          from the production single-row head.
// decode:  the first token is prefilled, then every later token is fed
//          through the decode step; row i is the frontier after token i.
// Both write [tokens][vocab] FP32, row i predicting token i + 1.
// --taps (prefill) writes position P's per-layer rows in the record format of
// tools/flint/compare_taps.py, to locate the first diverging layer.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/flint/kernels/rocm/device_model.hpp"
#include "src/models/flint/kernels/rocm/executor.hpp"
#include "src/models/flint/weights.hpp"

namespace flint = gufo::models::flint;

namespace {

std::vector<std::uint32_t> ReadTokens(const std::string& path) {
  std::ifstream in(path);
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  std::vector<std::uint32_t> tokens;
  std::uint64_t value = 0;
  bool digits = false;
  for (const char ch : text) {
    if (ch >= '0' && ch <= '9') {
      value = value * 10 + static_cast<std::uint64_t>(ch - '0');
      digits = true;
    } else if (digits) {
      tokens.push_back(static_cast<std::uint32_t>(value));
      value = 0;
      digits = false;
    }
  }
  if (digits)
    tokens.push_back(static_cast<std::uint32_t>(value));
  return tokens;
}

int Usage() {
  std::cerr << "usage: flint_gpu_probe --model M.gguf --tokens ids.txt "
               "--dump logits.f32 [--chunk N] [--mode prefill|decode] "
               "[--taps taps.bin --tap-position P]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, tokens_path, dump_path, taps_path, mode = "prefill";
  int chunk = 2048;
  std::size_t tap_position = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key = argv[i];
    const std::string value = argv[i + 1];
    if (key == "--model")
      model_path = value;
    else if (key == "--tokens")
      tokens_path = value;
    else if (key == "--dump")
      dump_path = value;
    else if (key == "--chunk")
      chunk = std::stoi(value);
    else if (key == "--mode")
      mode = value;
    else if (key == "--taps")
      taps_path = value;
    else if (key == "--tap-position")
      tap_position = std::stoul(value);
    else
      return Usage();
  }
  if (model_path.empty() || tokens_path.empty() || dump_path.empty() ||
      chunk < 1 || (mode != "prefill" && mode != "decode"))
    return Usage();
  const auto tokens = ReadTokens(tokens_path);
  if (tokens.empty()) {
    std::cerr << "no tokens in " << tokens_path << '\n';
    return 1;
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  const auto weights =
      reader ? flint::ModelWeights::Bind(*reader, &error) : std::nullopt;
  if (!weights) {
    std::cerr << "load: " << error << '\n';
    return 1;
  }
  const auto device =
      flint::rocm::DeviceModel::Upload(*weights, *reader, &error);
  const auto executor =
      device ? flint::rocm::Executor::Create(*device, chunk, 1, &error)
             : nullptr;
  const auto context = static_cast<std::uint32_t>(tokens.size());
  const auto cache = executor
                         ? flint::rocm::SessionCache::Create(
                               weights->config, context,
                               static_cast<std::uint32_t>(
                                   flint::rocm::SlidingRingCapacity(chunk)),
                               &error)
                         : nullptr;
  if (!cache) {
    std::cerr << "device: " << error << '\n';
    return 1;
  }
  const std::size_t vocab = static_cast<std::size_t>(device->output().rows);
  std::vector<float> logits(tokens.size() * vocab);
  std::vector<float> frontier(vocab);
  const auto start = std::chrono::steady_clock::now();
  const std::span<const std::uint32_t> all(tokens);
  // Taps: the rows of `tap_position`, in compare_taps.py's record format.
  std::ofstream taps;
  std::size_t chunk_start = 0;
  std::vector<float> row;
  std::vector<float> ffn_inp;
  if (!taps_path.empty()) {
    taps.open(taps_path, std::ios::binary);
    const auto write = [&](std::string_view name, std::int32_t layer,
                           std::span<const float> values) {
      const auto size = static_cast<std::uint32_t>(name.size());
      const auto position = static_cast<std::uint32_t>(tap_position);
      const auto count = static_cast<std::uint32_t>(values.size());
      taps.write(reinterpret_cast<const char*>(&size), 4);
      taps.write(name.data(), size);
      taps.write(reinterpret_cast<const char*>(&layer), 4);
      taps.write(reinterpret_cast<const char*>(&position), 4);
      taps.write(reinterpret_cast<const char*>(&count), 4);
      taps.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size_bytes()));
    };
    executor->SetTap([&, write](std::string_view name, std::uint32_t layer,
                                const float* rows, int width) {
      if (tap_position < chunk_start ||
          tap_position >= chunk_start + static_cast<std::size_t>(chunk))
        return;
      row.resize(static_cast<std::size_t>(width));
      (void)hipMemcpy(row.data(), rows + (tap_position - chunk_start) * width,
                      row.size() * sizeof(float), hipMemcpyDeviceToHost);
      const auto il = static_cast<std::int32_t>(layer);
      write(name, il, row);
      if (name == "ffn_inp") {
        ffn_inp = row;
      } else if (name == "ffn_out") {
        for (std::size_t i = 0; i < row.size(); ++i)
          row[i] += ffn_inp[i];
        write("l_out", il, row);
      }
    });
  }
  if (mode == "prefill") {
    for (std::size_t at = 0; at < tokens.size();) {
      const std::size_t n = std::min<std::size_t>(
          static_cast<std::size_t>(chunk), tokens.size() - at);
      if (executor->Prefill(*cache, all.subspan(at, n),
                            static_cast<std::uint32_t>(at), frontier.data(), {},
                            &error, logits.data() + at * vocab) !=
          flint::rocm::Executor::Status::kOk) {
        std::cerr << "prefill: " << error << '\n';
        return 1;
      }
      at += n;
      chunk_start = at;
    }
  } else {
    if (executor->Prefill(*cache, all.first(1), 0, logits.data(), {}, &error) !=
        flint::rocm::Executor::Status::kOk) {
      std::cerr << "prefill: " << error << '\n';
      return 1;
    }
    for (std::size_t i = 1; i < tokens.size(); ++i) {
      flint::rocm::SessionCache* caches[] = {cache.get()};
      const std::uint32_t token[] = {tokens[i]};
      const std::uint32_t position[] = {static_cast<std::uint32_t>(i)};
      float* rows[] = {logits.data() + i * vocab};
      if (!executor->Decode(caches, token, position, rows, &error)) {
        std::cerr << "decode: " << error << '\n';
        return 1;
      }
    }
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::ofstream out(dump_path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(logits.data()),
            static_cast<std::streamsize>(logits.size() * sizeof(float)));
  if (!out) {
    std::cerr << "cannot write " << dump_path << '\n';
    return 1;
  }
  std::cerr << mode << " tokens=" << tokens.size() << " chunk=" << chunk
            << " seconds=" << seconds << '\n';
  return 0;
}
