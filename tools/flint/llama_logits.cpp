// Independent oracle for the Flint reference: evaluates token ids with
// llama.cpp and dumps float32 logits for every position, optionally with the
// graph's intermediate rows for first-divergence debugging.
//
// Qualified against llama.cpp 970b51ce7c9b (Flint support a37cc461), HIP
// build: BUILD_SHARED_LIBS=ON GGML_HIP=ON AMDGPU_TARGETS=gfx1151
// GGML_HIP_NO_VMM=ON GGML_HIP_GRAPHS=ON GGML_NATIVE=ON, Release. In that
// checkout (one command):
//
//   g++ -std=c++17 -O2 -Iinclude -Iggml/include
//     GUFO/tools/flint/llama_logits.cpp -Lbuild/bin -lllama -lggml
//     -lggml-base -Wl,-rpath,$PWD/build/bin -o llama_logits
//
//   llama_logits --model M.gguf --tokens ids.txt --dump logits.f32
//                [--ngl 999] [--batch 512] [--ubatch 512]
//                [--taps taps.bin --tap-position P]
//
// Defaults keep llama.cpp's context defaults (F16 KV, flash attention auto).
// --ngl 0 runs entirely on the CPU backend (no operation offload), a second
// llama.cpp numerical path for separating implementation noise from bugs.
// Compare with tools/flint/compare_logits.py and compare_taps.py.
//
// Taps use the record layout of flint_reference_probe: u32 name length, name,
// i32 layer, u32 position, u32 count, `count` float32 values. llama.cpp
// tensors named `<name>-<layer>` (or `<name>` for layer -1) are recorded;
// each name may be recorded several times per layer (the graph reuses names
// for successive views), the last record being the final value.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

namespace {

std::vector<llama_token> ParseIds(const std::string& text) {
  std::vector<llama_token> ids;
  long value = 0;
  bool in_number = false;
  for (const char ch : text) {
    if (ch >= '0' && ch <= '9') {
      value = value * 10 + (ch - '0');
      in_number = true;
    } else if (in_number) {
      ids.push_back(static_cast<llama_token>(value));
      value = 0;
      in_number = false;
    }
  }
  if (in_number) {
    ids.push_back(static_cast<llama_token>(value));
  }
  return ids;
}

struct TapState {
  std::ofstream out;
  long position = -1;
  std::uint32_t batch_start = 0;  // position of the current ubatch's first row
  std::vector<float> buffer;
};

bool TapCallback(ggml_tensor* t, bool ask, void* user) {
  auto* state = static_cast<TapState*>(user);
  if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
    return !ask;
  }
  if (ask) {
    return true;
  }
  // Rows are the outermost non-unit dimension: [ne0, tokens] or
  // [ne0, ne1, tokens].
  std::int64_t tokens = t->ne[1];
  std::int64_t width = t->ne[0];
  if (t->ne[2] > 1) {
    tokens = t->ne[2];
    width = t->ne[0] * t->ne[1];
  }
  if (t->ne[3] > 1) {
    return true;
  }
  const std::string full = t->name;
  std::string name = full;
  std::int32_t layer = -1;
  const auto dash = full.rfind('-');
  if (dash != std::string::npos && dash + 1 < full.size() &&
      std::all_of(full.begin() + dash + 1, full.end(),
                  [](char c) { return c >= '0' && c <= '9'; })) {
    name = full.substr(0, dash);
    layer = std::stoi(full.substr(dash + 1));
  }
  state->buffer.resize(static_cast<std::size_t>(ggml_nelements(t)));
  ggml_backend_tensor_get(t, state->buffer.data(), 0, ggml_nbytes(t));
  for (std::int64_t r = 0; r < tokens; ++r) {
    const auto position = state->batch_start + static_cast<std::uint32_t>(r);
    if (state->position >= 0 &&
        position != static_cast<std::uint32_t>(state->position)) {
      continue;
    }
    const auto size = static_cast<std::uint32_t>(name.size());
    const auto count = static_cast<std::uint32_t>(width);
    state->out.write(reinterpret_cast<const char*>(&size), 4);
    state->out.write(name.data(), size);
    state->out.write(reinterpret_cast<const char*>(&layer), 4);
    state->out.write(reinterpret_cast<const char*>(&position), 4);
    state->out.write(reinterpret_cast<const char*>(&count), 4);
    state->out.write(
        reinterpret_cast<const char*>(state->buffer.data() + r * width),
        static_cast<std::streamsize>(width * sizeof(float)));
  }
  return true;
}

int Usage() {
  std::cerr << "usage: llama_logits --model M.gguf --tokens ids.txt "
               "--dump logits.f32 [--ngl N] [--batch N] [--ubatch N] "
               "[--taps taps.bin --tap-position P]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, tokens_path, dump_path, taps_path;
  int ngl = 999;
  std::uint32_t batch = 512;
  std::uint32_t ubatch = 512;
  TapState taps;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string arg = argv[i];
    const std::string value = argv[i + 1];
    if (arg == "--model") {
      model_path = value;
    } else if (arg == "--tokens") {
      tokens_path = value;
    } else if (arg == "--dump") {
      dump_path = value;
    } else if (arg == "--ngl") {
      ngl = std::stoi(value);
    } else if (arg == "--batch") {
      batch = static_cast<std::uint32_t>(std::stoul(value));
    } else if (arg == "--ubatch") {
      ubatch = static_cast<std::uint32_t>(std::stoul(value));
    } else if (arg == "--taps") {
      taps_path = value;
    } else if (arg == "--tap-position") {
      taps.position = std::stol(value);
    } else {
      return Usage();
    }
  }
  if (argc % 2 == 0 || model_path.empty() || tokens_path.empty() ||
      dump_path.empty()) {
    return Usage();
  }
  std::ifstream in(tokens_path);
  const std::vector<llama_token> ids =
      ParseIds(std::string(std::istreambuf_iterator<char>(in), {}));
  if (ids.empty()) {
    std::cerr << "no token ids in " << tokens_path << "\n";
    return 1;
  }
  if (!taps_path.empty()) {
    // One ubatch per decode keeps tap rows aligned with positions.
    ubatch = batch;
    taps.out.open(taps_path, std::ios::binary);
  }

  llama_backend_init();
  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = ngl;
  llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
  if (model == nullptr) {
    std::cerr << "cannot load " << model_path << "\n";
    return 1;
  }
  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = static_cast<std::uint32_t>((ids.size() + 255) / 256 * 256);
  cparams.n_batch = batch;
  cparams.n_ubatch = ubatch;
  cparams.n_seq_max = 1;
  // With --ngl 0 the run stays on the CPU backend: llama.cpp would otherwise
  // offload large-batch host operations to the GPU.
  cparams.op_offload = ngl > 0;
  if (!taps_path.empty()) {
    cparams.cb_eval = TapCallback;
    cparams.cb_eval_user_data = &taps;
  }
  llama_context* ctx = llama_init_from_model(model, cparams);
  if (ctx == nullptr) {
    std::cerr << "cannot create context\n";
    return 1;
  }
  const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
  std::fprintf(stderr, "n_ctx %u batch %u ubatch %u flash_attn %s\n",
               llama_n_ctx(ctx), batch, ubatch,
               llama_flash_attn_type_name(cparams.flash_attn_type));

  std::ofstream dump(dump_path, std::ios::binary);
  llama_batch b = llama_batch_init(static_cast<std::int32_t>(batch), 0, 1);
  for (std::size_t start = 0; start < ids.size(); start += batch) {
    const std::size_t count = std::min<std::size_t>(batch, ids.size() - start);
    b.n_tokens = static_cast<std::int32_t>(count);
    for (std::size_t i = 0; i < count; ++i) {
      b.token[i] = ids[start + i];
      b.pos[i] = static_cast<llama_pos>(start + i);
      b.n_seq_id[i] = 1;
      b.seq_id[i][0] = 0;
      b.logits[i] = 1;
    }
    taps.batch_start = static_cast<std::uint32_t>(start);
    if (llama_decode(ctx, b) != 0) {
      std::cerr << "llama_decode failed at " << start << "\n";
      return 1;
    }
    for (std::size_t i = 0; i < count; ++i) {
      const float* row =
          llama_get_logits_ith(ctx, static_cast<std::int32_t>(i));
      dump.write(reinterpret_cast<const char*>(row),
                 static_cast<std::streamsize>(vocab * sizeof(float)));
    }
  }
  llama_batch_free(b);
  llama_free(ctx);
  llama_model_free(model);
  if (!dump) {
    std::cerr << "write failed\n";
    return 1;
  }
  return 0;
}
