#include "src/models/flint/config.hpp"

#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace flint = gufo::models::flint;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct BoolArray {
  std::vector<bool> values;
};
using Value = std::variant<std::uint32_t, float, bool, std::string, BoolArray>;
using Metadata = std::map<std::string, Value>;

constexpr std::uint32_t kHidden = 8;
constexpr std::uint32_t kVocab = 16;
constexpr std::uint32_t kLayers = 6;

// Metadata plus one F32 token_embd tensor: no model or GPU is needed.
std::optional<flint::Config> Parse(const Metadata& fields,
                                   std::string* error = nullptr) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{1});
  pod(std::uint64_t{fields.size()});
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& item) {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t{4});
            pod(item);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t{6});
            pod(item);
          } else if constexpr (std::is_same_v<T, bool>) {
            pod(std::uint32_t{7});
            pod(static_cast<std::uint8_t>(item));
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else {
            pod(std::uint32_t{9});
            pod(std::uint32_t{7});
            pod(std::uint64_t{item.values.size()});
            for (const bool entry : item.values)
              pod(static_cast<std::uint8_t>(entry));
          }
        },
        value);
  }
  text("token_embd.weight");
  pod(std::uint32_t{2});
  pod(std::uint64_t{kHidden});
  pod(std::uint64_t{kVocab});
  pod(std::uint32_t{0});  // F32
  pod(std::uint64_t{0});
  bytes.resize((bytes.size() + 31) / 32 * 32);
  bytes.resize(bytes.size() + kHidden * kVocab * sizeof(float));

  std::string open_error;
  const auto reader = gufo::core::GgufReader::OpenMemory(
      bytes.data(), bytes.size(), &open_error);
  Require(reader != nullptr, "synthetic GGUF did not open");
  std::string parse_error;
  auto result = flint::Config::FromGguf(*reader, &parse_error);
  Require(result.has_value() == parse_error.empty(),
          "error text must accompany every rejection");
  if (error != nullptr)
    *error = parse_error;
  return result;
}

Metadata ValidMetadata() {
  Metadata fields{{"general.architecture", std::string{"flint"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", kLayers},
           {"context_length", 262144},
           {"embedding_length", kHidden},
           {"feed_forward_length", 24},
           {"attention.head_count", 4},
           {"attention.head_count_kv", 2},
           {"attention.key_length", 4},
           {"attention.value_length", 4},
           {"attention.sliding_window", 5},
           {"expert_count", 8},
           {"expert_used_count", 2},
           {"expert_gating_func", 2},
           {"expert_feed_forward_length", 4},
           {"expert_shared_feed_forward_length", 6},
           {"expert_shared_count", 2},
           {"leading_dense_block_count", 2}}) {
    fields["flint." + std::string(key)] = value;
  }
  fields["flint.rope.freq_base"] = 10000.0F;
  fields["flint.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["flint.expert_weights_norm"] = true;
  fields["flint.expert_weights_scale"] = 1.0F;
  fields["flint.attention.sliding_window_pattern"] =
      BoolArray{{false, false, true, true, true, false}};
  fields["flint.attention.rope_pattern"] =
      BoolArray{{false, false, true, true, true, false}};
  return fields;
}

void ParsesLayerPatterns() {
  std::string error;
  const auto config = Parse(ValidMetadata(), &error);
  Require(config.has_value(), error.c_str());
  Require(config->vocab_size == kVocab, "vocabulary not read from embedding");
  Require(config->num_layers == kLayers, "layer count");
  Require(
      !config->IsSliding(0) && config->IsSliding(2) && !config->IsSliding(5),
      "sliding-window pattern not applied per layer");
  Require(!config->UsesRope(1) && config->UsesRope(4) && !config->UsesRope(5),
          "rope pattern not applied per layer");
  Require(config->IsDense(1) && !config->IsDense(2),
          "leading dense layers not respected");
  Require(config->sliding_window == 5 && config->QueryWidth() == 16 &&
              config->KvWidth() == 8,
          "attention geometry");
}

void RejectsUnsupportedSemantics() {
  const auto rejects = [](const char* key, Value value, const char* message) {
    auto fields = ValidMetadata();
    fields[key] = std::move(value);
    Require(!Parse(fields).has_value(), message);
  };
  rejects("general.architecture", std::string{"qwen3moe"},
          "another architecture accepted");
  rejects("flint.expert_gating_func", std::uint32_t{1},
          "softmax routing accepted");
  rejects("flint.expert_weights_norm", false,
          "unnormalized routing weights accepted");
  rejects("flint.attention.value_length", std::uint32_t{8},
          "value length differing from key length accepted");
  rejects("flint.attention.head_count_kv", std::uint32_t{3},
          "query heads not divisible by KV heads accepted");
  rejects("flint.attention.sliding_window_pattern",
          BoolArray{{false, true, true}},
          "sliding-window pattern shorter than the layer count accepted");
  rejects("flint.leading_dense_block_count", std::uint32_t{kLayers + 1},
          "more dense layers than blocks accepted");

  auto fields = ValidMetadata();
  fields.erase("flint.expert_weights_scale");
  const auto config = Parse(fields);
  Require(config.has_value() && config->expert_weights_scale == 1.0F,
          "absent expert weight scale must default to one");
  fields.erase("flint.attention.rope_pattern");
  Require(!Parse(fields).has_value(), "missing rope pattern accepted");
}

}  // namespace

int main() {
  try {
    ParsesLayerPatterns();
    RejectsUnsupportedSemantics();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "flint config: OK\n";
  return 0;
}
