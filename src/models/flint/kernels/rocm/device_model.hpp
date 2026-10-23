#ifndef GUFO_MODELS_FLINT_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_FLINT_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/flint/kernels/rocm/kernels.hpp"
#include "src/models/flint/weights.hpp"

namespace gufo::models::flint::rocm {

/// One decoder block resident on the device, weights in their GGUF
/// encodings.
struct DeviceLayer {
  bool sliding = false;
  bool rope = false;
  bool dense = false;
  const float* attn_norm = nullptr;
  const float* q_norm = nullptr;
  const float* k_norm = nullptr;
  const float* sinks = nullptr;
  const float* ffn_norm = nullptr;
  /// q, k and v rows stacked: [hidden -> 4096 + 512 + 512]. Every Flint
  /// layer stores the three in one type, and each output row is the same
  /// dot product as in the separate matrices.
  QuantMatrix qkv;
  QuantMatrix o;
  QuantMatrix gate;  ///< dense layers
  QuantMatrix up;
  QuantMatrix down;
  const std::uint16_t* router = nullptr;  ///< BF16 [experts][hidden]
  const float* router_bias = nullptr;     ///< optional
  RoutedExperts experts;
  QuantMatrix shared_gate;
  QuantMatrix shared_up;
  QuantMatrix shared_down;
};

/// Flint's weights uploaded to device memory.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Rejects tensor encodings the HIP kernels lack, then streams every
  /// tensor from the reader's mapping into one placement decided up front.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] const void* token_embd() const noexcept { return token_embd_; }
  [[nodiscard]] const float* output_norm() const noexcept {
    return output_norm_;
  }
  [[nodiscard]] const QuantMatrix& output() const noexcept { return output_; }
  [[nodiscard]] const std::vector<DeviceLayer>& layers() const noexcept {
    return layers_;
  }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }

private:
  DeviceModel() = default;

  Config config_;
  const void* token_embd_ = nullptr;
  const float* output_norm_ = nullptr;
  QuantMatrix output_;
  std::vector<DeviceLayer> layers_;
  std::vector<void*> allocations_;
  std::size_t bytes_ = 0;
};

}  // namespace gufo::models::flint::rocm

#endif  // GUFO_MODELS_FLINT_KERNELS_ROCM_DEVICE_MODEL_HPP_
