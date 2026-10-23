#ifndef GUFO_TESTS_MODELS_FLINT_OPS_TEST_SUPPORT_HPP_
#define GUFO_TESTS_MODELS_FLINT_OPS_TEST_SUPPORT_HPP_

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/flint/kernels/rocm/kernels.hpp"

namespace gufo::models::flint::test {

inline void CheckHip(hipError_t error, const std::string& what) {
  if (error != hipSuccess) {
    throw std::runtime_error(what + ": " + hipGetErrorString(error));
  }
}

/// Owning device allocation.
template<typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { Resize(count); }
  explicit DeviceBuffer(std::span<const T> host) {
    Resize(host.size());
    Upload(host);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      (void)hipFree(ptr_);
    }
  }
  void Resize(std::size_t count) {
    if (ptr_ != nullptr) {
      CheckHip(hipFree(ptr_), "hipFree");
      ptr_ = nullptr;
    }
    count_ = count;
    if (count > 0) {
      CheckHip(hipMalloc(&ptr_, count * sizeof(T)), "hipMalloc");
    }
  }
  void Upload(std::span<const T> host) {
    if (host.size() > count_) {
      throw std::runtime_error("upload larger than buffer");
    }
    CheckHip(hipMemcpy(ptr_, host.data(), host.size() * sizeof(T),
                       hipMemcpyHostToDevice),
             "upload");
  }
  std::vector<T> Download(std::size_t count) const {
    std::vector<T> host(count);
    CheckHip(
        hipMemcpy(host.data(), ptr_, count * sizeof(T), hipMemcpyDeviceToHost),
        "download");
    return host;
  }
  std::vector<T> Download() const { return Download(count_); }
  void Fill(unsigned char byte) {
    CheckHip(hipMemset(ptr_, byte, count_ * sizeof(T)), "hipMemset");
  }
  T* get() const { return ptr_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

/// Counts failures and prints the first few of each check.
class Checker {
public:
  bool Expect(bool ok, const std::string& what) {
    if (!ok) {
      ++failures_;
      if (failures_ <= 200) {
        std::cerr << "FAIL: " << what << '\n';
      }
    }
    return ok;
  }
  int failures() const { return failures_; }
  int Finish(const char* name) const {
    if (failures_ == 0) {
      std::cout << "PASS: " << name << '\n';
      return 0;
    }
    std::cerr << name << ": " << failures_ << " failure(s)\n";
    return 1;
  }

private:
  int failures_ = 0;
};

inline std::uint16_t FloatToHalfBits(float value) {
  const _Float16 h = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

inline float HalfBitsToFloat(std::uint16_t bits) {
  return quant::Fp16ToFloat(bits);
}

inline core::GgmlType ToGgml(rocm::WeightType type) {
  return static_cast<core::GgmlType>(static_cast<int>(type));
}

inline const char* Name(rocm::WeightType type) {
  switch (type) {
    case rocm::WeightType::kQ8_0:
      return "Q8_0";
    case rocm::WeightType::kQ4_K:
      return "Q4_K";
    case rocm::WeightType::kQ5_K:
      return "Q5_K";
    case rocm::WeightType::kQ6_K:
      return "Q6_K";
    case rocm::WeightType::kIQ3_S:
      return "IQ3_S";
    case rocm::WeightType::kIQ4_XS:
      return "IQ4_XS";
  }
  return "?";
}

inline constexpr rocm::WeightType kAllTypes[] = {
    rocm::WeightType::kQ8_0,  rocm::WeightType::kQ4_K,
    rocm::WeightType::kQ5_K,  rocm::WeightType::kQ6_K,
    rocm::WeightType::kIQ3_S, rocm::WeightType::kIQ4_XS};

/// Random but well-formed GGUF blocks: every byte random, then each FP16
/// scale field replaced by a positive normal value giving weights of
/// realistic magnitude (RMS ~0.02-0.1): [2^-12, 2^-11) for Q8_0 and
/// [2^-14, 2^-13) for the types with sub-block scales.
inline std::vector<std::uint8_t> RandomWeights(rocm::WeightType type,
                                               std::size_t rows, int cols,
                                               std::mt19937& rng) {
  const std::size_t row_bytes = rocm::WeightRowBytes(type, cols);
  std::vector<std::uint8_t> data(rows * row_bytes);
  for (auto& b : data) {
    b = static_cast<std::uint8_t>(rng());
  }
  std::size_t block = 0;
  std::vector<std::size_t> halves;
  switch (type) {
    case rocm::WeightType::kQ8_0:
      block = 34;
      halves = {0};
      break;
    case rocm::WeightType::kQ4_K:
      block = 144;
      halves = {0, 2};
      break;
    case rocm::WeightType::kQ5_K:
      block = 176;
      halves = {0, 2};
      break;
    case rocm::WeightType::kQ6_K:
      block = 210;
      halves = {208};
      break;
    case rocm::WeightType::kIQ3_S:
      block = 110;
      halves = {0};
      break;
    case rocm::WeightType::kIQ4_XS:
      block = 136;
      halves = {0};
      break;
  }
  std::uniform_real_distribution<float> mantissa(1.0f, 2.0f);
  const float base =
      std::ldexp(1.0f, type == rocm::WeightType::kQ8_0 ? -12 : -14);
  for (std::size_t off = 0; off < data.size(); off += block) {
    for (std::size_t h : halves) {
      const std::uint16_t bits = FloatToHalfBits(mantissa(rng) * base);
      std::memcpy(&data[off + h], &bits, sizeof(bits));
    }
  }
  return data;
}

/// Dequantizes `rows` rows with the CPU reference decoder.
inline std::vector<float> Dequantize(rocm::WeightType type,
                                     const std::uint8_t* data, std::size_t rows,
                                     int cols) {
  const std::size_t row_bytes = rocm::WeightRowBytes(type, cols);
  std::vector<float> out(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const void* src = data + r * row_bytes;
    float* dst = out.data() + r * cols;
    switch (type) {
      case rocm::WeightType::kQ8_0:
        quant::DequantizeQ8_0(src, dst, cols);
        break;
      case rocm::WeightType::kQ4_K:
        quant::DequantizeQ4_K(src, dst, cols);
        break;
      case rocm::WeightType::kQ5_K:
        quant::DequantizeQ5_K(src, dst, cols);
        break;
      case rocm::WeightType::kQ6_K:
        quant::DequantizeQ6_K(src, dst, cols);
        break;
      case rocm::WeightType::kIQ3_S:
        quant::DequantizeIQ3_S(src, dst, cols);
        break;
      case rocm::WeightType::kIQ4_XS:
        quant::DequantizeIQ4_XS(src, dst, cols);
        break;
    }
  }
  return out;
}

/// Activations that Q8_1 represents exactly: each 32-value block is
/// 2^e * q with e in [-9, -3], integer q in [-127, 127], one |q| == 127 and
/// |sum q| <= 2000, so the block scale, codes and FP16 sums are all exact
/// and the device result differs from the exact product only by FP32
/// accumulation.
inline std::vector<float> ExactActivations(std::size_t rows, int cols,
                                           std::mt19937& rng) {
  std::vector<float> x(rows * cols);
  std::uniform_int_distribution<int> code(-127, 127);
  std::uniform_int_distribution<int> exponent(-9, -3);
  std::uniform_int_distribution<int> lane(0, 31);
  for (std::size_t b = 0; b < x.size() / 32; ++b) {
    int q[32];
    int sum = 0;
    do {
      sum = 0;
      for (int& v : q) {
        v = code(rng);
      }
      q[lane(rng)] = (rng() & 1) != 0 ? 127 : -127;
      for (int v : q) {
        sum += v;
      }
    } while (std::abs(sum) > 2000);
    const float scale = std::ldexp(1.0f, exponent(rng));
    for (int i = 0; i < 32; ++i) {
      x[b * 32 + i] = scale * static_cast<float>(q[i]);
    }
  }
  return x;
}

inline std::vector<float> RandomFloats(std::size_t count, float lo, float hi,
                                       std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> out(count);
  for (float& v : out) {
    v = dist(rng);
  }
  return out;
}

/// Double-precision w . x and sum |w x| for one row pair.
struct Dot {
  double value = 0.0;
  double magnitude = 0.0;
};

inline Dot DotRow(const float* w, const float* x, int n) {
  Dot d;
  for (int i = 0; i < n; ++i) {
    const double p = static_cast<double>(w[i]) * x[i];
    d.value += p;
    d.magnitude += std::fabs(p);
  }
  return d;
}

/// Projection tolerance (see projection_ops_test.cpp): FP32 accumulation on
/// Q8_1-exact activations.
inline constexpr double kProjRelTol = 1e-4;
inline constexpr double kProjAbsTol = 1e-6;

inline bool HasMins(rocm::WeightType t) {
  return t == rocm::WeightType::kQ4_K || t == rocm::WeightType::kQ5_K;
}

/// Random quantized weights of `groups` stacked [m][k] matrices (experts),
/// uploaded once and decoded per group on first use.
class QuantTensor {
public:
  QuantTensor(rocm::WeightType t, int groups, int m, int k, std::mt19937& rng)
      : QuantTensor(t, groups, m, k,
                    RandomWeights(t, std::size_t(groups) * m, k, rng)) {}

  /// Existing GGUF bytes of `groups` stacked [m][k] matrices.
  QuantTensor(rocm::WeightType t, int groups, int m, int k,
              std::vector<std::uint8_t> bytes)
      : type(t),
        groups(groups),
        m(m),
        k(k),
        bytes_(std::move(bytes)),
        device_(std::span<const std::uint8_t>(bytes_)),
        full_(groups),
        scale_(groups),
        min_(groups) {}

  rocm::QuantMatrix View() const { return {device_.get(), type, m, k}; }

  /// Row o of group g, decoded by core::quant.
  const float* Row(int g, int o) {
    Decode(g);
    return &full_[g][std::size_t(o) * k];
  }

  /// Allowed |device - exact| for row o of group g against input x. MMQ
  /// tiles hold Q4_K/Q5_K sub-block scale d*sc and min dmin*m as FP16
  /// (relative 2^-11), so `mmq` rows of those types also allow 2^-11 *
  /// sum(|scale part * x| + |min part * x|), the parts decoded separately.
  double Tolerance(int g, int o, const float* x, bool mmq, const Dot& exact) {
    double tol = kProjRelTol * exact.magnitude + kProjAbsTol;
    if (mmq && HasMins(type)) {
      Decode(g);
      const std::size_t base = std::size_t(o) * k;
      tol += std::ldexp(1.0, -11) * (DotRow(&scale_[g][base], x, k).magnitude +
                                     DotRow(&min_[g][base], x, k).magnitude);
    }
    return tol;
  }

  /// The part of row o that multiplies quantized activation codes: the
  /// whole row, or for Q4_K/Q5_K the scale part (the min part multiplies
  /// the unquantized block sums).
  const float* ScaleRow(int g, int o) {
    Decode(g);
    return HasMins(type) ? &scale_[g][std::size_t(o) * k]
                         : &full_[g][std::size_t(o) * k];
  }

  rocm::WeightType type;
  int groups;
  int m;
  int k;

private:
  void Decode(int g) {
    if (!full_[g].empty()) {
      return;
    }
    const std::size_t group_bytes =
        std::size_t(m) * rocm::WeightRowBytes(type, k);
    std::vector<std::uint8_t> block(bytes_.begin() + g * group_bytes,
                                    bytes_.begin() + (g + 1) * group_bytes);
    full_[g] = Dequantize(type, block.data(), m, k);
    if (HasMins(type)) {
      const std::size_t stride = type == rocm::WeightType::kQ4_K ? 144 : 176;
      // Field 2 (dmin) zeroed gives the scale part, field 0 (d) the min part.
      for (const auto [field, out] : {std::pair{std::size_t{2}, &scale_[g]},
                                      std::pair{std::size_t{0}, &min_[g]}}) {
        std::vector<std::uint8_t> part = block;
        for (std::size_t off = 0; off < part.size(); off += stride) {
          part[off + field] = 0;
          part[off + field + 1] = 0;
        }
        *out = Dequantize(type, part.data(), m, k);
      }
    }
  }

  std::vector<std::uint8_t> bytes_;
  DeviceBuffer<std::uint8_t> device_;
  std::vector<std::vector<float>> full_;
  std::vector<std::vector<float>> scale_;
  std::vector<std::vector<float>> min_;
};

/// Skip return code for model-dependent checks.
inline constexpr int kSkip = 77;

}  // namespace gufo::models::flint::test

#endif  // GUFO_TESTS_MODELS_FLINT_OPS_TEST_SUPPORT_HPP_
