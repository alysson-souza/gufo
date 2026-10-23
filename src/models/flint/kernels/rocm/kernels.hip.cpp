// Flint device operators. Compiled with -ffp-contract=off and without
// fast-math: every fused multiply-add below is an explicit fmaf and every
// transcendental is the accurate library function.
#include "src/models/flint/kernels/rocm/kernels.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "src/models/flint/kernels/rocm/mmq/flint_mmq.h"

namespace gufo::models::flint::rocm {
namespace {

constexpr int kWarp = 32;
constexpr int kGroup = kQHeads / kKvHeads;  // query heads per KV head
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kLog2e = 1.4426950408889634074f;
// Query scale for base-2 softmax: fp32(1/sqrt(128) * log2(e)).
constexpr float kQScale = 0.12751743082459868f;
constexpr std::size_t kAlign = 256;

constexpr std::size_t AlignUp(std::size_t bytes) {
  return (bytes + kAlign - 1) / kAlign * kAlign;
}

hipError_t Status(int rc) {
  if (rc != 0) {
    return rc == -1 ? hipErrorInvalidValue : hipErrorLaunchFailure;
  }
  return hipGetLastError();
}

hipError_t LaunchStatus() {
  return hipGetLastError();
}

int Type(const QuantMatrix& w) {
  return static_cast<int>(w.type);
}

bool ValidMatrix(const QuantMatrix& w) {
  return w.data != nullptr && w.rows > 0 && w.cols > 0 && w.cols % 256 == 0 &&
         flint_mmq::weight_type_supported(Type(w));
}

/// Carves consecutive aligned regions out of caller scratch.
class Carver {
public:
  Carver(void* base, std::size_t bytes)
      : base_(static_cast<char*>(base)), bytes_(bytes) {}
  template<typename T>
  T* Take(std::size_t bytes) {
    if (base_ == nullptr || bytes > bytes_ - used_) {
      ok_ = false;
      return nullptr;
    }
    T* out = reinterpret_cast<T*>(base_ + used_);
    used_ += std::min(AlignUp(bytes), bytes_ - used_);
    return out;
  }
  bool ok() const { return ok_; }
  std::size_t remaining() const { return ok_ ? bytes_ - used_ : 0; }

private:
  char* base_;
  std::size_t bytes_;
  std::size_t used_ = 0;
  bool ok_ = true;
};

/// The vector (MMVQ) tier serves up to kVectorRows rows unless the matrix
/// tier is requested.
bool UseVector(int rows, bool matrix) {
  return !matrix && rows <= kVectorRows;
}

std::size_t ActivationBytes(int rows, int cols, bool matrix = false) {
  return UseVector(rows, matrix) ? flint_mmq::q8_1_vec_bytes(rows, cols)
                                 : flint_mmq::q8_1_mmq_bytes(rows, cols);
}

__device__ __forceinline__ float WarpSum(float v) {
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset, kWarp);
  }
  return v;
}

/// Block-wide sum of per-thread partials: warp trees, then the warp sums
/// added in warp order. Every thread receives the same value.
template<int kThreads>
__device__ float BlockSum(float v, float* shared) {
  constexpr int kWarps = kThreads / kWarp;
  v = WarpSum(v);
  const int warp = threadIdx.x / kWarp;
  if (threadIdx.x % kWarp == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = 0.0f;
  for (int i = 0; i < kWarps; ++i) {
    total += shared[i];
  }
  __syncthreads();
  return total;
}

__device__ __forceinline__ float Silu(float g) {
  return g / (1.0f + expf(-g));
}

__device__ __forceinline__ float Bf16ToFloat(std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

// --------------------------------------------------------------------------
// Elementwise kernels

__global__ void SwiGluKernel(const float* gate, const float* up, float* out,
                             std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = Silu(gate[i]) * up[i];
  }
}

hipError_t LaunchSwiGlu(const float* gate, const float* up, float* out,
                        std::size_t count, hipStream_t stream) {
  if (count == 0) {
    return hipSuccess;
  }
  const unsigned blocks = static_cast<unsigned>((count + 255) / 256);
  SwiGluKernel<<<blocks, 256, 0, stream>>>(gate, up, out, count);
  return LaunchStatus();
}

__global__ void EmbedQ8_0Kernel(const std::uint8_t* table,
                                const std::int32_t* tokens, int hidden,
                                float* out) {
  const int t = blockIdx.x;
  const std::size_t row_bytes = static_cast<std::size_t>(hidden) / 32 * 34;
  const std::uint8_t* row = table + tokens[t] * row_bytes;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const std::uint8_t* block = row + (i / 32) * 34;
    const __half d = *reinterpret_cast<const __half*>(block);
    const auto q = static_cast<std::int8_t>(block[2 + i % 32]);
    out[static_cast<std::size_t>(t) * hidden + i] =
        __half2float(d) * static_cast<float>(q);
  }
}

/// Q8_1 block (fp16 d, fp16 sum, 32 codes) of the vector kernels.
struct Q8_1Block {
  __half d;
  __half s;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_1Block) == 36, "block_q8_1 layout");

/// The 32 lanes of a warp encode one Q8_1 block from their values with the
/// arithmetic of the vendored quantize_q8_1 (warp max/sum trees,
/// d = amax / 127, q = round(v / d)), so the bytes are identical.
__device__ __forceinline__ void QuantizeWarpBlock(float v, Q8_1Block* block) {
  const int lane = threadIdx.x % kWarp;
  float amax = fabsf(v);
  float sum = v;
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, offset, kWarp));
    sum += __shfl_xor(sum, offset, kWarp);
  }
  const float d = amax / 127.0f;
  block->qs[lane] =
      static_cast<std::int8_t>(amax == 0.0f ? 0.0f : roundf(v / d));
  if (lane == 0) {
    block->d = __float2half(d);
    block->s = __float2half(sum);
  }
}

/// Padded inputs per Q8_1 row of the vector kernels (MATRIX_ROW_PADDING).
constexpr int kQ8RowPadding = 512;
__host__ __device__ constexpr int PaddedCols(int cols) {
  return (cols + kQ8RowPadding - 1) / kQ8RowPadding * kQ8RowPadding;
}

template<bool kAdd>
__global__ __launch_bounds__(256) void RmsNormKernel(
    float* residual, const float* delta, const float* x, const float* w,
    float* out, Q8_1Block* out_q8, int dim, float eps) {
  __shared__ float partial[256 / kWarp];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * dim;
  const float* src = kAdd ? residual + base : x + base;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += 256) {
    float v = src[i];
    if constexpr (kAdd) {
      v = v + delta[base + i];
      residual[base + i] = v;
    }
    sum = fmaf(v, v, sum);
  }
  sum = BlockSum<256>(sum, partial);
  const float scale = 1.0f / sqrtf(sum / static_cast<float>(dim) + eps);
  if (out_q8 == nullptr) {
    for (int i = threadIdx.x; i < dim; i += 256) {
      out[base + i] = src[i] * scale * w[i];
    }
    return;
  }
  // Element i belongs to block i / 32 = warp-uniform: the warp's lanes hold
  // one whole block per pass. Padding blocks encode zeros.
  const int padded = PaddedCols(dim);
  Q8_1Block* row_q8 =
      out_q8 + static_cast<std::size_t>(blockIdx.x) * (padded / 32);
  for (int i = threadIdx.x; i < padded; i += 256) {
    float v = 0.0f;
    if (i < dim) {
      v = src[i] * scale * w[i];
      out[base + i] = v;
    }
    QuantizeWarpBlock(v, &row_q8[i / 32]);
  }
}

__global__ void MoeCombineKernel(const float* expert_out, const float* weights,
                                 const float* shared, const float* residual,
                                 float* out, int used, int hidden) {
  const int t = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= hidden) {
    return;
  }
  const float* e = expert_out + static_cast<std::size_t>(t) * used * hidden;
  float acc = 0.0f;
  for (int j = 0; j < used; ++j) {
    acc = fmaf(weights[t * used + j],
               e[static_cast<std::size_t>(j) * hidden + d], acc);
  }
  const std::size_t i = static_cast<std::size_t>(t) * hidden + d;
  if (shared != nullptr) {
    acc = shared[i] + acc;
  }
  out[i] = residual != nullptr ? residual[i] + acc : acc;
}

// --------------------------------------------------------------------------
// Router

/// Each warp scores kE experts for kT tokens. Lane l always covers inputs
/// [256 i + 8 l, 256 i + 8 l + 8) in ascending FMA order and the warp sum
/// is one tree, so a score does not depend on the launch shape.
template<int kE, int kT, int kWarps>
__global__ __launch_bounds__(kWarps* kWarp) void RouterScoresKernel(
    const std::uint16_t* w, const float* x, float* scores, int tokens,
    int hidden, int experts) {
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int e0 = (blockIdx.x * kWarps + warp) * kE;
  const int t0 = blockIdx.y * kT;
  float acc[kE][kT] = {};
  // Deep unrolling keeps several row loads in flight for the narrow
  // shapes; the 16-token shape already has enough independent work.
  constexpr int kUnroll = kT >= 16 ? 1 : 8;
#pragma unroll kUnroll
  for (int k = lane * 8; k < hidden; k += kWarp * 8) {
    float wv[kE][8];
#pragma unroll
    for (int i = 0; i < kE; ++i) {
      const int e = min(e0 + i, experts - 1);
      const uint4 packed = *reinterpret_cast<const uint4*>(
          w + static_cast<std::size_t>(e) * hidden + k);
      const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        wv[i][2 * j] = Bf16ToFloat(static_cast<std::uint16_t>(words[j]));
        wv[i][2 * j + 1] =
            Bf16ToFloat(static_cast<std::uint16_t>(words[j] >> 16));
      }
    }
#pragma unroll
    for (int j = 0; j < kT; ++j) {
      const int t = min(t0 + j, tokens - 1);
      const float4* xp = reinterpret_cast<const float4*>(
          x + static_cast<std::size_t>(t) * hidden + k);
      const float4 a = xp[0];
      const float4 b = xp[1];
      const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int i = 0; i < kE; ++i) {
#pragma unroll
        for (int d = 0; d < 8; ++d) {
          acc[i][j] = fmaf(wv[i][d], xv[d], acc[i][j]);
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < kE; ++i) {
#pragma unroll
    for (int j = 0; j < kT; ++j) {
      const float logit = WarpSum(acc[i][j]);
      const int e = e0 + i;
      const int t = t0 + j;
      if (lane == 0 && e < experts && t < tokens) {
        scores[static_cast<std::size_t>(t) * experts + e] =
            1.0f / (1.0f + expf(-logit));
      }
    }
  }
}

constexpr int kRouterMaxPerLane = 32;  // experts <= 1024

/// True when (a, ia) outranks (b, ib): higher value, then lower index.
__device__ __forceinline__ bool Outranks(float a, int ia, float b, int ib) {
  return a > b || (a == b && ia < ib);
}

__global__ __launch_bounds__(kWarp) void RouterTopKKernel(
    const float* scores, const float* bias, float scale, int experts, int used,
    std::int32_t* ids, float* weights) {
  const int t = blockIdx.x;
  const int lane = threadIdx.x;
  const float* row = scores + static_cast<std::size_t>(t) * experts;
  float key[kRouterMaxPerLane];
  float raw[kRouterMaxPerLane];
  std::uint32_t taken = 0;
  // Lane l holds experts l, l + 32, ...: ascending index within a lane.
#pragma unroll
  for (int i = 0; i < kRouterMaxPerLane; ++i) {
    const int e = lane + i * kWarp;
    float v = -kInf;
    raw[i] = 0.0f;
    if (e < experts) {
      raw[i] = row[e];
      v = bias != nullptr ? raw[i] + bias[e] : raw[i];
      if (v != v) {
        v = -kInf;
      }
    } else {
      taken |= 1u << i;
    }
    key[i] = v;
  }
  // Every lane follows each round's winner; lane j keeps round j's choice,
  // so no result is read back from memory.
  float sum = 0.0f;
  int my_id = 0;
  float my_score = 0.0f;
  for (int j = 0; j < used; ++j) {
    float best = -kInf;
    float best_raw = 0.0f;
    int best_e = 0x7fffffff;
    int best_i = -1;
#pragma unroll
    for (int i = 0; i < kRouterMaxPerLane; ++i) {
      const int e = lane + i * kWarp;
      if (!(taken & (1u << i)) && Outranks(key[i], e, best, best_e)) {
        best = key[i];
        best_raw = raw[i];
        best_e = e;
        best_i = i;
      }
    }
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
      const float other = __shfl_xor(best, offset, kWarp);
      const float other_raw = __shfl_xor(best_raw, offset, kWarp);
      const int other_e = __shfl_xor(best_e, offset, kWarp);
      if (Outranks(other, other_e, best, best_e)) {
        best = other;
        best_raw = other_raw;
        best_e = other_e;
      }
    }
    if (best_i >= 0 && lane + best_i * kWarp == best_e) {
      taken |= 1u << best_i;
    }
    sum += best_raw;
    if (lane == j) {
      my_id = best_e;
      my_score = best_raw;
    }
  }
  if (lane < used) {
    const float denom = sum + 1e-20f;
    ids[t * used + lane] = my_id;
    weights[t * used + lane] = my_score / denom * scale;
  }
}

// --------------------------------------------------------------------------
// Attention

/// fp32(theta^(-i/64)) for the 64 NeoX pairs, computed on the host in
/// double and passed by value.
struct RopeFrequencies {
  float f[kHeadDim / 2];
};

RopeFrequencies MakeRopeFrequencies(float theta) {
  RopeFrequencies freq{};
  for (int i = 0; i < kHeadDim / 2; ++i) {
    freq.f[i] = static_cast<float>(
        std::pow(static_cast<double>(theta), -static_cast<double>(i) / 64.0));
  }
  return freq;
}

/// Shared NeoX angles of one position: accurate FP32 cos/sin of
/// fp32(p) * fp32(theta^(-i/64)).
__device__ void RopeAngles(int position, const RopeFrequencies& freq,
                           float* cos_out, float* sin_out) {
  for (int i = threadIdx.x; i < kHeadDim / 2; i += blockDim.x) {
    const float angle = static_cast<float>(position) * freq.f[i];
    sincosf(angle, &sin_out[i], &cos_out[i]);
  }
}

/// One block per row: 32 q heads and 4 k heads, one warp per head task.
/// Lane l holds dims {2l, 2l+1} and {64+2l, 65+2l}, i.e. the NeoX pairs
/// i = 2l and 2l + 1.
/// Inputs are rows `*_stride` floats apart; q_out is [rows][32][128] and
/// may alias q_in when q_stride == 32 * 128.
__global__ __launch_bounds__(256) void PrepareAttentionKernel(
    const float* q_in, int q_stride, float* q_out, const float* k, int k_stride,
    const float* v, int v_stride, const float* q_norm, const float* k_norm,
    const std::int32_t* positions, const std::int32_t* row_seq,
    const KvSequence* seqs, bool rope, RopeFrequencies freq, float eps) {
  __shared__ float cos_s[kHeadDim / 2];
  __shared__ float sin_s[kHeadDim / 2];
  const int r = blockIdx.x;
  const int position = positions[r];
  const KvSequence seq = seqs[row_seq != nullptr ? row_seq[r] : 0];
  if (rope) {
    RopeAngles(position, freq, cos_s, sin_s);
  }
  __syncthreads();
  const std::size_t slot = static_cast<std::size_t>(position % seq.capacity);
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int lo = 2 * lane;
  const int hi = kHeadDim / 2 + 2 * lane;
  for (int task = warp; task < kQHeads + kKvHeads; task += 256 / kWarp) {
    const bool is_q = task < kQHeads;
    const int head = is_q ? task : task - kQHeads;
    const float* src =
        is_q ? q_in + static_cast<std::size_t>(r) * q_stride + head * kHeadDim
             : k + static_cast<std::size_t>(r) * k_stride + head * kHeadDim;
    const float* gamma = is_q ? q_norm : k_norm;
    float x[4] = {src[lo], src[lo + 1], src[hi], src[hi + 1]};
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sum = fmaf(x[i], x[i], sum);
    }
    sum = WarpSum(sum);
    const float scale = 1.0f / sqrtf(sum / static_cast<float>(kHeadDim) + eps);
    x[0] = x[0] * scale * gamma[lo];
    x[1] = x[1] * scale * gamma[lo + 1];
    x[2] = x[2] * scale * gamma[hi];
    x[3] = x[3] * scale * gamma[hi + 1];
    if (rope) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const float c = cos_s[lo + i];
        const float s = sin_s[lo + i];
        const float x0 = x[i];
        const float x1 = x[2 + i];
        x[i] = x0 * c - x1 * s;
        x[2 + i] = x0 * s + x1 * c;
      }
    }
    if (is_q) {
      float* dst =
          q_out + (static_cast<std::size_t>(r) * kQHeads + head) * kHeadDim;
      dst[lo] = x[0];
      dst[lo + 1] = x[1];
      dst[hi] = x[2];
      dst[hi + 1] = x[3];
    } else {
      const std::size_t at =
          (static_cast<std::size_t>(head) * seq.capacity + slot) * kHeadDim;
      __half* kd = seq.k + at;
      kd[lo] = __float2half_rn(x[0]);
      kd[lo + 1] = __float2half_rn(x[1]);
      kd[hi] = __float2half_rn(x[2]);
      kd[hi + 1] = __float2half_rn(x[3]);
      const float* vs =
          v + static_cast<std::size_t>(r) * v_stride + head * kHeadDim;
      __half* vd = seq.v + at;
      for (int d = lane; d < kHeadDim; d += kWarp) {
        vd[d] = __float2half_rn(vs[d]);
      }
    }
  }
}

__device__ __forceinline__ void HalvesToFloats(const uint4& packed,
                                               float* out) {
  const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[2 * i] =
        __half2float(__ushort_as_half(static_cast<unsigned short>(words[i])));
    out[2 * i + 1] = __half2float(
        __ushort_as_half(static_cast<unsigned short>(words[i] >> 16)));
  }
}

struct DecodeSpan {
  int lo;
  int chunk;
  int chunks;
};

__device__ __forceinline__ DecodeSpan DecodeChunks(int position, int window) {
  DecodeSpan span{};
  span.lo = window > 0 ? max(0, position - window + 1) : 0;
  const int length = position - span.lo + 1;
  const int per = (length + kDecodeMaxChunks - 1) / kDecodeMaxChunks;
  span.chunk = max(32, (per + 31) / 32 * 32);
  span.chunks = (length + span.chunk - 1) / span.chunk;
  return span;
}

/// 2^(a - m) for a running maximum m, zero when a is -inf.
__device__ __forceinline__ float Weight(float a, float m) {
  return a == -kInf ? 0.0f : exp2f(a - m);
}

constexpr int kDecodeWarps = 4;
constexpr int kDecodeThreads = kDecodeWarps * kWarp;
constexpr int kDecodeTile = 32;  // positions staged per step
constexpr int kDecodePerWarp = kDecodeTile / kDecodeWarps;

/// One block per (chunk, KV head, row), four warps. The block stages 32
/// positions of K/V through LDS (the next tile prefetched into registers);
/// warp w then walks positions 8w .. 8w + 7 of each tile. Lane l serves
/// query head l / 4 of the group and dims 32 (l % 4) .. + 32 and keeps an
/// online softmax; the warps merge in warp order at the end.
__global__ __launch_bounds__(kDecodeThreads) void AttentionDecodeChunkKernel(
    const float* q, const std::int32_t* positions, const std::int32_t* row_seq,
    const KvSequence* seqs, int window, float* part_o, float* part_ml) {
  const int chunk_index = blockIdx.x;
  const int kvh = blockIdx.y;
  const int r = blockIdx.z;
  const int position = positions[r];
  const DecodeSpan span = DecodeChunks(position, window);
  if (chunk_index >= span.chunks) {
    return;
  }
  const KvSequence seq = seqs[row_seq != nullptr ? row_seq[r] : 0];
  const int start = span.lo + chunk_index * span.chunk;
  const int end = min(position + 1, start + span.chunk);

  // The K/V tiles and the final cross-warp merge share one LDS buffer.
  static_assert(2 * kDecodeTile * kHeadDim * sizeof(__half) ==
                    kDecodeWarps * kGroup * kHeadDim * sizeof(float),
                "tile and merge buffers alias exactly");
  __shared__
      __attribute__((aligned(16))) float lds[kDecodeWarps * kGroup * kHeadDim];
  auto* k_s = reinterpret_cast<__half(*)[kHeadDim]>(lds);
  auto* v_s = k_s + kDecodeTile;
  auto* red_o = reinterpret_cast<float (*)[kGroup][kHeadDim]>(lds);
  __shared__ float red_m[kDecodeWarps][kGroup];
  __shared__ float red_l[kDecodeWarps][kGroup];

  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int g = lane / 4;
  const int d0 = (lane % 4) * 32;
  float qv[32];
  const float* qr =
      q +
      (static_cast<std::size_t>(r) * kQHeads + kvh * kGroup + g) * kHeadDim +
      d0;
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    qv[d] = qr[d] * kQScale;
  }
  float m = -kInf;
  float l = 0.0f;
  float o[32];
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    o[d] = 0.0f;
  }
  // Staging: thread tid moves dims 32 (tid % 4) .. + 32 of tile position
  // tid / 4 for both K and V.
  const int stage_j = threadIdx.x / 4;
  const int stage_d = (threadIdx.x % 4) * 32;
  uint4 kr0, kr1, kr2, kr3, vr0, vr1, vr2, vr3;
#define FLINT_DECODE_LOAD(tile_start)                                          \
  do {                                                                         \
    const int t_ = (tile_start) + stage_j;                                     \
    if (t_ < end) {                                                            \
      const std::size_t base_ =                                                \
          (static_cast<std::size_t>(kvh) * seq.capacity + t_ % seq.capacity) * \
              kHeadDim +                                                       \
          stage_d;                                                             \
      const uint4* kp_ = reinterpret_cast<const uint4*>(seq.k + base_);        \
      const uint4* vp_ = reinterpret_cast<const uint4*>(seq.v + base_);        \
      kr0 = kp_[0];                                                            \
      kr1 = kp_[1];                                                            \
      kr2 = kp_[2];                                                            \
      kr3 = kp_[3];                                                            \
      vr0 = vp_[0];                                                            \
      vr1 = vp_[1];                                                            \
      vr2 = vp_[2];                                                            \
      vr3 = vp_[3];                                                            \
    }                                                                          \
  } while (0)
  FLINT_DECODE_LOAD(start);
  for (int tile = start; tile < end; tile += kDecodeTile) {
    __syncthreads();
    {
      uint4* kd = reinterpret_cast<uint4*>(&k_s[stage_j][stage_d]);
      uint4* vd = reinterpret_cast<uint4*>(&v_s[stage_j][stage_d]);
      kd[0] = kr0;
      kd[1] = kr1;
      kd[2] = kr2;
      kd[3] = kr3;
      vd[0] = vr0;
      vd[1] = vr1;
      vd[2] = vr2;
      vd[3] = vr3;
    }
    __syncthreads();
    if (tile + kDecodeTile < end) {
      FLINT_DECODE_LOAD(tile + kDecodeTile);
    }
    const int count = min(kDecodePerWarp, end - tile - warp * kDecodePerWarp);
    for (int j = 0; j < count; ++j) {
      const int row = warp * kDecodePerWarp + j;
      const uint4* kp = reinterpret_cast<const uint4*>(&k_s[row][d0]);
      float s = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float kf[8];
        HalvesToFloats(kp[i], kf);
#pragma unroll
        for (int d = 0; d < 8; ++d) {
          s = fmaf(qv[i * 8 + d], kf[d], s);
        }
      }
      s += __shfl_xor(s, 1, kWarp);
      s += __shfl_xor(s, 2, kWarp);
      if (s > m) {
        const float corr = Weight(m, s);
        m = s;
        l *= corr;
#pragma unroll
        for (int d = 0; d < 32; ++d) {
          o[d] *= corr;
        }
      }
      const float p = exp2f(s - m);
      l += p;
      const uint4* vp = reinterpret_cast<const uint4*>(&v_s[row][d0]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float vf[8];
        HalvesToFloats(vp[i], vf);
#pragma unroll
        for (int d = 0; d < 8; ++d) {
          o[i * 8 + d] = fmaf(p, vf[d], o[i * 8 + d]);
        }
      }
    }
  }
#undef FLINT_DECODE_LOAD
  __syncthreads();  // the last tile's readers are done with the buffer
  if (lane % 4 == 0) {
    red_m[warp][g] = m;
    red_l[warp][g] = l;
  }
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    red_o[warp][g][d0 + d] = o[d];
  }
  __syncthreads();
  // 128 threads: head tid / 16, dims 8 * (tid % 16) .. + 8.
  const int h = threadIdx.x / 16;
  const int e0 = (threadIdx.x % 16) * 8;
  float mm = -kInf;
  for (int w = 0; w < kDecodeWarps; ++w) {
    mm = fmaxf(mm, red_m[w][h]);
  }
  float ll = 0.0f;
  float oo[8] = {};
  for (int w = 0; w < kDecodeWarps; ++w) {
    const float a = Weight(red_m[w][h], mm);
    ll = fmaf(red_l[w][h], a, ll);
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      oo[d] = fmaf(red_o[w][h][e0 + d], a, oo[d]);
    }
  }
  const std::size_t part =
      (static_cast<std::size_t>(r) * kQHeads + kvh * kGroup + h) *
          kDecodeMaxChunks +
      chunk_index;
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    part_o[part * kHeadDim + e0 + d] = oo[d];
  }
  if (e0 == 0) {
    part_ml[part * 2] = mm;
    part_ml[part * 2 + 1] = ll;
  }
}

/// One block per (row, q head), one thread per output dim. The chunk
/// weights 2^(m_c - M) are computed once into LDS; denominators and outputs
/// then sum the chunks in ascending order, the sink last.
__global__ __launch_bounds__(kHeadDim) void AttentionDecodeMergeKernel(
    const std::int32_t* positions, const float* sinks, int window,
    const float* part_o, const float* part_ml, float* out, Q8_1Block* out_q8) {
  __shared__ float weight[kDecodeMaxChunks];
  __shared__ float maxima[kHeadDim];
  const int r = blockIdx.x;
  const int h = blockIdx.y;
  const int d = threadIdx.x;
  const DecodeSpan span = DecodeChunks(positions[r], window);
  const std::size_t base =
      (static_cast<std::size_t>(r) * kQHeads + h) * kDecodeMaxChunks;
  const float sink = sinks[h] * kLog2e;
  float local = -kInf;
  for (int c = d; c < span.chunks; c += kHeadDim) {
    local = fmaxf(local, part_ml[(base + c) * 2]);
  }
  maxima[d] = local;
  __syncthreads();
  float mm = sink;
  for (int i = 0; i < kHeadDim; ++i) {
    mm = fmaxf(mm, maxima[i]);
  }
  for (int c = d; c < span.chunks; c += kHeadDim) {
    weight[c] = Weight(part_ml[(base + c) * 2], mm);
  }
  __syncthreads();
  float ll = 0.0f;
  float oo = 0.0f;
#pragma unroll 8
  for (int c = 0; c < span.chunks; ++c) {
    const float a = weight[c];
    ll = fmaf(part_ml[(base + c) * 2 + 1], a, ll);
    oo = fmaf(part_o[(base + c) * kHeadDim + d], a, oo);
  }
  ll += Weight(sink, mm);
  const float value = oo / ll;
  out[(static_cast<std::size_t>(r) * kQHeads + h) * kHeadDim + d] = value;
  if (out_q8 != nullptr) {
    // Each warp holds one 32-value block of the row's 4096 outputs.
    QuantizeWarpBlock(
        value,
        &out_q8[(static_cast<std::size_t>(r) * kQHeads + h) * (kHeadDim / 32) +
                d / 32]);
  }
}

using V16h = _Float16 __attribute__((ext_vector_type(16)));
using V8f = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ V8f Wmma(const V16h& a, const V16h& b,
                                    const V8f& c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// Exact power-of-two lifts keep the FP16 q and p values out of the subnormal
// range; products are scaled back exactly.
constexpr float kQLift = 256.0f;
constexpr float kPLift = 1024.0f;
constexpr int kPrefillPositions = 16;
constexpr int kPrefillKeys = 32;
constexpr int kKStride = kHeadDim + 8;       // halves per staged K row
constexpr int kVtStride = kPrefillKeys + 8;  // halves per staged V^T row
constexpr int kPStride = kPrefillKeys + 8;
constexpr int kStageHalves = kPrefillKeys * kHeadDim / 256;  // per thread

/// One block per (absolute 16-position tile 16t .. 16t + 15, KV head); wave w
/// computes query head 8 kvh + w for the tile's positions that belong to the
/// call. Wave32 fragments: A holds row L % 16 with 16 contiguous k, B column
/// L % 16 with 16 contiguous k, accumulator element i is row 2i + L / 16,
/// column L % 16. The query (pre-scaled by log2(e) / sqrt(128)) and the
/// probabilities enter the products as FP16 (lifted by exact powers of two to
/// stay normal), like the FP16 K and V they multiply; accumulation and the
/// online softmax are FP32.
///
/// Split invariance: keys before the tile stream through LDS in 32-key
/// blocks aligned to absolute positions, keys outside the tile's window
/// staged as zeros, so every product a row's WMMA sums sees the same operands
/// whatever first_position and rows are. The tile's own 16 keys (some of
/// which may not be written yet when a call ends inside the tile) are
/// applied last with per-row FP32 FMAs over exactly the visible keys.
__global__ __launch_bounds__(256) void AttentionPrefillKernel(
    const float* q, int first, int rows, KvSequence seq, const float* sinks,
    int window, float* out) {
  __shared__
      __attribute__((aligned(16))) _Float16 k_lds[kPrefillKeys][kKStride];
  __shared__ __attribute__((aligned(16))) _Float16 v_lds[kHeadDim][kVtStride];
  __shared__ __attribute__((
      aligned(16))) _Float16 p_lds[8][kPrefillPositions][kPStride];
  const int kvh = blockIdx.y;
  const int wave = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int col = lane % 16;
  const int half_id = lane / 16;
  const int head = kvh * kGroup + wave;
  const int pos0 = (first / kPrefillPositions + blockIdx.x) * kPrefillPositions;
  const int last = first + rows - 1;
  const int key_lo = window > 0 ? max(0, pos0 - window + 1) : 0;
  const int key_end = pos0 - 1;  // last key before the tile
  V16h q_hi[kHeadDim / 16];
  {
    const int pos = pos0 + col;
    const bool live = pos >= first && pos <= last;
    const float* qr =
        q +
        (static_cast<std::size_t>(live ? pos - first : 0) * kQHeads + head) *
            kHeadDim;
#pragma unroll
    for (int ks = 0; ks < kHeadDim / 16; ++ks) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float v = live ? qr[ks * 16 + i] * kQScale * kQLift : 0.0f;
        q_hi[ks][i] = static_cast<_Float16>(v);
      }
    }
  }
  V8f o_acc[kHeadDim / 16];
#pragma unroll
  for (int nt = 0; nt < kHeadDim / 16; ++nt) {
    o_acc[nt] = V8f{};
  }
  float m[8];
  float l[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    m[i] = -kInf;
    l[i] = 0.0f;
  }
  const int row_base = pos0 + half_id;
  const auto row_live = [&](int rp) { return rp >= first && rp <= last; };
  const int stage_key = threadIdx.x / (256 / kPrefillKeys);
  const int stage_d = (threadIdx.x % (256 / kPrefillKeys)) * kStageHalves;
  uint4 k_next[kStageHalves / 8];
  uint4 v_next[kStageHalves / 8];
  const auto load = [&](int kb) {
    const int kp = kb + stage_key;
    const bool live = kp >= key_lo && kp <= key_end;
    const std::size_t off = (static_cast<std::size_t>(kvh) * seq.capacity +
                             (live ? kp : 0) % seq.capacity) *
                                kHeadDim +
                            stage_d;
#pragma unroll
    for (int j = 0; j < kStageHalves / 8; ++j) {
      k_next[j] = live ? reinterpret_cast<const uint4*>(seq.k + off)[j]
                       : make_uint4(0, 0, 0, 0);
      v_next[j] = live ? reinterpret_cast<const uint4*>(seq.v + off)[j]
                       : make_uint4(0, 0, 0, 0);
    }
  };
  // Keys before the tile, in absolute 32-key blocks.
  const int key_first = key_lo / kPrefillKeys * kPrefillKeys;
  if (key_first <= key_end) {
    load(key_first);
  }
  for (int kb = key_first; kb <= key_end; kb += kPrefillKeys) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kStageHalves / 8; ++j) {
      *reinterpret_cast<uint4*>(&k_lds[stage_key][stage_d + 8 * j]) = k_next[j];
      const _Float16* vh = reinterpret_cast<const _Float16*>(&v_next[j]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        v_lds[stage_d + 8 * j + i][stage_key] = vh[i];
      }
    }
    __syncthreads();
    if (kb + kPrefillKeys <= key_end) {
      load(kb + kPrefillKeys);
    }
    V8f s_acc[kPrefillKeys / 16];
#pragma unroll
    for (int sub = 0; sub < kPrefillKeys / 16; ++sub) {
      s_acc[sub] = V8f{};
#pragma unroll
      for (int ks = 0; ks < kHeadDim / 16; ++ks) {
        const V16h kf =
            *reinterpret_cast<const V16h*>(&k_lds[sub * 16 + col][ks * 16]);
        s_acc[sub] = Wmma(q_hi[ks], kf, s_acc[sub]);
        __builtin_amdgcn_sched_barrier(0);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int rp = row_base + 2 * i;
      float s[kPrefillKeys / 16];
      float tile_max = -kInf;
#pragma unroll
      for (int sub = 0; sub < kPrefillKeys / 16; ++sub) {
        const int kp = kb + sub * 16 + col;
        const bool visible = kp >= key_lo && kp <= key_end && row_live(rp) &&
                             (window <= 0 || kp > rp - window);
        s[sub] = visible ? s_acc[sub][i] * (1.0f / kQLift) : -kInf;
        tile_max = fmaxf(tile_max, s[sub]);
      }
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 8, kWarp));
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 4, kWarp));
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 2, kWarp));
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 1, kWarp));
      if (tile_max > m[i]) {
        const float corr = Weight(m[i], tile_max);
        m[i] = tile_max;
        l[i] *= corr;
#pragma unroll
        for (int nt = 0; nt < kHeadDim / 16; ++nt) {
          o_acc[nt][i] *= corr;
        }
      }
      float sum = 0.0f;
#pragma unroll
      for (int sub = 0; sub < kPrefillKeys / 16; ++sub) {
        const float p = Weight(s[sub], m[i]);
        sum += p;
        p_lds[wave][2 * i + half_id][sub * 16 + col] =
            static_cast<_Float16>(p * kPLift);
      }
      sum += __shfl_xor(sum, 8, kWarp);
      sum += __shfl_xor(sum, 4, kWarp);
      sum += __shfl_xor(sum, 2, kWarp);
      sum += __shfl_xor(sum, 1, kWarp);
      l[i] += sum;
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
#pragma unroll
    for (int sub = 0; sub < kPrefillKeys / 16; ++sub) {
      const V16h pf =
          *reinterpret_cast<const V16h*>(&p_lds[wave][col][sub * 16]);
#pragma unroll
      for (int nt = 0; nt < kHeadDim / 16; ++nt) {
        const V16h vf =
            *reinterpret_cast<const V16h*>(&v_lds[nt * 16 + col][sub * 16]);
        o_acc[nt] = Wmma(pf, vf, o_acc[nt]);
        __builtin_amdgcn_sched_barrier(0);
      }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  }
  // The tile's own keys pos0 .. pos0 + 15 (written up to `last`).
  __syncthreads();
  {
    const int key = threadIdx.x / 16;
    const int d = (threadIdx.x % 16) * 8;
    const int kp = pos0 + key;
    const bool live = kp <= last;
    const std::size_t off = (static_cast<std::size_t>(kvh) * seq.capacity +
                             (live ? kp : 0) % seq.capacity) *
                                kHeadDim +
                            d;
    const uint4 kk = live ? *reinterpret_cast<const uint4*>(seq.k + off)
                          : make_uint4(0, 0, 0, 0);
    const uint4 vv = live ? *reinterpret_cast<const uint4*>(seq.v + off)
                          : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(&k_lds[key][d]) = kk;
    const _Float16* vh = reinterpret_cast<const _Float16*>(&vv);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v_lds[d + i][key] = vh[i];
    }
  }
  __syncthreads();
  {
    V8f s_acc = {};
#pragma unroll
    for (int ks = 0; ks < kHeadDim / 16; ++ks) {
      const V16h kf = *reinterpret_cast<const V16h*>(&k_lds[col][ks * 16]);
      s_acc = Wmma(q_hi[ks], kf, s_acc);
      __builtin_amdgcn_sched_barrier(0);
    }
    // Lifted probabilities as FP32, rows of 17 floats in the wave's P buffer.
    float* p32 = reinterpret_cast<float*>(&p_lds[wave][0][0]);
    constexpr int kP32Stride = 17;
    static_assert(
        kPrefillPositions * kP32Stride * sizeof(float) <= sizeof(p_lds[0]),
        "FP32 diagonal probabilities fit the P buffer");
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int rp = row_base + 2 * i;
      const bool visible = pos0 + col <= rp && row_live(rp);
      const float s = visible ? s_acc[i] * (1.0f / kQLift) : -kInf;
      float tile_max = s;
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 8, kWarp));
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 4, kWarp));
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 2, kWarp));
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 1, kWarp));
      if (tile_max > m[i]) {
        const float corr = Weight(m[i], tile_max);
        m[i] = tile_max;
        l[i] *= corr;
#pragma unroll
        for (int nt = 0; nt < kHeadDim / 16; ++nt) {
          o_acc[nt][i] *= corr;
        }
      }
      const float p = Weight(s, m[i]);
      float sum = p;
      sum += __shfl_xor(sum, 8, kWarp);
      sum += __shfl_xor(sum, 4, kWarp);
      sum += __shfl_xor(sum, 2, kWarp);
      sum += __shfl_xor(sum, 1, kWarp);
      l[i] += sum;
      p32[(2 * i + half_id) * kP32Stride + col] = p * kPLift;
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    // Row 2i + half_id sees diagonal keys j = 0 .. 2i + half_id.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = 2 * i + half_id;
      const float* pr = p32 + r * kP32Stride;
#pragma unroll
      for (int nt = 0; nt < kHeadDim / 16; ++nt) {
        const _Float16* vr = &v_lds[nt * 16 + col][0];
        float acc = o_acc[nt][i];
        for (int j = 0; j <= r; ++j) {
          acc = fmaf(pr[j], static_cast<float>(vr[j]), acc);
        }
        o_acc[nt][i] = acc;
      }
    }
  }
  const float sink = sinks[head] * kLog2e;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rp = row_base + 2 * i;
    if (!row_live(rp)) {
      continue;
    }
    const float mm = fmaxf(m[i], sink);
    const float a = Weight(m[i], mm);
    const float denom = fmaf(l[i], a, Weight(sink, mm));
    float* dst =
        out +
        (static_cast<std::size_t>(rp - first) * kQHeads + head) * kHeadDim +
        col;
#pragma unroll
    for (int nt = 0; nt < kHeadDim / 16; ++nt) {
      dst[nt * 16] = o_acc[nt][i] * (1.0f / kPLift) * a / denom;
    }
  }
}

// --------------------------------------------------------------------------
// Projection helpers

/// Quantizes x for `targets` sharing one MMQ layout and runs them.
hipError_t ProjectGroup(std::span<const ProjectionTarget> targets,
                        const float* x, int rows, int cols, void* xq,
                        std::size_t xq_bytes, hipStream_t stream,
                        bool matrix = false) {
  if (UseVector(rows, matrix)) {
    hipError_t err = Status(flint_mmq::quantize_vec(x, xq, rows, cols, stream));
    for (const ProjectionTarget& t : targets) {
      if (err != hipSuccess) {
        return err;
      }
      err = Status(flint_mmq::dense_vec(Type(t.w), t.w.data, nullptr, xq, t.y,
                                        t.w.rows, rows, cols, stream));
    }
    return err;
  }
  bool done[16] = {};
  for (std::size_t i = 0; i < targets.size(); ++i) {
    if (done[i]) {
      continue;
    }
    const int layout = flint_mmq::mmq_layout(Type(targets[i].w));
    hipError_t err = Status(flint_mmq::quantize_mmq(
        Type(targets[i].w), x, nullptr, xq, xq_bytes, rows, cols, stream));
    for (std::size_t j = i; j < targets.size() && err == hipSuccess; ++j) {
      if (done[j] || flint_mmq::mmq_layout(Type(targets[j].w)) != layout) {
        continue;
      }
      done[j] = true;
      err = Status(flint_mmq::dense_mmq(Type(targets[j].w), targets[j].w.data,
                                        xq, targets[j].y, targets[j].w.rows,
                                        rows, cols, stream));
    }
    if (err != hipSuccess) {
      return err;
    }
  }
  return hipSuccess;
}

}  // namespace

std::optional<WeightType> WeightTypeFromGgml(std::uint32_t ggml_type) {
  if (!flint_mmq::weight_type_supported(static_cast<int>(ggml_type))) {
    return std::nullopt;
  }
  return static_cast<WeightType>(ggml_type);
}

std::size_t WeightRowBytes(WeightType type, int cols) {
  const std::size_t c = static_cast<std::size_t>(cols);
  switch (type) {
    case WeightType::kQ8_0:
      return c / 32 * 34;
    case WeightType::kQ4_K:
      return c / 256 * 144;
    case WeightType::kQ5_K:
      return c / 256 * 176;
    case WeightType::kQ6_K:
      return c / 256 * 210;
    case WeightType::kIQ3_S:
      return c / 256 * 110;
    case WeightType::kIQ4_XS:
      return c / 256 * 136;
  }
  return 0;
}

std::size_t ProjectionScratchBytes(int rows, int cols) {
  return AlignUp(ActivationBytes(rows, cols));
}

hipError_t Project(const QuantMatrix& w, const float* x, float* y, int rows,
                   void* scratch, std::size_t scratch_bytes,
                   hipStream_t stream) {
  const ProjectionTarget target{w, y};
  return ProjectShared({&target, 1}, x, rows, w.cols, scratch, scratch_bytes,
                       stream);
}

namespace {

hipError_t ProjectSharedImpl(std::span<const ProjectionTarget> targets,
                             const float* x, int rows, int cols, void* scratch,
                             std::size_t scratch_bytes, hipStream_t stream,
                             bool matrix) {
  if (targets.empty() || targets.size() > 16 || x == nullptr || rows <= 0 ||
      scratch == nullptr ||
      scratch_bytes < AlignUp(ActivationBytes(rows, cols, matrix))) {
    return hipErrorInvalidValue;
  }
  for (const ProjectionTarget& t : targets) {
    if (!ValidMatrix(t.w) || t.w.cols != cols || t.y == nullptr) {
      return hipErrorInvalidValue;
    }
  }
  return ProjectGroup(targets, x, rows, cols, scratch, scratch_bytes, stream,
                      matrix);
}

std::size_t SwiGluBytes(int rows, int cols, int inner, bool matrix) {
  const std::size_t staged =
      AlignUp(static_cast<std::size_t>(rows) * inner * sizeof(float));
  return 2 * staged + AlignUp(ActivationBytes(rows, cols, matrix));
}

hipError_t ProjectSwiGluImpl(const QuantMatrix& gate, const QuantMatrix& up,
                             const float* x, float* y, int rows, void* scratch,
                             std::size_t scratch_bytes, hipStream_t stream,
                             bool matrix) {
  if (!ValidMatrix(gate) || !ValidMatrix(up) || gate.rows != up.rows ||
      gate.cols != up.cols || x == nullptr || y == nullptr || rows <= 0 ||
      scratch_bytes < SwiGluBytes(rows, gate.cols, gate.rows, matrix)) {
    return hipErrorInvalidValue;
  }
  if (!matrix && rows == 1 && gate.type == up.type) {
    const hipError_t err =
        Status(flint_mmq::quantize_vec(x, scratch, 1, gate.cols, stream));
    if (err != hipSuccess) {
      return err;
    }
    return Status(flint_mmq::dense_vec(Type(up), up.data, gate.data, scratch, y,
                                       up.rows, 1, up.cols, stream));
  }
  Carver carve(scratch, scratch_bytes);
  const std::size_t staged =
      static_cast<std::size_t>(rows) * gate.rows * sizeof(float);
  float* g = carve.Take<float>(staged);
  float* u = carve.Take<float>(staged);
  const std::size_t xq_bytes =
      AlignUp(ActivationBytes(rows, gate.cols, matrix));
  void* xq = carve.Take<char>(xq_bytes);
  if (!carve.ok()) {
    return hipErrorInvalidValue;
  }
  const ProjectionTarget targets[2] = {{gate, g}, {up, u}};
  const hipError_t err =
      ProjectGroup(targets, x, rows, gate.cols, xq, xq_bytes, stream, matrix);
  if (err != hipSuccess) {
    return err;
  }
  return LaunchSwiGlu(g, u, y, static_cast<std::size_t>(rows) * gate.rows,
                      stream);
}

}  // namespace

hipError_t ProjectShared(std::span<const ProjectionTarget> targets,
                         const float* x, int rows, int cols, void* scratch,
                         std::size_t scratch_bytes, hipStream_t stream) {
  return ProjectSharedImpl(targets, x, rows, cols, scratch, scratch_bytes,
                           stream, false);
}

std::size_t SwiGluScratchBytes(int rows, int cols, int inner) {
  return SwiGluBytes(rows, cols, inner, false);
}

hipError_t ProjectSwiGlu(const QuantMatrix& gate, const QuantMatrix& up,
                         const float* x, float* y, int rows, void* scratch,
                         std::size_t scratch_bytes, hipStream_t stream) {
  return ProjectSwiGluImpl(gate, up, x, y, rows, scratch, scratch_bytes, stream,
                           false);
}

std::size_t MatrixProjectionScratchBytes(int rows, int cols) {
  return AlignUp(ActivationBytes(rows, cols, true));
}

hipError_t ProjectMatrix(const QuantMatrix& w, const float* x, float* y,
                         int rows, void* scratch, std::size_t scratch_bytes,
                         hipStream_t stream) {
  const ProjectionTarget target{w, y};
  return ProjectSharedImpl({&target, 1}, x, rows, w.cols, scratch,
                           scratch_bytes, stream, true);
}

hipError_t ProjectSharedMatrix(std::span<const ProjectionTarget> targets,
                               const float* x, int rows, int cols,
                               void* scratch, std::size_t scratch_bytes,
                               hipStream_t stream) {
  return ProjectSharedImpl(targets, x, rows, cols, scratch, scratch_bytes,
                           stream, true);
}

std::size_t MatrixSwiGluScratchBytes(int rows, int cols, int inner) {
  return SwiGluBytes(rows, cols, inner, true);
}

hipError_t ProjectSwiGluMatrix(const QuantMatrix& gate, const QuantMatrix& up,
                               const float* x, float* y, int rows,
                               void* scratch, std::size_t scratch_bytes,
                               hipStream_t stream) {
  return ProjectSwiGluImpl(gate, up, x, y, rows, scratch, scratch_bytes, stream,
                           true);
}

hipError_t ProjectQuantized(const QuantMatrix& w, const void* xq, float* y,
                            int rows, hipStream_t stream) {
  if (!ValidMatrix(w) || xq == nullptr || y == nullptr || rows <= 0 ||
      rows > kVectorRows) {
    return hipErrorInvalidValue;
  }
  return Status(flint_mmq::dense_vec(Type(w), w.data, nullptr, xq, y, w.rows,
                                     rows, w.cols, stream));
}

hipError_t ProjectSwiGluQuantized(const QuantMatrix& gate,
                                  const QuantMatrix& up, const void* xq,
                                  float* y, int rows, void* scratch,
                                  std::size_t scratch_bytes,
                                  hipStream_t stream) {
  if (!ValidMatrix(gate) || !ValidMatrix(up) || gate.rows != up.rows ||
      gate.cols != up.cols || xq == nullptr || y == nullptr || rows <= 0 ||
      rows > kVectorRows) {
    return hipErrorInvalidValue;
  }
  if (rows == 1 && gate.type == up.type) {
    return Status(flint_mmq::dense_vec(Type(up), up.data, gate.data, xq, y,
                                       up.rows, 1, up.cols, stream));
  }
  const std::size_t staged =
      static_cast<std::size_t>(rows) * gate.rows * sizeof(float);
  Carver carve(scratch, scratch_bytes);
  float* g = carve.Take<float>(staged);
  float* u = carve.Take<float>(staged);
  if (!carve.ok()) {
    return hipErrorInvalidValue;
  }
  hipError_t err =
      Status(flint_mmq::dense_vec(Type(gate), gate.data, nullptr, xq, g,
                                  gate.rows, rows, gate.cols, stream));
  if (err == hipSuccess) {
    err = Status(flint_mmq::dense_vec(Type(up), up.data, nullptr, xq, u,
                                      up.rows, rows, up.cols, stream));
  }
  if (err != hipSuccess) {
    return err;
  }
  return LaunchSwiGlu(g, u, y, static_cast<std::size_t>(rows) * gate.rows,
                      stream);
}

hipError_t SwiGlu(const float* gate, const float* up, float* out,
                  std::size_t count, hipStream_t stream) {
  if ((gate == nullptr || up == nullptr || out == nullptr) && count != 0) {
    return hipErrorInvalidValue;
  }
  return LaunchSwiGlu(gate, up, out, count, stream);
}

namespace {

int RoutedVecRows(const RoutedExperts& e) {
  return std::min({flint_mmq::moe_vec_max_rows(Type(e.gate)),
                   flint_mmq::moe_vec_max_rows(Type(e.up)),
                   flint_mmq::moe_vec_max_rows(Type(e.down))});
}

bool ValidExperts(const RoutedExperts& e) {
  return ValidMatrix(e.gate) && ValidMatrix(e.up) && ValidMatrix(e.down) &&
         e.experts > 0 && e.gate.rows == e.up.rows &&
         e.gate.cols == e.up.cols && e.down.cols == e.gate.rows &&
         e.down.rows == e.gate.cols;
}

}  // namespace

std::size_t RoutedProjectScratchBytes(const QuantMatrix& w, int experts,
                                      int tokens, int used,
                                      bool per_slot_input) {
  const std::size_t slots = static_cast<std::size_t>(tokens) * used;
  if (tokens <= kVectorRows) {
    const int chunk = std::min(tokens, flint_mmq::moe_vec_max_rows(Type(w)));
    return AlignUp(flint_mmq::q8_1_vec_bytes(
        per_slot_input ? chunk * used : chunk, w.cols));
  }
  return 2 * AlignUp(slots * sizeof(std::int32_t)) +
         AlignUp((experts + 1) * sizeof(std::int32_t)) +
         AlignUp(flint_mmq::q8_1_mmq_bytes(static_cast<int>(slots), w.cols));
}

hipError_t RoutedProject(const QuantMatrix& w, int experts, const float* x,
                         bool per_slot_input, const std::int32_t* ids,
                         int tokens, int used, float* out, void* scratch,
                         std::size_t scratch_bytes, hipStream_t stream) {
  if (!ValidMatrix(w) || experts <= 0 || x == nullptr || ids == nullptr ||
      out == nullptr || tokens <= 0 || used <= 0 ||
      scratch_bytes <
          RoutedProjectScratchBytes(w, experts, tokens, used, per_slot_input)) {
    return hipErrorInvalidValue;
  }
  const std::size_t slots = static_cast<std::size_t>(tokens) * used;
  const int per_token = per_slot_input ? used : 1;
  Carver carve(scratch, scratch_bytes);
  hipError_t err = hipSuccess;
  if (tokens <= kVectorRows) {
    const int chunk_max =
        std::min(tokens, flint_mmq::moe_vec_max_rows(Type(w)));
    void* xq = carve.Take<char>(carve.remaining());
    for (int t0 = 0; t0 < tokens && err == hipSuccess; t0 += chunk_max) {
      const int n = std::min(chunk_max, tokens - t0);
      const std::size_t s0 = static_cast<std::size_t>(t0) * used;
      err = Status(flint_mmq::quantize_vec(
          x + static_cast<std::size_t>(t0) * per_token * w.cols, xq,
          n * per_token, w.cols, stream));
      if (err == hipSuccess) {
        err = Status(flint_mmq::moe_vec(Type(w), w.data, nullptr, xq, ids + s0,
                                        out + s0 * w.rows, w.rows, w.cols, n,
                                        used, experts, per_slot_input, stream));
      }
    }
    return err;
  }
  auto* gather = carve.Take<std::int32_t>(slots * sizeof(std::int32_t));
  auto* scatter = carve.Take<std::int32_t>(slots * sizeof(std::int32_t));
  auto* bounds = carve.Take<std::int32_t>((experts + 1) * sizeof(std::int32_t));
  const std::size_t xq_bytes = carve.remaining();
  void* xq = carve.Take<char>(xq_bytes);
  if (!carve.ok()) {
    return hipErrorInvalidValue;
  }
  const int rows = static_cast<int>(slots);
  err = Status(flint_mmq::moe_maps(ids, gather, scatter, bounds,
                                   per_slot_input ? rows : tokens,
                                   per_slot_input ? 1 : used, experts, stream));
  if (err == hipSuccess) {
    err = Status(flint_mmq::quantize_mmq(Type(w), x, gather, xq, xq_bytes, rows,
                                         w.cols, stream));
  }
  if (err == hipSuccess) {
    err = Status(flint_mmq::moe_mmq(Type(w), w.data, xq, scatter, bounds, out,
                                    w.rows, w.cols, rows, experts, tokens,
                                    stream));
  }
  return err;
}

namespace {

std::size_t RoutedBytes(const RoutedExperts& e, int tokens, int used,
                        bool matrix) {
  const int hidden = e.gate.cols;
  const int inner = e.gate.rows;
  const std::size_t slots = static_cast<std::size_t>(tokens) * used;
  const std::size_t staged = AlignUp(slots * inner * sizeof(float));
  if (UseVector(tokens, matrix)) {
    const int chunk = std::min(tokens, RoutedVecRows(e));
    const std::size_t xq =
        std::max(flint_mmq::q8_1_vec_bytes(chunk, hidden),
                 flint_mmq::q8_1_vec_bytes(chunk * used, inner));
    return 2 * staged + AlignUp(xq);
  }
  const std::size_t maps = 2 * AlignUp(slots * sizeof(std::int32_t)) +
                           AlignUp((e.experts + 1) * sizeof(std::int32_t));
  const std::size_t xq =
      std::max(flint_mmq::q8_1_mmq_bytes(static_cast<int>(slots), hidden),
               flint_mmq::q8_1_mmq_bytes(static_cast<int>(slots), inner));
  return maps + 2 * staged + AlignUp(xq);
}

}  // namespace

std::size_t RoutedScratchBytes(const RoutedExperts& e, int tokens, int used) {
  return RoutedBytes(e, tokens, used, false);
}

std::size_t MatrixRoutedScratchBytes(const RoutedExperts& e, int tokens,
                                     int used) {
  return RoutedBytes(e, tokens, used, true);
}

namespace {

/// RoutedForward for up to kVectorRows tokens from float rows `x` or from
/// Q8_1 rows `x_q8` (vector layout).
hipError_t RoutedVector(const RoutedExperts& e, const float* x,
                        const void* x_q8, const std::int32_t* ids, int tokens,
                        int used, float* expert_out, void* scratch,
                        std::size_t scratch_bytes, hipStream_t stream) {
  const int hidden = e.gate.cols;
  const int inner = e.gate.rows;
  const std::size_t slots = static_cast<std::size_t>(tokens) * used;
  Carver carve(scratch, scratch_bytes);
  hipError_t err = hipSuccess;
  {
    const int chunk_max = std::min(tokens, RoutedVecRows(e));
    float* g = carve.Take<float>(slots * inner * sizeof(float));
    float* u = carve.Take<float>(slots * inner * sizeof(float));
    const std::size_t xq_bytes = carve.remaining();
    void* xq = carve.Take<char>(xq_bytes);
    if (!carve.ok()) {
      return hipErrorInvalidValue;
    }
    for (int t0 = 0; t0 < tokens && err == hipSuccess; t0 += chunk_max) {
      const int n = std::min(chunk_max, tokens - t0);
      const std::size_t s0 = static_cast<std::size_t>(t0) * used;
      const std::int32_t* chunk_ids = ids + s0;
      float* h = g + s0 * inner;
      // Pre-quantized rows are used in place; float rows are quantized.
      const void* xin = x_q8 != nullptr
                            ? static_cast<const char*>(x_q8) +
                                  static_cast<std::size_t>(t0) *
                                      flint_mmq::q8_1_vec_bytes(1, hidden)
                            : xq;
      if (x_q8 == nullptr) {
        err = Status(flint_mmq::quantize_vec(
            x + static_cast<std::size_t>(t0) * hidden, xq, n, hidden, stream));
      }
      if (err == hipSuccess && n == 1 && e.gate.type == e.up.type) {
        err = Status(flint_mmq::moe_vec(Type(e.up), e.up.data, e.gate.data, xin,
                                        chunk_ids, h, inner, hidden, 1, used,
                                        e.experts, false, stream));
      } else if (err == hipSuccess) {
        err = Status(flint_mmq::moe_vec(Type(e.gate), e.gate.data, nullptr, xin,
                                        chunk_ids, h, inner, hidden, n, used,
                                        e.experts, false, stream));
        if (err == hipSuccess) {
          err = Status(flint_mmq::moe_vec(
              Type(e.up), e.up.data, nullptr, xin, chunk_ids, u + s0 * inner,
              inner, hidden, n, used, e.experts, false, stream));
        }
        if (err == hipSuccess) {
          err =
              LaunchSwiGlu(h, u + s0 * inner, h,
                           static_cast<std::size_t>(n) * used * inner, stream);
        }
      }
      if (err == hipSuccess) {
        err = Status(flint_mmq::quantize_vec(h, xq, n * used, inner, stream));
      }
      if (err == hipSuccess) {
        err = Status(flint_mmq::moe_vec(Type(e.down), e.down.data, nullptr, xq,
                                        chunk_ids, expert_out + s0 * hidden,
                                        hidden, inner, n, used, e.experts, true,
                                        stream));
      }
    }
    return err;
  }
}

}  // namespace

namespace {

hipError_t RoutedForwardImpl(const RoutedExperts& e, const float* x,
                             const std::int32_t* ids, int tokens, int used,
                             float* expert_out, void* scratch,
                             std::size_t scratch_bytes, hipStream_t stream,
                             bool matrix) {
  if (!ValidExperts(e) || x == nullptr || ids == nullptr ||
      expert_out == nullptr || tokens <= 0 || used <= 0 ||
      scratch_bytes < RoutedBytes(e, tokens, used, matrix)) {
    return hipErrorInvalidValue;
  }
  const int hidden = e.gate.cols;
  const int inner = e.gate.rows;
  const std::size_t slots = static_cast<std::size_t>(tokens) * used;
  Carver carve(scratch, scratch_bytes);
  hipError_t err = hipSuccess;
  if (UseVector(tokens, matrix)) {
    return RoutedVector(e, x, nullptr, ids, tokens, used, expert_out, scratch,
                        scratch_bytes, stream);
  }
  auto* gather = carve.Take<std::int32_t>(slots * sizeof(std::int32_t));
  auto* scatter = carve.Take<std::int32_t>(slots * sizeof(std::int32_t));
  auto* bounds =
      carve.Take<std::int32_t>((e.experts + 1) * sizeof(std::int32_t));
  float* g = carve.Take<float>(slots * inner * sizeof(float));
  float* u = carve.Take<float>(slots * inner * sizeof(float));
  const std::size_t xq_bytes = carve.remaining();
  void* xq = carve.Take<char>(xq_bytes);
  if (!carve.ok()) {
    return hipErrorInvalidValue;
  }
  const int rows = static_cast<int>(slots);
  err = Status(flint_mmq::moe_maps(ids, gather, scatter, bounds, tokens, used,
                                   e.experts, stream));
  if (err == hipSuccess) {
    err = Status(flint_mmq::quantize_mmq(Type(e.gate), x, gather, xq, xq_bytes,
                                         rows, hidden, stream));
  }
  if (err == hipSuccess) {
    err = Status(flint_mmq::moe_mmq(Type(e.gate), e.gate.data, xq, scatter,
                                    bounds, g, inner, hidden, rows, e.experts,
                                    tokens, stream));
  }
  if (err == hipSuccess && flint_mmq::mmq_layout(Type(e.up)) !=
                               flint_mmq::mmq_layout(Type(e.gate))) {
    err = Status(flint_mmq::quantize_mmq(Type(e.up), x, gather, xq, xq_bytes,
                                         rows, hidden, stream));
  }
  if (err == hipSuccess) {
    err = Status(flint_mmq::moe_mmq(Type(e.up), e.up.data, xq, scatter, bounds,
                                    u, inner, hidden, rows, e.experts, tokens,
                                    stream));
  }
  if (err == hipSuccess) {
    err = LaunchSwiGlu(g, u, g, slots * inner, stream);
  }
  // Sorted slot i of the down pass reads hidden row scatter[i] (= t*used+j),
  // so the gate/up scatter map is also the down gather map.
  if (err == hipSuccess) {
    err = Status(flint_mmq::quantize_mmq(Type(e.down), g, scatter, xq, xq_bytes,
                                         rows, inner, stream));
  }
  if (err == hipSuccess) {
    err = Status(flint_mmq::moe_mmq(Type(e.down), e.down.data, xq, scatter,
                                    bounds, expert_out, hidden, inner, rows,
                                    e.experts, tokens, stream));
  }
  return err;
}

}  // namespace

hipError_t RoutedForward(const RoutedExperts& e, const float* x,
                         const std::int32_t* ids, int tokens, int used,
                         float* expert_out, void* scratch,
                         std::size_t scratch_bytes, hipStream_t stream) {
  return RoutedForwardImpl(e, x, ids, tokens, used, expert_out, scratch,
                           scratch_bytes, stream, false);
}

hipError_t RoutedForwardMatrix(const RoutedExperts& e, const float* x,
                               const std::int32_t* ids, int tokens, int used,
                               float* expert_out, void* scratch,
                               std::size_t scratch_bytes, hipStream_t stream) {
  return RoutedForwardImpl(e, x, ids, tokens, used, expert_out, scratch,
                           scratch_bytes, stream, true);
}

hipError_t RoutedForwardQuantized(const RoutedExperts& experts,
                                  const void* x_q8, const std::int32_t* ids,
                                  int tokens, int used, float* expert_out,
                                  void* scratch, std::size_t scratch_bytes,
                                  hipStream_t stream) {
  if (!ValidExperts(experts) || x_q8 == nullptr || ids == nullptr ||
      expert_out == nullptr || tokens <= 0 || tokens > kVectorRows ||
      used <= 0 || scratch_bytes < RoutedScratchBytes(experts, tokens, used)) {
    return hipErrorInvalidValue;
  }
  return RoutedVector(experts, nullptr, x_q8, ids, tokens, used, expert_out,
                      scratch, scratch_bytes, stream);
}

hipError_t MoeCombine(const float* expert_out, const float* weights,
                      const float* shared, const float* residual, float* out,
                      int tokens, int used, int hidden, hipStream_t stream) {
  if (expert_out == nullptr || weights == nullptr || out == nullptr ||
      tokens <= 0 || used <= 0 || hidden <= 0) {
    return hipErrorInvalidValue;
  }
  const dim3 grid((hidden + 255) / 256, tokens);
  MoeCombineKernel<<<grid, 256, 0, stream>>>(expert_out, weights, shared,
                                             residual, out, used, hidden);
  return LaunchStatus();
}

std::size_t RouterScratchBytes(int tokens, int experts) {
  return AlignUp(static_cast<std::size_t>(tokens) * experts * sizeof(float));
}

hipError_t RouterTopK(const std::uint16_t* weight_bf16, const float* x,
                      const float* bias, float scale, int tokens, int hidden,
                      int experts, int used, std::int32_t* ids, float* weights,
                      void* scratch, std::size_t scratch_bytes,
                      hipStream_t stream) {
  if (weight_bf16 == nullptr || x == nullptr || ids == nullptr ||
      weights == nullptr || scratch == nullptr || tokens <= 0 || hidden <= 0 ||
      hidden % 256 != 0 || experts <= 0 ||
      experts > kRouterMaxPerLane * kWarp || used <= 0 || used > experts ||
      scratch_bytes < RouterScratchBytes(tokens, experts)) {
    return hipErrorInvalidValue;
  }
  auto* scores = static_cast<float*>(scratch);
  if (tokens == 1) {
    constexpr int kE = 1;
    constexpr int kT = 1;
    constexpr int kWarps = 2;
    const dim3 grid((experts + kWarps * kE - 1) / (kWarps * kE), 1);
    RouterScoresKernel<kE, kT, kWarps><<<grid, kWarps * kWarp, 0, stream>>>(
        weight_bf16, x, scores, tokens, hidden, experts);
  } else if (tokens <= 8) {
    constexpr int kE = 1;
    constexpr int kT = 8;
    constexpr int kWarps = 4;
    const dim3 grid((experts + kWarps * kE - 1) / (kWarps * kE),
                    (tokens + kT - 1) / kT);
    RouterScoresKernel<kE, kT, kWarps><<<grid, kWarps * kWarp, 0, stream>>>(
        weight_bf16, x, scores, tokens, hidden, experts);
  } else {
    constexpr int kE = 4;
    constexpr int kT = 16;
    constexpr int kWarps = 8;
    const dim3 grid((experts + kWarps * kE - 1) / (kWarps * kE),
                    (tokens + kT - 1) / kT);
    RouterScoresKernel<kE, kT, kWarps><<<grid, kWarps * kWarp, 0, stream>>>(
        weight_bf16, x, scores, tokens, hidden, experts);
  }
  hipError_t err = LaunchStatus();
  if (err != hipSuccess) {
    return err;
  }
  RouterTopKKernel<<<tokens, kWarp, 0, stream>>>(scores, bias, scale, experts,
                                                 used, ids, weights);
  return LaunchStatus();
}

hipError_t EmbedQ8_0(const void* table, const std::int32_t* tokens, int count,
                     int hidden, float* out, hipStream_t stream) {
  if (table == nullptr || tokens == nullptr || out == nullptr || count <= 0 ||
      hidden <= 0 || hidden % 32 != 0) {
    return hipErrorInvalidValue;
  }
  EmbedQ8_0Kernel<<<count, 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(table), tokens, hidden, out);
  return LaunchStatus();
}

hipError_t RmsNorm(const float* x, const float* w, float* out, int rows,
                   int dim, float eps, hipStream_t stream) {
  if (x == nullptr || w == nullptr || out == nullptr || rows <= 0 || dim <= 0) {
    return hipErrorInvalidValue;
  }
  RmsNormKernel<false><<<rows, 256, 0, stream>>>(nullptr, nullptr, x, w, out,
                                                 nullptr, dim, eps);
  return LaunchStatus();
}

hipError_t RmsNorm(const float* x, const float* w, float* out, void* out_q8,
                   int rows, int dim, float eps, hipStream_t stream) {
  if (x == nullptr || w == nullptr || out == nullptr || out_q8 == nullptr ||
      rows <= 0 || dim <= 0 || dim % 32 != 0) {
    return hipErrorInvalidValue;
  }
  RmsNormKernel<false><<<rows, 256, 0, stream>>>(
      nullptr, nullptr, x, w, out, static_cast<Q8_1Block*>(out_q8), dim, eps);
  return LaunchStatus();
}

hipError_t AddRmsNorm(float* residual, const float* delta, const float* w,
                      float* out, int rows, int dim, float eps,
                      hipStream_t stream) {
  if (residual == nullptr || delta == nullptr || w == nullptr ||
      out == nullptr || rows <= 0 || dim <= 0) {
    return hipErrorInvalidValue;
  }
  RmsNormKernel<true><<<rows, 256, 0, stream>>>(residual, delta, nullptr, w,
                                                out, nullptr, dim, eps);
  return LaunchStatus();
}

hipError_t AddRmsNorm(float* residual, const float* delta, const float* w,
                      float* out, void* out_q8, int rows, int dim, float eps,
                      hipStream_t stream) {
  if (residual == nullptr || delta == nullptr || w == nullptr ||
      out == nullptr || out_q8 == nullptr || rows <= 0 || dim <= 0 ||
      dim % 32 != 0) {
    return hipErrorInvalidValue;
  }
  RmsNormKernel<true>
      <<<rows, 256, 0, stream>>>(residual, delta, nullptr, w, out,
                                 static_cast<Q8_1Block*>(out_q8), dim, eps);
  return LaunchStatus();
}

std::size_t VectorActivationBytes(int rows, int cols) {
  return flint_mmq::q8_1_vec_bytes(rows, cols);
}

hipError_t PrepareAttention(float* q, const float* k, const float* v,
                            const float* q_norm, const float* k_norm,
                            const std::int32_t* positions,
                            const std::int32_t* row_seq, const KvSequence* seqs,
                            int rows, bool rope, float theta, float eps,
                            hipStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || q_norm == nullptr ||
      k_norm == nullptr || positions == nullptr || seqs == nullptr ||
      rows <= 0) {
    return hipErrorInvalidValue;
  }
  PrepareAttentionKernel<<<rows, 256, 0, stream>>>(
      q, kQHeads * kHeadDim, q, k, kKvHeads * kHeadDim, v, kKvHeads * kHeadDim,
      q_norm, k_norm, positions, row_seq, seqs, rope,
      rope ? MakeRopeFrequencies(theta) : RopeFrequencies{}, eps);
  return LaunchStatus();
}

hipError_t PrepareAttention(const float* qkv, int qkv_stride, float* q_out,
                            const float* q_norm, const float* k_norm,
                            const std::int32_t* positions,
                            const std::int32_t* row_seq, const KvSequence* seqs,
                            int rows, bool rope, float theta, float eps,
                            hipStream_t stream) {
  constexpr int kQkvWidth = (kQHeads + 2 * kKvHeads) * kHeadDim;
  if (qkv == nullptr || qkv_stride < kQkvWidth || q_out == nullptr ||
      q_norm == nullptr || k_norm == nullptr || positions == nullptr ||
      seqs == nullptr || rows <= 0) {
    return hipErrorInvalidValue;
  }
  constexpr int kOffsetK = kQHeads * kHeadDim;
  constexpr int kOffsetV = kOffsetK + kKvHeads * kHeadDim;
  PrepareAttentionKernel<<<rows, 256, 0, stream>>>(
      qkv, qkv_stride, q_out, qkv + kOffsetK, qkv_stride, qkv + kOffsetV,
      qkv_stride, q_norm, k_norm, positions, row_seq, seqs, rope,
      rope ? MakeRopeFrequencies(theta) : RopeFrequencies{}, eps);
  return LaunchStatus();
}

std::size_t AttentionDecodeScratchBytes(int rows) {
  const std::size_t parts =
      static_cast<std::size_t>(rows) * kQHeads * kDecodeMaxChunks;
  return AlignUp(parts * kHeadDim * sizeof(float)) +
         AlignUp(parts * 2 * sizeof(float));
}

namespace {

hipError_t AttentionDecodeImpl(const float* q, const std::int32_t* positions,
                               const std::int32_t* row_seq,
                               const KvSequence* seqs, const float* sinks,
                               int window, float* out, Q8_1Block* out_q8,
                               int rows, void* scratch,
                               std::size_t scratch_bytes, hipStream_t stream) {
  if (q == nullptr || positions == nullptr || seqs == nullptr ||
      sinks == nullptr || out == nullptr || rows <= 0 || window < 0 ||
      scratch_bytes < AttentionDecodeScratchBytes(rows)) {
    return hipErrorInvalidValue;
  }
  Carver carve(scratch, scratch_bytes);
  const std::size_t parts =
      static_cast<std::size_t>(rows) * kQHeads * kDecodeMaxChunks;
  float* part_o = carve.Take<float>(parts * kHeadDim * sizeof(float));
  float* part_ml = carve.Take<float>(parts * 2 * sizeof(float));
  if (!carve.ok()) {
    return hipErrorInvalidValue;
  }
  // Chunks hold at least 32 positions, so a window bounds the chunk count
  // and the grid need not carry blocks that would exit at once.
  const int max_chunks = window > 0
                             ? std::min(kDecodeMaxChunks, (window + 31) / 32)
                             : kDecodeMaxChunks;
  const dim3 grid(max_chunks, kKvHeads, rows);
  AttentionDecodeChunkKernel<<<grid, kDecodeThreads, 0, stream>>>(
      q, positions, row_seq, seqs, window, part_o, part_ml);
  hipError_t err = LaunchStatus();
  if (err != hipSuccess) {
    return err;
  }
  AttentionDecodeMergeKernel<<<dim3(rows, kQHeads), kHeadDim, 0, stream>>>(
      positions, sinks, window, part_o, part_ml, out, out_q8);
  return LaunchStatus();
}

}  // namespace

hipError_t AttentionDecode(const float* q, const std::int32_t* positions,
                           const std::int32_t* row_seq, const KvSequence* seqs,
                           const float* sinks, int window, float* out, int rows,
                           void* scratch, std::size_t scratch_bytes,
                           hipStream_t stream) {
  return AttentionDecodeImpl(q, positions, row_seq, seqs, sinks, window, out,
                             nullptr, rows, scratch, scratch_bytes, stream);
}

hipError_t AttentionDecode(const float* q, const std::int32_t* positions,
                           const std::int32_t* row_seq, const KvSequence* seqs,
                           const float* sinks, int window, float* out,
                           void* out_q8, int rows, void* scratch,
                           std::size_t scratch_bytes, hipStream_t stream) {
  if (out_q8 == nullptr) {
    return hipErrorInvalidValue;
  }
  return AttentionDecodeImpl(q, positions, row_seq, seqs, sinks, window, out,
                             static_cast<Q8_1Block*>(out_q8), rows, scratch,
                             scratch_bytes, stream);
}

hipError_t AttentionPrefill(const float* q, int first_position,
                            const KvSequence& seq, const float* sinks,
                            int window, float* out, int rows,
                            hipStream_t stream) {
  if (q == nullptr || sinks == nullptr || out == nullptr || rows <= 0 ||
      first_position < 0 || window < 0 || seq.k == nullptr ||
      seq.v == nullptr || seq.capacity <= 0) {
    return hipErrorInvalidValue;
  }
  const int last = first_position + rows - 1;
  const dim3 grid(
      last / kPrefillPositions - first_position / kPrefillPositions + 1,
      kKvHeads);
  AttentionPrefillKernel<<<grid, 256, 0, stream>>>(q, first_position, rows, seq,
                                                   sinks, window, out);
  return LaunchStatus();
}

}  // namespace gufo::models::flint::rocm
