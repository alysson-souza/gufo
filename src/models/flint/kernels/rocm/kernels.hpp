#ifndef GUFO_MODELS_FLINT_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_FLINT_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

/// Flint's model-private HIP primitives.
///
/// Every launcher is asynchronous on `stream`, performs no allocation or
/// host synchronization (so it can be captured in a HIP graph) and returns
/// hipErrorInvalidValue for unsupported shapes, otherwise the launch status.
/// Pointers are device memory; scratch must be 256-byte aligned (as
/// hipMalloc returns). Activations are row-major FP32
/// [rows][features]; weights are the GGUF encodings, uploaded unchanged.
/// Results are deterministic: a row's output depends only on that row's
/// inputs and, where noted, on whether more than kVectorRows rows share the
/// launch. Up to kVectorRows rows, every operator's per-row result is
/// bitwise the same for any row count (flint.batch_invariance_ops), so
/// batched decode replays single-row decode exactly.
namespace gufo::models::flint::rocm {

inline constexpr int kHidden = 2048;
inline constexpr int kQHeads = 32;
inline constexpr int kKvHeads = 4;
inline constexpr int kHeadDim = 128;
/// Keys a sliding layer sees: the current position plus 512 previous.
inline constexpr int kSlidingWindow = 513;

/// GGUF weight types the projections accept; values are ggml type ids.
enum class WeightType : int {
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kIQ3_S = 21,
  kIQ4_XS = 23,
};

/// The WeightType for a ggml type id, or nullopt if unsupported.
std::optional<WeightType> WeightTypeFromGgml(std::uint32_t ggml_type);

/// Encoded bytes of one row of `cols` inputs.
std::size_t WeightRowBytes(WeightType type, int cols);

/// A quantized matrix of `rows` outputs by `cols` inputs (GGUF ne1 x ne0),
/// row-major GGUF blocks. `cols` must be a multiple of 256. A routed tensor
/// stacks its experts contiguously: expert e starts at
/// data + e * rows * WeightRowBytes(type, cols).
struct QuantMatrix {
  const void* data = nullptr;
  WeightType type = WeightType::kQ8_0;
  int rows = 0;
  int cols = 0;
};

// ---------------------------------------------------------------------------
// Dense projections
//
// y[r][o] = sum_i W[o][i] * x[r][i]. Activations are quantized to Q8_1
// (32-value blocks, scale amax/127, rounded codes; the scale and the block
// sum are stored as FP16, so a block's |sum x| must stay below 65504) in
// scratch, then multiplied with integer dot products and FP32 scales:
// llama.cpp's MMVQ kernels for rows <= 8 and MMQ tiles above. MMQ holds the
// Q4_K/Q5_K sub-block scale d*sc and min dmin*m as FP16 (relative 2^-11).
// Rows 1..kVectorRows give bitwise identical per-row results (the MMVQ tier
// is built without fast-math or contraction); above kVectorRows a row's value
// may differ in the last bits.

/// Rows handled by the vector (decode) kernels; larger counts use MMQ.
inline constexpr int kVectorRows = 8;

/// Scratch for Project/ProjectShared over `rows` inputs of `cols` values.
std::size_t ProjectionScratchBytes(int rows, int cols);

/// y [rows][w.rows] = x [rows][w.cols] projected by `w`.
hipError_t Project(const QuantMatrix& w, const float* x, float* y, int rows,
                   void* scratch, std::size_t scratch_bytes,
                   hipStream_t stream);

/// One output of ProjectShared.
struct ProjectionTarget {
  QuantMatrix w;
  float* y = nullptr;  ///< [rows][w.rows]
};

/// Several projections of the same x (e.g. q/k/v): the activation is
/// quantized once per distinct Q8_1 layout. Every target's w.cols == cols.
hipError_t ProjectShared(std::span<const ProjectionTarget> targets,
                         const float* x, int rows, int cols, void* scratch,
                         std::size_t scratch_bytes, hipStream_t stream);

/// Scratch for ProjectSwiGlu with `inner` = gate.rows.
std::size_t SwiGluScratchBytes(int rows, int cols, int inner);

/// y [rows][inner] = silu(gate . x) * (up . x). One row uses the fused
/// vector kernel; more rows stage both projections in scratch.
hipError_t ProjectSwiGlu(const QuantMatrix& gate, const QuantMatrix& up,
                         const float* x, float* y, int rows, void* scratch,
                         std::size_t scratch_bytes, hipStream_t stream);

// Pre-quantized activations (decode). Up to kVectorRows rows can be
// quantized once by their producer (RmsNorm, AddRmsNorm, AttentionDecode
// overloads with `out_q8`) into the vector kernels' Q8_1 layout and fed to
// several projections. The bytes equal what the float-input functions
// quantize internally, so results are bit-identical to them.

/// Bytes of `rows` Q8_1 rows of `cols` values (each row padded to 512).
std::size_t VectorActivationBytes(int rows, int cols);

/// Project over pre-quantized rows (rows <= kVectorRows).
hipError_t ProjectQuantized(const QuantMatrix& w, const void* xq, float* y,
                            int rows, hipStream_t stream);

/// ProjectSwiGlu over pre-quantized rows (rows <= kVectorRows). One row of
/// equal-type gate/up needs no scratch; otherwise SwiGluScratchBytes.
hipError_t ProjectSwiGluQuantized(const QuantMatrix& gate,
                                  const QuantMatrix& up, const void* xq,
                                  float* y, int rows, void* scratch,
                                  std::size_t scratch_bytes,
                                  hipStream_t stream);

/// out[i] = silu(gate[i]) * up[i] with silu(g) = g / (1 + exp(-g)), FP32.
hipError_t SwiGlu(const float* gate, const float* up, float* out,
                  std::size_t count, hipStream_t stream);

// ---------------------------------------------------------------------------
// Routed experts

/// Stacked routed experts: gate/up have `rows` = expert dim (512) and
/// `cols` = hidden; down has rows = hidden, cols = expert dim.
struct RoutedExperts {
  QuantMatrix gate;
  QuantMatrix up;
  QuantMatrix down;
  int experts = 0;
};

/// Scratch for RoutedProject.
std::size_t RoutedProjectScratchBytes(const QuantMatrix& w, int experts,
                                      int tokens, int used,
                                      bool per_slot_input);

/// One routed projection: out [tokens][used][w.rows] = W_e . x with
/// e = ids[t][j] and x = x[t] ([tokens][w.cols]) or, with per_slot_input,
/// x[t][j] ([tokens][used][w.cols]). `w` stacks `experts` matrices; ids must
/// name valid experts, distinct within a token. Same kernels and row-count
/// behaviour as RoutedForward's unfused stages.
hipError_t RoutedProject(const QuantMatrix& w, int experts, const float* x,
                         bool per_slot_input, const std::int32_t* ids,
                         int tokens, int used, float* out, void* scratch,
                         std::size_t scratch_bytes, hipStream_t stream);

/// Scratch for RoutedForward over `tokens` rows with `used` slots each.
std::size_t RoutedScratchBytes(const RoutedExperts& experts, int tokens,
                               int used);

/// expert_out [tokens][used][hidden] = down_e(silu(gate_e x) * up_e x) for
/// e = ids[t][j]. ids [tokens][used] must name valid experts, distinct within
/// a token (as top-k routing produces). Up to kVectorRows tokens run the
/// MMVQ kernels in groups of the type's routed-vector limit (4 for K-quants
/// and IQ3_S, 6 for IQ4_XS, 8 for Q8_0), a single token with gate and up of
/// one type fusing silu(gate) * up into one kernel; more tokens run the
/// expert-sorted MMQ path. The silu(gate) * up rows are re-quantized to
/// Q8_1 for the down projection. Per-row results are identical for 1 to
/// kVectorRows tokens (fused or grouped) and differ in the last bits above.
hipError_t RoutedForward(const RoutedExperts& experts, const float* x,
                         const std::int32_t* ids, int tokens, int used,
                         float* expert_out, void* scratch,
                         std::size_t scratch_bytes, hipStream_t stream);

/// RoutedForward over pre-quantized token rows (tokens <= kVectorRows).
hipError_t RoutedForwardQuantized(const RoutedExperts& experts,
                                  const void* x_q8, const std::int32_t* ids,
                                  int tokens, int used, float* expert_out,
                                  void* scratch, std::size_t scratch_bytes,
                                  hipStream_t stream);

// Matrix-tier entry points (prefill). Same results as the functions above
// at more than kVectorRows rows, for any row count: a row's output is
// bitwise independent of how many rows share the call, so prefill can be
// split into chunks of any size (including 1..kVectorRows rows) without
// changing any row.

std::size_t MatrixProjectionScratchBytes(int rows, int cols);
hipError_t ProjectMatrix(const QuantMatrix& w, const float* x, float* y,
                         int rows, void* scratch, std::size_t scratch_bytes,
                         hipStream_t stream);
hipError_t ProjectSharedMatrix(std::span<const ProjectionTarget> targets,
                               const float* x, int rows, int cols,
                               void* scratch, std::size_t scratch_bytes,
                               hipStream_t stream);
std::size_t MatrixSwiGluScratchBytes(int rows, int cols, int inner);
hipError_t ProjectSwiGluMatrix(const QuantMatrix& gate, const QuantMatrix& up,
                               const float* x, float* y, int rows,
                               void* scratch, std::size_t scratch_bytes,
                               hipStream_t stream);
std::size_t MatrixRoutedScratchBytes(const RoutedExperts& experts, int tokens,
                                     int used);
hipError_t RoutedForwardMatrix(const RoutedExperts& experts, const float* x,
                               const std::int32_t* ids, int tokens, int used,
                               float* expert_out, void* scratch,
                               std::size_t scratch_bytes, hipStream_t stream);

/// out[t][d] = residual[t][d] + (shared[t][d] + sum_j w[t][j] *
/// expert_out[t][j][d]), the sum taken as an FMA chain over j ascending from
/// zero. `shared` and `residual` may be null (treated as zero); `out` may
/// alias `residual`.
hipError_t MoeCombine(const float* expert_out, const float* weights,
                      const float* shared, const float* residual, float* out,
                      int tokens, int used, int hidden, hipStream_t stream);

// ---------------------------------------------------------------------------
// Router

/// Scratch for RouterTopK: FP32 scores [tokens][experts].
std::size_t RouterScratchBytes(int tokens, int experts);

/// Sigmoid top-k routing. logits = W x with W BF16 [experts][hidden]
/// (GGUF ffn_gate_inp) and FP32 accumulation; scores = 1 / (1 + exp(-logit))
/// are left in scratch. Experts are chosen by score + bias[e] (bias may be
/// null), highest first, ties broken by the lower expert index; a NaN key
/// ranks as -inf. ids [tokens][used] lists them in that order and
/// weights[t][j] = score_j / (sum_j score_j + 1e-20) * scale, using the
/// unbiased scores summed in slot order. Results do not depend on `tokens`.
/// Requires experts <= 1024, used <= experts, hidden % 256 == 0.
hipError_t RouterTopK(const std::uint16_t* weight_bf16, const float* x,
                      const float* bias, float scale, int tokens, int hidden,
                      int experts, int used, std::int32_t* ids, float* weights,
                      void* scratch, std::size_t scratch_bytes,
                      hipStream_t stream);

// ---------------------------------------------------------------------------
// Elementwise

/// out[t][d] = fp16(d_b) * q for Q8_0 table rows of `hidden` values.
hipError_t EmbedQ8_0(const void* table, const std::int32_t* tokens, int count,
                     int hidden, float* out, hipStream_t stream);

/// out[r] = x[r] * w / sqrt(mean(x[r]^2) + eps), FP32, fixed reduction tree.
hipError_t RmsNorm(const float* x, const float* w, float* out, int rows,
                   int dim, float eps, hipStream_t stream);

/// residual[r] += delta[r]; out[r] = RmsNorm(residual[r]). Same arithmetic
/// as RmsNorm on the updated residual.
hipError_t AddRmsNorm(float* residual, const float* delta, const float* w,
                      float* out, int rows, int dim, float eps,
                      hipStream_t stream);

/// RmsNorm that also writes `out` as Q8_1 rows (VectorActivationBytes;
/// dim % 32 == 0).
hipError_t RmsNorm(const float* x, const float* w, float* out, void* out_q8,
                   int rows, int dim, float eps, hipStream_t stream);

/// AddRmsNorm that also writes `out` as Q8_1 rows.
hipError_t AddRmsNorm(float* residual, const float* delta, const float* w,
                      float* out, void* out_q8, int rows, int dim, float eps,
                      hipStream_t stream);

// ---------------------------------------------------------------------------
// Attention
//
// KV storage, per layer and sequence, owned by the engine: separate K and V
// arrays of FP16 [kKvHeads][capacity][kHeadDim] (head-major: each KV head's
// positions are contiguous, 256 bytes per position, so decode streams each
// head linearly). Position p lives in slot p % capacity of every head.
// Full-attention layers use capacity >= context, so slots equal positions.
// Sliding layers use a ring of at least SlidingRingCapacity(rows) slots,
// rows being the largest number of rows written for one sequence by a single
// PrepareAttention call (the prefill chunk): a chunk then never overwrites
// keys that AttentionPrefill still reads for it (it reads from the start of
// each row's absolute 16-position tile, up to 15 positions before the
// chunk).

/// Minimum ring capacity of a sliding layer for chunks of up to `rows` rows.
constexpr int SlidingRingCapacity(int rows) {
  return kSlidingWindow + 14 + rows;
}

/// One sequence's cache for one layer (device-resident when passed by
/// array).
struct KvSequence {
  __half* k = nullptr;
  __half* v = nullptr;
  std::int32_t capacity = 0;
};

/// Per-head RMSNorm of q [rows][32][128] (in place) and of k [rows][4][128]
/// with the [128] weights, then NeoX RoPE on both when `rope` (pairs
/// (i, i + 64), angle = fp32(position) * fp32(theta^(-i/64)) rounded to
/// FP32, accurate FP32 sin/cos of that angle). Writes the
/// normalized/rotated k and the raw v [rows][4][128] as FP16 (round to
/// nearest even) into seqs[row_seq[r]] at slot positions[r] % capacity.
/// `seqs` is a device array; `row_seq` may be null (all rows sequence 0).
hipError_t PrepareAttention(float* q, const float* k, const float* v,
                            const float* q_norm, const float* k_norm,
                            const std::int32_t* positions,
                            const std::int32_t* row_seq, const KvSequence* seqs,
                            int rows, bool rope, float theta, float eps,
                            hipStream_t stream);

/// PrepareAttention over fused projection rows: qkv [rows][qkv_stride]
/// holds q (32 x 128) at offset 0, k (4 x 128) at 4096 and v (4 x 128) at
/// 4608, qkv_stride >= 5120. The normalized/rotated q goes to q_out
/// [rows][32][128]; K/V are written as above. Same arithmetic.
hipError_t PrepareAttention(const float* qkv, int qkv_stride, float* q_out,
                            const float* q_norm, const float* k_norm,
                            const std::int32_t* positions,
                            const std::int32_t* row_seq, const KvSequence* seqs,
                            int rows, bool rope, float theta, float eps,
                            hipStream_t stream);

/// Split-K partitions per decode row; bounds the scratch below.
inline constexpr int kDecodeMaxChunks = 256;

/// Scratch for AttentionDecode over `rows` rows (about 4.1 MiB per row).
std::size_t AttentionDecodeScratchBytes(int rows);

/// Attention for independent rows, typically one per decoding sequence:
/// row r (query q [rows][32][128], position p = positions[r]) reads
/// seqs[row_seq[r]] over positions [lo, p], lo = 0 for full layers
/// (window == 0) or max(0, p - window + 1). With s_j = (q . k_j) /
/// sqrt(128) and m = max(max_j s_j, sinks[h]),
/// out = sum_j exp(s_j - m) v_j / (sum_j exp(s_j - m) + exp(sinks[h] - m)).
/// Arithmetic is FP32 in base 2: q is scaled once by fp32(log2(e) /
/// sqrt(128)), dot products are FMA chains over 32 dims joined by a 4-lane
/// tree, and weights are exp2 of score differences. The span is split into
/// at most kDecodeMaxChunks chunks of a multiple of 32 positions, chosen
/// from the span alone, then merged in chunk order with the sink last, so a
/// row's result does not depend on the other rows. The key for p must
/// already be written. out [rows][32][128] FP32.
hipError_t AttentionDecode(const float* q, const std::int32_t* positions,
                           const std::int32_t* row_seq, const KvSequence* seqs,
                           const float* sinks, int window, float* out, int rows,
                           void* scratch, std::size_t scratch_bytes,
                           hipStream_t stream);

/// AttentionDecode that also writes `out` as Q8_1 rows of 4096 values (the
/// o projection's input).
hipError_t AttentionDecode(const float* q, const std::int32_t* positions,
                           const std::int32_t* row_seq, const KvSequence* seqs,
                           const float* sinks, int window, float* out,
                           void* out_q8, int rows, void* scratch,
                           std::size_t scratch_bytes, hipStream_t stream);

/// Causal attention for `rows` consecutive positions first_position.. of
/// one sequence (prefill), with AttentionDecode's visibility and formula.
/// Runs on the WMMA cores: the scaled query and the probabilities enter the
/// products as FP16 (lifted by exact powers of two to stay normal), the
/// precision of the stored K and V; accumulation and the online softmax over
/// 32-key blocks are FP32. Measured against an FP64 oracle on the same FP16
/// cache the outputs differ by up to ~4e-4 * max|v| (AttentionDecode: ~1e-7).
/// Key blocks are aligned to absolute positions, so each row's result is
/// bitwise independent of first_position and rows (prefill split anywhere,
/// including single rows, gives identical rows). All keys up to the last row
/// must already be written.
hipError_t AttentionPrefill(const float* q, int first_position,
                            const KvSequence& seq, const float* sinks,
                            int window, float* out, int rows,
                            hipStream_t stream);

}  // namespace gufo::models::flint::rocm

#endif  // GUFO_MODELS_FLINT_KERNELS_ROCM_KERNELS_HPP_
