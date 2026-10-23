// SPDX-License-Identifier: MIT
// Raw-pointer adapter over Flint's vendored llama.cpp MMVQ/MMQ kernels.
//
// Weight types use ggml_type numbering (Q8_0 = 8, Q4_K = 12, Q5_K = 13,
// Q6_K = 14, IQ3_S = 21, IQ4_XS = 23). Weights are GGUF rows: `m` rows of `k`
// inputs, experts stacked contiguously. Every function enqueues work on
// `stream` only, allocates nothing and returns 0 on success, nonzero on an
// unsupported type or shape. Buffers are caller-owned device memory.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flint_mmq {

/// True for the six GGUF weight types Flint ships.
bool weight_type_supported(int type);

/// Inputs per row padding for quantized activations (MATRIX_ROW_PADDING).
int padded_k(int k);

/// Most rows the vector kernels take in one launch for a routed projection.
int moe_vec_max_rows(int type);

/// Bytes of `rows` activation rows in the vector (block_q8_1) layout.
std::size_t q8_1_vec_bytes(int rows, int k);

/// Bytes of `rows` activation rows in the MMQ layout, including the tail
/// slack the tile loader reads past the last row.
std::size_t q8_1_mmq_bytes(int rows, int k);

/// MMQ activation layouts differ by weight type; types reporting the same
/// value can share one quantized activation buffer.
int mmq_layout(int type);

/// x [rows][k] f32 -> block_q8_1 rows of padded_k(k) values (zero padded).
int quantize_vec(const float* x, void* xq, int rows, int k, hipStream_t stream);

/// x rows -> MMQ layout for `type`. With `gather`, output row r reads x row
/// gather[r]. Zeroes the tail slack so the final tile reads deterministic
/// zeros. `xq_bytes` must be at least q8_1_mmq_bytes(rows, k).
int quantize_mmq(int type, const float* x, const std::int32_t* gather,
                 void* xq, std::size_t xq_bytes, int rows, int k,
                 hipStream_t stream);

/// out[r][m] = W[m] . x[r] for rows <= 8 from quantize_vec rows. With a
/// non-null `gate` (rows == 1 only) out = (W . x) * silu(gate . x).
int dense_vec(int type, const void* w, const void* gate, const void* xq,
              float* out, int m, int rows, int k, hipStream_t stream);

/// out[r][m] = W[m] . x[r] from quantize_mmq rows.
int dense_mmq(int type, const void* w, const void* xq, float* out, int m,
              int rows, int k, hipStream_t stream);

/// Routed vector projection for tokens <= moe_vec_max_rows(type).
/// ids [tokens][used]; out [tokens][used][m]. With `per_slot_input` the
/// input holds one quantize_vec row per (token, slot), otherwise one per
/// token. A non-null `gate` (tokens == 1 only) fuses up * silu(gate).
/// Negative ids write zeros.
int moe_vec(int type, const void* w, const void* gate, const void* xq,
            const std::int32_t* ids, float* out, int m, int k, int tokens,
            int used, int experts, bool per_slot_input, hipStream_t stream);

/// Expert-sorted maps for routed MMQ: ids [tokens][used] ->
/// gather [tokens*used] (input row per sorted slot), scatter [tokens*used]
/// (token*used + slot per sorted slot) and bounds [experts + 1].
int moe_maps(const std::int32_t* ids, std::int32_t* gather,
             std::int32_t* scatter, std::int32_t* bounds, int tokens,
             int used, int experts, hipStream_t stream);

/// Routed MMQ over `rows` = tokens*used sorted slots: xq holds the rows
/// quantize_mmq gathered with `gather`; out[scatter[i]][m] receives each
/// slot's projection. `max_expert_rows` bounds one expert's bucket (the
/// token count when a token never repeats an expert) and only sizes the
/// launch grid; the column tile follows the mean bucket. Neither changes
/// any output value.
int moe_mmq(int type, const void* w, const void* xq,
            const std::int32_t* scatter, const std::int32_t* bounds,
            float* out, int m, int k, int rows, int experts,
            int max_expert_rows, hipStream_t stream);

}  // namespace flint_mmq
