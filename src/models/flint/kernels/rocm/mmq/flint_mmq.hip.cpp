// SPDX-License-Identifier: MIT
// Flint adapter over the vendored MMVQ/MMQ templates; see flint_mmq.h.
#include "flint_mmq_prelude.h"
namespace flint_mmq {
#include "common.hip.hpp"
#include "mmid.hip.hpp"
#include "mmq.hip.hpp"
#include "mmvq.hip.hpp"
#include "quantize.hip.hpp"
}  // namespace flint_mmq

#include "flint_mmq.h"

namespace flint_mmq {
namespace {

int device_cc() {
  return flint_ggml_hip_info().devices[flint_ggml_hip_get_device()].cc;
}

// One context per device. Its pool is only reached by stream-K, which
// RDNA never selects, so the adapter itself performs no allocation.
flint_ggml_hip_context& context() {
  static flint_ggml_hip_context* contexts[FLINT_GGML_HIP_MAX_DEVICES] = {};
  const int device = flint_ggml_hip_get_device();
  if (contexts[device] == nullptr) {
    contexts[device] = new flint_ggml_hip_context(device);
  }
  return *contexts[device];
}

int launch_status() {
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

bool valid_shape(int type, int m, int rows, int k) {
  return weight_type_supported(type) && m > 0 && rows > 0 && k > 0 &&
         k % 256 == 0;
}

std::int64_t blocks_per_row(int type, int k) {
  return k / ggml_blck_size(static_cast<ggml_type>(type));
}

void run_mmq(int type, const mmq_args& args, hipStream_t stream) {
  flint_ggml_hip_context& ctx = context();
  switch (type) {
    case GGML_TYPE_Q8_0:
      mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
      break;
    case GGML_TYPE_Q4_K:
      mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
      break;
    case GGML_TYPE_Q5_K:
      mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
      break;
    case GGML_TYPE_Q6_K:
      mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
      break;
    case GGML_TYPE_IQ4_XS:
      mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
      break;
    case GGML_TYPE_IQ3_S:
      mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
      break;
    default:
      break;
  }
}

}  // namespace

bool weight_type_supported(int type) {
  switch (type) {
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_IQ3_S:
      return true;
    default:
      return false;
  }
}

int padded_k(int k) { return GGML_PAD(k, MATRIX_ROW_PADDING); }

int moe_vec_max_rows(int type) {
  return get_mmvq_mmid_max_batch(static_cast<ggml_type>(type), device_cc());
}

std::size_t q8_1_vec_bytes(int rows, int k) {
  return static_cast<std::size_t>(rows) * padded_k(k) / QK8_1 *
         sizeof(block_q8_1);
}

std::size_t q8_1_mmq_bytes(int rows, int k) {
  return static_cast<std::size_t>(rows) * padded_k(k) / (4 * QK8_1) *
             sizeof(block_q8_1_mmq) +
         static_cast<std::size_t>(get_mmq_x_max_host(device_cc())) *
             sizeof(block_q8_1_mmq);
}

int mmq_layout(int type) {
  return static_cast<int>(mmq_get_q8_1_ds_layout(static_cast<ggml_type>(type)));
}

int quantize_vec(const float* x, void* xq, int rows, int k,
                 hipStream_t stream) {
  if (x == nullptr || xq == nullptr || rows <= 0 || k <= 0 || k % QK8_1) {
    return -1;
  }
  flint_quantize_row_q8_1_hip(x, nullptr, xq, GGML_TYPE_Q8_0, k, k, 0, 0,
                              padded_k(k), rows, 1, 1, stream);
  return launch_status();
}

int quantize_mmq(int type, const float* x, const std::int32_t* gather,
                 void* xq, std::size_t xq_bytes, int rows, int k,
                 hipStream_t stream) {
  if (!weight_type_supported(type) || x == nullptr || xq == nullptr ||
      rows <= 0 || k <= 0 || k % 256 || xq_bytes < q8_1_mmq_bytes(rows, k)) {
    return -1;
  }
  const std::size_t data = static_cast<std::size_t>(rows) * padded_k(k) /
                           (4 * QK8_1) * sizeof(block_q8_1_mmq);
  if (hipMemsetAsync(static_cast<char*>(xq) + data, 0,
                     q8_1_mmq_bytes(rows, k) - data, stream) != hipSuccess) {
    return -2;
  }
  flint_quantize_mmq_q8_1_hip(x, gather, xq, static_cast<ggml_type>(type), k,
                              k, 0, 0, padded_k(k), rows, 1, 1, stream);
  return launch_status();
}

int dense_vec(int type, const void* w, const void* gate, const void* xq,
              float* out, int m, int rows, int k, hipStream_t stream) {
  if (!valid_shape(type, m, rows, k) || w == nullptr || xq == nullptr ||
      out == nullptr || rows > MMVQ_MAX_BATCH_SIZE ||
      (gate != nullptr && rows != 1)) {
    return -1;
  }
  flint_ggml_hip_mm_fusion_args_device fusion{};
  fusion.gate = gate;
  fusion.glu_op = GGML_GLU_OP_SWIGLU;
  const int row_blocks = static_cast<int>(blocks_per_row(type, k));
  const int y_stride = padded_k(k) / QK8_1;
  mul_mat_vec_q_switch_type(w, static_cast<ggml_type>(type), xq, nullptr,
                            fusion, out, k, m, rows, row_blocks, y_stride, m,
                            1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, stream);
  return launch_status();
}

int dense_mmq(int type, const void* w, const void* xq, float* out, int m,
              int rows, int k, hipStream_t stream) {
  if (!valid_shape(type, m, rows, k) || w == nullptr || xq == nullptr ||
      out == nullptr) {
    return -1;
  }
  const std::int64_t y_channel = static_cast<std::int64_t>(rows) *
                                 padded_k(k) * sizeof(block_q8_1) /
                                 (QK8_1 * sizeof(int));
  mmq_args args{};
  args.x = static_cast<const char*>(w);
  args.type_x = static_cast<ggml_type>(type);
  args.y = static_cast<const int*>(xq);
  args.dst = out;
  args.ncols_x = k;
  args.nrows_x = m;
  args.ncols_dst = rows;
  args.stride_row_x = blocks_per_row(type, k);
  args.ncols_y = rows;
  args.nrows_dst = m;
  args.nchannels_x = 1;
  args.nchannels_y = 1;
  args.stride_channel_y = y_channel;
  args.nsamples_x = 1;
  args.nsamples_y = 1;
  args.stride_sample_y = y_channel;
  args.use_stream_k = false;
  args.ncols_max = rows;
  run_mmq(type, args, stream);
  return launch_status();
}

int moe_vec(int type, const void* w, const void* gate, const void* xq,
            const std::int32_t* ids, float* out, int m, int k, int tokens,
            int used, int experts, bool per_slot_input, hipStream_t stream) {
  if (!valid_shape(type, m, tokens, k) || w == nullptr || xq == nullptr ||
      ids == nullptr || out == nullptr || used <= 0 || experts <= 0 ||
      tokens > moe_vec_max_rows(type) || (gate != nullptr && tokens != 1)) {
    return -1;
  }
  flint_ggml_hip_mm_fusion_args_device fusion{};
  fusion.gate = gate;
  fusion.glu_op = GGML_GLU_OP_SWIGLU;
  const int row_blocks = static_cast<int>(blocks_per_row(type, k));
  const int y_row = padded_k(k) / QK8_1;
  const int y_channels = per_slot_input ? used : 1;
  mul_mat_vec_q_switch_type(
      w, static_cast<ggml_type>(type), xq, ids, fusion, out, k, m, tokens,
      row_blocks, y_channels * y_row, used * m, experts, y_channels, used,
      m * row_blocks, y_row, m, 1, 1, 0, 0, 0, used, stream);
  return launch_status();
}

int moe_maps(const std::int32_t* ids, std::int32_t* gather,
             std::int32_t* scatter, std::int32_t* bounds, int tokens,
             int used, int experts, hipStream_t stream) {
  if (ids == nullptr || gather == nullptr || scatter == nullptr ||
      bounds == nullptr || tokens <= 0 || used <= 0 || experts <= 0) {
    return -1;
  }
  // Slots of dropped (negative) ids keep valid zero indices for the gather.
  const std::size_t slots = static_cast<std::size_t>(tokens) * used;
  if (hipMemsetAsync(gather, 0, slots * sizeof(std::int32_t), stream) !=
          hipSuccess ||
      hipMemsetAsync(scatter, 0, slots * sizeof(std::int32_t), stream) !=
          hipSuccess) {
    return -2;
  }
  flint_ggml_hip_launch_mm_ids_helper(ids, gather, scatter, bounds, experts,
                                      tokens, used, 1, used, 1, stream);
  return launch_status();
}

int moe_mmq(int type, const void* w, const void* xq,
            const std::int32_t* scatter, const std::int32_t* bounds,
            float* out, int m, int k, int rows, int experts,
            int max_expert_rows, hipStream_t stream) {
  if (!valid_shape(type, m, rows, k) || w == nullptr || xq == nullptr ||
      scatter == nullptr || bounds == nullptr || out == nullptr ||
      experts <= 0 || max_expert_rows <= 0) {
    return -1;
  }
  const std::int64_t row_blocks = blocks_per_row(type, k);
  const std::int64_t y_channel =
      static_cast<std::int64_t>(padded_k(k)) * sizeof(block_q8_1) /
      (QK8_1 * sizeof(int));
  mmq_args args{};
  args.x = static_cast<const char*>(w);
  args.type_x = static_cast<ggml_type>(type);
  args.y = static_cast<const int*>(xq);
  args.ids_dst = scatter;
  args.expert_bounds = bounds;
  args.dst = out;
  args.ncols_x = k;
  args.nrows_x = m;
  args.ncols_dst = rows;
  args.stride_row_x = row_blocks;
  args.ncols_y = rows;
  args.nrows_dst = m;
  args.nchannels_x = experts;
  args.nchannels_y = experts;
  args.stride_channel_x = static_cast<std::int64_t>(m) * row_blocks;
  args.stride_channel_y = y_channel;
  args.nsamples_x = 1;
  args.nsamples_y = 1;
  args.stride_sample_y = y_channel * rows;
  args.use_stream_k = false;
  args.ncols_max = rows;
  args.ncols_grid_max = std::min(rows, max_expert_rows);
  // Column tile ~4/3 of the mean bucket, in 16-column steps up to the
  // device maximum: measured on gfx1151 at 512 and 2048 tokens over 256
  // experts (16 and 64 columns beat the dense default of 80 by 35% / 6%).
  const int mean_bucket = (4 * rows + 3 * experts - 1) / (3 * experts);
  args.mmq_x_request = std::min(80, std::max(16, (mean_bucket + 15) / 16 * 16));
  run_mmq(type, args, stream);
  return launch_status();
}

}  // namespace flint_mmq
