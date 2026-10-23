#pragma once
#include "common.hip.hpp"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void flint_ggml_hip_op_abs(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_sgn(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_neg(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_step(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_gelu(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_silu(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_silu_back(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_gelu_erf(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_gelu_quick(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_tanh(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_relu(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_sigmoid(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_hardsigmoid(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_exp(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_hardswish(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_leaky_relu(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_sqr(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_sqrt(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_sin(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_cos(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_log(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_expm1(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_softplus(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_elu(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_floor(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_ceil(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_round(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_trunc(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_reglu(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_geglu(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_swiglu(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_swiglu_oai(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_geglu_erf(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_geglu_quick(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_xielu(flint_ggml_hip_context & ctx, ggml_tensor * dst);

void flint_ggml_hip_op_unary_mul(flint_ggml_hip_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);

void flint_ggml_hip_op_relu_sqr(flint_ggml_hip_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);

__device__ __forceinline__ float flint_ggml_hip_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float flint_ggml_hip_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float flint_ggml_hip_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}
