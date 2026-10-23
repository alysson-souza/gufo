# Quantized HIP kernels

Adapted from `ggml-org/llama.cpp` at
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT, The ggml authors; see
`licenses/llama.cpp.txt`), originally under `ggml/src/ggml-cuda`. The files
were copied from Gufo's DeepSeek V4 Flash import of the same pin
(`src/models/deepseek_v4_flash/kernels/rocm/mmq`), which already carries the
HIP-only conversion: native HIP runtime and numeric types, AMD WMMA/MFMA
fragments, no CUDA alias layer, NVIDIA/MUSA branches and ggml graph entry
points removed. Its local tuning in `mmq.hip.hpp` and `mmid.hip.cpp`
(comments marked `ds4`, MIT, see
`src/models/deepseek_v4_flash/LICENSE.ds4`) is retained unchanged; the SoA
weight loaders and the MoE epilogue fields it adds are not used by Flint.

Flint's changes:

- Every translation unit wraps the tier in `namespace flint_mmq` after
  `flint_mmq_prelude.h` pulls the system headers in at global scope, so this
  copy cannot clash with the other models' copies in one binary. Runtime
  symbols use the `flint_` prefix instead of `ds4_`.
- `mmvq.hip.cpp` dispatches only Flint's weight types (Q8_0, Q4_K, Q5_K,
  Q6_K, IQ4_XS, IQ3_S); `mmq_<type>.hip.cpp` instantiates `mul_mat_q_case`
  once per type.
- `mmid.hip.cpp` no longer logs the large-batch id-map path.
- The DS4 activation layout (Q4_K/Q5_K MMQ) stores each 32-value block sum
  times 2^-8 and the WMMA consumer scales it back exactly
  (`FLINT_MMQ_DS4_SUM_SCALE`): upstream's FP16 sum overflows past 65504,
  which real silu(gate) * up rows reach. Only the gfx1151 WMMA consumer is
  adjusted; the DP4A/NVIDIA consumers are not built for this target.
- The tier is compiled without fast-math and with FP contraction off
  (upstream and the other models use `-ffast-math`): every row-count, tile
  width and fused/unfused instance then evaluates a row identically
  (flint.batch_invariance_ops, flint.split_invariance_ops), and Flint's
  fused Q8_1 encoders reproduce `quantize.hip.cpp` bytes. Cost: prefill MMQ
  1-3% slower, decode MMVQ unchanged within noise.
- `flint_mmq.h` / `flint_mmq.hip.cpp` are Gufo's adapter: caller-owned
  scratch, no allocation, every launch on the caller's stream. Routed MMQ
  bounds its column-tile grid by the token count.

The public interface is `src/models/flint/kernels/rocm/kernels.hpp`; only
`kernels.hip.cpp` includes the adapter. When importing an upstream fix,
keep the namespace wrap and the adapter, update the pin here and in
`THIRD_PARTY_NOTICES.md`, and rerun `ctest -L flint -L hip` (the
`flint.projection_ops` and `flint.moe_ops` checks cover every type and path).
