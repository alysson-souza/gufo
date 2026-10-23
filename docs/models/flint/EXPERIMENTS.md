# Recursive Flint experiments

Initial gfx1151 bring-up, October 22, 2026. Kernel times are standalone
microbenchmarks with weights/KV rotated past the 32 MiB MALL; engine rates
are `gufo bench` on the same GGUF. DRAM read peak is 226 GB/s.

## Engine

| Experiment | Outcome |
| --- | --- |
| Stack q/k/v rows into one 5120×2048 matrix per layer | Retained: one projection and one activation quantization per layer; every row's arithmetic is unchanged and bit-identical. |
| Quantize-once decode: norms and attention emit Q8_1 consumed by the projections | Retained: 990 → 795 dispatches per token; producer-side encoding reproduces `quantize_q8_1` bytes. |
| One HIP graph per decode batch width 1–8 | Retained: the only host sync per step is the logits copy; host gap between steps ≈ 0.13 ms. |
| Prefill chunk 2048 versus 1024 | Retained 2048: pp2048 2600 vs 2207 tok/s, pp2048 at 16K 1677 vs 1552. 4096 was not tried: it doubles the sliding ring and scratch and pp2048 cannot use it. |
| Matrix (MMQ) tier for every prefill row count | Retained: required for split invariance. Before it, chunk 64 versus 2048 differed by KL 0.0042 (llama.cpp ubatch 64 vs 512: 0.0028). |
| Pinned double-buffered snapshot staging on a model-owned stream | Retained: a pageable save on a per-call stream took 99 ms while another session decoded and failed the batching check in 2 of 3 runs; 3/3 pass after. |
| Uninitialized snapshot storage | Retained: 977 MB save at 32K 65 → 47 ms. |
| Re-prefill generated tokens from the last prefill-written position when a Sync extends them, restoring the window before it from a per-session anchor | Retained for cold-equal continuations (Q8_0 forked chat diverged at step 7). Decode cost 0: tg1024 Q8_0 59.4 tok/s, C4 52.4 per session, as before. |
| Re-prefill every 512 decoded tokens during generation | Rejected: tg1024 Q8_0 −3.8%, a pause every 512 tokens that stalls every batched session, and wasted work for replies whose reasoning clients drop. |

## Kernels

| Experiment | Outcome |
| --- | --- |
| Head-major K/V `[head][slot][128]` versus token-major | Retained: decode attention streams ≈200 GB/s versus ≈60 GB/s. |
| Routed MMQ Q8_1 block sums scaled by 2^-8 | Retained: layer-46 prefill repro 8192 non-finite → 0; bit-identical in the normal range. |
| AttentionPrefill 32-key tiles | Retained: 2048 rows at 16K 56.4 → 43.0 ms; d0 3.60 → 2.94 ms. |
| AttentionPrefill q and P as plain FP16 instead of hi/lo pairs | Retained: 2048 rows at 16K 43.0 → 27.7 ms (21.1 TFLOP/s), at 30K 95.3 → 48.2 ms. Error versus FP64 ≈ 4e-4·max\|v\|, the order of FP16 K/V storage. |
| Split-invariant attention tiles at absolute 16-position boundaries | Retained for exact chunked prefill: d0 1.80 → 1.91 ms, sliding 0.89 → 0.99 ms, 30K unchanged. |
| Vendored MMQ/MMVQ tier without fast-math, `-ffp-contract=off` | Retained for row-count and split invariance (16 failing groups → 0); 2048-row MMQ 1–3% slower, decode unchanged. |
| Sliding decode grid bounded by `ceil(window/32)` chunks | Retained: 18.6 → 16.6 µs. |
| RoPE frequencies computed on the host | Retained: PrepareAttention 1 row 6.44 → 4.66 µs; ≤1 ulp angle change. |
| Sliding decode 16-position chunks; 8-warp decode chunks | Rejected: 16.6 → 18.4 µs; 16.6 → 17.0 µs and 4K full 53 → 62 µs. |
| MMVQ RDNA4 parameter table (8 warps) | Rejected: small GEMVs 10–40% slower. |
| GEMV quantizing its input in LDS | Superseded by quantize-once: small shapes 2–4 µs faster, large and routed shapes slower. |
| Fused router (scores and top-k in the last block) | Rejected: 1 token 19.6 → 18.6 µs, 4 tokens 25 → 34 µs, needs non-reentrant device counters. |
| Routed MMQ SwiGLU epilogue; row tile 128; column-warp split 2 | Rejected: 1% at 2048 tokens; 2.88 → 3.15 ms; gate 1.40 → 2.58 ms. |
| Prefill attention split-K | Not adopted: −8% for 512 rows at 32K, mixed at 128 rows; needs a scratch API. |
