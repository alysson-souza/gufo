# Recursive Flint benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL`
single-file GGUF. HTTP, greedy, seed 1, measured October 23, 2026 on the
`performance` power profile. Flint has no thinking
switch: on both servers the model opens `<think>` and spends all 128 generated
tokens reasoning; the timed scope counts every generated token. Both sides
are Nix builds: the production package at commit `526736b` and
`.#llama-cpp-flint-reference` (llama.cpp `970b51ce`, build 11827). Flint has
no speculative mode in Gufo or in this llama.cpp reference, so the card has no
speculative tables.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

Both servers report identical prompt and cache counts at every depth (131,072:
frontiers one token apart). Completion hashes cover the streamed reasoning
and content. All 63 Gufo C1–C8 concurrency completions match its C1 output,
which also equals the single-user depth-0 completion.

## Single user, autoregressive

pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| Flint Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2211.31 | 1561.41 | +41.6% | 65.46 | 51.58 | +26.9% |
| 4,096 | 1915.50 | 1372.28 | +39.6% | 61.92 | 49.86 | +24.2% |
| 8,192 | 1734.97 | 1270.89 | +36.5% | 60.02 | 48.46 | +23.9% |
| 12,288 | 1549.48 | 1194.55 | +29.7% | 57.85 | 47.15 | +22.7% |
| 16,384 | 1478.56 | 1111.24 | +33.1% | 56.08 | 46.05 | +21.8% |
| 32,768 | 1133.66 | 901.34 | +25.8% | 49.53 | 41.70 | +18.8% |
| 65,536 | 740.86 | 641.14 | +15.6% | 40.37 | 35.05 | +15.2% |
| 131,072 | 450.04 | 406.37 | +10.7% | 29.79 | 26.82 | +11.1% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
Prefill every session before timed decoding; sum individual request decode
rates; mean of three measured cohorts per point (single-user rows: one warmed
sample).

<!-- bench:multi-ar -->
| Flint Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 65.42 | 51.97 | +25.9% |
| 2 | 114.49 | 93.20 | +22.8% |
| 4 | 187.16 | 140.77 | +33.0% |
| 6 | 226.81 | 170.40 | +33.1% |
| 8 | 252.93 | 192.72 | +31.2% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Loading time

C1, capacity 262144, AR. Cold model file (`POSIX_FADV_DONTNEED`, zero resident
pages verified with `mincore`) to HTTP readiness; runtime libraries may be warm.

<!-- bench:loading -->
| Flint Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 4.26 | 5.24 | +23.0% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, AR, capacity 133121 on both servers; peak device-global HIP memory in use
during each workload, including 3.01 GiB already in use before either server
started (Gufo's own share: 28.1 GiB). Both servers
allocate KV for the full capacity before the measured requests, so neither
footprint grows with the prefix.

<!-- bench:memory -->
| Flint Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 31.15 | 30.74 | -1.3% |
| 16K prefix, pp4096 + tg128 | 31.15 | 30.74 | -1.3% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)
