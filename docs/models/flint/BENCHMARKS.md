# Recursive Flint benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL`
single-file GGUF. HTTP, greedy, seed 1, measured October 23, 2026 on the
`performance` power profile. Flint has no thinking
switch: on both servers the model opens `<think>` and spends all 128 generated
tokens reasoning; the timed scope counts every generated token. Gufo is a
non-Nix release-preset build of commit `09c6149`; llama.cpp is an
ad-hoc ROCm build of `970b51ce` because the pinned `.#llama-cpp-reference`
(b11069) predates Flint support. Flint has no speculative mode in Gufo or in
this llama.cpp reference, so the card has no speculative tables.

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
| 0 | 2372.91 | 1665.81 | +42.4% | 67.65 | 53.77 | +25.8% |
| 4,096 | 2045.10 | 1439.61 | +42.1% | 63.95 | 51.78 | +23.5% |
| 8,192 | 1830.03 | 1331.18 | +37.5% | 62.24 | 50.44 | +23.4% |
| 12,288 | 1621.75 | 1236.89 | +31.1% | 59.85 | 49.01 | +22.1% |
| 16,384 | 1543.43 | 1132.69 | +36.3% | 58.05 | 47.80 | +21.4% |
| 32,768 | 1173.63 | 851.67 | +37.8% | 51.31 | 43.30 | +18.5% |
| 65,536 | 765.26 | 568.27 | +34.7% | 41.90 | 36.46 | +14.9% |
| 131,072 | 460.80 | 319.65 | +44.2% | 30.99 | 27.77 | +11.6% |
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
| 1 | 67.60 | 53.89 | +25.4% |
| 2 | 118.66 | 95.53 | +24.2% |
| 4 | 197.28 | 145.63 | +35.5% |
| 6 | 241.34 | 173.91 | +38.8% |
| 8 | 270.94 | 194.85 | +39.1% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Loading time

C1, capacity 262144, AR. Cold model file (`POSIX_FADV_DONTNEED`, zero resident
pages verified with `mincore`) to HTTP readiness; runtime libraries may be warm.

<!-- bench:loading -->
| Flint Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 4.11 | 5.30 | +29.0% |
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
| pp2048 + tg128 | 31.12 | 30.72 | -1.3% |
| 16K prefix, pp4096 + tg128 | 31.12 | 30.72 | -1.3% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)
