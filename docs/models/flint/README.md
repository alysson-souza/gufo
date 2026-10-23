# Recursive Flint

Recursive's mixture-of-experts text model for coding agents (39.8B total,
about 2.5B active per token; Apache-2.0, source
[`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint/tree/2a247be0c0aec895ccb0ac7ca92229ec6df37288)).
GGUF `general.architecture = flint`. Supported targets from
`unsloth/flint-GGUF`: **UD-Q4_K_XL** (25,054,688,704 bytes) and **Q8_0**
(42,301,076,928 bytes), each one file.
Text only, autoregressive (AR) only. Agreement with the original BF16
checkpoint is not qualified; see [Quality](QUALITY.md).

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Architecture

49 pre-norm layers, hidden 2048, vocabulary 150,144, native context 262,144.

- Attention: 32 query / 4 KV heads of 128, per-head QK RMSNorm and one learned
  sink logit per query head (denominator only). 35 sliding-window layers see the
  current token plus 512 previous tokens and use NeoX RoPE (theta 10000);
  the 14 full-attention layers have no positional encoding.
- Feed-forward: layers 0–1 dense SwiGLU (6144). Layers 2–48 route each token
  to 6 of 256 SwiGLU experts (512) by sigmoid score, normalized over the chosen
  six, plus an always-active shared expert (1024).
- Untied Q8_0 output head; byte-level BPE tokenizer with Flint's
  pre-tokenizer; no BOS.

## Load and run

```sh
nix develop -c hf download unsloth/flint-GGUF flint-UD-Q4_K_XL.gguf \
  --revision 20468edfc38ab43d6914cc88b08cbecb05d0fc62 --local-dir models/flint
nix build
MODEL=models/flint/flint-UD-Q4_K_XL.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo prompt --model "$MODEL" -p "What is 17*23?"
./result/bin/gufo serve llm --model "$MODEL" --sessions 4 --context 32768
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 -d 0,16384 -c 1,4,8
```

SHA-256: UD-Q4_K_XL
`63c2ecc461ca545134b670c660e76fa20f47240f8d39b6510bfd21c4dd6e072f`, Q8_0
`5df42b56859c7b5142228005d7d63c327ec9e34c99c72548016d805309a0d284`
(download `flint-Q8_0.gguf` the same way).
Loading validates the pinned chat template, then uploads every tensor to the
GPU. The HIP kernels implement Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS and IQ3_S matrices
(input width a multiple of 256, one encoding shared by q/k/v in each layer),
a Q8_0 token embedding, BF16 routers and F32 norms/sinks. Other quantizations
and sharded GGUFs are rejected at load with the offending tensor named.
ROCm (gfx1151) is required; CPU builds refuse Flint.

`--speculative` accepts only `off`: Flint has no MTP head or draft model in
Gufo. `--mmproj` and `--enable-yarn` are rejected. `serve` accepts
`--context` up to 262,144 and batches up to eight sessions per decode step.
`prompt` and `chat` use a fixed 4,096-token context. `bench` measures pp/tg at
`-d` depths (prefix restored from a snapshot) and C1–C8 with `-c`.

Q8_0 needs no other option. Native `gufo bench` / `llama-bench -fa 1` on the
same file, Nix builds of both (`.#llama-cpp-flint-reference`), mean of two runs
(not the HTTP card in [Benchmarks](BENCHMARKS.md)): pp2048 2315 vs 1330 tok/s,
tg128 58.6 vs 47.8, pp2048 at depth 16,384 1594 vs 1022, tg128 at depth
16,384 50.2 vs 42.5.

## Reasoning, tools and sampling

Flint has no thinking switch. The model decides on each turn whether to open
`<think>`; the llama.cpp oracle gives `<think>` p≈1 as the first assistant
token on every prompt tried. `--think`, `reasoning_effort` and
`chat_template_kwargs.enable_thinking` change neither the prompt nor the
sampling defaults, as in Recursive's SGLang parser. A reasoning budget
(`--reasoning-budget N` or `"reasoning_budget_tokens": N`) waits for the
model's `<think>` and counts from it; `0` closes it immediately. The budget is
a cap, not a clean thinking-off: after a forced `</think>` the model often
continues its reasoning as content. Reasoning is returned as
`reasoning_content` and counted in `usage.reasoning_tokens`; a tool call
written inside `<think>` stays reasoning. See the
[reasoning controls](../../SERVER.md#reasoning-controls).

Tool calls use Flint's GLM-style syntax,
`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`:
string values raw, other values JSON, typed by the tool schema. Constrained
calls follow the reference SGLang grammar, except that `required` and named
`tool_choice` allow only an optional `<think>…</think>` before the call, no
text (Gufo's tool-choice contract). See the server's
[Chat Completions adapter](../../SERVER.md#chat-completions-adapter).

Defaults come from `generation_config.json`: temperature 1.0, top-p 0.95,
top-k 20 (the GGUF agrees). `--temperature 0` is greedy; `bench` stays greedy.

## Memory

Weights stay resident on the GPU (23.33 GiB of tensors for UD-Q4_K_XL,
39.40 GiB for Q8_0, plus scratch).
Each session holds FP16 K/V: 14 full layers × context × 2 KiB plus
35 sliding layers × min(context, 2575) × 2 KiB, about 0.3 GB at 4K,
1.1 GB at 32K and 7.7 GB at 262K, plus a 36.7 MB copy of the sliding window
taken when decoding starts. Admission reserves the configured capacity
for every session before creating it. With one session at capacity 133,121,
peak device-global memory in use during the benchmark workloads is
31.12 GiB, of which 3.01 GiB was already in use before the server started:
28.1 GiB is the server's (weights, KV and scratch). See
[Benchmarks](BENCHMARKS.md#memory-occupation).

## Tools and artifacts

Model code is in `src/models/flint`, tests in `tests/models/flint`, and
reference/golden tools in `tools/flint`. [Quality](QUALITY.md) lists the
test targets and reproduction commands. Retained result summaries belong in
`artifacts/`; logit dumps and traces stay in the ignored top-level `artifacts/`.
