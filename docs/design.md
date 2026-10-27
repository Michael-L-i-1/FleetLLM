# Design

cinfer is one engine (scheduler, paged KV, prefix sharing, speculation) over
pluggable backends (Metal for Apple GPUs, CPU for the Pi path). This file
describes the pieces; `src/*.h` headers document the APIs.

## Request path

`tools/serve` (OpenAI-compatible HTTP, `src/server.c`) -> one engine thread
that continuously batches every in-flight request (`src/engine.c`) -> backend
forward over the mixed batch of decode rows, draft rows and prefill chunks.

## KV cache and sharing

Paged (16 tokens per page), reference-counted, copy-on-write (`src/kv.c`).
Two pools: full-attention layers and sliding-window layers (pages released
once outside the window). A new sequence forks the longest matching prefix of
a live, cached (finished) or still-prefilling sequence; forks copy page ids,
not data. Under memory pressure cached sequences are evicted first, then
running ones are preempted (KV freed, recomputed later).

## Weights and kernels

Weights load from GGUF or HF/MLX safetensors (`src/model.c`). On the M5 the
4-bit affine weights are repacked at load into 64x64 tiles with inline scales
(`src/metal/metal.m`, `kernels_nax.metal`): a tiled mat-vec for 1-4 rows and a
NAX (tensor unit) GEMM, one simdgroup per 32 weight rows, for more; q|k|v|gate
and gate|up are fused. GGUF K-quants still use the older kernels in
`kernels.metal`.

## Matmul dispatch

| Rows in the step | Kernel |
|---|---|
| 1-8 | fused mat-vec on the packed K-quant blocks |
| 9-64 | fused GEMM: weight tiles decoded into threadgroup memory, simdgroup MMA |
| > 64 (prefill) | dequantize once into an f16 scratch matrix (256 MB, row-chunked), dense GEMM |

Crossovers are env-tunable: `CINFER_MV_MAX`, `CINFER_FUSED_MAX`.

MLX affine 4-bit weights (repacked at load into 64-row x 64-K blocks, scales/biases inline):

| Rows in the step | Kernel |
|---|---|
| 1-2 | tiled mat-vec (8 rows per simdgroup, 4 K tiles of loads in flight), split-K |
| 3-4 | same, with the 16-value partial dots in f16 (codes centered at 8; perplexity unchanged) |
| 5-16 / 17+ | NAX GEMM, one simdgroup per 32 weight rows x 16 / 32 tokens, split-K |

q/k/v/gate and ffn gate/up are fused into single matrices at repack time; the
norm, attention and SiLU kernels write f16 matmul inputs directly; the residual
add, post-norm and next pre-norm run as one kernel.

## Speculative decoding

DFlash drafter (`metal_draft`): context rows write the drafter's KV from the
target's hidden states (captured after layers 1/13/25/37/49), block rows
(anchor + masks) are denoised in one pass; the target verifies all drafts in
one batched forward and rejected KV is truncated. Drafts per agent shrink as
more agents decode (`spec_rows` target).

## Source layout

```
src/gguf.*        GGUF reader
src/safetensors.* safetensors reader (sharded); src/json.* minimal JSON parser
src/quant.*       reference dequantization (ggml port)
src/tokenizer.*   BPE + hand-written pre-tokenizers; src/unicode_data.c is generated
src/chat.*        chat formats (Llama 3, ChatML/Qwen, Glimmer channels)
src/model.*       generic decoder description (flags per architecture)
src/kv.*          paged ref-counted copy-on-write KV
src/engine.*      sequences, scheduler, prefix sharing, sampling
src/cpu.c         CPU reference backend
src/metal/        Metal backend (metal.m) and kernels (kernels.metal; kernels_nax.metal for the
                  M5 neural accelerators; vendored MLX headers in mlx/), embedded at build
tools/            generate, bench, ppl, tokcheck, dequant_dump
scripts/          verification scripts, Unicode table generator
bench/tasks.txt   8 heterogeneous agent tasks used by bench
```
