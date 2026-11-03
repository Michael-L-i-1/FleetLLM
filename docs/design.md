# Design

FleetLLM is one engine (scheduler, paged KV, prefix sharing, speculation) over
pluggable backends (Metal for Apple GPUs, CPU for the Pi path). This file
describes the pieces; `src/*.h` headers document the APIs.

## Request path

`fleet serve` (OpenAI-compatible HTTP, `src/server.c`) -> one engine thread
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

Crossovers are env-tunable: `FLEET_MV_MAX`, `FLEET_FUSED_MAX`.

MLX affine 4-bit weights (repacked at load into 64-row x 64-K blocks, scales/biases inline):

| Rows in the step | Kernel |
|---|---|
| 1-2 | tiled mat-vec (8 rows per simdgroup, 4 K tiles of loads in flight), split-K |
| 3-4 | same, with the 16-value partial dots in f16 (codes centered at 8; perplexity unchanged) |
| 5-16 / 17+ | NAX GEMM, one simdgroup per 32 weight rows x 16 / 32 tokens, split-K |

q/k/v/gate and ffn gate/up are fused into single matrices at repack time; the
norm, attention and SiLU kernels write f16 matmul inputs directly; the residual
add, post-norm and next pre-norm run as one kernel.

Attention, chosen per row (`row_t.input`) so a row's result never depends on
its batch or on how its prompt was chunked:

| Rows | Kernel |
|---|---|
| input (prompt, appended text, recompute), M5 | `attn_nax`: 64 rows x 1 head per threadgroup, QK^T and PV as MPP 16x32x16 matmuls, K/V from the pages into registers; fixed 8192-position spans merged by `attn_reduce` |
| sampled / drafted (decode, verify), M5 | `attn_naxs`: a sequence's decode row + drafts x the G heads fill the 16 query slots of an MPP matmul tile; fixed 512-position spans, and row groups of sequences holding the same pages in a span share one threadgroup and one dequantized K/V tile |
| sampled / drafted, GPUs without NAX | `attn_vec`: 1 row x its G heads per threadgroup, lanes across the head dim, row-local splits |
| input, GPUs without NAX | `attn_mma`: up to 8 consecutive rows x G heads share K/V tiles in threadgroup memory, 8x8 simdgroup MMA |

All kernels read 16-bit or 8-bit (per-token, per-head scale) KV pages directly.

## Speculative decoding

DFlash drafter (`metal_draft`): context rows write the drafter's KV from the
target's hidden states (captured after layers 1/13/25/37/49), block rows
(anchor + masks) are denoised in one pass; the target verifies all drafts in
one batched forward and rejected KV is truncated. Drafts per agent shrink as
more agents decode (`spec_rows` target).

The CPU backend also implements DFlash: it captures the selected target-layer
residuals, projects their concatenation into fixed context features, and runs
anchor/mask blocks with bidirectional attention and the draft model's sliding
window. Draft weights use the existing affine 4-bit CPU tiles; verification,
rejection rollback, and draft-length scheduling remain in the shared engine.
`scripts/cpudraftcheck.py` compares the draft hidden states/tokens to an
independent dense PyTorch reference and exercises acceptance/rejection with
float32, float16, and int8 KV under address/undefined-behavior sanitizers.

Q2_K and Q3_K use compact CPU tiles and NEON integer dot products, preserving
the source codes/scales. Q2_K retains its original storage size; Q3_K uses
116 rather than 110 bytes per 256 weights for simpler per-half scale access.
On Linux, release each repacked GGUF source range from the mapping and file
cache immediately to avoid keeping two resident copies during loading.
Q5 and sub-4-bit affine weights without native tiles retain their original
format and use row-wise dequantization plus the NEON float dot product, rather
than expanding to CQ8. See the [Pi Q2_K tests](../results/pi5/2026-11-02/glimmer-q2k/README.md).

## Precision and hardware profiles

CPU target precision defaults to 32-value int8 activation scales and all model
layers/features. The optional 256-value scale uses integer K-quant superblock
accumulation; it changes activation rounding. Explicit layer and FFN down-input
pruning controls are experimental, disabled by default, and validated before
inference. They change the target's answers. Speculative verification preserves
the selected target, including any approximation, rather than restoring an
unmodified model's output.

The engine owns verification and sequence/KV coordination. Pi model paths,
thread counts and demo choices belong in `scripts/pi-*.sh`, with measured
tradeoffs in [Pi tuning](pi-tuning.md). Generic `fleet eval` records task outputs
and token IDs; `bench/pi_sweep.py` owns Linux experiment orchestration, not the
inference engine. This keeps platform/demo choices separate from reusable
inference controls and future multi-agent experiments.

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
tools/            `fleet` subcommands (serve, generate, eval, bench, ppl, reuse_eval, tokcheck,
                  dequant_dump; built as build/fleet-<name>); tools/fleet/ is the dispatcher
scripts/          verification scripts, Unicode table generator
bench/tasks.txt   8 heterogeneous agent tasks used by `fleet bench`
```
