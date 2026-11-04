# Native Qwen3.5 text inference

Fleet can execute the text tower of `mlx-community/Qwen3.5-4B-4bit` directly in
its CPU and Metal backends. This is native inference, not an HTTP wrapper around
MLX-LM, llama.cpp, or vLLM. MLX-LM is used only by the development parity check.

The validated checkpoint is revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`:
affine four-bit weights, group size 64. The vision tower is not loaded.

```sh
hf download mlx-community/Qwen3.5-4B-4bit --revision 0e7ffd5c629ef7719d4cbc04069232580bfa9d9c
make -j4
./build/fleet serve --model /path/to/the/snapshot --device metal \
  --name qwen35-4b-mlx4 --budget-gb 10 --kv-gb 4 --max-active 4 --admit
```

Use `reasoning_effort: "none"` to disable thinking, or leave thinking enabled.
Qwen3.5 does not have Glimmer's low/medium/high reasoning-strength control.
Text requests and the site's JSON-action agent harness are supported. Qwen3.5's
native XML function-call template/parser and image/video inputs are not implemented.
The chat endpoint rejects native tool definitions instead of advertising the old
Qwen tool-call format for this model.

## Model operations

The implementation adds depthwise causal convolution and Gated DeltaNet recurrent
updates, per-head gated RMS normalization, interleaved full-attention query/gate
projections, and partial rotary embedding. Linear state stays in FP32. Full
attention uses Fleet's paged FP16 or optional INT8 KV cache. Metal executes both
linear and full-attention operations on the GPU; CPU and CPU-reference are
separate selectable backends.

This initial loader targets converted MLX dense checkpoints with 128-dimensional
linear key/value heads, a four-token convolution, 256-dimensional attention heads,
and a 64-dimensional rotary subspace. Raw unsanitized Hugging Face weights, GGUF,
MoE variants and other dimensions are not claimed as supported. Converted MLX
normalization weights already include the source model's zero-centered offset.

## State ownership and memory

A Qwen3.5-4B sequence needs approximately 52.69 MB of recurrent state in addition
to attention KV. Fleet reserves bounded state slots from the supplied KV budget
(one quarter by default, rounded to at least one slot and capped at 64 slots). The rest
backs paged attention. Cached states are evicted first when slots are needed;
active states can be discarded and exactly recomputed under pressure.

Each available endpoint is independently copied on a normal fork. Reusing arbitrary
intermediate attention pages does not provide the corresponding recurrent state.
An unavailable intermediate fork therefore recomputes its prefix. The optional
shared-prefix scheduler below stops a prefill leader at an exact fork boundary.
Each child still owns independent mutable recurrent state.

N-gram speculation, DFlash, span reuse and recurrent disk offload need additional
checkpoint/rollback support. Their CLI switches fail explicitly for this model.
They remain available for the previously supported architectures. This initial
support does not imply all Fleet optimizations compose with hybrid recurrence.

## Verification

`build/fleet-model-probe` exports logits for direct, reproducible comparisons.
`scripts/qwen35check.py` compares tokenization and teacher-forced logits against
MLX-LM, including one-token and chunked prefill and optional INT8 KV. Run it with
a development Python environment containing MLX-LM and NumPy:

```sh
python scripts/qwen35check.py --model /path/to/snapshot \
  --reference-f32 --output results/qwen35-parity.json
cc -O2 -g -Isrc -DCI_METAL=1 scripts/qwen35statecheck.c build/libfleetllm.a \
  -lpthread -lm -framework Metal -framework Foundation -o build/qwen35statecheck
build/qwen35statecheck --model /path/to/snapshot --device metal --budget-gb 8 --kv-gb 2
```

FP32 MLX provides the closest arithmetic reference for Fleet. The checkpoint's
usual BF16 MLX execution also produces a separate audit. Optimized CPU inference
quantizes activations to eight bits, so its distribution-error check is distinct
from the stricter CPU-reference and Metal logit checks. The reports preserve the
initial strict-check failures rather than silently relabeling them as passes.

The state driver checks concurrent generation, endpoint fork versus recomputation,
parent/child isolation, slot reset, unavailable intermediate forks and memory
pressure. These checks do not establish coding quality or long-context accuracy;
the website task is evaluated separately with preserved transcripts and browser checks.

Reference implementations consulted: Apple's `mlx_lm.models.qwen3_5`,
`qwen3_next` and `gated_delta`, installed MLX-LM 0.32.0. The standalone Metal
kernels implement the mathematical recurrence using Fleet's existing buffers,
matmul dispatch and scheduler interfaces.

## Optional prompt checkpoints

`--prompt-checkpoints` preserves an exact recurrent snapshot immediately before
an assistant generation header. Later agent turns can reuse that common prefix
even when the client omits the preceding turn's reasoning text. Snapshots consume
the same bounded state-slot budget and share attention pages through COW. A
checkpoint is skipped when there is no slot; preserving it is not counted as a
served prefix-cache hit. The feature is opt-in and currently wired to Qwen3.5's
text chat format. It does not provide arbitrary intermediate rollback or SPAN.

The Metal implementation keeps one GPU wrapper per resident recurrent state and
releases it with that state. Recurrence dispatch groups four independent value
channels into each threadgroup. These are exact execution changes, not layer
skipping or activation sparsity. A controlled 128-token generation preserved the
output byte-for-byte and improved decode from 31.20 to 90.26 tok/s on this M5 Pro;
this compares two versions of Fleet's new implementation, not competing engines.

## Optional throughput experiments

The following switches are opt-in; normal Fleet defaults are unchanged. They
operate on exact recurrent states, without layer skipping, activation sparsity,
or context deletion.

| Setting | Behavior | Cost or limitation |
|---|---|---|
| `--recurrent-prefix-dedup` | Coalesces concurrent shared-prefix prefill at an exact leader boundary, then forks each waiter. | Each child needs its own state slot; missing/canceled leaders fall back to recomputation. |
| `--recurrent-state-move` | Transfers ownership of a reusable inactive cached endpoint instead of copying it and releasing the source. | Consumes that cached endpoint; never transfers a live mutable parent or a source with waiting children. |
| `--recurrent-cache-priority` | Reclaims inactive generation endpoints before useful prompt checkpoints when state slots fill. | Checkpoints still compete within the same bounded cache budget. |
| `--recurrent-fraction 0.5` | Reserves half the supplied KV budget for recurrent states instead of the default quarter. | Leaves fewer attention pages. Valid range 0.05–0.90, maximum 64 slots. |
| `FLEET_Q35_NAX_ATTN=1` | Uses tiled NAX matrix operations for 256-dimensional full-attention prefill. | Metal only; decode keeps the vector path. Floating-point rounding can change. |

For example, on the validated Mac checkpoint:

```sh
FLEET_Q35_NAX_ATTN=1 ./build/fleet serve --model /path/to/snapshot \
  --device metal --name qwen35-4b-mlx4 --budget-gb 16 --kv-gb 4 \
  --max-active 8 --admit --row-budget 128 --prompt-checkpoints \
  --recurrent-prefix-dedup --recurrent-state-move --recurrent-cache-priority
```

At a 4 GB KV budget, the default split provides 18 state slots; a 0.5 split
provides 37. Start with the default for eight active sessions; the larger split
is tested separately for sixteen. Compare total budgets and KV precision within
each experiment rather than treating different memory allocations as free gains.

`FLEET_TRACE_STATS=1` emits cumulative per-request work, shared tokens, checkpoint
captures/skips, state moves, slot misses and eviction counters. It is intended for
diagnosis and benchmark attribution. `scripts/qwen35dedupcheck.c` exercises tiny
scheduler fixtures; `scripts/qwen35_attention_check.py` checks HD256 span-boundary
logits. See the [throughput experiment report](../results/throughput-exploration/README.md)
for workload limits and measured results. These experiments do not establish
successful autonomous coding or performance against another inference engine.
