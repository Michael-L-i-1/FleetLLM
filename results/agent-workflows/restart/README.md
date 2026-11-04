# Qwen3.5-4B GPU restart results

Native Fleet inference on the M5 Pro GPU works. Exact prompt checkpoints help repeated agent turns; the selected four-agent coding configuration did not complete the website task.

## Fixed-history inference comparison

Three recorded turns per worker, four concurrent workers, 128-token output cap, fresh server per trial. Two sweeps ran in opposite order. Every measured run generated 1,275 tokens. Generated actions were not executed, and this replay does not measure coding correctness.

| Configuration | Mean seconds | Aggregate tok/s | Relative throughput | Cached prompt fraction | Identical replies vs control |
|---|---:|---:|---:|---:|---|
| control | 14.31 | 89.1 | 1.000× | 0.0% | 12/12; 12/12 |
| checkpoints | 12.22 | 104.3 | 1.171× | 26.7% | 12/12; 12/12 |
| mask | 14.09 | 90.5 | 1.016× | 0.0% | 11/12; 11/12 |
| combined | 12.41 | 102.7 | 1.153× | 21.1% | 11/12; 11/12 |

**Interpretation:** checkpoints improve replay throughput by about 17% with identical content and reasoning in these samples. Masking alone gives about 2%, within plausible run-to-run noise; combining it with checkpoints adds no gain. Leave masking disabled by default. The combination reused 21.1% of prompt tokens versus 26.7% with checkpoints alone; edited prefixes invalidate some saved state, which likely contributes to the weaker combined gain. Only two repetitions were run; there is no confidence interval or general speedup claim.

## Actual coding task

Same Fieldnotes restyling task, unchanged acceptance gates at 390/1440 px, one coordinator and four model-delegated file owners. Each agent had 20 calls, 4,096 generated tokens per call and a three-consecutive-invalid-action stop rule. The coordinator alone must pass integration checks; workers finish their owned artifacts. These clarified instructions apply to both restarted runs.

| Configuration | Seconds | Generated tokens | Aggregate tok/s | Task success |
|---|---:|---:|---:|---|
| baseline | 151.33 | 13,286 | 87.8 | No |
| checkpoints + masking | 161.90 | 15,299 | 94.5 | No |

Both coordinators exhausted their 20-call allowance. Independent artifact checks also failed: both generated HTML files omitted the required `search` and `email` IDs. The task is not solved. Higher aggregate token throughput is not a successful-task speedup: the combined run produced more output and took longer. This experiment evaluates the Fleet + checkpoint + protocol configuration; it does not establish that Qwen3.5 intrinsically cannot do the task.

## Model, hardware and flags

- Model: `mlx-community/Qwen3.5-4B-4bit`, revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`; affine 4-bit, group 64. Already downloaded and hash-audited. Text tower only.
- Native C/Metal backend on Apple M5 Pro, 20 GPU cores, 48 GB unified RAM. MLX-LM is used for independent validation, not Fleet inference.
- Same 10 GB runtime budget, 4 GB combined recurrent-state/attention-cache budget, max-active 4, admission enabled, row budget 128. Attention KV is FP16; recurrent state is FP32.
- Greedy sampling. Qwen thinking enabled; `low` selects no special Qwen reasoning-strength level.
- N-gram speculation, DFlash, SPAN and recurrent disk offload are disabled because this initial hybrid implementation does not implement the necessary rollback/offload support.
- Checkpoint variant adds `--prompt-checkpoints`. Masking variant adds `--context-prune`: mask superseded observations only after a confirmed write/integration and only when at least 512 characters long. Immutable app.js and current feedback remain.
- Both restarted task runs use the same narrowly audited JSON final-brace recovery. Neither needed that recovery. No generated file content is automatically repaired.
- Sampled peak process RSS was approximately 5.4 GiB for each complete coding trial. RSS is not a complete accounting of Metal/unified-memory allocations.
- Model startup, fixture preflight and warmup are excluded from workload timing. Full task timings include model reasoning, JSON actions, tools and integration. Replays contain inference only; they have no tool execution or task grading.

## Native support and correctness

Native support adds gated delta recurrence, causal convolution, gated attention, partial RoPE, bounded per-sequence state and safe endpoint forks. One GPU wrapper is retained for each recurrent allocation; four value channels share a recurrent threadgroup. A separate controlled 128-token test improved decode from 31.20 to 90.26 tok/s while preserving the output byte-for-byte. That 2.89× gain is against the first Fleet implementation of this new architecture, not against llama.cpp, vLLM or MLX-LM.

Validation: 24 teacher-forced configurations across four prompts passed against the FP32 MLX-LM reference; all sampled argmax choices matched. Metal mean absolute logit error stayed below 0.008 for FP16 KV. State tests cover endpoint fork/resume, parent isolation, concurrent equivalence, reset and memory-pressure recomputation. Existing CPU speculation regressions and four fake-model harness tests passed. These tests do not establish broad long-context or coding quality. The earlier BF16/strict-logit failures remain recorded; optimized CPU uses a distinct activation-quantization distribution-error bound.

## Artifacts and reproducibility

- `coding-comparison.csv`, `replay-comparison.csv`, `analysis.json`, `replay-comparison.json`: measurements and individual runs.
- Per-label config JSON files: exact server/harness commands; per-label logs, calls, events and resource CSV files.
- `setup.json`: pinned configuration and source/binary hashes. `../validation-summary.json` and referenced raw audits: correctness evidence.
- `../runs.zip`: complete raw histories, generated files, patches and screenshots, including failed/paused historical attempts. Transcript messages plus masked indices reconstruct active histories.
- Engine support commit `5b203f1`; GPU/state/checkpoint followup `22ca445`; benchmark harness `24de3f3`; replay metadata guard `8226e98`. All changes are on `experiment/agent-workflows`, separate from the original workspace.
- Historical pre-restart runs used a different worker prompt and are not apples-to-apples comparisons. The first old run also had inaccurate pending-prefix cache accounting; it is excluded here. The first replay setup attempt failed before model requests because of Path metadata serialization, was preserved, and was replaced by v2 labels.
- No external-engine benchmark was performed for this model. Program-aware priority scheduling was deferred because this single workflow admits all four workers and then waits for all of them.

Paper motivation: [Continuum](https://arxiv.org/abs/2511.02230) and [CacheScout](https://arxiv.org/abs/2608.14624) motivate retaining useful agent state; [Cache Engineering for Agents](https://arxiv.org/abs/2610.28577) motivates explicit context editing. These prototypes implement limited, conservative components, not the complete paper systems or their published gains.

All benchmark servers are stopped after the sweep.
