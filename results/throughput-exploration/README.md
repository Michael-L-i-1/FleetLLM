# Qwen3.5 throughput exploration on Apple M5 Pro

Status: all 12 matrix trials completed without timeouts, failed requests, or
censored comparisons. See `matrix/progress.json` and `matrix/cleanup.json`.

## Baseline and scope

Model: `mlx-community/Qwen3.5-4B-4bit`, revision
`0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`, affine 4-bit/group 64.
Hardware: Apple M5 Pro, 20 GPU cores, 48 GB unified memory. Native Fleet Metal
execution. Model startup and an unrelated warmup are excluded from trace timing;
scripted tool waits are included. Runs are serial, with one owned model server.

The frozen normal Fleet binary is from `9c8dbef`; `control.json` records its hash.
It retains native quantized matmuls, persistent recurrent wrappers, continuous
batching, paged KV, available exact prefix caching, and admission. This is not an
unoptimized implementation. Prompt checkpoints (previously experimental), shared
prefix coalescing, state ownership moves, phase-aware retention, and HD256 NAX
prefill are disabled in the control. No context masking is used in this matrix.
The optimized frozen binary contains native code from `e96d18e`. Per-run configs
record its hash; `source_commit` records the checkout at launch and may include
later harness-only changes. A subsequent repeated-identical-prompt checkpoint
fix (`00dfdd2`) is deliberately excluded from these frozen matrix binaries.
A final confirmation using that fix completed the N8 review trace in 24.00 s,
with the same 2,048 tokens and 83.25% cache fraction; it is reported separately
as `final-fix-confirm-f16-8`, not folded into the frozen trial means.

Within each control/optimized pair, weights, workload, output cap, total cache
budget, KV precision, concurrency, and row budget match. The optimized N16/N32
cases reserve half the same cache budget for recurrent state; this reduces the
attention-page allocation. This is a memory-policy change, not extra memory.
N8 uses default quarter allocation. All new features remain opt-in.

## Workloads and quality interpretation

`workload32.json` freezes 32 sessions with four turns each, a shared source prefix,
growing tool observations, seeded 0–1 second waits, and reference assistant
histories. We select 8, 16, or 32 sessions per case. Generated replies never become
subsequent prompts or executed edits. The cap64 review trace is entirely truncated
at its output cap. It measures repeated-context serving throughput, including
reasoning tokens, rather than successful coding or useful completed reviews.

`workload32-qa.json` uses the same frozen source with focused, graded retrieval
questions and correct reference histories. Thinking is disabled; all 32 answers
in the N8 trials are correct and finish without truncation. This provides a
useful-answer check but does not establish autonomous planning/coding quality or
perplexity across a general corpus. Six additional short fact probes are run
separately after each timed trace and excluded from its throughput.

`workload32-cap256.json` is a clone of the review workload with only the output
cap changed. It tests whether the benefit persists with more decoding work.

## Completed matched comparisons

| Workload | Sessions | KV / total cache | Normal seconds | Optimized seconds | Speedup | Generated tokens per side | Quality / truncation |
|---|---:|---|---:|---:|---:|---:|---|
| Review, cap64 (repeat means) | 8 | FP16 / 4 GB | 135.04 | 23.68 | 5.70× | 2,048 | 32/32 capped; separate QA 6/6 |
| Graded retrieval | 8 | FP16 / 4 GB | 131.13 | 19.98 | 6.56× | 108 | 32/32 correct; no caps |
| Review, cap256 | 8 | FP16 / 4 GB | 142.55 | 40.43 | 3.53× | 8,083 | 29/32 capped; separate QA 6/6 |
| Review, cap64 | 16 | INT8 / 4 GB | 316.27 | 46.52 | 6.80× | 4,096 | 64/64 capped; separate QA 6/6 |
| Review, cap64 | 32 | INT8 / 8 GB | 611.97 | 86.89 | 7.04× | 8,192 | 128/128 capped; separate QA 6/6 |

All listed pairs have zero failed or missing requests. N8 repeat means use
`control-f16-8-a/b` and `final-all-f16-8-b/c`. Each trial remains available rather
than reporting only the best result. Cross-row memory/precision changes mean
this table is not a pure concurrency-scaling experiment. No llama.cpp, vLLM,
or Ollama performance claim follows from these Fleet-versus-Fleet comparisons.

## Mechanism and ablations

At N8, the final combination reuses about 83.25% of prompt tokens versus about
9–10% in normal Fleet. At the end of the 32-request trace, counters show 24 state
ownership moves, no checkpoint skips, no slot misses or preemptions, 24 endpoint
evictions, and zero checkpoint evictions. Retaining exact prompt-phase states
avoids repeated long prefills. It does not make fresh-token decoding 5–7× faster.

Initial isolated N8 probes: checkpoints alone 128.28 s, prefix coalescing alone
106.90 s, and HD256 NAX prefill alone 93.27 s, against 133.05 s control. These
are single exploratory runs, not repeated estimates. Final ablations give:

| N8 configuration | Seconds | Aggregate generated tok/s | Cache hit fraction |
|---|---:|---:|---:|
| Normal Fleet, repeat mean | 135.04 | 15.17 | ~9–10% |
| Checkpoints + ownership moves + retention priority | 53.88 | 38.01 | 68.41% |
| Above + concurrent shared-prefix coalescing | 35.15 | 58.27 | 83.25% |
| Above + HD256 NAX prefill, repeat mean | 23.68 | 86.48 | 83.25% |

Thus the cache-reuse combination is about 3.84× faster even on the original
attention kernel. NAX adds approximately 1.48× on top for this workload. These
ratios are conditional on this traffic and do not decompose into independent
universal per-feature multipliers. Every isolated/final ablation passed all six
short probes and completed all requests.

The changes do not deliberately approximate model math. Cache copying/moving
preserves exact independent state; NAX alters floating-point accumulation order.
The first NAX-only trial matched all 32 content/reasoning replies. Both final
N8 cap64 repeats matched 31/32; cap256 matched 27/32; N16 INT8 matched 55/64;
N32 INT8 matched 112/128.
The graded QA pair matched all 32 replies. Matching token counts alone is not a
quality test, and a changed ungraded review is not evidence of equal quality.

## Validation and outside feedback

- `nax256-parity.json`: 24 teacher-forced comparisons against FP32 MLX-LM,
  including chunking and INT8 KV; all pass.
- `nax256-boundaries.json`: 18 vector/NAX comparisons around 32/64/4096 span
  boundaries with FP16 and INT8 KV; all pass, matching next-token argmax.
- GPU state checks cover independent forks, parent isolation, fallback
  recomputation, reset, and pressure. Scheduler ASAN/UBSAN fixtures additionally
  cover cancellation, ownership callbacks, finite slot budgets, and retention.
- The repeated-identical-prompt lifecycle fix passes ASAN/UBSAN and real Metal
  API checks: four correct complete replies, cached tokens 0/2525/2525/2525
  out of 2532 prompt tokens. Growing-history reuse is unchanged in the fixture.
- Final build and all six model-free harness tests pass. The CPU DFlash
  hidden-state/token reference and engine rejection/acceptance regression passes
  with KV1/KV2/KV4, confirming the existing path for other architectures remains
  functional. This does not enable DFlash for Qwen3.5.
- `validation/stale-smoke-note.txt` identifies an early stale-binary run that
  must not be counted as validation.
- `opus/verified-provenance.json` verifies the requested `claude-opus-5-5`
  consultation. The reviewer prioritized measured state-slot pressure and exact
  ownership/retention before approximations, and audited the HD256 kernel.
  Speculative numerical projections are not substituted for measurements.

Recurrent SSD swapping, activation sparsity, layer skipping, and model-level
speculation were not implemented in this phase. Unsupported Qwen3.5 speculative
and SPAN paths remain explicitly rejected. Earlier website task runs still failed
their completion criteria; this report does not relabel them as successful.

## Reproduction and artifacts

`bench/throughput_workflows.py` prepares, freezes, clones, and runs workloads.
`bench/throughput_experiments.py` owns one server, samples resources, runs the
trace and separate quality probes, and preserves partial work on failures.
`matrix/plan.json` records exact arguments and `matrix/progress.json` completion.
`comparison.csv` and `paired-comparisons.json` can be regenerated with
`python3 bench/summarize_throughput.py results/throughput-exploration`.
Per-run configs include executable hashes, arguments, workload hashes, warmup,
response usage, TTFT/latency, completeness, caps and quality. Process RSS is an
approximate resource measure, not complete Metal allocation accounting.


Raw frozen inputs, per-request JSON, streaming events, and Claude response
metadata are preserved in `raw-artifacts.tar.gz` (1,060 files, approximately
2.4 MB compressed). `raw-manifest.json` lists every member's SHA-256 and the
archive hash; all archived members were read back and verified. Unpacked copies
remain available locally but are ignored to avoid duplicating large prompts in
git. Restore them before rerunning a frozen workload or the summary tool:

```sh
tar -xzf results/throughput-exploration/raw-artifacts.tar.gz \
  -C results/throughput-exploration
python3 bench/summarize_throughput.py results/throughput-exploration
```

All benchmark-owned servers were stopped. The original checkout and its Pi/GPQA
changes were left untouched; experiments live on `experiment/agent-workflows` in
the separate worktree.
