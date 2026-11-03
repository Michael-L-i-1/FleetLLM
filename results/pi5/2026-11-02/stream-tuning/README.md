# Exact CPU memory-stream tuning on Pi 5

Base: `dfd72cb` (standard 32-value activation scales, all layers and FFN features).
No production inference code/default has been changed for these experiments.
The candidate changes only memory-fetch hints or task partitioning; it does not
change arithmetic, activation rounding, weight storage, output selection or KV.

## Software prefetch: rejected

Eight generated variants prefetched packed codes and subscales 4/8/16/32 blocks
ahead, with ARM streaming and cache-keep hints. Five rotated rounds used four
threads, grain 64, the same repacked weights and the same quantized activations.
All 360 variant/tensor/batch/round results were bitwise equal to the baseline.

| Largest matrix for each type | Baseline one-row ms | Candidate speedup range |
| --- | ---: | ---: |
| Q2_K `token_embd.weight` | 41.219 | 0.957–0.988x |
| Q3_K `blk.0.ffn_down.weight` | 5.575 | 0.950–0.977x |
| Q4_K `blk.0.attn_v.weight` | 0.030 | 0.897–0.947x |
| Q6_K `output.weight` | 97.706 | 0.935–0.952x |

A value below 1x is slower. Four-input-row controls were approximately 0.98–1.005x
on the large matrices; the prototype only enables prefetch for one input row.
The extra branch/code layout can still affect those control timings.

**The Q2 embedding matrix is not a dense-matmul decode hot path:** inference looks
up embedding rows. Its size made it the inherited microbenchmark's selection,
but its timing must not be translated into a decode speedup. A decoder-only
selector excludes `token_embd.weight` for the follow-up screen.

These results do not justify a full-model prefetch trial or a new inference
setting. The added fetch hints and instructions lost on the initial screen. Hardware
prefetch, instruction overhead and code layout can all affect the result. This does not
rule out other layouts or other CPU architectures.

Raw timings: [glimmer.csv](glimmer.csv), medians/ratios:
[glimmer-summary.json](glimmer-summary.json). No startup, activation preparation
or weight-repacking time is included in kernel timings. Most configurations
have only five short observations, and the tiny Q4 matrix is particularly noisy.

## Decoder-only repeat and row-task partitioning

The follow-up excludes the embedding matrix. Its Q2 matrix is the actual
`blk.0.ffn_gate.weight` decode matmul. All 360 decoder-prefetch results and all
360 row-grain results were bitwise equal to their baseline (1,080 checks across
all three screens). The final candidates use the unmodified kernel library;
only the number of output rows assigned per threadpool task changes.

| Decoder matrix | Default grain 64, one-row ms | Grain 16 speedup | Four contiguous partitions speedup |
| --- | ---: | ---: | ---: |
| Q2_K FFN gate | 4.138 | 0.989x | 0.972x |
| Q3_K FFN down | 5.640 | 1.027x | 0.974x |
| Q4_K attention V | 0.030 | 0.987x | 0.884x |
| Q6_K output | 98.022 | 1.036x | 0.992x |

Other grains were 32/128/256/512/1024/4096. No grain consistently improved all
major matrices. Grain 4096 substantially under-utilized threads for the smaller
matrices. Four-row controls were near baseline for grain 16 and static4 on the
large matrices; large grains also regressed some of those controls. A small
kernel win cannot justify an end-to-end improvement claim.

Decoder-only prefetch again lost on Q2 (0.958–0.990x), Q4 and Q6. Its Q3 result
was inconsistent with the initial screen: apparent median gains up to 1.153x
came with large within-run variation. The baseline samples were
8.655/7.005/6.317/6.480/7.030 ms; `pf8l3` samples were
6.046/5.958/10.194/7.667/6.076 ms. By comparison, the initial baseline was
5.495–5.607 ms and consistently beat `pf8l3` at 5.687–5.898 ms; the subsequent
grain screen's unmodified Q3 baseline was back to 5.509–5.752 ms. These are
**noisy/conflicting observations**, not a repeatable 15% kernel improvement.

Decision: **retain the existing kernels and grain 64**. No full-model trial was
run, because neither experiment produced a broad, repeatable microbenchmark
improvement. No accuracy, perplexity or model quality was sacrificed; no new
speedup is claimed. This is evidence against the tested prefetch/partition
choices on this Pi, not against all CPU kernel tuning.

Raw follow-ups: [decoder-prefetch.csv](decoder-prefetch.csv),
[decoder-grain.csv](decoder-grain.csv), with per-variant sample counts and
medians in their adjacent `*-summary.json` files.

## Reproduction

The isolated driver, variant generator and analyzer live in
[`bench/experiments/pi-stream`](../../../../bench/experiments/pi-stream/README.md).
First compilation of all specialized kernel variants takes several minutes;
compile and benchmark under the shared Pi lock so jobs do not overlap.
The first serial build preceded the lock requirement, while root kept other
benchmarks paused; the measurements held the lock. The decoder-only follow-up
held the lock for both its tiny driver build and measurements. All model runs
were serialized by the root agent. No production Pi checkout or demo launcher
was changed. The experiment ran under `~/fleetllm-experiments/cpu-tuning` on
the Pi; its temporary directory was removed after archiving these results. It
installed no system service or long-running process.

The baseline was four CPU threads, group-32 activation scaling, row grain 64,
and the existing native NEON/dot-product K-quant kernels at `dfd72cb`. Input
floats are deterministic `sin(i*.13) + .2*cos(i*.019)` values. Repacking and
activation preparation are outside the timer. Timing is monotonic wall clock;
variant order rotates by one each round, with five samples per case. Each
candidate compares the entire float output buffer with the baseline using
`memcmp`, failing immediately on any differing byte. These synthetic
activations test arithmetic identity and timing, not task-level quality.

[source-manifest.json](source-manifest.json) records the base commit and source
hashes. The first screen used the driver before adding decoder/grain selection;
the generated prefetch kernels and timing/equality methodology are unchanged.
Final driver syntax checks passed in both modes, Python lint/compile passed,
and documentation links were checked locally. No additional runtime dependency
is introduced into the inference engine.
