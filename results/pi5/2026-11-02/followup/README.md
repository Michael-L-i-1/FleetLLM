# Follow-up: isolated single-agent Pi experiments

Baseline commit: `dfd72cb`. The existing Pi checkout and demo launchers were
kept separate from experiment directories under `~/fleetllm-experiments/`.
Compilation and benchmark runs were serialized on the Pi; microbenchmarks
do not substitute for full-model speed or quality measurements.

## Direct-answer quality baseline

Three self-written questions supplement the earlier short smoke suite. Each
requests a JSON answer plus one short explanatory sentence, under the same
direct-answer chat template as the original speed tests. These are not GPQA
questions, not a representative accuracy benchmark, and not a comparison of
long chain-of-thought protocols.

The expected answers can be checked analytically:

- Bayes: `0.9 * 0.1 / (0.9 * 0.1 + 0.1 * 0.9) = 0.5`.
- Collision: initial energy `13.5 J`; conserved momentum gives final velocity
  `1 m/s`, final energy `1.5 J`, so converted energy is `12 J`.
- Buffer: `4.76 + log10(0.010 / 0.10) = 3.76`.

Glimmer Q2_K with the default 32-value activation scales passed **1/3 answer
checks**. It returned `0.5`, `18`, and `4.76`. The buffer explanation derived
`3.76` while the answer field said `4.76`. All replies were valid JSON; these
two failures were incorrect answer values rather than JSON formatting errors.
The run took 283.66 s including loading and prompt processing, with decode
about 0.96 tok/s, peak RSS 11.94 GB and zero system swap growth.

This establishes a baseline-quality limitation for this checkpoint and prompt
protocol. A speed optimization preserving these tokens also preserves these
mistakes; it does not establish that the model is adequate for GPQA.

Raw answers: [glimmer-reasoning](glimmer-reasoning/), machine checks:
[reasoning-checks.json](reasoning-checks.json). `json_fields` grades the answer
field while permitting an explanation; the explanation still needs inspection.

## Exact memory-stream tuning: no candidate promoted

Software prefetch and alternate row-task partitioning did not provide a broad,
repeatable improvement. All 1,080 output-buffer comparisons were bitwise equal,
but prefetch regressed the streamed Q2/Q4/Q6 matrices. Smaller task chunks gave
minor Q3/Q6 kernel wins while regressing Q2; static partitions also lost. No
full-model speedup is claimed and no production source/default changed.

The [stream-tuning report](../stream-tuning/README.md) retains all measurements,
including noisy Q3 observations, plus a reproducible benchmark driver. Its
experimental worktree and branch were removed after archiving the findings.
