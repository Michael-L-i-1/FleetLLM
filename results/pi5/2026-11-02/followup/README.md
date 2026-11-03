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

Qwen3-8B Q4_K_M also struggled under the same direct-answer protocol. It passed
**0/3 strict JSON-field checks**: none of the replies was a JSON object. Reading
the answers manually, its Bayes value `0.5` was correct; its collision value
`4.5` and buffer value `4.76` were incorrect. Decode remained about 2.23–2.24
tok/s. These findings limit the earlier six-task smoke result; neither short
suite establishes GPQA accuracy. Raw replies: [qwen-reasoning](qwen-reasoning/).

An independent Qwen collision probe sent **the identical 96 input token IDs**
to llama.cpp, with the same GGUF, greedy sampling, and the same 128-token output
cap. llama.cpp returned the exact same visible answer and explanation (`4.5`,
incorrect), also without JSON. The raw `same_text` flag is false only because
Fleet's saved text includes its end-of-turn marker. This reproduces that failure
outside Fleet; it does not establish general equivalence of the engines or
evaluate a reasoning-enabled protocol. See
[qwen-collision-llama](qwen-collision-llama/) and [quality-diagnostic.json](quality-diagnostic.json).

`bench/pi_llama_compare.py --max-tokens 128` selects this quality-probe mode.
Omitting the option retains matched-output-count timing mode. Do not infer a
broad performance or accuracy ranking from this single incorrect reply.

## Exact memory-stream tuning: no candidate promoted

Software prefetch and alternate row-task partitioning did not provide a broad,
repeatable improvement. All 1,080 output-buffer comparisons were bitwise equal,
but prefetch regressed the streamed Q2/Q4/Q6 matrices. Smaller task chunks gave
minor Q3/Q6 kernel wins while regressing Q2; static partitions also lost. No
full-model speedup is claimed and no production source/default changed.

The [stream-tuning report](../stream-tuning/README.md) retains all measurements,
including noisy Q3 observations, plus a reproducible benchmark driver. Its
experimental worktree and branch were removed after archiving the findings.

## Draft-only vocabulary shortlist: rejected

Shortlisting 32,768 or 8,192 draft-head rows reduced drafting cost by about
22–28%, but reduced acceptance from 15/30 to 13/34 or 13/33. The extra target
verification work made total decode **6–7% slower**. Unrestricted controls at
both ends were stable (1.10945 and 1.11052 tok/s); all output tokens matched the
ordinary Glimmer target, and swap growth was zero.

No draft-shortlist production code or option was retained. The
[draft-vocab report](../draft-vocab/README.md) saves the four runs, local/Pi
sanitizer validations, 24 engine acceptance/rejection checks and a reproducible
prototype patch. Its experimental branch and worktree were removed.

## T-MAC-inspired subset-sum lookup: rejected

Two row layouts (four and sixteen output rows) and exact/rounded lookup entries
were tested on real streamed Glimmer Q2_K matrices. The best exact layout
produced **0.8015 tok/s versus 0.9595 baseline** in the 24-token full-model probe:
16.5% lower throughput, with identical input/output token IDs and no swap growth.
Loading and prefill also regressed and are recorded separately. Rounded lookup
provided only a 2.6% isolated FFN-matrix gain, no attention-gate gain, and changed
outputs; it did not justify a quality tradeoff or further adoption testing.

This was an independently written prototype inspired by T-MAC's bitplane and
activation subset-sum strategy, not an integration of the upstream generated
runtime. The result rejects this implementation on this model/Pi, not every
possible T-MAC implementation or checkpoint. The
[T-MAC prototype report](../tmac-prototype/README.md) retains upstream source pin,
raw measurements, sanitizer checks, hashes and a complete reproduction patch.
Its code was not merged into production; the experimental branch and worktree
were removed as requested.

## Final state

No additional inference optimization from this follow-up was promoted. Production
inference source and Pi demo launchers remain identical to `dfd72cb`. The useful
repository changes are isolated-build benchmark support, additional quality
checks, reproducible experiment drivers, and archived positive/negative evidence.
No experiment server, system service or hardware/network configuration change is
left behind. Model files and the existing launch commands are retained.
The four temporary Pi experiment directories were removed after copying all
results and reproduction sources into the repository. Final idle state and
production-source hashes are in [final-system-state.json](final-system-state.json).
