# Small-model agent workflow experiments

Branch: experiment/agent-workflows. Engine base: afc999e.
The existing site-refactor harness was copied from the main workspace because it is not yet committed there.
Provenance and source hashes: preparation.json. Original work remains untouched in the main workspace.

## Model and first run

Selected: Qwen3.5-4B, MLX affine four-bit/group 64, pinned revision
`0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`. This required native hybrid
architecture support; see `docs/qwen35.md`. Before any coding inference:
1. Inspect the model architecture and quantization against Fleet's loader/backend support.
2. Download the requested model, pin its revision and file hashes, and record its quantization.
3. Build Fleet with Metal and record hardware/backend, server argv, harness hashes and memory budget.
4. Run the unmodified harness with the small model before implementing optimizations.
   Do not assume the small model can follow the JSON protocol or coordinate four workers.

Experiments are complete. See [the restart report](restart/README.md) for the final
comparison, limitations, CSVs and preserved artifacts.
Precision audits and state checks are saved alongside this plan.
Speculation/span/offload are explicitly disabled until hybrid-state support exists.

## Recorded coding task

Use exactly demos/site-refactor/brief.txt and fixture/. Restyle the plain Fieldnotes workshop
website while preserving workshop names, IDs, search filtering, newsletter validation and app.js.
One coordinator delegates four workers, each owning one file:
- index.html: page structure and accessibility.
- theme.css: typography, palette and shared tokens.
- styles.css: component styling.
- responsive.css: layout and mobile behavior.

Acceptance: existing structure checks and Chromium behavior checks at 390/1440 pixels;
inspect final screenshots separately for aesthetics. Preserve transcripts, tool feedback,
patches, code, screenshots, server logs, per-call metrics and CSV summaries, including failures.

## Candidate experiments, after baseline

A. Program-aware scheduling, inspired by ThunderAgent/TAO:
   explicit workflow/session/parent IDs and dependency/ready state; schedule ready work by
   workflow progress and avoid starving the job's blocking turn. Existing batching and prefix
   sharing remain the control. Task decisions and file assignments must come from the model.

B. Tool-aware cache retention, inspired by Continuum/CacheScout:
   an explicit tool-wait/resume signal, bounded retention and reuse-aware eviction under the
   same KV budget. Record whether retention actually changes an eviction or avoids prefill.
   Fleet already retains finished sequences, so this needs to beat that existing policy.

C. Controlled context editing, inspired by Cache Engineering for Agents:
   explicit, auditable removal of obsolete tool observations and corresponding cache-state
   provenance. This changes active context and is a separate opt-in quality experiment.
   Start with an exact re-prefill reference for the edited history; any shortcut that keeps
   history-dependent KV must be labeled approximate rather than called exact forgetting.

References:
- https://arxiv.org/abs/2602.13692
- https://arxiv.org/abs/2610.37230
- https://arxiv.org/abs/2511.02230
- https://arxiv.org/abs/2608.14624
- https://arxiv.org/abs/2610.28577

## Comparison rules

Keep the checkpoint, quantization, prompt, checks, sampling, token/step budgets and backend
fixed within each comparison. Record any supported prompt lookup/drafter settings explicitly;
use no drafter unless a compatible one exists for the chosen small model. Span reuse cannot
currently compose with DFlash. Do not silently change models, tool protocol or grading.

Run fresh-server baseline repetitions first, then separate A/B/C ablations and the best
compatible combination. Only one model server benchmarks at a time. Include success rate,
time to checked result, generated tokens, task-wide and per-phase token rates, TTFT, tool/JSON
errors, failed checks, repairs, cache hit/prefill work, evictions and swap traffic. Use a fixed
recorded trajectory replay to isolate engine effects from different model decisions where feasible.

The original four-worker task may provide little scheduling or retention headroom: its
coordinator waits for all four workers, tools are cheap, and the prior run had no cache pressure.
If a mechanism does not engage, report zero/insufficient opportunity. Any long-context or
concurrent-workflow stress test must be a separately labeled additional workload, not a
replacement presented as the original coding task.

## Implemented experiment scope

Two paper-informed variants are selected for this workload:

- Exact prompt-boundary recurrent checkpoints (`--prompt-checkpoints`), informed by
  tool/phase-aware state retention. This enables reuse at a safe agent-turn boundary
  rather than trying to copy unavailable intermediate recurrent states.
- Optional masking of superseded editable-file observations (`--context-prune`).
  Mask observations of at least 512 characters with a small receipt only after a confirmed write to that file or after completed
  worker integration. Keep immutable app.js, current feedback, and the full audit
  history. The edited active history preserves message roles and is re-prefilled normally; no approximate KV
  splice is performed. This can change model decisions and is graded separately.

The scheduling proposal is deferred: this single workflow admits all four workers,
then waits for all of them, so prioritizing its ready queue has little opportunity.

Before these comparisons, the new native Metal implementation needed a performance
fix: persistent GPU wrappers and four-channel recurrent threadgroups. This change
is present in BOTH the new baseline and experimental builds. Its isolated 2.89x
microbenchmark gain is against the initial Fleet implementation, not llama.cpp/vLLM.

The initial unmodified run `qwen35-baseline-01` was stopped after repeated malformed
HTML-write actions and preserved as a failure. Its pending-prefix cache accounting
was incorrect; do not use its reported API cache-hit counts. Native telemetry now
counts actual reuse. The original harness/source hashes are in preparation.json.

Subsequent runs use the same minimal protocol recovery and stall guard for both
baseline and variants: allow ONLY a missing final object brace after a complete
JSON write payload (record a protocol_repair event); never edit the generated file
contents. Reject incomplete strings/other repairs. Stop after three consecutive
invalid tool actions. Keep the 20-call/4096-token limits and all grading checks.
The independent MLX-LM direct-generation probe produced valid JSON on the same
HTML prompt; Fleet's FP16/FP32 arithmetic can produce a different greedy trajectory.
Disabling thinking alone did not fix the native action's missing final delimiter.
