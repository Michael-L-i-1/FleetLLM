# Real coding tasks with Muse Glimmer

This benchmark executes generated file edits and protected acceptance checks.
It is distinct from the fixed-history serving traces: model responses determine
subsequent actions, repairs, token counts, artifacts and completion time.

Two task types use fresh copies of the same starter per run:

- **Coffee website:** eight HTML/CSS sections form a local nonprofit cafe site.
  Tests cover content, responsive layouts, navigation, search and signup.
  Captures preserve the actual development states; aesthetic quality is reviewed
  separately from automated correctness.
- **Ledger migration:** eight Rust modules must reproduce a small C ledger CLI.
  Protected tests cover APIs, edge cases and integrated generated CSV reports,
  cross-checked against C and an independent Python oracle. No external crates.

The three primary modes are Fleet with eight workers, vLLM-Metal with the same
eight workers, and vLLM-Metal with one agent implementing the whole task. The
harness schedules fixed, disjoint assignments; a model integration agent repairs
failures after workers finish. It does not supply solution code. The single
agent has all eight files and the same overall call/token/wall budgets.

## Common conditions

All modes use `mlx-community/Muse-Glimmer-30B-4bit` revision
`3e7677d7a40d348a3daba263a2b1c0aa41910710` (affine four-bit, group64),
temperature0, low reasoning, a4096token response limit and the same JSON tools.
Each complete task has56calls,96000generated tokens and2700seconds maximum.
Eight workers have six calls each before integration. Context is bounded with
the same logged history compaction in every mode. Generated histories naturally
differ; initial eight-worker request bodies match exactly between engines.

Only owned editable paths accept writes. Tests and assembly logic stay outside
the writable contract. Files are replaced atomically; full requests, responses,
actions, hashes, failures and check outputs are durable. Task success requires
all protected checks, not a model's claim. This is per-run filesystem isolation,
not a hosted agent service.

One model server runs at a time. Startup and an unrelated warmup are excluded
from task time. Website capture initialization and check execution are included.
Report time to the first passing check alongside total wall time. Failed attempts
have no time-to-success. The primary demonstration is one trial per task/mode;
per-call distributions are descriptive, not statistical proof of a general win.

## Serving presets and compatibility

Both target engines use the exact same MLX target weights and4GB target KV
allowance. Fleet's selected preset adds the cached target-verified DFlash
assistant (revision `c676cce91658f9568aaf264a416211e2782735b7`), k3 and0.5GB draft
KV. Its extra memory is measured rather than hidden. Approximate SPAN reuse is
not combined with DFlash; that combination is unsupported. Qwen recurrent
checkpoint switches are not advertised as Glimmer optimizations.

vLLM uses prefix caching, native Metal/NAX, compiled MLP and the native decode
pipeline. Prompt lookup was tested and slower on the coding pilot. Its installed
DFlash loader accepts the z-lab Qwen3 checkpoint format, not this Muse Glimmer
assistant. That is implementation coverage, not an exclusive Fleet algorithm.

Installed vLLM-Metal0.32.0 needs the small isolated patch in `compat/` for Glimmer
paged attention, normalization, scaling, gating and wrapper metadata. Both vLLM
arms use the same patch; the global package is unchanged. Sixteen reference
checks cover local/global attention, native BF16 and sliding windows. An initial
strict long-prefill FP32 tolerance failure is preserved: the native mixed-
precision path has small rounding differences; actual serving uses BF16. The
short FP32 checks retain their original stricter tolerance.

## Commands and artifacts

Exact frozen arguments, task hashes and preset decisions are under
`results/real-agent-tasks/`. Run from the repository root with a new output path:

```sh
python3 demos/agent-task-bench/serve.py \
  --config results/real-agent-tasks/configs/fleet-eight.json \
  --task demos/agent-task-bench/tasks/ledger-rust \
  --output /tmp/new-ledger-trial --agents 8
python3 demos/agent-task-bench/test_harness.py
```

`serve.py` owns model startup/shutdown and samples memory. `run.py` owns the actual
agent/tool workflow. `visual/` records sampled screenshots and composes three
lanes on one equally accelerated timeline. It never fabricates a completed site
or assumes Fleet wins. Chromium capture uses CPU rendering.

Darwin memory sampling records per-process physical footprint (including charged
GPU/IOKit memory), RSS, CPU and process coverage at1Hz. Model and whole-run trees
are sampled separately. Sums may overlap shared allocations and miss brief
transients; they are not total machine memory. RSS alone underreports some Metal
allocations. Preserve both startup and task-window peaks.

## Visibility correction

The initial coffee-sitev1grader allowed an opaque pseudo-element to hide hero
text while DOM/interaction tests passed. Those initial runs remain preserved.
A corrected `coffee-site-v2` task adds neutral visible-pixel contribution checks
for the hero heading and calls to action at both viewport widths. The corrected
website comparison reruns every mode from the same plain starter and criteria.
Do not mix v1automatic pass times with v2completion claims. Beauty, contrast
certification and broad coding ability remain separate from these bounded tests.
