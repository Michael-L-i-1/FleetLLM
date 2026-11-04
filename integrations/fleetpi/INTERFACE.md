# FleetPi integration contract

Implementation: pinned Pi fork, `packages/fleetpi`. The public entry point exports `runFestival`, `EngineSupervisor`, server specs, model registration, guarded tools and Fleet telemetry. The new package adds the harness without changing upstream Pi core.

## Programmatic run

```ts
await runFestival({
  mode: "fleet8", // or vllm8 / vllm1
  taskDir, outputDir, baseUrl, model: "glimmer-mlx4",
  onEvent, signal, resume: false,
  beforeRun: async ({ workspace, outputDir }) => { /* optional initial capture */ },
});
```

The caller owns the engine. `runFestival` never starts a model server. The CLI supervises one owned engine process group and serializes comparison modes. A non-resume output directory must be new, apart from setup `logs/`. `beforeRun` is awaited after workspace preparation and before Pi sessions or model requests start. Its time is excluded from task wall. An explicit `startTimeUnix` anchors event timestamps; otherwise the task/event clock starts after the hook.

Common limits: 96,000 context tokens, 4,096 output tokens per response, 40 turns per agent/round, one runner-issued repair pass per round, two browser-checker permits, 1,500 active seconds per round and 2,700 per run. Pi performs its real token-based compaction (16,384 reserved; 20,000 recent). There is no character-based transcript truncation.

`passed` requires every staged milestone to pass. `finalPassed` separately records the final cumulative site score, which can pass after earlier failures.

`totalsScope: "attempt"` identifies current-execution totals. `runTotals` is cumulative only when all attempts have complete accounting; it is null after an unclean interruption. `accountingComplete`, `unknownWallInterval` and `wallBudgetPolicy` make that explicit. Active-time checkpoints exclude downtime; a crash can lose the interval after the last checkpoint. Resume validates persisted model/limits/settings and preserves turn, repair and active-time budgets. Frozen version-1 benchmark states are immutable evidence; the current version-2 runner rejects reinterpretation.

## Models and tools

All modes use one isolated local provider (`openai-completions`), identical model registration and guarded Pi built-in `read`, `edit`, `write`, `ls`, plus `check_site`. Facts, requirements and cumulative browser tests are common. A single agent owns all pages; eight agents own one each. Writable scope is enforced below the tool implementation, with identical tool schemas/descriptions. Global Pi extensions, skills and context files are disabled in the benchmark.

Enhanced mode appends common project context to the system prompt, loads the Fleet admission/ledger extension, and schedules per-agent ready checks and repairs. Baselines place the same project facts in the initial user message, omit that extension and use ordinary barrier repairs. Both baseline modes retain default vLLM performance settings. Glimmer attention and streaming-parser compatibility corrections are isolated, pinned and documented separately.

## Durable output

- `workspace/`: actual model-edited page fragments and protected supplied files.
- `sessions/<agent>/`: authoritative Pi SessionManager transcripts.
- `ledger/<agent>.json`: checks, verified page hashes, pending work, budgets and session link; mirrored into the Pi session.
- `state.json`, `attempts/`: atomic run state and immutable attempt summaries.
- `events.jsonl`, `calls.jsonl`: lifecycle and provider usage/latency evidence.
- `checks/<unique-id>/`: immutable checker snapshots and outcomes; memo keys hash the bytes actually checked.
- `official/round-<n>.json`, `summary.json`: cumulative acceptance results and run summary.

Ledger passes require matching current page bytes; writes or out-of-band changes invalidate verified status. A failed or aborted provider task remains pending. Ctrl-C aborts requests, saves progress and stops only owned processes; the checker has owned process-group termination/escalation. A second Ctrl-C forces engine termination.

## Events, telemetry and metrics

Events include `event`, `t_unix`, `elapsed`, `mode`, `round`, optional `round_title` and `phase`. Core events: `run_start`, `run_resume`, `round_start`, `agent_state`, `admission`, `call_end`, `tool_end`, `file_write`, `check`, `compaction`, `ledger`, `engine_stats`, `round_complete`, `run_end`. `file_write` carries the actual relative path, agent, SHA-256 and byte length, allowing a recorder to follow real edits.

Fleet endpoints are `/v1/fleet/capabilities` and `/v1/fleet/stats`. Unknown fields remain null/unknown. In particular, native `context_window: null` does not imply zero or the model's advertised maximum. The deployment profile supplies the common context budget. KV pool headroom is not total system free memory. Native tools are buffered; first observed delta measures delivery latency, not first GPU token.

Headline throughput is provider-reported output tokens divided by actual task wall, including tools/checks/retries/compaction. Per-call request throughput uses full request duration and is labeled separately; stream-rate inference is unavailable/null. Concurrent request durations must not be summed and labeled aggregate throughput. Checker execution costs are counted centrally, rather than multiplied by result subscribers. Memory is sampled separately as Darwin process-tree physical footprint, including charged GPU memory.

The checked-in trial protocol and frozen patch identify the exact benchmark version. Subsequent reliability/accounting fixes do not rewrite raw measurements.
