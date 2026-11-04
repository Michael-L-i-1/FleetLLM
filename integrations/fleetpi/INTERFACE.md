# FleetPi interface (Pi fork/CLI worker -> root, Muse, Astra)

Status: interface v1, written before heavy implementation. Owner of this file:
the Pi fork/CLI worker. Root owns `UPSTREAM.json`, the upstream pin, provenance
and bootstrap metadata. Muse owns the C engine (`src/chat.c`, `src/server.c`,
`src/engine.c`, `tools/common.h`).

Implementation status: v2 (see `HANDOFF.md`). v2 amendments supersede anything below:

* Common profile for all modes: `maxOutputTokens 4096`, `contextWindow 96000`
  (models.json `maxTokens`/`contextWindow`, Fleet `--max-tokens 4096`). Pi compaction
  therefore triggers at ~79.6K context tokens (96000 - 16384 reserve).
* `runFestival({..., beforeRun?: ({workspace, outputDir}) => Promise<void>})`: awaited after
  the workspace is prepared, before any session opens or model call. If `startTimeUnix` is
  omitted, the event clock starts right after the hook; `totals.wallSeconds` and round
  `wallSeconds` are monotonic task time excluding setup and the hook.
* `outputDir` may already contain `logs/` (supervisor/server logs); anything else is refused.
* Metrics: `decode*` fields removed. Headline is `workflowOutputTokensPerSecond`
  (= outputTokens / task wall). `streamTokensPerSecond`/`streamSeconds`/`streamedOutputTokens`
  are client-observed (first non-empty text/reasoning/tool-argument delta to message end), not
  engine decode speed. `call_end` adds `firstTokenMs`.
* `CheckOutcome.pageHashes`: hashes of the exact snapshot checked; memo key uses the same
  bytes; ledger records these hashes.
* Fleet `capabilities.context_window: null` (or missing) means unknown and is ignored; the
  common 96000 limit is used either way. `max_active` null falls back to 8.

## 0. Environment blocker for this worker (read first)

In this Claude Code session every Bash command except trivial allow-listed ones
(`pwd`, `node --version`) is auto-denied: no `npm`, `node <script>`, `git`,
`ls`. I therefore cannot `npm ci`, build, type-check, run vitest, or commit in
the vendor fork. All code below was written against the actual pinned Pi
sources (read with file tools) but is **unexecuted**. Root (or a session with
shell approval) must run section 9 and report failures back through
`integrations/fleetpi/ROOT-NOTES.md`; I will fix them.

## 1. Where the code lives and how it is built

* Fork augmentation = a new workspace package inside the pinned fork:
  `integrations/fleetpi/vendor/pi/packages/fleetpi/` (package name
  `@earendil-works/fleetpi`, private, version 1.5.0, binary `fleetpi`).
  It imports only public SDK exports of `@earendil-works/pi-coding-agent` /
  `@earendil-works/pi-ai`. **No upstream core file is modified**, so the two
  baseline modes run genuinely upstream Pi 1.5.0 code; the enhancement is a Pi
  extension (`fleetExtension`) plus the FleetPi host. The only edits outside
  `packages/fleetpi/` are: root `package.json` `build` chain gains
  `&& cd ../fleetpi && npm run build`, and `package-lock.json` must be refreshed
  for the new workspace (`npm install --package-lock-only --ignore-scripts`).
* No new third-party dependencies. Runtime deps are workspace packages only.
* Build: `npm ci --ignore-scripts && npm run build` at fork root (builds dist of
  every package including fleetpi). CLI: `node packages/fleetpi/dist/cli.js` or
  the `fleetpi` bin after `npm link`/workspace bin.
* Tests (model-free, faux provider, no network):
  `cd packages/fleetpi && node ../../node_modules/vitest/dist/cli.js --run`.

## 2. Programmatic runner API (for root's benchmark wrapper)

```ts
import { runFestival } from "@earendil-works/fleetpi"; // dist/index.js

type FestivalMode = "fleet8" | "vllm8" | "vllm1";

interface RunFestivalOptions {
  mode: FestivalMode;
  taskDir: string;          // demos/agent-task-bench/tasks/festival-site
  outputDir: string;        // must not exist or be empty; everything is written here
  baseUrl: string;          // e.g. http://127.0.0.1:8090/v1 (server already running)
  model: string;            // served model id, e.g. "glimmer-mlx4"
  onEvent?: (e: FleetEvent) => void;   // same objects as events.jsonl lines
  signal?: AbortSignal;     // cancellation: aborts sessions, saves state, resolves
  limits?: Partial<RunLimits>;          // identical defaults for all modes
  resume?: boolean;         // continue from outputDir/state.json + session files
  pythonPath?: string;      // default "python3" for the protected checker
  startTimeUnix?: number;   // shared clock origin for visual recorder (default now)
  // Model-free test seams (production leaves unset): modelRuntime, piModel,
  // checkCommand, sessionSettings, telemetry.
}

interface RunLimits {
  roundWallSeconds: number;       // default 1500
  runWallSeconds: number;         // default 2700
  maxTurnsPerAgentRound: number;  // default 40 (assistant responses incl. retries)
  repairPassesPerRound: number;   // default 1 (runner-issued failing-check follow-up)
  checkerConcurrency: number;     // default 2 (common checker semaphore, all modes)
  maxOutputTokens: number;        // default 8192 (model maxTokens in models.json)
  contextWindow: number;          // 131072 (advertised; Pi token-based compaction)
}

interface FestivalResult {
  mode: FestivalMode;
  passed: boolean;                // final official round-3 check passed
  infrastructureError?: string;   // protected-hash mismatch, checker crash, server loss
  cancelled: boolean;             // user cancellation; state.json + ledgers allow resume
  timedOut: boolean;              // runWallSeconds reached (current round still officially checked)
  totals: Totals;                 // see section 6
  rounds: RoundResult[];          // per round: wall, official check, per-page pass, totals
  quality: { round: number; checksPassed: number; checksTotal: number; failures: string[] }[];
  outputDir: string;
}
```

`runFestival` never starts or stops a model server. Server ownership:

* `fleet8`: the CLI (`fleetpi demo festival`) owns `fleet-serve` through
  `EngineSupervisor`. Root's wrapper may instead start Fleet itself and call
  `runFestival({mode:"fleet8"})`; the runner then only reads
  `/v1/fleet/capabilities` and `/v1/fleet/stats`.
* `vllm8`, `vllm1`: root's wrapper owns vLLM (existing
  `results/glimmer-festival/configs/vllm-default.json` plus
  `--enable-auto-tool-choice --tool-call-parser muse_glimmer`).
* Exactly one server at a time. `fleetpi demo festival --compare` runs the three
  modes sequentially and refuses to start if the port answers `/health`.

## 3. CLI

```
fleetpi                                 interactive: starts engine, opens a Pi
                                        session (8-slot admission) in cwd
fleetpi demo festival [--mode fleet8|vllm8|vllm1] [--compare]
        [--output DIR] [--task-dir DIR] [--json] [--port 8090]
        [--fleet-bin PATH] [--model-path PATH] [--vllm-bin PATH] [--resume]
```

* Default terminal output: a live in-place dashboard (agents, state, elapsed,
  tok/s, KV/cache reuse, check progress). Everything else goes to
  `outputDir/logs/*.log`. `--json` prints only `events.jsonl` lines to stdout.
* Ctrl-C: first press aborts sessions (`session.abort()`), saves state and
  shuts the engine down gracefully; second press SIGKILLs the engine group.
* `--compare` runs fleet8 (owned Fleet), then vllm8, then vllm1 (owned vLLM via
  the existing default config) into `DIR/{fleet8,vllm8,vllm1}`, writes
  `DIR/comparison.json`. Root asked for one validated 3-way run, no sweep.

Hardware profile: only `mac-m5pro-48gb-glimmer30b-4bit` is defined. It is
selected when `process.platform==="darwin"`, `arch==="arm64"` and total memory
is >= 44 GiB; otherwise the CLI refuses (`--profile` cannot force a different
unsafe profile). Fleet command line for that profile:

```
build/fleet-serve --model <HF snapshot 3e7677d7...> --device metal
  --name glimmer-mlx4 --budget-gb 28 --kv-gb 4 --max-active 8 --admit
  --row-budget 128 --ngram --reasoning low --max-tokens 8192 --port 8090
  --spec-k 15 --spec-rows 128 --shared-prefix-coalescing
```

No learned drafter, no approximate SPAN. `--max-tokens 8192` (old configs had
4096) so Fleet and vLLM honour the same per-response cap Pi requests.

## 4. Three fair modes

| | fleet8 (enhanced) | vllm8 (baseline) | vllm1 (baseline) |
|---|---|---|---|
| Pi code | fork, upstream core + `fleetExtension` | upstream core | upstream core |
| agents | 8 role sessions, disjoint page ownership | same 8 roles | 1 session owning all 8 pages |
| model registration | common `models.json` (section 5) | same | same |
| tools | guarded built-in `read`,`edit`,`write`,`ls` + `check_site` | same | same |
| path guard | common `PathGuard` | same | same |
| compaction | Pi default, token-based, contextWindow 131072, reserve 16384, keep 20000 | same | same |
| retries | Pi default auto-retry | same | same |
| limits | `RunLimits` defaults | same | same |
| shared project context | stable system-prompt section, byte-identical across agents, before assignment; survives compaction | ordinary Pi: brief+content as the first user message, then assignment | same as vllm8 |
| scheduling | `AdmissionQueue`: permits = engine `max_active`; held only while a provider request streams, released during tool waits (checker/browser), KV-pressure gate from `/v1/fleet/stats`, priority+aging; per-agent ready repair as soon as that agent settles | all 8 sessions prompt concurrently; round barrier then repair pass | single session; round barrier then repair pass |
| progress ledger | durable ledger re-injected after each compaction and on resume | ledger persisted for metrics/resume only, not injected | same as vllm8 |

Same task text, same weights, same tools, same limits, same compaction and
path guard. No artificial delays, baselines are not serialized, vLLM's native
default prefix cache stays on, model replies are never cached. Checker results
are memoized by (workspace page hashes, round) in all modes (that is checker
work, not model output).

Task text: taken from `task.json`, with exactly two sentences that describe the
old toy JSON harness removed ("Use the harness JSON tools: ..." and "Finish
with the done action ..."), replaced by one common Pi sentence: use read/edit/
write on owned files, call `check_site`, reply with a short plain-text summary
when done. `adaptInstruction()` fails closed if any toy-protocol wording
remains. Round r instructions are delivered only when round r is released
(after round r-1 official check); nothing from later rounds is visible earlier.

## 5. Common model registration (all modes)

`outputDir/pi-agent/models.json` (agentDir is isolated from `~/.pi`):

```json
{"providers":{"fleetpi-local":{
  "baseUrl":"<baseUrl>","api":"openai-completions","apiKey":"local",
  "compat":{"supportsDeveloperRole":false,"supportsStore":false,
            "maxTokensField":"max_tokens","supportsReasoningEffort":true,
            "thinkingFormat":"openai","supportsUsageInStreaming":true},
  "models":[{"id":"<model>","name":"Muse Glimmer 30B 4-bit","reasoning":true,
    "input":["text"],"contextWindow":131072,"maxTokens":8192,
    "cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0},
    "samplingParams":{"temperature":0,"chat_template_kwargs":{"reasoning_strength":"low"}}}]}}}
```

Thinking level `low` -> `reasoning_effort: "low"`. No `promptCache` entry, so
Pi never sends cache-warming requests. Settings (in-memory): compaction
defaults, retry defaults, `cacheWarming: off`.

## 6. Usage, timing and metrics contract

Per provider call (from Pi's finalized assistant `message_end`):
`input = prompt_tokens - cached_tokens`, `cacheRead = cached_tokens`,
`output = completion_tokens` (includes reasoning), `prompt = input + cacheRead`,
`total = prompt + output`. Thinking text is kept separate from actions
(`thinkingChars`, `textChars`, `toolCalls`). Recorded per call in
`calls.jsonl`: agent, round, phase, kind (`turn` | `compaction` | `retry`),
start/firstToken/end unix ms, tokens, stopReason, error. Compaction summary
usage comes from `CompactionResult.usage`; compaction wall time from
`compaction_start`..`compaction_end`; retries from `auto_retry_*`. All are
included in `totals`:

```ts
interface Totals { wallSeconds; calls; promptTokens; cachedTokens; inputTokens;
  outputTokens; totalTokens; reasoningTokens; compactions; compactionTokens;
  compactionSeconds; retries; toolCalls; toolSeconds; checkerRuns;
  checkerSeconds; admissionWaitSeconds; decodedOutputTokens; decodeSeconds;
  decodeTokensPerSecond }
```

`decodeTokensPerSecond` is per stream (first token to end of each turn call);
aggregate throughput for a run is `outputTokens / wallSeconds`. `calls` and
`outputTokens` include compaction summary calls (`kind: "compaction"`).
Memoized checker results do not count as checker runs.

## 7. Events (`outputDir/events.jsonl`, one JSON object per line)

Every line: `{"event","t_unix","elapsed","round","round_title","phase","mode", ...}`.
`elapsed` is seconds since `startTimeUnix`. Compatible with
`demos/agent-task-bench/visual/record.mjs` (`round_start`, `phase`,
`round_complete`, `file_write`).

| event | extra fields |
|---|---|
| `run_start` / `run_resume` | taskDir, workspace, agents[{agent, owns}], limits, baseUrl, model |
| `engine` | state: telemetry (runner, with capabilities) or starting/ready/stopping/stopped/exited (CLI supervisor) |
| `round_start` | title |
| `phase` | phase: `working` / `repair` (baselines) / `official_check` / `done` |
| `agent_state` | agent, state: queued/streaming/tool/checking/compacting/settled/failed/aborted, outcome? |
| `admission` | agent, action: wait/grant/release/abort, waitMs, permitsFree, waiting, kvFreeFraction (fleet8 only) |
| `call_end` | agent, kind turn/retry/compaction, prompt, cached, output, ms, tokPerSec, stopReason, toolCalls |
| `tool_end` | agent, tool, ms, isError |
| `file_write` | agent, path (workspace-relative, e.g. `pages/home.html`), sha256, bytes, tool (write/edit) |
| `guard_block` | agent, tool, kind (outside-workspace/not-writable/symlink), path |
| `check` | agent, official, passed, ownedPassed, checksPassed, checksTotal, ownedChecksPassed, ownedChecksTotal, seconds, memoized, infrastructureError |
| `compaction` | agent, reason, tokensBefore, ms, aborted, usage |
| `retry` | agent, attempt, error |
| `ledger` | agent, completedPages, pendingTask (kind or null) |
| `agent_error` | agent, error |
| `no_tool_calls` | agent, kind, lastText — a settled task made no tool call (typical symptom of missing native tool parsing; reported, never hidden) |
| `round_complete` | title, passed, checksPassed, checksTotal, pagesPassed{}, timedOut |
| `engine_stats` | active, queued, kvFreeFraction, kvFull, sharedPrefixTokens, prefillTokens, generatedTokens (fleet8, <= 1 Hz) |
| `infrastructure_error` | error |
| `run_end` | passed, cancelled, timedOut, infrastructureError, totals |

`round` is null until the first `round_start`. Official round checks are run by the
runner after all agents of a round settle (and after repairs), on a snapshot, with
results in `official/round-N.json`. They are the scored quality numbers.

Other outputs: `state.json` (resumable run state), `ledger/<agent>.json`,
`sessions/<agent>/*.jsonl` (actual Pi session files, outside workspace),
`calls.jsonl`, `checks/<id>/results.json`, `official/round-N/results.json`,
`workspace/` (agent-editable copy of the fixture), `summary.json`, `logs/`.

## 8. What FleetPi needs from Fleet (Muse) and root

Required for tool use (blocker if missing; FleetPi reports, never degrades):

1. `POST /v1/chat/completions` with `stream:true` accepting OpenAI `tools`
   (function schemas), `tool_choice` absent/`auto`, `messages` with roles
   `system`, `user`, `assistant` (with `tool_calls`, content may be null/""),
   `tool` (`tool_call_id`, content string). Glimmer ATEM formatting/parsing per
   official template.
2. SSE output: `delta.reasoning_content` for to=self reasoning, `delta.content`
   for answers, `delta.tool_calls[{index,id,type:"function",function:{name,
   arguments}}]` (buffered single delta per call is fine), final
   `finish_reason: "tool_calls" | "stop" | "length"`, then a usage chunk with
   `usage.prompt_tokens`, `usage.completion_tokens`,
   `usage.prompt_tokens_details.cached_tokens`, then `data: [DONE]`.
   Truncated/length calls must not be emitted as tool calls.
3. Accept `max_tokens` (<= 8192), `temperature`, `reasoning_effort`,
   `chat_template_kwargs`, `stream_options.include_usage`; ignore unknown keys.
4. Client disconnect cancels the request (Pi aborts via AbortSignal).
5. `GET /health` -> 200 when ready.

Telemetry (used by admission/UI; absent fields are tolerated and shown as n/a):

```
GET /v1/fleet/capabilities -> {"engine":"fleet","model":"glimmer-mlx4",
  "max_active":8,"context_window":131072,
  "features":{"native_tools":true,"tool_streaming":"buffered",
              "shared_prefix_coalescing":true,"ngram_spec":true,
              "learned_drafter":false}}
GET /v1/fleet/stats -> {"active":3,"queued":1,
  "kv":{"total_bytes":4294967296,"free_bytes":1073741824,"full":false},
  "tokens":{"prefill":123,"generated":456,"shared_prefix":789},
  "requests":{"completed":12}}
```

Admission gate: when `kv.full` or `free_bytes/total_bytes < 0.08` while at
least one FleetPi request is streaming, new requests wait for a permit release
or the next stats sample (500 ms) rather than piling onto the engine queue.

From root: built `build/fleet-serve`; vLLM command (section 2); a confirmed
Playwright install for `check.py`; running section 9.

## 9. Commands root must run (I could not)

```
cd integrations/fleetpi/vendor/pi
npm install --package-lock-only --ignore-scripts   # new workspace package
npm ci --ignore-scripts
npm run build
cd packages/fleetpi && node ../../node_modules/vitest/dist/cli.js --run
cd ../.. && npx tsc --noEmit -p packages/fleetpi/tsconfig.json
```
