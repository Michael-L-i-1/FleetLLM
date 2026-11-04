# Independent FleetPi correctness review

## Completed follow-up

After the native worker handed over ownership, this reviewer completed the C fixes and focused tests. **17/17 exact-byte goldens now pass**, including the actual exported Pi SDK schemas. CPU ASAN/UBSAN checks pass for the parser, actual server reply/routing/cancellation/telemetry functions, and deterministic SWA engine fixtures. See `NATIVE-VALIDATION.md`, `native-checks-final.log`, and `template-parity-final/results.json`. The original findings and 11/15 failure record below are retained as the initial audit; they do not describe the final fixed state.

The follow-up changed native C and tests, including a backward-compatible JSON node flag used only by Glimmer formatting. It still did not run any GPU/model job. Root separately reported successful real-Pi native-tool smoke tests against both engines.

## Initial audit (preserved)

Reviewed 2026-11-04. This is an independent review of an in-progress implementation, not a release approval. Native C and Pi TypeScript files were not edited. No model, GPU, Metal build, or inference server was started. The only compilation was a CPU executable linking `chat.c` and `json.c` with tokenizer detection stubs.

## Formatter parity result

**11/15 fixtures pass exact UTF-8 byte comparison.** The reference is the checkpoint's cached `glimmer-chat-template.jinja`, rendered through Transformers' actual `_compile_jinja_template`. The four root-provided fixtures remain unchanged. Eleven additional independent cases are in `reference/template-extra-golden-fixtures.json`.

Artifacts:

- `check-template-parity.py`: independent fixture-to-C driver and optional extra-golden generator.
- `template-parity-initial/results.log`: compiler output and case results.
- `template-parity-initial/results.json`: input/source SHA-256s, outcomes, first differing bytes and snippets.
- `template-parity-initial/*.expected.txt`, `*.actual.txt`, `*.diff.txt`: exact inputs to comparisons and failures.
- `template-parity-initial/generated-fixtures.c`: independently generated C initializers.

The driver snapshots C sources and headers before compilation; none changed during this run. It derives no expected text from the native implementation. It independently resolves tool result names using explicit `name`, then matching `tool_call_id`, then ID fallback. Therefore a pass does **not** prove that HTTP request ingestion populates those fields correctly, nor that native ID resolution works when `name` is omitted. Those require separate server integration coverage.

Passing cases include representative Pi schemas, multiple historical calls, reversed result order, incoming reasoning, Unicode/HTML whitespace, nested arrays/objects, an assistant call followed by another assistant message, explicit result-name precedence, absent/late system messages, text content arrays, and shared tool namespaces.

Failures in the captured native source:

1. `adjacent-assistant-and-explicit-tool-name` and `extra-explicit-end-turn-false`: explicit false ends with `<|eot|>` instead of `<|eom|>`. The ternary `m->end_turn ? true : m->end_turn < 0 ? false : ...` treats the documented `-1` value as true.
2. `extra-empty-reasoning`: an empty, non-NULL `reasoning_content` emits an empty self block. Official Jinja emits it only for a truthy/nonempty value.
3. `extra-float-values`: `1.0`, `-0.0`, and `1e-07` become `1`, `-0`, and `9.9999999999999995e-08`. This is a general exact-template fidelity failure. The current Pi file-tool arguments are principally strings and integer offsets, so this is less immediately relevant than the framing failures. JavaScript also normalizes integral floating values before transmission; fractional/exponent spelling still merits coverage against actual wire inputs.

The compile succeeded with one warning about unused `INVOKE_END`. No sanitizers were used in this formatter-only run.

Reproduce from the worktree root, preserving the initial failure record:

```sh
python3 results/fleetpi/setup/check-template-parity.py --output results/fleetpi/setup/template-parity-recheck
```

Regenerate only the extra independent goldens, using the installed Transformers environment:

```sh
/opt/homebrew/Cellar/vllm-metal/0.32.0/libexec/bin/python3.12 results/fleetpi/setup/check-template-parity.py --generate-extra
```

The driver can accept alternate saved goldens with repeated `--golden PATH`. Replace the representative schemas with exported real SDK schemas in a separate fixture file before the benchmark; do not overwrite expected outputs to match native output.

## Execution and protocol blockers

- **Preserve output-cap status.** Pi `packages/agent/src/agent-loop.ts:272–279` does not execute any tool call from a response with final `stopReason: "length"`; it emits synthetic failed results and continues. Tool streaming events alone do not execute tools. Current native `reply_fields()` still changes the finish reason to `tool_calls` whenever any call parses. A complete first call followed by an incomplete second call defeats Pi's safeguard. The installed vLLM streaming path changes `stop` to `tool_calls` only when the underlying finish reason is `stop`.
- **Cancellation ownership.** The new timed disconnect checks improve buffered requests. However, canceled queued requests are removed only inside the admission loop, which is disabled while all `max_active` slots are occupied. Such cancellation can wait for unrelated work to finish. `cancelled` is also written under the request mutex and read outside that mutex; the shared flag needs synchronized/atomic access. The engine must acknowledge removal/retirement before the connection thread frees a request.
- **ATEM coercion.** The general native `json_parse()` accepts a valid prefix and ignores trailing bytes. The new ATEM-specific full-value recognizer is necessary; retain tests proving `123 Main Street`, `true story`, malformed JSON and whitespace-preserving raw strings survive unchanged. Do not infer types from schemas differently between engines.
- **Channel scoping.** Only real tool-recipient messages may create calls. The installed parser's permissive unframed fallback is not a safe rule to copy into Fleet. Match `self` and `user` exactly, and do not let an incomplete invoke borrow a closing tag from a different channel.
- **Historical reasoning.** Official Jinja consumes `reasoning_content`. Installed vLLM accepts both spellings and restores that alias before template rendering. Pi preserves the incoming reasoning field as replay metadata. Fleet currently reads `reasoning_content`; accepting `reasoning` too is needed for robust cross-provider replay. No Pi hook is required solely to repair vLLM's alias.
- **Multiple calls.** The native response path currently has eight call slots. Define and test behavior beyond eight; silently dropping calls is unacceptable. IDs must remain distinct and results must bind to the corresponding historical name.

## Remaining acceptance gates

1. Exact formatter parity after fixes, including actual exported Pi tool definitions. Separately exercise HTTP JSON ingestion with string-valued `function.arguments`, reverse-order tool results, incoming reasoning and explicit false `end_turn`.
2. Parser fixtures: invoke examples in self/user/bare content, valid tool channel, damaged boundaries, Unicode, literal HTML, JSON/raw values, duplicate parameters, multiple calls, and unknown names. Normalize only a registered doubled bare name such as `write.write` to `write`; never remap arbitrary namespace leaves.
3. Fake-model Pi execution: read/edit/write using actual built-in tools; cap after a complete first call; abruptly end stream; abort after a streamed call but before final status. Assert no file write on capped/error/aborted responses.
4. Cancellation before admission, during prefill, during buffered generation and with every active slot occupied. Verify bounded acknowledgment, zero execution, capacity reclamation and successful next request.
5. Persist, reopen, compact twice and resume a real Pi session using a fake provider. Preserve an early unique requirement and durable completed-page state. Count summarization/retry tokens and time in benchmark totals.
6. Root-owned real-model two-agent/two-turn native-tool pilot on each engine before the eight-agent comparison. Capture actual wire requests and compare rendering/usage behavior.
7. Same pinned target weights, Pi tools/path guard, task, output budgets and token-based compaction across lanes. Keep vLLM's default performance/cache settings. `--enable-auto-tool-choice --tool-call-parser muse_glimmer` enables required model functionality and is not performance tuning. The previous JSON-harness timing remains separate historical evidence.

The common read/edit/write path guard should prevent writes outside assigned files and to protected checkers/assembly. Do not reintroduce unrestricted bash. Preserve fresh workspaces/sessions, cumulative 187/219/251 checks, failed trials and actual captured screenshots.

## Sources and scope

Local official references: `reference/glimmer-chat-template.jinja`, `reference/vllm-glimmer-tool-parser.py`. Pi checkout reviewed: `9e870f6f002ca6c21ce010fdb99122afe3738b28`. Installed vLLM sources were read under `/opt/homebrew/Cellar/vllm-metal/0.32.0/libexec/lib/python3.12/site-packages/vllm/` to verify finish-reason and reasoning-alias behavior.

Upstream contracts: [custom provider](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/custom-provider.md), [SDK lifecycle](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/sdk.md), [compaction](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/compaction.md).

Extensions/SDK composition suffice for Fleet lifecycle, hardware queue, status UI and common guards while retaining Pi's actual tool/session/compaction implementation. Native engine/server changes are required for Glimmer ATEM protocol support and reliable observability/cancellation. This review does not certify those untested server changes or the inference engine.
