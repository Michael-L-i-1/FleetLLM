# FleetPi native protocol and scheduling validation

The native completion implements the checkpoint's Glimmer ATEM chat/tool contract. It preserves real Pi assistant calls and tool results, reasoning history, explicit turn boundaries, tool schemas and recipients. The server's tool channel retains framing; reasoning/user echoes and unframed markup never become calls. A capped or canceled response emits no executable tool-call fields, even when an earlier call in the same response is complete.

Tool binding collapses `write.write` only if `write` is registered and `write.write` is not itself registered. Unknown names remain unchanged for Pi to reject. Response capacity is derived from the number of possible call markers in the reply, replacing the fixed eight-call limit. IDs include process-instance time/PID and an atomic counter to avoid ordinary restart collisions.

Cancellation uses a C11 atomic flag, timed peer-disconnect checks, and a queue drain independent of active-slot availability. The engine detaches canceled requests before acknowledging completion; the connection then owns freeing them. Active cancellation is observed between engine steps. Buffered tool requests remain explicitly advertised as buffered.

The stale-prefix fix permits waiting only on a source that can still produce the required future endpoint. A longer cached/done match with missing sliding-window pages no longer masks a shorter usable source or a live prefill leader. No model arithmetic changes were introduced by this fix.

## Completed checks

- **17/17 exact UTF-8 formatter goldens:** four initial independent reference cases, eleven extra cases, and two cases with actual exported Pi SDK tool schemas. Includes Unicode, nested JSON, historical calls and reversed results, reasoning, empty reasoning, explicit false turn ending, namespace metadata, float formatting and text-only behavior. Expected bytes come from the actual Transformers Jinja compiler, not native C.
- **Parser/formatter ASAN+UBSAN:** strict channel selection, JSON-versus-raw parameter types, Unicode/whitespace, malformed boundary separation, more than eight calls, exact/unknown/registered doubled names, native historical ID resolution, and ChatML/Llama parser and formatter regressions.
- **Server ASAN+UBSAN:** actual `reply_fields`, token routing, UTF-8 chunk assembly, twelve calls, distinct IDs, complete-first/truncated-second suppression, canceled responses, direct-answer mode, socketpair disconnect detection, canceled requests behind a live queue head while the active limit is occupied, acknowledgment/free ordering, and structured telemetry snapshots.
- **Actual engine/fake backend ASAN+UBSAN:** aligned/unaligned/staggered eight-way SWA prefix sharing, deterministic output and retained-KV parity, COW, leader removal/fallback, canceled waiters, memory pressure and explicit full/SWA page reclamation. Four new stale-source cases cover done/cached sources and immediately usable/future-active alternatives; all reuse the expected 2,500 tokens.
- `git diff --check` passes. The whole CPU library was compiled with sanitizers in `build-fleetpi-asan`, separate from root's Metal build. Darwin leak sanitization is disabled; fixtures free their allocations and the engine fixture verifies complete KV pool reclamation.

Exact formatter evidence is in `template-parity-final/results.json` and `results.log`. The JSON records hashes of the tested inputs and captured native sources. Native sanitizer output is in `native-checks-final.log`; the first failure log is preserved in `native-checks-initial.log` (the new engine test initially overshot its intended prefill endpoint; the fixture was corrected to construct an exact endpoint).

Reproduce CPU checks from the repository root:

```sh
sh results/fleetpi/setup/run-native-checks.sh
python3 results/fleetpi/setup/check-template-parity.py \
  --golden results/fleetpi/setup/reference/template-golden-fixtures.json \
  --golden results/fleetpi/setup/reference/template-extra-golden-fixtures.json \
  --golden results/fleetpi/setup/reference/template-pi-golden-fixtures.json \
  --output results/fleetpi/setup/template-parity-recheck
```

The generic JSON serializer is unchanged. A new `jnode.is_float` flag preserves JSON integer-versus-floating-token provenance; Glimmer uses a local shortest-round-trip formatter to match the exercised Jinja float cases. This is fixture-backed fidelity, not an exhaustive proof over all JSON values or every floating-point bit pattern.

## Telemetry contract and limits

`/v1/fleet/capabilities` includes the shared `engine`, `model`, `max_active`, and `features` keys, alongside detailed diagnostic fields. `features.tool_streaming` is `buffered`. `context_window` is `null`: native `model_cfg` does not retain a model context limit, and physical KV capacity must not be substituted for it.

`/v1/fleet/stats` includes `active`, `queued`, `kv.total_bytes`, `kv.free_bytes`, `kv.full`, `tokens.prefill/generated/shared_prefix`, and `requests.completed`. Engine counters and active counts are copied by the engine owner under the server mutex; HTTP readers use that snapshot. KV bytes describe allocated full/SWA attention pools, not model context length, total machine memory or process footprint. `memory.rss_bytes` and `memory.physical_footprint_bytes` are `null`; the external sampler remains authoritative for those measurements.

These tests do not claim incremental tool streaming, zero-latency cancellation during a GPU kernel, generic multimodal support, or benchmark speedups. The model-free server tests exercise the real internal functions with synthetic inputs rather than a full weighted HTTP server. Root independently reported successful real-Pi read/edit/read and follow-up-history smoke tests on both Fleet and default vLLM-Metal, including Unicode and HTML, before the timed comparison.

No GPU/model job was run by this reviewer. Native sources were declared stable before root's isolated Metal build; subsequent edits were tests and documentation only.
