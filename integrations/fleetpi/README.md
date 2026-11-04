# FleetPi

Pi coding agents and Fleet inference as one local system. This demo targets Muse Glimmer 30B, MLX affine four-bit, on an Apple silicon Mac with 48 GB unified memory. It uses real Pi tool execution, session files, retries and token-based compaction. The website is built by the local model; the recorder takes screenshots of those files.

## Start

```sh
python3 integrations/fleetpi/bootstrap.py
make -j6
./fleetpi
```

`fleetpi` starts Fleet, opens Pi's terminal interface with the local Glimmer model, shows engine activity and KV headroom in its footer, and stops its engine when the session closes. It keeps its interactive configuration and logs in `~/.fleetpi`. The pinned model must already be in the Hugging Face cache; no weights or credentials are bundled.

For the eight-agent festival website demo:

```sh
./fleetpi demo festival --mode fleet8
```

Use Ctrl-C once to cancel and save progress, then `--resume --output <same-directory>` to continue. Each agent has an actual Pi session and a JSON progress ledger. Completed rounds and page checks are recorded with atomic state publication. Verified page status is invalidated when content changes. Current-attempt and cumulative usage are reported separately; crash-incomplete accounting stays explicitly unknown. Overall success requires every staged round to pass. A second Ctrl-C forces termination. Only the process group started by this invocation is stopped.

## Compare and record

The optional vLLM-Metal baseline needs the locally installed 0.32.0 package. Its Glimmer attention implementation needs the isolated correctness overlay already documented in `demos/agent-task-bench/compat`; the global package is not modified.

```sh
python3 integrations/fleetpi/configure.py
./fleetpi demo festival --compare
```

For an interrupted comparison, resume its individual case with `--mode <mode> --resume --output <case-directory>`; the completed case outputs remain intact.

The comparison runs one engine at a time: FleetPi/Fleet with eight agents, upstream Pi/vLLM with eight agents, and upstream Pi/vLLM with one agent. All use the same checkpoint, file tools, task facts, protected browser checker and common token limits. vLLM retains its native default caching and batching. Required model/tool/reasoning parser flags are compatibility settings.

For the recorded benchmark, install the existing visual harness dependencies and use the isolated Metal build:

```sh
npm ci --prefix demos/site-refactor
make -j6 BUILD=build-fleetpi build-fleetpi/fleet-serve
node integrations/fleetpi/scripts/compare.mjs results/fleetpi/comparison
```

The runner refuses existing result directories. It saves actual workspace screenshots, events, Pi sessions, checker outcomes, provider token usage, and Darwin physical-footprint samples. `results/fleetpi/protocol.json` freezes the primary trial's source, task and binary hashes. Task wall includes tools/checks/retries and excludes engine load and the initial screenshot hook. Workflow output tokens per second are not GPU decode speed. A single trial does not establish statistical significance.

## What is integrated

- Stable common project context before agent-specific roles, enabling shared-prefix reuse.
- Fleet capability and KV-pressure telemetry feeding a bounded admission queue, with repair priority and aging.
- A request holds its permit during inference and releases it for real tool work.
- Each finished agent can check and repair its page while its peers continue.
- A durable progress ledger is restored after Pi compaction, while Pi's SessionManager remains authoritative.
- Native Glimmer ATEM tools, historical calls/results/reasoning, safe truncation and disconnect handling in Fleet.

Baseline modes use the same guarded Pi built-in read/edit/write/ls tools and checker, with the Fleet extension omitted. The benchmark permits writes only to assigned page fragments; it does not expose bash. Single-agent mode owns all eight pages. Each mode receives all three stakeholder rounds and one runner repair pass per round. No learned drafter, layer skipping, activation sparsity or approximate SPAN is enabled. Fleet's n-gram proposals are verified by the target model.

## Reproducible fork

`UPSTREAM.json` pins the MIT-licensed Pi revision. `fleetpi.patch` contains the new workspace package and build integration; upstream core files are unchanged. `bootstrap.py` clones that revision, checks/applies the patch, verifies the offline model-catalog archive and builds the fork. Vendor source, dependencies, generated files and local machine configuration are ignored. In a fresh checkout, `bootstrap.py --patch results/fleetpi/frozen/fleetpi-trial.patch` reconstructs the measured harness version instead of subsequent robustness fixes. The upstream MIT notice remains in the checkout and is copied to `PI-LICENSE` here.

Model-free tests live in the fork's `packages/fleetpi/test`. They exercise real Pi sessions against a faux provider, compaction/restart, guarded tool edits, admission/cancellation, immutable checker snapshots and full three-round orchestration. `scripts/smoke.mjs` separately checks two concurrent real Glimmer agents on either engine before the timed trial.
