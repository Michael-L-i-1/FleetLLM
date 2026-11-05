# FleetLLM

**More throughput. Same hardware.**

FleetLLM combines local inference with agent workflow coordination. Run multiple agents through one model, reuse shared context, and schedule work around available memory. The engine supports Apple Silicon Metal and CPU inference; FleetPi adds a terminal coding-agent experience built on the Pi SDK.

## Latest results

### GPQA Diamond: 65.7%

**130/198 correct** with Muse Glimmer 30B Q2_K and phased output budgets. The experiment began on a **16 GB Raspberry Pi 5**. The uniform Pi-only run scored 58.1% at a 2,048-token cap.

[Results and method](results/gpqa/2026-11-04-glimmer-q2k-phased/README.md) · [Per-question results](results/gpqa/2026-11-04-glimmer-q2k-phased/questions.csv)

### Raspberry Pi: agents building a website

Eight agents generated sections of a coffee website using Glimmer 30B Q2_K on the Pi CPU. The harness and browser capture ran on the laptop. The run passed **55/63 final checks**; it demonstrates real generated progress, rather than a completed-task speed benchmark.

[Watch the timelapse](results/pi5/2026-11-04/coffee-site-v2-fleet8/video/pi-website-timelapse.mp4) · [Website and metrics](results/pi5/2026-11-04/README.md)

### Mac M5 Pro: eight-agent throughput

Same Muse Glimmer 30B MLX four-bit checkpoint, same eight-agent website task, 48 GB unified memory.

| System | Task time | Output tok/s | Engine peak memory | Stages passed |
|---|---:|---:|---:|---:|
| **FleetPi + FleetLLM** | **5m 39s** | **112.7** | 18.19 GiB | 3/3 |
| Pi + vLLM-Metal | 17m 09s | 38.0 | 17.99 GiB | 3/3 |

**3.03× faster task completion** in this recorded trial. Both passed all 251 final checks. Output tok/s includes tools and checks; this is a complete-system comparison, with native vLLM batching and caching retained. Individual optimization contributions have not been isolated.

[Video](results/fleetpi/presentation/annotated-demo/comparison.mp4) · [Method and all three modes](results/fleetpi/README.md) · [CSV](results/fleetpi/summary.csv)

## What we optimize

| Technique | Purpose |
|---|---|
| Shared context and KV reuse | Avoid repeated prompt processing across agents |
| Memory-aware admission | Keep work within the available KV budget |
| Independent agent checks | Start checks and repairs as each agent becomes ready |
| Verified n-gram speculation | Accept matching draft tokens with fewer target passes |
| Hardware-tuned low-bit kernels | Accelerate inference on Metal and ARM CPUs |

These techniques combine in the demo; several are also available in other engines. The measured result is for this workload and configuration.

## Try it

```sh
make                              # Apple Silicon Metal
# make NO_METAL=1                 # CPU-only, including Raspberry Pi
python3 integrations/fleetpi/bootstrap.py
./fleetpi                         # interactive terminal coding agent
./fleetpi demo festival            # eight-agent website task and live dashboard
```

The terminal demo requires the cached Muse Glimmer 30B MLX four-bit checkpoint and a supported Node runtime (>=22.19). See the [FleetPi guide](integrations/fleetpi/README.md) for setup, comparison runs and resume commands.

For an OpenAI-compatible endpoint without the harness:

```sh
./build/fleet serve --model /path/to/model.gguf --budget-gb 14.8 --kv-gb 2 --kv-int8 --port 8090
```

## Explore

- [Today's results and raw evidence](results/2026-11-04/README.md), including extended-budget GPQA phases and earlier task trials.
- [Architecture](docs/design.md), [historical benchmarks](docs/results.md), and [Pi tuning](docs/pi-tuning.md).
- [Presentation](docs/demo/FleetLLM-Demo.pptx).

Engine: `src/` · Harness: `integrations/fleetpi/` · Workloads: `demos/` · Evaluation: `evals/` · Evidence: `results/`.
