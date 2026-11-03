# First Pi 5 deployment: 2026-11-02

Raspberry Pi 5, 16 GB RAM, four Cortex-A76 cores, NVMe root disk, wired
Stanford residence connection. Source: `83f416f40ea017c12b7cff664577cbb05692917f`.
Built with `make NO_METAL=1 -j4`; CPU reports `neon+dotprod`, four threads.
Model: `unsloth/Llama-3.2-1B-Instruct-GGUF`,
`Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807,694,368 bytes.

## Validation

Generation answered the France-capital prompt correctly. Batched/sequential
greedy output check passed 8/8 tasks at `--gen 64` (49.06 s sequential,
20.10 s batched). See raw logs beside this file.

## Initial throughput

From `~/fleetllm`, with `M` set to the model path:

```sh
./build/fleet bench --model "$M" --check --gen 64 --budget-gb 8 --kv-gb 2
./build/fleet bench --model "$M" --levels 1,4,16 --gen 128 --reps 1 --budget-gb 8 --kv-gb 2
~/llama.cpp/build/bin/llama-bench -m "$M" -t 4 -p 512 -n 128 -r 2
```

| FleetLLM concurrent sequences | Wall seconds | Aggregate tokens/sec |
| --- | ---: | ---: |
| 1 | 11.73 | 10.9 |
| 4 | 17.46 | 29.3 |
| 16 | 55.09 | 37.2 |

FleetLLM figures include prompt processing and use `bench/tasks.txt`, with
sharing and speculation off. One repetition; these are initial measurements.
The separate llama.cpp microbenchmark reported 83.27 tokens/sec for pp512
and 14.78 tokens/sec for tg128 (build `785322d`). Different prompts and timing
definitions mean these numbers do **not** establish a comparative speedup.

Final temperature 57.1°C; `vcgencmd get_throttled` returned `0x0`. During the
run, the fan reported 1,741 RPM at 49.9°C and 2,369 RPM at 58.2°C. No long-term
reliability claim is implied by this short successful session.

A subsequent [matched HTTP comparison](comparison/README.md) measured both engines
on identical tokenized prompts and fixed output lengths. Use that comparison
for FleetLLM/llama.cpp ratios rather than the independent microbenchmarks above.
