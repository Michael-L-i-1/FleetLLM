# FleetLLM vs llama.cpp on Raspberry Pi 5

Measured 2026-11-02 on the same Pi 5 (16 GB, four Cortex-A76 cores), using the
same local Llama 3.2 1B Instruct Q4_K_M GGUF file. FleetLLM engine commit
`83f416f40ea017c12b7cff664577cbb05692917f`; llama.cpp commit
`785322d17f2d51e5185cf996a1f4e46d87640841`, Release build, native CPU and OpenMP.

| Concurrent requests | FleetLLM aggregate tok/s | llama.cpp aggregate tok/s | FleetLLM / llama.cpp |
| --- | ---: | ---: | ---: |
| 1 | 10.81 | 10.62 | 1.02x |
| 4 | 28.86 | 26.10 | 1.11x |
| 16 | 36.42 | 26.95 | 1.35x |

These are end-to-end HTTP completion rates, including prompt processing,
generation, and localhost request overhead. Each request generated exactly
128 tokens; prompt-token counts matched for every paired request. Table shows
the median of two repetitions. One-request performance is effectively tied
given the small sample; this is an initial comparison, not a broad engine claim.

## Method

Run `bench/pi_compare.py` on the provisioned Pi:

```sh
python3 ~/fleetllm/bench/pi_compare.py --output ~/pi-comparison-20261102
```

The script starts only one engine at a time on localhost, sets both to four CPU
threads, and uses identical tasks from `bench/tasks.txt` with a request for long
answers. Inference is greedy. Model load/startup and a four-token warmup are
excluded. Early per-request labels limit cross-request/repetition prefix reuse.
Requests ask to disable prompt caching; all recorded cached-token counts were 0.
Both engines process the same explicit Llama chat token sequence through the raw
completion API. llama.cpp automatically adds BOS, so its request string omits
the explicit BOS already present in FleetLLM's request string. A preliminary
llama.cpp run had duplicate BOS and failed the prompt-count assertion; those
numbers are excluded here. Full server commands are recorded in each JSON file.

FleetLLM uses its default FP16 KV representation with a 2 GB KV allocation and
8 GB overall budget. llama.cpp uses its default FP16 KV, unified 16,384-token
context, 16 slots, CPU only, and disabled RAM prompt cache. Actual live contexts
fit in both; this is not a matched-memory-capacity stress test. Sharing,
speculation, swapping, and long multi-turn sessions were not separately tested.

Final temperature was 58.7°C and `vcgencmd get_throttled` was `0x0`. No system
upgrades were performed. Both temporary benchmark servers were stopped after
completion. Raw per-request timings, token usage, output text, and commands are
in `fleetllm.json` and `llama.cpp.json`; medians are in `summary.json`.

vLLM was not benchmarked. Its Arm CPU backend exists, but its documented CPU
quantization list does not include this GGUF/Arm combination, so llama.cpp was
used for the same-file comparison. See the official
[Arm installation guide](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/cpu.arm.inc.md)
and [CPU quantization FAQ](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/cpu.md#which-quantization-configs-does-vllm-cpu-support).
