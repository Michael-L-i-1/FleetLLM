#!/bin/bash
# FleetLLM vs llama.cpp on the Raspberry Pi 5 (4x Cortex-A76), run on the Pi. Log: ~/pi-bench.log
# Expects the source in ~/fleetllm (git archive --prefix=fleetllm/), llama.cpp built in
# ~/llama.cpp/build, and an eval text at ~/eval.txt (e.g. cat README.md docs/*.md).
exec > ~/pi-bench.log 2>&1
set -x
cd ~
M=~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf; M0=~/models/Llama-3.2-1B-Instruct-Q4_0.gguf
mkdir -p models
[ -s $M ] || curl -sfL -o $M https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
[ -s $M0 ] || curl -sfL -o $M0 https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf
ls -la models
# build FleetLLM (CPU only) and llama.cpp's batched bench
cd ~/fleetllm && make NO_METAL=1 -j4 2>&1 | grep -E "error|warning"; ls build/fleet-bench
(cd ~/llama.cpp && cmake --build build -j4 --target llama-batched-bench llama-bench 2>&1 | tail -1)
vcgencmd measure_temp; vcgencmd get_throttled
B="--budget-gb 8 --kv-gb 2"
# correctness on the Pi
./build/fleet bench --model $M --check $B 2>&1 | tail -1
./build/fleet ppl --model $M --file ~/eval.txt --ctx 512 --max-tokens 2048 $B 2>&1 | grep -E '^\[cpu\]|^ppl'
# single-sequence speed
./build/fleet generate --model $M --prompt "Tell me a long story about a dragon." --n 256 --temp 0 $B 2>&1 | grep stats
./build/fleet generate --model $M0 --prompt "Tell me a long story about a dragon." --n 256 --temp 0 $B 2>&1 | grep -E '^\[cpu\]|stats'
~/llama.cpp/build/bin/llama-bench -m $M,$M0 -t 4 -p 512 -n 128 -r 2 2>&1 | grep -E '\|'
# many sequences: decode rate = N*256 / (wall(gen 384) - wall(gen 128))
for g in 128 384; do ./build/fleet bench --model $M --levels 1,4,16 --gen $g --reps 1 $B 2>&1 | grep -E '^ +[0-9]'; done
~/llama.cpp/build/bin/llama-batched-bench -m $M -t 4 -c 16384 -b 2048 -ub 512 -npp 64 -ntg 128 -npl 1,4,16 2>&1 | grep -E '^\|'
vcgencmd measure_temp; vcgencmd get_throttled
echo "== DONE $(date)"
