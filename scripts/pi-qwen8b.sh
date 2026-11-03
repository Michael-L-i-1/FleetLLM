#!/bin/sh
# Optional Pi demo model. This does not change the Glimmer launcher or defaults.
set -eu
prompt=${1:-Explain why rainbows have different colors, in two sentences.}
if [ "$#" -gt 0 ]; then shift; fi
exec "$HOME/fleetllm/build/fleet" generate "$@" \
  --model "$HOME/models/Qwen3-8B-Q4_K_M.gguf" \
  --backend cpu --threads 4 --budget-gb 14.8 --kv-gb 0.25 \
  --row-budget 64 --no-pin --n 128 --temp 0 --prompt "$prompt"
