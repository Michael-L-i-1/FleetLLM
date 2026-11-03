#!/bin/sh
# One-agent Glimmer on the provisioned Pi. Quote the prompt; optional Fleet flags follow.
set -eu
prompt=${1:-Explain how a rainbow forms in a few sentences.}
if [ "$#" -gt 0 ]; then shift; fi
exec "$HOME/fleetllm/build/fleet" generate "$@" \
  --model "$HOME/models/Muse-Glimmer-30B-Q2_K.gguf" \
  --backend cpu --threads 4 --budget-gb 14.8 --kv-gb 0.25 \
  --row-budget 64 --no-pin --n 64 --temp 0 --prompt "$prompt"
