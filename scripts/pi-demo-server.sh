#!/bin/sh
# Keep one demo model loaded; clients can use the OpenAI-compatible chat API.
set -eu
case "${1:-}" in
  qwen)
    model="$HOME/models/Qwen3-8B-Q4_K_M.gguf"
    depth=7
    rows=8
    ;;
  glimmer)
    model="$HOME/models/Muse-Glimmer-30B-Q2_K.gguf"
    depth=15
    rows=16
    ;;
  *)
    echo 'usage: serve-demo qwen|glimmer [Fleet server options]' >&2
    exit 1
    ;;
esac
shift
exec "$HOME/fleetllm/build/fleet" serve "$@" \
  --model "$model" --backend cpu --threads 4 --budget-gb 14.8 --kv-gb 0.25 \
  --row-budget 64 --no-pin --ngram --spec-k "$depth" --spec-rows "$rows" \
  --host 127.0.0.1 --port 8090 --name pi-demo --reasoning none --max-active 1
