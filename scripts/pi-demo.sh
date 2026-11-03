#!/bin/sh
# Verified-copy demo profiles; target precision/layers remain at their defaults.
set -eu
if [ "$#" -lt 2 ]; then
  echo 'usage: run-demo qwen|glimmer "prompt" [Fleet options]' >&2
  exit 1
fi
case "${1:-}" in
  qwen)
    shift
    exec "$HOME/fleetllm/scripts/pi-qwen8b.sh" "$@" --ngram --spec-k 7 --spec-rows 8
    ;;
  glimmer)
    shift
    exec "$HOME/fleetllm/scripts/pi-glimmer.sh" "$@" --ngram --spec-k 15 --spec-rows 16 --n 128
    ;;
  *)
    echo 'usage: run-demo qwen|glimmer "prompt" [Fleet options]' >&2
    exit 1
    ;;
esac
