#!/bin/sh
# The four GPQA Diamond runs used to compare engines (same harness, same model):
# all 198 questions with 16 agents greedy and at T=0.5, plus the same 16
# questions with 1 and with 16 agents in flight.
#   sh evals/gpqa_suite.sh BASE_URL MODEL_ID LABEL_PREFIX
set -e
cd "$(dirname "$0")/.."
URL=$1 MODEL=$2 PREFIX=$3
P=evals/.venv/bin/python
$P evals/gpqa.py --base-url "$URL" --model "$MODEL" --workers 16 --label "$PREFIX-w16"
$P evals/gpqa.py --base-url "$URL" --model "$MODEL" --workers 16 --temperature 0.5 --label "$PREFIX-w16-t0.5"
$P evals/gpqa.py --base-url "$URL" --model "$MODEL" --workers 1 --num-examples 16 --label "$PREFIX-subset16-w1"
$P evals/gpqa.py --base-url "$URL" --model "$MODEL" --workers 16 --num-examples 16 --label "$PREFIX-subset16-w16"
