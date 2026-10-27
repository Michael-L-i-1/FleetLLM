#!/bin/sh
# Prepare the eval environment: a local venv and a pinned checkout of OpenAI's
# simple-evals (MIT), the harness behind the GPQA numbers in many model reports.
#   sh evals/setup.sh
set -e
cd "$(dirname "$0")"
SIMPLE_EVALS_COMMIT=652c89d0ca9df547706735883097e9537d40dc47
if [ ! -d third_party/simple_evals ]; then
    mkdir -p third_party
    git clone -q https://github.com/openai/simple-evals third_party/simple_evals
fi
git -C third_party/simple_evals checkout -q "$SIMPLE_EVALS_COMMIT"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
echo "ready: evals/.venv/bin/python evals/gpqa.py --help"
