#!/bin/bash
# Agent-swarm comparison on CPU: FleetLLM vs llama-server, same model and KV budget.
cd /Users/michaelli/code.tmp/GithubDevelopment/constrained-inference
M=$(ls ~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/*/Llama-3.2-1B-Instruct-Q4_K_M.gguf)
LS=$CLAUDE_JOB_DIR/tmp/llama.cpp/build-a76/bin/llama-server
OUT=$CLAUDE_JOB_DIR/tmp/swarm; mkdir -p $OUT
wait_up() { for i in $(seq 1 120); do curl -s -o /dev/null http://127.0.0.1:$1/v1/models && return 0; sleep 1; done; return 1; }
run() { # name port agents
  python3 bench/agents.py --base http://127.0.0.1:$2/v1 --model llama1b --agents $3 --turns 6 > $OUT/$1-$3.json 2> $OUT/$1-$3.err
  echo "$1 agents=$3: $(python3 -c "import json;d=json.load(open('$OUT/$1-$3.json'));print(f\"wall {d['wall_secs']}s  p50 turn {d['turn_p50_secs']}s  p95 {d['turn_p95_secs']}s  gen {d['gen_tok_per_sec']} tok/s  cache hit {d.get('cache_hit')}\")" 2>&1)"
}
until mkdir /tmp/fleet-bench.lock 2>/dev/null; do sleep 5; done
trap 'rmdir /tmp/fleet-bench.lock 2>/dev/null; kill $P 2>/dev/null' EXIT
for cfg in "fleet:--kv-int8 --ngram" "fleet-reuse:--kv-int8 --ngram --reuse-spans"; do
  name=${cfg%%:*}; flags=${cfg#*:}
  ./build/fleet serve --model llama1b --name llama1b --device cpu --threads 6 --budget-gb 8 --kv-gb 1.5 $flags --port 8090 > $OUT/$name.log 2>&1 & P=$!
  wait_up 8090 || { echo "$name failed to start"; tail $OUT/$name.log; }
  for n in 8 16; do run $name 8090 $n; done
  kill $P; wait $P 2>/dev/null
done
$LS -m $M -t 6 -np 16 -c 90000 -kvu -ctk q8_0 -ctv q8_0 --cache-reuse 256 --port 8091 --alias llama1b > $OUT/llama.log 2>&1 & P=$!
wait_up 8091 || { echo "llama-server failed"; tail $OUT/llama.log; }
for n in 8 16; do run llama-server 8091 $n; done
kill $P; wait $P 2>/dev/null
