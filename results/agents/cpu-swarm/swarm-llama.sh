#!/bin/bash
# llama-server half of the CPU agent-swarm comparison (see swarm.sh); lock taken per run.
cd /Users/michaelli/code.tmp/GithubDevelopment/constrained-inference
M=$(ls ~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/*/Llama-3.2-1B-Instruct-Q4_K_M.gguf)
LS=$CLAUDE_JOB_DIR/tmp/llama.cpp/build-a76/bin/llama-server
OUT=$CLAUDE_JOB_DIR/tmp/swarm; mkdir -p $OUT
wait_up() { for i in $(seq 1 120); do curl -sf -o /dev/null http://127.0.0.1:$1/health && return 0; sleep 1; done; return 1; }
trap 'rmdir /tmp/fleet-bench.lock 2>/dev/null; kill $P 2>/dev/null' EXIT
for n in 8 16; do
  until mkdir /tmp/fleet-bench.lock 2>/dev/null; do sleep 5; done
  $LS -m $M -t 6 -np 16 -c 90000 -kvu -ctk q8_0 -ctv q8_0 --cache-reuse 256 --port 8091 --alias llama1b > $OUT/llama-$n.log 2>&1 & P=$!
  wait_up 8091 || { echo "llama-server failed"; tail $OUT/llama-$n.log; }
  t0=$(date +%s)
  python3 bench/agents.py --base http://127.0.0.1:8091/v1 --model llama1b --agents $n --turns 6 > $OUT/llama-server-$n.json 2> $OUT/llama-server-$n.err
  echo "llama-server agents=$n ($(( $(date +%s) - t0 ))s): $(python3 -c "import json;d=json.load(open('$OUT/llama-server-$n.json'));print(f\"wall {d['wall_secs']}s  p50 turn {d['turn_p50_secs']}s  p95 {d['turn_p95_secs']}s  gen {d['gen_tok_per_sec']} tok/s  cache hit {d.get('cache_hit')}\")" 2>&1)"
  kill $P; wait $P 2>/dev/null
  rmdir /tmp/fleet-bench.lock
done
