#!/bin/bash
# GPQA Diamond on this machine (macOS, Metal): Fleet's model server plus the resumable
# simple-evals runner, restarted after a crash. Same model file, quantization, KV format,
# sampling and harness as the Pi runs (evals/pi_gpqa.sh); only the host and backend differ.
#
#   evals/run_gpqa_local.sh RUN_DIR      RUN_DIR/checkpoint/examples.json frozen beforehand
#                                        (evals/gpqa_extend.py select ... --unfinished)
#   echo pause|run|stop > RUN_DIR/checkpoint/control
#
# Logs: RUN_DIR/supervisor.log, server.log, runner.log. Finished questions are never rerun.
set -u
RUN=$(cd "$1" && pwd); cd "$(dirname "$0")/.."
MODEL=${MODEL:-$HOME/models/Muse-Glimmer-30B-Q2_K.gguf}; PORT=${PORT:-18097}; NAME=glimmer-q2k
MAXTOK=${MAXTOK:-32768}; WORKERS=${WORKERS:-16}; KVGB=${KVGB:-12}
FINGERPRINT="Muse-Glimmer-30B-Q2_K sha256:af4b15f0f3268d23e0f48f796a4ab563454b9cf8bab60d2bbe251b314b342fb5 kv-int8"
SERVER=(./build/fleet serve --model "$MODEL" --budget-gb 32 --kv-gb "$KVGB" --kv-int8 --ngram --spec-k 15
        --spec-rows 16 --host 127.0.0.1 --port "$PORT" --name $NAME --reasoning low --max-active "$WORKERS"
        --admit --max-tokens "$MAXTOK")
CLIENT=(evals/.venv/bin/python -u evals/gpqa.py --base-url "http://127.0.0.1:$PORT/v1" --model $NAME
        --workers "$WORKERS" --repeats 1 --max-tokens "$MAXTOK" --temperature 0 --stream --reasoning-effort low
        --request-timeout 43200 --label gpqa --out-dir "$RUN" --checkpoint-dir "$RUN/checkpoint"
        --model-fingerprint "$FINGERPRINT")
CK=$RUN/checkpoint

log() { echo "$(date '+%F %T') $*" | tee -a "$RUN/supervisor.log"; }
finished() { awk -F, 'NR > 1 && $3 ~ /^(correct|wrong|capped)$/' "$CK/questions.csv" 2>/dev/null | wc -l | tr -d ' '; }
total() { awk 'END { print NR - 1 }' "$CK/questions.csv" 2>/dev/null; }
healthy() { curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; }
server_pid=; client_pid=
stop_server() { [ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null && wait "$server_pid" 2>/dev/null; server_pid=; }
trap 'log "stopped by signal"; [ -n "$client_pid" ] && kill "$client_pid" 2>/dev/null; wait; stop_server; exit 0' TERM INT

echo $$ > "$RUN/supervisor.pid"
log "supervisor started on $(hostname -s): ${SERVER[*]}"
strikes=0; last=$(finished)
while [ "$strikes" -lt 3 ]; do
    [ "$(cat "$CK/control" 2>/dev/null)" = stop ] && { log "stop requested"; break; }
    "${SERVER[@]}" >> "$RUN/server.log" 2>&1 & server_pid=$!
    log "server starting (pid $server_pid)"
    for _ in $(seq 1 120); do healthy && break; kill -0 "$server_pid" 2>/dev/null || break; sleep 5; done
    if ! healthy; then log "server did not come up"; stop_server; strikes=$((strikes + 1)); sleep 30; continue; fi
    log "server ready"
    "${CLIENT[@]}" >> "$RUN/runner.log" 2>&1 & client_pid=$!
    log "runner started (pid $client_pid), $(finished)/$(total) finished"
    while kill -0 "$client_pid" 2>/dev/null; do
        sleep 10
        kill -0 "$server_pid" 2>/dev/null || { log "server died; stopping runner"; kill "$client_pid" 2>/dev/null; break; }
    done
    wait "$client_pid"; rc=$?; client_pid=
    log "runner exited ($rc), $(finished)/$(total) finished"
    stop_server
    [ "$rc" = 0 ] && { log "run complete"; break; }
    now=$(finished); if [ "$now" -gt "$last" ]; then strikes=0; else strikes=$((strikes + 1)); fi; last=$now
    sleep 30
done
[ "$strikes" -ge 3 ] && log "giving up: 3 failures in a row without progress"
rm -f "$RUN/supervisor.pid"; log "supervisor exiting"
