#!/bin/bash
# GPQA Diamond on the Raspberry Pi 5, unattended: Fleet's model server plus the
# resumable simple-evals runner (evals/gpqa.py --checkpoint-dir), supervised.
#
#   evals/pi_gpqa.sh prepare [RUN_DIR]   freeze the 198 questions; no model calls
#   evals/pi_gpqa.sh start   [RUN_DIR]   launch detached (survives SSH logout)
#   evals/pi_gpqa.sh pause|run|stop [RUN_DIR]   hold new questions / continue / stop
#   FROM_RUN=~/gpqa-run MAXTOK=8192 WORKERS=8 KVGB=2 evals/pi_gpqa.sh extend ~/gpqa-run-8k
#        run the questions that hit the cap again with a larger cap (then: start)
#
# Progress: RUN_DIR/checkpoint/questions.csv (one row per question, updated as it
# streams), RUN_DIR/checkpoint/responses/ (every response, partial ones included),
# RUN_DIR/supervisor.log. Finished questions are never rerun.
#
# If the server or the runner dies, both are restarted and the run resumes from the
# CSV (questions in flight start over). Backoff 1, 2, 5 minutes. It gives up after 3
# crashes in a row with no newly finished question, and stops everything if free
# memory falls below 600 MB, so a broken setup cannot reload 11 GB all night.
set -u
cmd=${1:-}; RUN=${2:-$HOME/gpqa-run}
MAXTOK=${MAXTOK:-2048}; WORKERS=${WORKERS:-16}; KVGB=${KVGB:-1.5}
F=$HOME/fleetllm; PORT=18096; NAME=pi-gpqa-glimmer
MODEL=$HOME/models/Muse-Glimmer-30B-Q2_K.gguf
FINGERPRINT="Muse-Glimmer-30B-Q2_K sha256:af4b15f0f3268d23e0f48f796a4ab563454b9cf8bab60d2bbe251b314b342fb5 kv-int8"
SERVER=("$F/build/fleet" serve --model "$MODEL" --backend cpu --threads 4 --budget-gb 14.8
        --kv-gb $KVGB --kv-int8 --row-budget 64 --no-pin --ngram --spec-k 15 --spec-rows 16
        --host 127.0.0.1 --port $PORT --name $NAME --reasoning low --max-active $WORKERS --admit
        --max-tokens $MAXTOK)
CLIENT=("$F/evals/.venv/bin/python" -u "$F/evals/gpqa.py" --base-url "http://127.0.0.1:$PORT/v1"
        --model $NAME --workers $WORKERS --repeats 1 --max-tokens $MAXTOK --temperature 0 --stream
        --reasoning-effort low --request-timeout 43200 --label gpqa --out-dir "$RUN"
        --checkpoint-dir "$RUN/checkpoint" --model-fingerprint "$FINGERPRINT")
CK=$RUN/checkpoint

log() { echo "$(date '+%F %T') $*" | tee -a "$RUN/supervisor.log"; }
finished() { awk -F, 'NR > 1 && $3 ~ /^(correct|wrong|capped)$/' "$CK/questions.csv" 2>/dev/null | wc -l; }
total() { awk 'END { print NR - 1 }' "$CK/questions.csv" 2>/dev/null; }
control() { cat "$CK/control" 2>/dev/null || echo run; }
healthy() { curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; }
avail_mb() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }

server_pid=; client_pid=
stop_server() {
    [ -n "$server_pid" ] && kill -TERM "$server_pid" 2>/dev/null && wait "$server_pid" 2>/dev/null
    server_pid=
}
start_server() {
    "${SERVER[@]}" >> "$RUN/server.log" 2>&1 & server_pid=$!
    log "server starting (pid $server_pid)"
    for _ in $(seq 1 180); do  # loading and repacking the 11 GB model takes minutes
        healthy && { log "server ready"; return 0; }
        kill -0 "$server_pid" 2>/dev/null || { log "server exited while loading"; server_pid=; return 1; }
        sleep 5
    done
    log "server not ready after 15 minutes"; stop_server; return 1
}

supervise() {
    mkdir -p "$RUN"; exec 9> "$RUN/supervisor.lock"
    flock -n 9 || { echo "already running for $RUN"; exit 1; }
    echo $$ > "$RUN/supervisor.pid"
    trap 'log "supervisor stopped by signal"; [ -n "$client_pid" ] && kill -TERM "$client_pid" 2>/dev/null; wait; stop_server; exit 0' TERM INT
    log "supervisor started: ${SERVER[*]}"
    strikes=0; last=$(finished); backoff=(60 120 300)
    while :; do
        [ "$(control)" = stop ] && { log "stop requested"; break; }
        if ! healthy; then stop_server; start_server || { strikes=$((strikes + 1)); }; fi
        if healthy; then
            "${CLIENT[@]}" >> "$RUN/runner.log" 2>&1 & client_pid=$!
            log "runner started (pid $client_pid), $(finished)/$(total) finished"
            reason=
            while kill -0 "$client_pid" 2>/dev/null; do
                sleep 10
                if ! kill -0 "$server_pid" 2>/dev/null; then reason="server died"; break; fi
                if [ "$(avail_mb)" -lt 600 ]; then reason="low memory: $(avail_mb) MB available"; break; fi
            done
            if [ -n "$reason" ]; then
                log "$reason; stopping runner"; kill -TERM "$client_pid" 2>/dev/null
            fi
            wait "$client_pid"; rc=$?; client_pid=
            log "runner exited ($rc), $(finished)/$(total) finished"
            if [ "$rc" = 0 ]; then log "run complete"; break; fi
            case "$reason" in low*) log "giving up: memory"; break;; esac
            [ "$(control)" = stop ] && { log "stop requested"; break; }
            stop_server
            now=$(finished)
            if [ "$now" -gt "$last" ]; then strikes=0; else strikes=$((strikes + 1)); fi
            last=$now
        fi
        if [ "$strikes" -ge 3 ]; then log "giving up: 3 failures in a row without progress"; break; fi
        wait_s=${backoff[$(( strikes > 2 ? 2 : (strikes > 0 ? strikes - 1 : 0) ))]}
        log "restarting in ${wait_s}s (failures in a row: $strikes)"; sleep "$wait_s"
    done
    stop_server; rm -f "$RUN/supervisor.pid"; log "supervisor exiting"
}

case "$cmd" in
    prepare) mkdir -p "$RUN"; "${CLIENT[@]}" --prepare-only ;;
    extend)  # Questions that hit the cap in FROM_RUN, plus CONTROLS that stopped on their
             # own (their greedy output must come out identical), become a new run in RUN.
             FROM=${FROM_RUN:?set FROM_RUN}; mkdir -p "$CK"
             "$F/evals/.venv/bin/python" "$F/evals/gpqa_extend.py" select "$FROM/checkpoint" "$CK/examples.json" --controls "${CONTROLS:-2}"
             "${CLIENT[@]}" --prepare-only ;;
    start)   mkdir -p "$RUN"; [ -f "$CK/control" ] && [ "$(control)" = stop ] && echo run > "$CK/control"
             setsid nohup "$0" supervise "$RUN" > /dev/null 2>&1 < /dev/null &
             echo "started; follow $RUN/supervisor.log" ;;
    supervise) supervise ;;
    pause|run|stop) mkdir -p "$CK"; echo "$cmd" > "$CK/control"; echo "control: $cmd" ;;
    *) sed -n '2,19p' "$0"; exit 1 ;;
esac
