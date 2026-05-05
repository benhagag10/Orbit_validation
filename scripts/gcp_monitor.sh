#!/bin/bash
# GCP experiment monitor — runs alongside gcp_run.sh.
# Validates completed evals, restarts on failure, handles stalls.
#
# Usage: bash gcp_monitor.sh <MODEL> <DEFENSE>
# (Called by gcp_run.sh, not directly)
#
set -uo pipefail

MODEL="${1:?}"
DEFENSE="${2:?}"
LOGFILE="$HOME/gcp_monitor.log"
REPO_DIR="$HOME/InspectMAS"
STALL_THRESHOLD=1800  # 30 minutes
CHECK_INTERVAL=60

cd "$REPO_DIR"
set -a && source .env && set +a

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] MONITOR: $*" | tee -a "$LOGFILE"; }

# Determine expected eval count for this model+defense
EXPECTED=$(uv run python scripts/run_neurips_defense_experiments.py \
    --model "$MODEL" --defense "$DEFENSE" --dry-run 2>/dev/null \
    | grep -cE "DRY RUN|SKIPPED" || echo 0)
log "Expected evals: $EXPECTED for $MODEL/$DEFENSE"

count_good_evals() {
    uv run python scripts/validate_defense_evals.py neurips/logs_defenses/ --json 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['summary']['good_count'])" 2>/dev/null || echo 0
}

count_new_evals() {
    find neurips/logs_defenses -name "*.eval" -not -path "*_smoke*" -newer "$HOME/.last_check" 2>/dev/null | wc -l | tr -d ' '
}

restart_service() {
    log "Restarting BrowserART service..."
    bash scripts/browserart_service.sh down 2>/dev/null || true
    sleep 5
    bash scripts/browserart_service.sh up 2>&1 | tail -1
    sleep 15
    if [ -f "$HOME/behaviors_merged.json" ]; then
        docker cp "$HOME/behaviors_merged.json" \
            browserart-service:/app/orbit/scenarios/browser/browserart/hbb/behaviors.json 2>/dev/null
    fi
    for i in $(seq 1 20); do
        if curl -s http://localhost:7878/healthz | grep -q '"ok"'; then
            log "Service healthy after restart"
            return 0
        fi
        sleep 3
    done
    log "WARNING: Service not healthy after restart"
    return 1
}

restart_runner() {
    log "Restarting experiment runner..."
    pkill -f "run_neurips_defense_experiments" 2>/dev/null || true
    sleep 2
    pkill -9 -f "orbit browserart" 2>/dev/null || true
    sleep 2

    # Delete any bad evals so they get re-run
    uv run python scripts/validate_defense_evals.py neurips/logs_defenses/ --delete-bad 2>/dev/null | tee -a "$LOGFILE"

    nohup uv run python scripts/run_neurips_defense_experiments.py \
        --parallel 1 \
        --model "$MODEL" \
        --defense "$DEFENSE" >> "$HOME/gcp_run.log" 2>&1 &
    log "Runner restarted (PID $!)"
}

# ── Main monitor loop ───────────────────────────────────────────────
touch "$HOME/.last_check"
last_eval_time=$(date +%s)
log "Monitor starting"

while true; do
    sleep "$CHECK_INTERVAL"

    # 1. Check if all done
    good=$(count_good_evals)
    if [ "$good" -ge "$EXPECTED" ] && [ "$EXPECTED" -gt 0 ]; then
        log "ALL DONE: $good/$EXPECTED evals validated"
        break
    fi

    # 2. Check runner alive
    runner_alive=$(ps aux | grep "run_neurips_defense" | grep -v grep | wc -l | tr -d ' ')
    orbit_alive=$(ps aux | grep "orbit browserart" | grep -v grep | wc -l | tr -d ' ')

    if [ "$runner_alive" -eq 0 ] && [ "$good" -lt "$EXPECTED" ]; then
        log "Runner died ($good/$EXPECTED done), restarting..."
        restart_runner
        last_eval_time=$(date +%s)
        continue
    fi

    # 3. Check BrowserART service
    if ! curl -s --max-time 5 http://localhost:7878/healthz | grep -q '"ok"'; then
        log "Service unhealthy, restarting service + runner"
        restart_service
        restart_runner
        last_eval_time=$(date +%s)
        continue
    fi

    # 4. Check for new evals (stall detection)
    new_count=$(count_new_evals)
    if [ "$new_count" -gt 0 ]; then
        touch "$HOME/.last_check"
        last_eval_time=$(date +%s)
        log "Progress: $good/$EXPECTED good evals, $new_count new since last check"
    fi

    now=$(date +%s)
    stall_seconds=$((now - last_eval_time))
    if [ "$stall_seconds" -gt "$STALL_THRESHOLD" ]; then
        log "STALL DETECTED: no new eval in ${stall_seconds}s, restarting service + runner"
        restart_service
        restart_runner
        last_eval_time=$(date +%s)
    fi
done

log "Monitor exiting — all evals validated"
