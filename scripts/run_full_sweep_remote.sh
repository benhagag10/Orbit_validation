#!/usr/bin/env bash
# run_full_sweep_remote.sh — Run on the VM inside tmux.
#
# Runs each model's conditions one-at-a-time with Docker cleanup between
# each to avoid daemon state degradation.
#
# Usage (on the VM):
#   tmux new -s sweep
#   bash scripts/run_full_sweep_remote.sh 2>&1 | tee logs/full_sweep.log

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd ~/orbit
set -a && source .env && set +a

# Docker Compose API timeout — prevents CLI hanging when the daemon is
# slow to respond under heavy container load (default is 60s).
export COMPOSE_HTTP_TIMEOUT=300

# Limit concurrent Docker CLI commands.  Inspect defaults to CPU*2 (=32
# on this 16-vCPU VM), which hammers the Docker socket with 30+ concurrent
# compose build/up calls, causing http2 connection resets and buildkit
# context-canceled errors.  4 concurrent CLI calls avoids daemon contention
# while still allowing parallel sample execution.
export INSPECT_DOCKER_CLI_CONCURRENCY=4

# Patch Inspect's hardcoded COMPOSE_WAIT from 120s to 300s.
# This controls the `docker compose up --wait-timeout` value.
# 120s is too short when many containers start concurrently.
COMPOSE_PY="$(uv run python -c 'import inspect_ai.util._sandbox.docker.compose as m; print(m.__file__)')"
if grep -q 'COMPOSE_WAIT = 120' "$COMPOSE_PY" 2>/dev/null; then
    sed -i 's/COMPOSE_WAIT = 120/COMPOSE_WAIT = 300/' "$COMPOSE_PY"
    echo "Patched COMPOSE_WAIT → 300s in $COMPOSE_PY"
fi

MAX_SAMPLES=10
SEED=42

MODELS=(
    "openai/gpt-4o|gpt4o"
    "openai/gpt-5.4|gpt54"
    "openai/gpt-5.4-mini|gpt54mini"
    "together/nim/meta/llama-3.2-90b-vision-instruct|llama32_90b"
    "together/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8|qwen3vl_235b"
    "anthropic/claude-sonnet-4-5|sonnet45"
)

CONDITIONS=(
    single_agent
    star_batch_relaxed
    star_batch
    star_step
    star_tool_2_specialist
    star_tool_3_specialist
    star_tool_specialist
    mesh_tool_round_robin
    mesh_tool_delegation
    tool_memory_own_reasoning
    tool_memory_full
    mesh_tool_delegation_cot
    mesh_tool_delegation_full
)

log() { echo ""; echo "$(date '+%Y-%m-%d %H:%M:%S') | $1"; }

# Check if a condition has a completed 44-sample eval.
condition_done() {
    local log_dir="$1" cond="$2"
    python3 -c "
import zipfile, json
from pathlib import Path
d = Path('$log_dir/$cond/misuse')
if not d.exists(): exit(1)
found = False
for ep in sorted(d.glob('*.eval'), reverse=True):
    try:
        with zipfile.ZipFile(ep) as z:
            if 'header.json' not in z.namelist(): continue
            h = json.loads(z.read('header.json'))
            s = [n for n in z.namelist() if n.startswith('samples/') and n.endswith('.json')]
            if h.get('status') == 'success' and len(s) >= 44:
                found = True
                break
    except Exception: pass
exit(0 if found else 1)
" 2>/dev/null
    return $?
}

docker_reset() {
    log "    Restarting Docker daemon..."
    sudo systemctl restart docker
    sleep 5
    docker container prune -f > /dev/null 2>&1
    # NOTE: Do NOT use 'docker image prune -a' here — it deletes the OSWorld
    # base image (16GB, 14min rebuild) causing all subsequent conditions to
    # timeout on compose up.  Only prune dangling (untagged) images.
    docker image prune -f > /dev/null 2>&1
    docker builder prune -f > /dev/null 2>&1
    docker network prune -f > /dev/null 2>&1
    log "    Docker reset. Disk: $(df -h / | tail -1 | awk '{print $4 " free"}')"
}

docker_cleanup() {
    docker stop $(docker ps -q) 2>/dev/null || true
    docker container prune -f > /dev/null 2>&1
    docker network prune -f > /dev/null 2>&1
}

# ── Main ─────────────────────────────────────────────────────────────────────

TOTAL_MODELS=${#MODELS[@]}
GRAND_OK=0
GRAND_FAIL=0

for i in "${!MODELS[@]}"; do
    IFS='|' read -r MODEL LOG_SUFFIX <<< "${MODELS[$i]}"
    MODEL_IDX=$((i + 1))
    LOG_DIR="logs/osharm_icml_v2_${LOG_SUFFIX}"

    log "================================================================"
    log "[$MODEL_IDX/$TOTAL_MODELS] Model: $MODEL"
    log "  Log dir: $LOG_DIR"

    # Fresh Docker for each model
    docker_reset

    MODEL_OK=0
    MODEL_FAIL=0
    MODEL_SKIP=0
    MODEL_START=$(date +%s)

    for cond in "${CONDITIONS[@]}"; do
        # Skip completed conditions
        if condition_done "$LOG_DIR" "$cond"; then
            echo "  [ OK ] $cond — already complete"
            MODEL_SKIP=$((MODEL_SKIP + 1))
            continue
        fi

        # Light cleanup between conditions
        docker_cleanup

        echo "  [RUN ] $cond — starting..."
        COND_START=$(date +%s)

        # Run the single condition
        sg docker -c "uv run python -u scripts/run_osharm_icml_experiments.py \
            --model '$MODEL' --seed $SEED --parallel 1 --max-samples $MAX_SAMPLES \
            --log-root '$LOG_DIR' \
            --condition '$cond'" 2>&1 || true

        COND_END=$(date +%s)
        COND_MINS=$(( (COND_END - COND_START) / 60 ))

        # Verify completion
        if condition_done "$LOG_DIR" "$cond"; then
            echo "  [ OK ] $cond — done (${COND_MINS}m)"
            MODEL_OK=$((MODEL_OK + 1))
        else
            echo "  [FAIL] $cond — incomplete after ${COND_MINS}m"
            MODEL_FAIL=$((MODEL_FAIL + 1))
            # Full restart after failure
            docker_reset
        fi
    done

    MODEL_END=$(date +%s)
    MODEL_MINS=$(( (MODEL_END - MODEL_START) / 60 ))

    # Prune between models — keep tagged images (OSWorld base), clean the rest
    log "  Pruning Docker (keeping base images)..."
    docker image prune -f > /dev/null 2>&1
    docker builder prune -f > /dev/null 2>&1

    log "  Model $MODEL: ${MODEL_OK} completed, ${MODEL_FAIL} failed, ${MODEL_SKIP} skipped (${MODEL_MINS}m total)"

    GRAND_OK=$((GRAND_OK + MODEL_OK + MODEL_SKIP))
    GRAND_FAIL=$((GRAND_FAIL + MODEL_FAIL))

    echo "================================================================"
done

log "============================================"
log "  ALL SWEEPS COMPLETE"
log "  Total OK: $GRAND_OK / $((GRAND_OK + GRAND_FAIL))"
log "  Failures: $GRAND_FAIL"
log "============================================"
