#!/usr/bin/env bash
# run_full_sweep.sh — Orchestrate sequential ICML sweeps across multiple models.
#
# Runs on the LOCAL machine. For each model, runs conditions one at a time
# with Docker daemon restart between each to avoid state degradation.
#
# Usage:
#   bash scripts/run_full_sweep.sh              # Run all models
#   bash scripts/run_full_sweep.sh --dry-run    # Preview what would run
#   bash scripts/run_full_sweep.sh --resume     # Skip models with 13 completed evals locally

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
PROJECT="mats-masec-arch-9"
ZONE="us-central1-a"
INSTANCE="osharm-sweep"
LOCAL_LOG_ROOT="logs/osharm_icml_v2"

MAX_SAMPLES=10
SEED=42

# Models to sweep, in order.
# Format: "model_string|log_suffix"
MODELS=(
    "openai/gpt-4o|gpt4o"
    "openai/gpt-5.4|gpt54"
    "openai/gpt-5.4-mini|gpt54mini"
    "together/nim/meta/llama-3.2-90b-vision-instruct|llama32_90b"
    "together/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8|qwen3vl_235b"
    "anthropic/claude-sonnet-4-5|sonnet45"
)

# All 13 conditions in order
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

# ── Helpers ──────────────────────────────────────────────────────────────────

DRY_RUN=false
RESUME=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --resume)  RESUME=true ;;
    esac
done

ssh_cmd() {
    $GCLOUD compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command="$1" 2>&1
}

scp_from() {
    $GCLOUD compute scp --recurse \
        "$INSTANCE:$1" "$2" \
        --project="$PROJECT" --zone="$ZONE" 2>&1
}

log() {
    echo ""
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1"
}

# Restart Docker daemon and prune everything.  Gives the daemon a fresh
# process with zero cached state — fixes compose timeouts caused by fd/network
# leaks after many containers.
docker_reset() {
    # NOTE: Do NOT use 'docker image prune -a' — it deletes the OSWorld base
    # image (16GB, 14min rebuild) causing all subsequent conditions to timeout
    # on compose up.  Only prune dangling (untagged) images.
    ssh_cmd "sudo systemctl restart docker && sleep 5 && \
        docker container prune -f > /dev/null 2>&1; \
        docker image prune -f > /dev/null 2>&1; \
        docker builder prune -f > /dev/null 2>&1; \
        docker network prune -f > /dev/null 2>&1; \
        echo 'Docker restarted. Disk:' && df -h / | tail -1"
}

# Check whether a condition already has a successful 44-sample eval on the VM.
condition_done() {
    local remote_log_dir="$1"
    local cond="$2"
    ssh_cmd "cd ~/orbit && export PATH=\$HOME/.local/bin:\$PATH && \
        python3 -c \"
import zipfile, json
from pathlib import Path
d = Path('$remote_log_dir/$cond/misuse')
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
\""
    return $?
}

# Run a single condition for a model.
run_condition() {
    local model="$1"
    local cond="$2"
    local remote_log_dir="$3"
    local log_suffix="$4"

    ssh_cmd "cd ~/orbit && \
        export PATH=\$HOME/.local/bin:\$PATH && \
        export COMPOSE_HTTP_TIMEOUT=300 && \
        set -a && source .env && set +a && \
        sg docker -c 'uv run python -u scripts/run_osharm_icml_experiments.py \
            --model \"$model\" --seed $SEED --parallel 1 --max-samples $MAX_SAMPLES \
            --log-root \"$remote_log_dir\" \
            --condition \"$cond\"' \
        2>&1 | tee -a logs/sweep_${log_suffix}.log"
}

# ── Pre-flight ───────────────────────────────────────────────────────────────

log "Checking VM status..."
VM_STATUS=$($GCLOUD compute instances describe "$INSTANCE" \
    --project="$PROJECT" --zone="$ZONE" --format='value(status)' 2>&1)

if [ "$VM_STATUS" != "RUNNING" ]; then
    echo "VM is $VM_STATUS. Starting..."
    $GCLOUD compute instances start "$INSTANCE" --project="$PROJECT" --zone="$ZONE"
    sleep 30
fi

log "VM is running. Verifying environment..."
ssh_cmd "cd ~/orbit && docker info > /dev/null 2>&1 && echo 'Docker: OK' || echo 'Docker: FAIL'"

# ── Main loop ────────────────────────────────────────────────────────────────

TOTAL_MODELS=${#MODELS[@]}
MODELS_OK=0
MODELS_FAIL=0
MODELS_SKIP=0

for i in "${!MODELS[@]}"; do
    IFS='|' read -r MODEL LOG_SUFFIX <<< "${MODELS[$i]}"
    MODEL_IDX=$((i + 1))
    REMOTE_LOG_DIR="logs/osharm_icml_v2_${LOG_SUFFIX}"
    LOCAL_LOG_DIR="${LOCAL_LOG_ROOT}_${LOG_SUFFIX}"

    log "================================================================"
    log "[$MODEL_IDX/$TOTAL_MODELS] Model: $MODEL"
    log "  Remote: $REMOTE_LOG_DIR  |  Local: $LOCAL_LOG_DIR"

    # ── Resume check (local) ─────────────────────────────────────────────
    if $RESUME && [ -d "$LOCAL_LOG_DIR" ]; then
        LOCAL_EVALS=$(find "$LOCAL_LOG_DIR" -name "*.eval" 2>/dev/null | wc -l)
        if [ "$LOCAL_EVALS" -ge 13 ]; then
            log "  SKIP: $LOCAL_EVALS eval files found locally (--resume)"
            MODELS_SKIP=$((MODELS_SKIP + 1))
            continue
        fi
    fi

    # ── Fresh Docker daemon for each model ───────────────────────────────
    if ! $DRY_RUN; then
        log "  Resetting Docker daemon..."
        docker_reset
    fi

    MODEL_START=$(date +%s)
    COND_OK=0
    COND_FAIL=0
    COND_SKIP=0

    for cond in "${CONDITIONS[@]}"; do
        COND_LABEL="[$MODEL_IDX/$TOTAL_MODELS] $MODEL | $cond"

        # ── Check if already done on VM ──────────────────────────────────
        if condition_done "$REMOTE_LOG_DIR" "$cond" > /dev/null 2>&1; then
            echo "  [ OK ] $cond — already complete, skipping"
            COND_SKIP=$((COND_SKIP + 1))
            continue
        fi

        if $DRY_RUN; then
            echo "  [DRY ] $cond — would execute"
            continue
        fi

        # ── Docker cleanup between conditions ────────────────────────────
        # Light cleanup: stop containers, prune stopped containers and
        # dangling networks.  Full daemon restart only between models.
        ssh_cmd "docker stop \$(docker ps -q) 2>/dev/null; \
            docker container prune -f > /dev/null 2>&1; \
            docker network prune -f > /dev/null 2>&1" > /dev/null 2>&1

        echo "  [RUN ] $cond — starting..."
        COND_START=$(date +%s)

        run_condition "$MODEL" "$cond" "$REMOTE_LOG_DIR" "$LOG_SUFFIX" > /dev/null 2>&1

        COND_END=$(date +%s)
        COND_DURATION=$(( (COND_END - COND_START) / 60 ))

        # ── Verify this condition completed ──────────────────────────────
        if condition_done "$REMOTE_LOG_DIR" "$cond" > /dev/null 2>&1; then
            echo "  [ OK ] $cond — done (${COND_DURATION}m)"
            COND_OK=$((COND_OK + 1))
        else
            echo "  [FAIL] $cond — incomplete after ${COND_DURATION}m"
            COND_FAIL=$((COND_FAIL + 1))

            # Restart Docker after a failure to recover
            log "  Restarting Docker after failure..."
            docker_reset > /dev/null 2>&1
        fi
    done

    MODEL_END=$(date +%s)
    MODEL_DURATION=$(( (MODEL_END - MODEL_START) / 60 ))

    log "  Model $MODEL: ${COND_OK} OK, ${COND_FAIL} failed, ${COND_SKIP} skipped (${MODEL_DURATION}m)"

    # ── Sync logs ────────────────────────────────────────────────────────
    if ! $DRY_RUN; then
        log "  Syncing logs to local machine..."
        mkdir -p "$LOCAL_LOG_DIR"
        scp_from "~/orbit/$REMOTE_LOG_DIR/*" "$LOCAL_LOG_DIR/" || true
        mkdir -p logs
        scp_from "~/orbit/logs/sweep_${LOG_SUFFIX}.log" "logs/" || true
        log "  Logs synced."

        # Prune between models — keep tagged images (OSWorld base), clean the rest
        log "  Pruning Docker (keeping base images)..."
        ssh_cmd "docker image prune -f > /dev/null 2>&1; \
            docker builder prune -f > /dev/null 2>&1; \
            echo 'Pruned. Disk:' && df -h / | tail -1"
    fi

    if [ "$COND_FAIL" -eq 0 ]; then
        MODELS_OK=$((MODELS_OK + 1))
    else
        MODELS_FAIL=$((MODELS_FAIL + 1))
    fi

    echo "================================================================"
done

# ── Summary ──────────────────────────────────────────────────────────────────

log "============================================"
log "  FULL SWEEP COMPLETE"
log "  Models: $TOTAL_MODELS"
log "  Clean:  $MODELS_OK"
log "  Errors: $MODELS_FAIL"
log "  Skipped: $MODELS_SKIP"
log "============================================"

if [ "$MODELS_FAIL" -gt 0 ]; then
    log "Some models had condition failures."
    log "Re-run with --resume to skip completed models and retry remaining."
fi
