#!/bin/bash
# GCP experiment runner with monitoring and auto-recovery.
#
# Usage: bash gcp_run.sh <FULL_MODEL> <DEFENSE>
# e.g.:  bash gcp_run.sh "openai/gpt-5.4" "system_prompt"
#
# Prerequisites (scp'd to ~ before running):
#   ~/keys.env               — API keys
#   ~/behaviors_merged.json  — merged harmful+benign behaviors
#
set -euo pipefail

MODEL="${1:?Usage: gcp_run.sh <MODEL> <DEFENSE>}"
DEFENSE="${2:?Usage: gcp_run.sh <MODEL> <DEFENSE>}"
LOGFILE="$HOME/gcp_run.log"
REPO_DIR="$HOME/InspectMAS"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

# ── 1. Clone + setup ────────────────────────────────────────────────
log "Starting GCP run: model=$MODEL defense=$DEFENSE"

if [ ! -d "$REPO_DIR" ]; then
    log "Cloning repo..."
    git clone https://github.com/anonymous/orbit.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git checkout test
git pull origin test

# Install uv if not present
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

log "Installing dependencies..."
uv sync --extra dev 2>&1 | tail -1

# ── 2. API keys ─────────────────────────────────────────────────────
if [ -f "$HOME/keys.env" ]; then
    cp "$HOME/keys.env" .env
    log "API keys copied"
else
    log "ERROR: ~/keys.env not found"
    exit 1
fi
set -a && source .env && set +a

# ── 3. BrowserART service ──────────────────────────────────────────
log "Starting BrowserART service..."
bash scripts/browserart_service.sh down 2>/dev/null || true
sleep 3
bash scripts/browserart_service.sh up 2>&1 | tail -3

# Wait for healthy
for i in $(seq 1 30); do
    if curl -s http://localhost:7878/healthz | grep -q '"ok"'; then
        log "Service healthy"
        break
    fi
    sleep 5
done

# Copy merged behaviors
if [ -f "$HOME/behaviors_merged.json" ]; then
    docker cp "$HOME/behaviors_merged.json" \
        browserart-service:/app/orbit/scenarios/browser/browserart/hbb/behaviors.json
    log "Merged behaviors copied to container"
else
    log "WARNING: ~/behaviors_merged.json not found, benign tasks may fail"
fi

# Verify service
if ! curl -s http://localhost:7878/healthz | grep -q '"ok"'; then
    log "ERROR: BrowserART service not healthy"
    exit 1
fi

# ── 4. Launch experiments + monitor ─────────────────────────────────
log "Launching experiments..."

# Start the monitor in background
bash scripts/gcp_monitor.sh "$MODEL" "$DEFENSE" &
MONITOR_PID=$!
log "Monitor started (PID $MONITOR_PID)"

# Run experiments (monitor will restart if this dies)
uv run python scripts/run_defense_experiments.py \
    --parallel 1 \
    --model "$MODEL" \
    --defense "$DEFENSE" 2>&1 | tee -a "$LOGFILE"

# Wait for monitor to confirm all validated
wait "$MONITOR_PID" 2>/dev/null || true

# ── 5. Package results ─────────────────────────────────────────────
SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
RESULTS_TAR="$HOME/results_${DEFENSE}_${SAFE_MODEL}.tar.gz"
tar czf "$RESULTS_TAR" logs/defenses/
log "Results packaged: $RESULTS_TAR"
log "DONE"
