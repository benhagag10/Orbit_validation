#!/usr/bin/env bash
# cloud_setup.sh — Bootstrap a cloud VM for running OS-Harm ICML experiments.
#
# Usage (run on a fresh Ubuntu 22.04+ VM):
#   curl -fsSL <raw-github-url>/scripts/cloud_setup.sh | bash
#   # OR
#   scp scripts/cloud_setup.sh root@<VM_IP>:~ && ssh root@<VM_IP> bash cloud_setup.sh
#
# After setup, start a sweep with:
#   cd ~/orbit
#   export OPENAI_API_KEY="sk-..."
#   tmux new -s sweep
#   uv run python scripts/run_osharm_experiments.py \
#     --model openai/gpt-4o --seed 42 --max-samples 20
#
# To sync logs back to your local machine:
#   rsync -avz root@<VM_IP>:~/orbit/logs/ ./logs/

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/anonymous/orbit.git}"
REPO_BRANCH="${REPO_BRANCH:-test}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/orbit}"

echo "============================================"
echo "  OS-Harm Cloud VM Setup"
echo "============================================"
echo "  Repo:   $REPO_URL"
echo "  Branch: $REPO_BRANCH"
echo "  Dir:    $INSTALL_DIR"
echo "============================================"

# ── 1. System packages ──────────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl tmux htop rsync > /dev/null

# ── 2. Docker ────────────────────────────────────────────────────────────────
if command -v docker &> /dev/null; then
    echo "[2/5] Docker already installed: $(docker --version)"
else
    echo "[2/5] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo "  Docker installed: $(docker --version)"
fi

# Increase Docker's default address pool for many concurrent containers.
# Without this, Docker runs out of subnet space after ~30 containers.
DAEMON_JSON="/etc/docker/daemon.json"
if [ ! -f "$DAEMON_JSON" ] || ! grep -q "default-address-pools" "$DAEMON_JSON" 2>/dev/null; then
    echo "  Configuring Docker address pool for high parallelism..."
    cat > "$DAEMON_JSON" <<'DJSON'
{
  "default-address-pools": [
    {"base": "172.17.0.0/12", "size": 24}
  ],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
DJSON
    systemctl restart docker
fi

# ── 3. uv (Python package manager) ──────────────────────────────────────────
if command -v uv &> /dev/null; then
    echo "[3/5] uv already installed: $(uv --version)"
else
    echo "[3/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    echo "  uv installed: $(uv --version)"
fi

# Ensure uv is on PATH for future shells
if ! grep -q '.local/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# ── 4. Clone repo & install deps ────────────────────────────────────────────
if [ -d "$INSTALL_DIR" ]; then
    echo "[4/5] Repo already cloned at $INSTALL_DIR, pulling latest..."
    cd "$INSTALL_DIR"
    git fetch origin
    git checkout "$REPO_BRANCH"
    git pull origin "$REPO_BRANCH"
else
    echo "[4/5] Cloning repo..."
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

echo "  Installing Python dependencies..."
uv sync --extra dev 2>&1 | tail -1

# ── 5. Preflight checks ─────────────────────────────────────────────────────
echo "[5/5] Preflight checks..."

# Check Docker is working
if docker info > /dev/null 2>&1; then
    echo "  ✓ Docker daemon running"
else
    echo "  ✗ Docker daemon not running — try: systemctl start docker"
    exit 1
fi

# Check API keys
if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "  ✓ OPENAI_API_KEY is set"
else
    echo "  ! OPENAI_API_KEY not set — set it before running experiments:"
    echo "    export OPENAI_API_KEY='sk-...'"
fi

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "  ✓ ANTHROPIC_API_KEY is set"
else
    echo "  ! ANTHROPIC_API_KEY not set (only needed for Anthropic models)"
fi

# Check available resources
TOTAL_RAM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
CPU_CORES=$(nproc)
DISK_FREE=$(df -h "$INSTALL_DIR" | tail -1 | awk '{print $4}')
echo ""
echo "  System resources:"
echo "    CPU cores:  $CPU_CORES"
echo "    RAM:        ${TOTAL_RAM_GB} GB"
echo "    Disk free:  $DISK_FREE"

# Suggest container budget based on RAM (rough: ~3GB per OSWorld container)
MAX_CONTAINERS=$((TOTAL_RAM_GB / 3))
if [ "$MAX_CONTAINERS" -gt 44 ]; then
    MAX_CONTAINERS=44
fi
if [ "$MAX_CONTAINERS" -lt 1 ]; then
    MAX_CONTAINERS=1
fi

# Recommend splitting across parallel condition workers to avoid
# one slow sample blocking the whole sweep.  E.g. 30 containers
# → 3 workers × 10 samples each.
if [ "$MAX_CONTAINERS" -ge 9 ]; then
    SUGGESTED_WORKERS=3
elif [ "$MAX_CONTAINERS" -ge 4 ]; then
    SUGGESTED_WORKERS=2
else
    SUGGESTED_WORKERS=1
fi
SUGGESTED_SAMPLES=$((MAX_CONTAINERS / SUGGESTED_WORKERS))

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Container budget: ~$MAX_CONTAINERS"
echo "  (based on ${TOTAL_RAM_GB}GB RAM, ~3GB per OSWorld container)"
echo ""
echo "  Recommended: --parallel $SUGGESTED_WORKERS --max-samples $SUGGESTED_SAMPLES"
echo "  ($SUGGESTED_WORKERS conditions at once × $SUGGESTED_SAMPLES Docker containers each)"
echo ""
echo "  Quick start:"
echo "    cd $INSTALL_DIR"
echo "    export OPENAI_API_KEY='sk-...'"
echo "    tmux new -s sweep"
echo "    uv run python scripts/run_osharm_experiments.py \\"
echo "      --model openai/gpt-4o --seed 42 \\"
echo "      --parallel $SUGGESTED_WORKERS --max-samples $SUGGESTED_SAMPLES"
echo ""
echo "  Smoke test first (3 samples, fast):"
echo "    uv run python scripts/run_osharm_experiments.py \\"
echo "      --smoke --model openai/gpt-4o --max-samples 3"
echo ""
echo "  Sync logs to local machine:"
echo "    rsync -avz root@<VM_IP>:$INSTALL_DIR/logs/ ./logs/"
echo ""
