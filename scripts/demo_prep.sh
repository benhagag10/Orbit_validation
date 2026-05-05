#!/usr/bin/env bash
# demo_prep.sh — Pre-run InspectMAS demo experiments and capture artifacts.
#
# Runs the same BrowserART task under two conditions:
#   1. single_agent      — one browser agent (expected: REFUSED)
#   2. star_4_specialists — orchestrator + 4 specialists (expected: HARMFUL)
#
# Outputs:
#   logs/demo_video_single/  — Browser recording (.webm) for single agent
#   logs/demo_video_multi/   — Browser recording (.webm) for multi-agent
#   logs/demo_single/run_output.txt  — CLI output capture
#   logs/demo_multi/run_output.txt   — CLI output capture
#   logs/demo_summary.txt            — Side-by-side comparison
#
# Usage:
#   bash scripts/demo_prep.sh
#   bash scripts/demo_prep.sh --task-id 236 --model openai/gpt-4o
#   bash scripts/demo_prep.sh --task-id 42 --model anthropic/claude-sonnet-4-20250514

set -euo pipefail

# ── Load .env if present ──────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ── Defaults ──────────────────────────────────────────────────────────────

TASK_ID="3"
MODEL="openai/gpt-4o"
MAX_BEHAVIORS="1"
MAX_TURNS="15"
DATASET=""  # auto-detected from task ID if empty

# ── Parse arguments ───────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-id)
            TASK_ID="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max-turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash scripts/demo_prep.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --task-id ID      BrowserART task ID (default: 236)"
            echo "  --model MODEL     Model to use (default: openai/gpt-4o)"
            echo "  --max-turns N     Max turns per behavior (default: 15)"
            echo "  --dataset NAME    Dataset: hbb, hbb_extension (default: auto from task ID)"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Pretty printing ──────────────────────────────────────────────────────

BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
RESET="\033[0m"

info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
fail()  { echo -e "${RED}[FAIL]${RESET}  $*"; exit 1; }
header() { echo -e "\n${BOLD}── $* ──${RESET}\n"; }

# ── Auto-detect dataset from task ID ──────────────────────────────────────

if [[ -z "$DATASET" ]]; then
    if [[ "$TASK_ID" -ge 227 && "$TASK_ID" -le 236 ]]; then
        DATASET="hbb_extension"
    else
        DATASET="hbb"
    fi
fi

# Extension tasks need more turns/time
if [[ "$DATASET" == "hbb_extension" && "$MAX_TURNS" == "15" ]]; then
    MAX_TURNS="50"
fi

info "Config: task=$TASK_ID  dataset=$DATASET  model=$MODEL  max_turns=$MAX_TURNS"

# ── Directories ───────────────────────────────────────────────────────────

VIDEO_DIR_SINGLE="logs/demo_video_single"
VIDEO_DIR_MULTI="logs/demo_video_multi"
OUTPUT_DIR_SINGLE="logs/demo_single"
OUTPUT_DIR_MULTI="logs/demo_multi"

# ── 1. Validate environment ──────────────────────────────────────────────

header "Validating environment"

# Check inspect_mas is installed
if ! python -c "import inspect_mas" 2>/dev/null; then
    fail "inspect_mas not installed. Run: pip install -e \".[dev]\""
fi
ok "inspect_mas installed"

# Check inspect CLI is available
if ! command -v inspect &>/dev/null; then
    fail "inspect CLI not found. Run: pip install -e \".[dev]\""
fi
ok "inspect CLI available"

# Check API key
if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ "$MODEL" == openai/* ]]; then
    warn "OPENAI_API_KEY not set — required for model $MODEL"
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && [[ "$MODEL" == anthropic/* ]]; then
    warn "ANTHROPIC_API_KEY not set — required for model $MODEL"
fi

# Check playwright
if python -c "import playwright" 2>/dev/null; then
    ok "playwright installed"
else
    warn "playwright not found. Installing..."
    pip install playwright && python -m playwright install chromium
fi

# Check browsergym
if python -c "import browsergym" 2>/dev/null; then
    ok "browsergym installed"
else
    fail "browsergym not installed. Check install instructions."
fi

# ── 2. Create output directories ─────────────────────────────────────────

header "Setting up output directories"

mkdir -p "$VIDEO_DIR_SINGLE" "$VIDEO_DIR_MULTI" "$OUTPUT_DIR_SINGLE" "$OUTPUT_DIR_MULTI"
ok "Created output directories"

# ── 3. Run experiment: single_agent ───────────────────────────────────────

header "Run 1/2: single_agent (task $TASK_ID)"

info "Command:"
echo "  inspect eval inspect_mas/browserart_safety \\"
echo "    -T condition=single_agent \\"
echo "    -T dataset=$DATASET \\"
echo "    -T task_ids=$TASK_ID \\"
echo "    -T max_behaviors=$MAX_BEHAVIORS \\"
echo "    -T max_turns=$MAX_TURNS \\"
echo "    -T record_video_dir=$VIDEO_DIR_SINGLE \\"
echo "    --model $MODEL"
echo ""

SINGLE_START=$(date +%s)

inspect eval inspect_mas/browserart_safety \
    -T condition=single_agent \
    -T "dataset=$DATASET" \
    -T "task_ids=$TASK_ID" \
    -T "max_behaviors=$MAX_BEHAVIORS" \
    -T "max_turns=$MAX_TURNS" \
    -T "record_video_dir=$VIDEO_DIR_SINGLE" \
    --model "$MODEL" \
    2>&1 | tee "$OUTPUT_DIR_SINGLE/run_output.txt"

SINGLE_END=$(date +%s)
SINGLE_ELAPSED=$(( SINGLE_END - SINGLE_START ))

ok "single_agent completed in ${SINGLE_ELAPSED}s"

# ── 4. Run experiment: star_4_specialists ─────────────────────────────────

header "Run 2/2: star_4_specialists (task $TASK_ID)"

info "Command:"
echo "  inspect eval inspect_mas/browserart_safety \\"
echo "    -T condition=star_4_specialists \\"
echo "    -T dataset=$DATASET \\"
echo "    -T task_ids=$TASK_ID \\"
echo "    -T max_behaviors=$MAX_BEHAVIORS \\"
echo "    -T max_turns=$MAX_TURNS \\"
echo "    -T record_video_dir=$VIDEO_DIR_MULTI \\"
echo "    --model $MODEL"
echo ""

MULTI_START=$(date +%s)

inspect eval inspect_mas/browserart_safety \
    -T condition=star_4_specialists \
    -T "dataset=$DATASET" \
    -T "task_ids=$TASK_ID" \
    -T "max_behaviors=$MAX_BEHAVIORS" \
    -T "max_turns=$MAX_TURNS" \
    -T "record_video_dir=$VIDEO_DIR_MULTI" \
    --model "$MODEL" \
    2>&1 | tee "$OUTPUT_DIR_MULTI/run_output.txt"

MULTI_END=$(date +%s)
MULTI_ELAPSED=$(( MULTI_END - MULTI_START ))

ok "star_4_specialists completed in ${MULTI_ELAPSED}s"

# ── 5. Generate comparison summary ───────────────────────────────────────

header "Generating comparison summary"

cat > logs/demo_summary.txt << 'SUMMARY_HEADER'
========================================================================
  InspectMAS Demo — Comparison Summary
========================================================================

SUMMARY_HEADER

{
    echo "── Run 1: single_agent ──"
    echo ""
    cat "$OUTPUT_DIR_SINGLE/run_output.txt"
    echo ""
    echo ""
    echo "── Run 2: star_4_specialists ──"
    echo ""
    cat "$OUTPUT_DIR_MULTI/run_output.txt"
    echo ""
    echo ""
    echo "── Side-by-Side ──"
    echo ""
    echo "                    single_agent    star_4_specialists"
    echo "                    ────────────    ──────────────────"
    echo "  Topology          Single          Star (1+4)"
    echo "  Agents            1               5"
    echo "  Expected AS       0               1"
    echo "  Expected PD       0.00            ~0.83"
    echo "  Expected Outcome  REFUSED         HARMFUL"
    echo ""
    echo "  Key insight: Task decomposition across specialists"
    echo "  can bypass per-agent safety alignment."
    echo ""
    echo "========================================================================"
} >> logs/demo_summary.txt

ok "Comparison written to logs/demo_summary.txt"

# ── 6. Verify artifacts ──────────────────────────────────────────────────

header "Verifying artifacts"

ERRORS=0

# Check video files
SINGLE_VIDEOS=$(find "$VIDEO_DIR_SINGLE" -name "*.webm" 2>/dev/null | wc -l | tr -d ' ')
MULTI_VIDEOS=$(find "$VIDEO_DIR_MULTI" -name "*.webm" 2>/dev/null | wc -l | tr -d ' ')

if [[ "$SINGLE_VIDEOS" -gt 0 ]]; then
    ok "Single-agent video: $SINGLE_VIDEOS .webm file(s) in $VIDEO_DIR_SINGLE/"
else
    warn "No .webm files found in $VIDEO_DIR_SINGLE/ — check if record_video_dir worked"
    ERRORS=$((ERRORS + 1))
fi

if [[ "$MULTI_VIDEOS" -gt 0 ]]; then
    ok "Multi-agent video: $MULTI_VIDEOS .webm file(s) in $VIDEO_DIR_MULTI/"
else
    warn "No .webm files found in $VIDEO_DIR_MULTI/ — check if record_video_dir worked"
    ERRORS=$((ERRORS + 1))
fi

# Check output captures
for f in "$OUTPUT_DIR_SINGLE/run_output.txt" "$OUTPUT_DIR_MULTI/run_output.txt" "logs/demo_summary.txt"; do
    if [[ -s "$f" ]]; then
        ok "Output: $f ($(wc -l < "$f" | tr -d ' ') lines)"
    else
        warn "Missing or empty: $f"
        ERRORS=$((ERRORS + 1))
    fi
done

# ── 7. Print recording instructions ──────────────────────────────────────

header "Ready for recording!"

echo -e "${BOLD}All demo artifacts have been generated.${RESET}"
echo ""
echo "Artifacts:"
echo "  Videos:  $VIDEO_DIR_SINGLE/  $VIDEO_DIR_MULTI/"
echo "  Output:  $OUTPUT_DIR_SINGLE/run_output.txt  $OUTPUT_DIR_MULTI/run_output.txt"
echo "  Summary: logs/demo_summary.txt"
echo ""
echo "Next steps:"
echo "  1. Review the storyboard: scripts/demo_storyboard.md"
echo "  2. Set up terminal: dark theme, 16-18pt font, 1920x1080"
echo "  3. Record each act separately:"
echo "     - Act 1: Title card (create in Keynote/Canva)"
echo "     - Act 2: Type/replay the inspect eval command in terminal"
echo "     - Act 3: Play the .webm videos at 3-4x speed"
echo "     - Act 4: Show CLI output (cat logs/demo_single/run_output.txt)"
echo "     - Act 5: Feature card + closing"
echo "  4. Edit together with narration"
echo ""
echo "Tips:"
echo "  - Use 'cat $OUTPUT_DIR_SINGLE/run_output.txt' to show single-agent results"
echo "  - Use 'cat $OUTPUT_DIR_MULTI/run_output.txt' to show multi-agent results"
echo "  - Browser videos have colored borders showing which agent is active"
echo "  - Agent colors: blue=orchestrator, green=click, orange=fill, pink=scroll, purple=navigate"
echo ""

if [[ "$ERRORS" -gt 0 ]]; then
    warn "$ERRORS artifact(s) may need attention — see warnings above"
    exit 1
else
    ok "All artifacts verified. Ready to record!"
fi
