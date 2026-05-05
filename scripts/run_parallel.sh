#!/bin/bash
# Run commands from file with N parallel jobs
# Usage: bash scripts/run_parallel.sh /tmp/jira_v3_cmds.txt 15

FILE=$1
JOBS=${2:-15}
TOTAL=$(wc -l < "$FILE")
DONE=0
FAIL=0

echo "Running $TOTAL commands with $JOBS parallel jobs..."

while IFS= read -r cmd; do
    while [ "$(jobs -r | wc -l)" -ge "$JOBS" ]; do
        wait -n 2>/dev/null
        DONE=$((DONE + 1))
        if [ $((DONE % 20)) -eq 0 ]; then
            echo "  [$DONE/$TOTAL] completed..."
        fi
    done
    eval "$cmd" &
done < "$FILE"

wait
echo "DONE: $TOTAL commands executed"
