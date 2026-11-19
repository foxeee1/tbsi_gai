#!/usr/bin/env bash
set -u
BASE_DIR=/root/autodl-tmp/TBSI_gai/TBSI
RUN_DIR="$BASE_DIR/output/experiments/MR-G1-FGLOSS-U936-S42-FULL15-FULL244-20261116-RETRY2"
STATUS_FILE="$RUN_DIR/candidate/status.json"
MR_PID="${1:?MR evaluate pid required}"
LOG="$RUN_DIR/logs/baseline_after_mr.log"
printf 'watch_started %s pid=%s\n' "$(date -Is)" "$MR_PID" >> "$LOG"
while :; do
    if grep -q '"status": "FULL15-FULL244-COMPLETE"' "$STATUS_FILE" 2>/dev/null; then
        printf 'mr_candidate_complete %s\n' "$(date -Is)" >> "$LOG"
        cd "$BASE_DIR"
        exec /root/miniconda3/bin/python -u tools/run_tbsi_v101_original_full_20261116.py --stage all >> "$LOG" 2>&1
    fi
    if grep -qE '"status": "(EXECUTION-FAILED|FAILED|INTERRUPTED)' "$STATUS_FILE" 2>/dev/null; then
        printf 'mr_candidate_failed %s\n' "$(date -Is)" >> "$LOG"
        exit 1
    fi
    if ! kill -0 "$MR_PID" 2>/dev/null; then
        printf 'mr_evaluate_process_gone_without_complete_status %s\n' "$(date -Is)" >> "$LOG"
        exit 1
    fi
    sleep 30
done
