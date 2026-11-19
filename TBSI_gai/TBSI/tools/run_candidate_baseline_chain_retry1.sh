#!/usr/bin/env bash
set -u

BASE_DIR=/root/autodl-tmp/TBSI_gai/TBSI
RUN_DIR="$BASE_DIR/output/experiments/MR-G1-FGLOSS-U936-S42-FULL15-FULL244-20261116-RETRY1"
STATUS_FILE="$RUN_DIR/candidate/status.json"
CANDIDATE_PID="${1:?candidate pid required}"
LOG="$RUN_DIR/logs/candidate_eval_baseline_chain.log"

printf 'chain_started %s pid=%s\n' "$(date -Is)" "$CANDIDATE_PID" >> "$LOG"
while :; do
    if grep -q '"status": "TRAINING-COMPLETE / FULL244-PENDING"' "$STATUS_FILE" 2>/dev/null; then
        printf 'candidate_training_complete %s\n' "$(date -Is)" >> "$LOG"
        cd "$BASE_DIR"
        /root/miniconda3/bin/python -u tools/run_mr_source_foreground_full_20261116_retry1.py --stage evaluate --arm candidate >> "$LOG" 2>&1
        break
    fi
    if grep -qE '"status": "(EXECUTION-FAILED|FAILED|INTERRUPTED)' "$STATUS_FILE" 2>/dev/null; then
        printf 'candidate_failed %s\n' "$(date -Is)" >> "$LOG"
        exit 1
    fi
    if ! kill -0 "$CANDIDATE_PID" 2>/dev/null; then
        printf 'candidate_training_process_gone_without_completion %s\n' "$(date -Is)" >> "$LOG"
        exit 1
    fi
    sleep 30
done

if ! grep -q '"status": "FULL15-FULL244-COMPLETE"' "$STATUS_FILE" 2>/dev/null; then
    printf 'candidate_evaluation_not_complete %s\n' "$(date -Is)" >> "$LOG"
    exit 1
fi
printf 'candidate_evaluation_complete %s\n' "$(date -Is)" >> "$LOG"
cd "$BASE_DIR"
exec /root/miniconda3/bin/python -u tools/run_tbsi_v101_original_full_20261116.py --stage all >> "$LOG" 2>&1
