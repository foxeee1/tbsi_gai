#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
OUT="/autodl-fs/data/tbsi_dcvr/full15_20261025"
RUN="$ROOT/output/experiments/M1_DCVR_FULL15_SOT_20261025"
TRAIN_PID="${1:?training PID is required}"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 30
done

echo "training_exit_utc=$(date -u +%FT%TZ)" >> "$OUT/provenance.txt"
CKPT="$RUN/checkpoints/TBSITrack_ep0015.pth.tar"
test -s "$CKPT"
echo "test_start_utc=$(date -u +%FT%TZ)" >> "$OUT/provenance.txt"
TBSI_DCVR_MODE=on \
TBSI_DCVR_ALPHA=0.25 \
TBSI_DCVR_DIAGNOSTICS_PATH="$OUT/dcvr_test_stats.jsonl" \
TBSI_CHECKPOINT_OVERRIDE="$CKPT" \
TBSI_TEMPLATE_UPDATE_MODE=off \
python -u "$ROOT/tracking/test.py" tbsi_track M1_DCVR_FULL15_SOT_20261025 \
  --dataset_name lasher_test --threads 0 --num_gpus 1 \
  > "$RUN/logs/test_full244.log" 2>&1
python -u "$ROOT/tracking/analysis_results.py" \
  --tracker_name tbsi_track --tracker_param M1_DCVR_FULL15_SOT_20261025 \
  --dataset_name lasher_test > "$RUN/logs/analysis_full244.log" 2>&1
echo "test_end_utc=$(date -u +%FT%TZ)" >> "$OUT/provenance.txt"
