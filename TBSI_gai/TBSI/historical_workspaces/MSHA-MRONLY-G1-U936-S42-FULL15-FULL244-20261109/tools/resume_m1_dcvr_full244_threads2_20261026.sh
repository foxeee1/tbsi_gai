#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
RUN="$ROOT/output/experiments/M1_DCVR_FULL15_SOT_20261025"
OUT="/autodl-fs/data/tbsi_dcvr/full15_20261025"
CONFIG="M1_DCVR_FULL15_SOT_20261025"
CKPT="$RUN/checkpoints/TBSITrack_ep0015.pth.tar"
START_INDEX="${1:-99}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TBSI_DCVR_MODE=on
export TBSI_DCVR_ALPHA=0.25
export TBSI_DCVR_DIAGNOSTICS_PATH="$OUT/dcvr_test_stats.jsonl"
export TBSI_CHECKPOINT_OVERRIDE="$CKPT"
export TBSI_TEMPLATE_UPDATE_MODE=off
export TBSI_INFER_FP16=0

cd "$ROOT"
echo "resume_start_utc=$(date -u +%FT%TZ) start_index=$START_INDEX threads=2 omp=1 mkll=1" >> "$OUT/provenance.txt"
test -s "$CKPT"

python -u tracking/resume_dataset.py \
  --start-index "$START_INDEX" \
  --tracker-name tbsi_track \
  --tracker-param "$CONFIG" \
  --dataset-name lasher_test \
  --threads 2 \
  --num-gpus 1 \
  > "$RUN/logs/test_full244_resume_threads2.log" 2>&1

python -u tracking/analysis_results.py \
  --tracker_name tbsi_track \
  --tracker_param "$CONFIG" \
  --dataset_name lasher_test \
  > "$RUN/logs/analysis_full244.log" 2>&1

echo "resume_end_utc=$(date -u +%FT%TZ)" >> "$OUT/provenance.txt"
