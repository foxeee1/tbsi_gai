#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
OUTPUT_ROOT="$ROOT/output"
CFG="完美基线全量"
RUN_ROOT="$ROOT/artifacts/baseline/current-reverted-c792048"
LOG_DIR="$RUN_ROOT/logs"
PYTHON="/root/miniconda3/bin/python"
RESULT_DIR="$OUTPUT_ROOT/test/tracking_results/tbsi_track/$CFG"

mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export TBSI_LASHER_PROTOCOL=legacy244
cd "$ROOT"

stage() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" | tee "$RUN_ROOT/stage.txt"
}

"$PYTHON" - <<'PY' | tee "$RUN_ROOT/python_env.txt"
import sys, torch
print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
PY

stage "current_reverted_training"
"$PYTHON" lib/train/run_training.py \
  --script tbsi_track --config "$CFG" --save_dir "$OUTPUT_ROOT" --seed 42 \
  2>&1 | tee "$LOG_DIR/train.log"

CHECKPOINT="$OUTPUT_ROOT/experiments/$CFG/checkpoints/TBSITrack_ep0015.pth.tar"
test -f "$CHECKPOINT"
sha256sum "$CHECKPOINT" | tee "$RUN_ROOT/ep15.sha256"

stage "current_reverted_legacy244_test"
"$PYTHON" tracking/test.py tbsi_track "$CFG" \
  --dataset_name lasher_test --threads 2 --num_gpus 1 \
  2>&1 | tee "$LOG_DIR/test.log"

count=$(find -L "$RESULT_DIR" -maxdepth 1 -type f -name '*.txt' ! -name '*_time.txt' | wc -l)
printf 'sequence_files=%s\n' "$count" | tee "$LOG_DIR/count.log"
if [[ "$count" -ne 244 ]]; then
  printf 'Expected 244 sequence outputs, found %s\n' "$count" >&2
  exit 1
fi

stage "current_reverted_analysis"
"$PYTHON" tracking/analysis_results.py \
  --tracker_name tbsi_track --tracker_param "$CFG" --dataset_name lasher_test \
  2>&1 | tee "$LOG_DIR/analysis.log"

stage "complete"
