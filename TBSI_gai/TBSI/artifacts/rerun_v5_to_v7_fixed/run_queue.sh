#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
cd "$ROOT"

RUN_ROOT="$ROOT/artifacts/rerun_v5_to_v7_fixed"
mkdir -p "$RUN_ROOT"

SUMMARY="$RUN_ROOT/summary.tsv"
if [[ ! -f "$SUMMARY" ]]; then
  printf "config\tbaseline_auc\tauc\tdelta\tstatus\n" > "$SUMMARY"
fi

BASELINE_AUC="55.46"
CONFIGS=(
  "v6.1.1-rtm-aligned-local-fixed"
  "v7.0.1-egir-only-fixed"
  "v7.1.2-rsm-v2-imbalance-fixed"
)

check_space() {
  local avail_kb
  avail_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  if [[ "$avail_kb" -lt 2500000 ]]; then
    echo "[STOP] Available disk below 2.5GB: ${avail_kb}KB"
    exit 2
  fi
}

extract_auc() {
  local log="$1"
  python - "$log" <<'PY'
import re
import sys
path = sys.argv[1]
text = open(path, 'r', encoding='utf-8', errors='ignore').read()
matches = re.findall(r'\|\s*([0-9]+\.[0-9]+)\s*\|\s*[0-9]+\.[0-9]+\s*\|\s*[0-9]+\.[0-9]+\s*\|\s*[0-9]+\.[0-9]+\s*\|\s*[0-9]+\.[0-9]+\s*\|', text)
if not matches:
    raise SystemExit(1)
print(matches[-1])
PY
}

for cfg in "${CONFIGS[@]}"; do
  check_space
  run_dir="$RUN_ROOT/$cfg"
  mkdir -p "$run_dir"
  train_log="$run_dir/train.log"
  test_log="$run_dir/test_lasher.log"
  eval_log="$run_dir/eval_lasher.log"

  if [[ ! -f "$ROOT/output/experiments/$cfg/checkpoints/TBSITrack_ep$(printf '%04d' "$(python - "$cfg" <<'PY'
import sys
from pathlib import Path
name = sys.argv[1]
text = Path(f'experiments/tbsi_track/{name}.yaml').read_text()
epoch = None
in_test = False
for line in text.splitlines():
    if line.startswith('TEST:'):
        in_test = True
    elif line and not line.startswith(' ') and not line.startswith('#'):
        in_test = False
    if in_test and line.strip().startswith('EPOCH:'):
        epoch = int(line.split(':', 1)[1].strip())
print(epoch)
PY
)").pth.tar" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] TRAIN $cfg"
    python -u lib/train/run_training.py \
      --script tbsi_track \
      --config "$cfg" \
      --save_dir ./output \
      --use_lmdb 0 \
      --use_wandb 0 \
      --seed 42 > "$train_log" 2>&1
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP TRAIN existing final checkpoint for $cfg"
  fi

  check_space
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] TEST $cfg"
  env TBSI_STRICT_CHECKPOINT=1 TBSI_TEST_CPU_THREADS=2 \
    python -u tracking/test.py tbsi_track "$cfg" \
      --dataset_name lasher_test \
      --threads 4 \
      --num_gpus 1 > "$test_log" 2>&1

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] EVAL $cfg"
  python tracking/analysis_results.py \
    --tracker_param "$cfg" \
    --dataset_name lasher_test > "$eval_log" 2>&1

  auc="$(extract_auc "$eval_log")"
  delta="$(python - "$auc" "$BASELINE_AUC" <<'PY'
import sys
auc = float(sys.argv[1])
base = float(sys.argv[2])
print(f"{auc - base:+.2f}")
PY
)"
  printf "%s\t%s\t%s\t%s\t%s\n" "$cfg" "$BASELINE_AUC" "$auc" "$delta" "done" >> "$SUMMARY"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE $cfg AUC=$auc delta=$delta"
done
