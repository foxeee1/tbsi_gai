#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
BASELINE="experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
LOG_DIR="output/experiments/template_expert_p1_20260928/logs"
SAVE_ROOT="output/template_expert_p1_retry/experiments"
export PYTHONPATH="$ROOT"
export TBSI_INFER_FP16=0

[[ -z "$(git diff -- "$BASELINE")" ]] || { echo "protected canonical YAML is modified" >&2; exit 3; }
for config in TEMPLATE_EXPERT_C_P1_seed42 TEMPLATE_EXPERT_G_P1_seed42 TEMPLATE_EXPERT_T_P1_seed42; do
  ckpt="$SAVE_ROOT/$config/checkpoints/TBSITrack_ep0003.pth.tar"
  [[ -f "$ckpt" ]] || { echo "missing checkpoint: $ckpt" >&2; exit 4; }
  result="output/test/tracking_results/tbsi_track/$config"
  [[ ! -e "$result" ]] || { echo "result directory exists: $result" >&2; exit 5; }
done

for config in TEMPLATE_EXPERT_C_P1_seed42 TEMPLATE_EXPERT_G_P1_seed42 TEMPLATE_EXPERT_T_P1_seed42; do
  ckpt="$SAVE_ROOT/$config/checkpoints/TBSITrack_ep0003.pth.tar"
  echo "TEST $config start $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/evaluation.log"
  TBSI_CHECKPOINT_OVERRIDE="$ROOT/$ckpt" "$PYTHON_BIN" -u tracking/test.py tbsi_track "$config" \
    --dataset_name lasher_test --threads 6 --num_gpus 1 2>&1 | tee "$LOG_DIR/${config}_test.log"
  "$PYTHON_BIN" -u tracking/analysis_results.py --tracker_name tbsi_track \
    --tracker_param "$config" --dataset_name lasher_test 2>&1 | tee "$LOG_DIR/${config}_analysis.log"
  echo "TEST $config exit=0 $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/evaluation.log"
done
