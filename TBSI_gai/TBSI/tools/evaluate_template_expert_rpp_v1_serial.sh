#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
MANIFEST_DIR="${ROOT_DIR}/experiments/proxy/lasher_rpp_v1"
SAVE_ROOT="${ROOT_DIR}/output/template_expert_p1_retry/experiments"
LOG_DIR="${ROOT_DIR}/output/experiments/template_expert_p1_20260928/logs"
BASELINE_YAML="${ROOT_DIR}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
TEST_LIST="${MANIFEST_DIR}/proxy_test_sequences.txt"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TBSI_RPP_TEST_LIST="$TEST_LIST"
export TBSI_INFER_FP16=0

[[ -z "$(git diff -- "$BASELINE_YAML")" ]] || { echo "protected canonical YAML is modified" >&2; exit 3; }
[[ -s "$TEST_LIST" ]] || { echo "missing frozen test manifest" >&2; exit 4; }
for config in TEMPLATE_EXPERT_C_P1_seed42 TEMPLATE_EXPERT_G_P1_seed42 TEMPLATE_EXPERT_T_P1_seed42; do
  ckpt="$SAVE_ROOT/$config/checkpoints/TBSITrack_ep0003.pth.tar"
  [[ -s "$ckpt" ]] || { echo "missing checkpoint: $ckpt" >&2; exit 5; }
  result="output/test/tracking_results/tbsi_track/$config"
  [[ ! -e "$result" ]] || { echo "refusing to overwrite result directory: $result" >&2; exit 6; }
done
if pgrep -af '[t]racking/(train|test)\.py|[l]ib/train/run_training\.py' >/tmp/tbsi_template_rpp_processes.$$ 2>/dev/null; then
  cat /tmp/tbsi_template_rpp_processes.$$ >&2
  rm -f /tmp/tbsi_template_rpp_processes.$$
  echo "another TBSI train/test process is running" >&2
  exit 7
fi
rm -f /tmp/tbsi_template_rpp_processes.$$

mkdir -p "$LOG_DIR"
echo "TEMPLATE EXPERT RPP-v1 SERIAL START $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/template_expert_rpp_v1_serial.log"
echo "TEST_MANIFEST_SHA256 $(sha256sum "$MANIFEST_DIR/proxy_test_manifest.json" | awk '{print $1}')" | tee -a "$LOG_DIR/template_expert_rpp_v1_serial.log"
echo "TEST_COUNT $(wc -l < "$TEST_LIST")" | tee -a "$LOG_DIR/template_expert_rpp_v1_serial.log"

for config in TEMPLATE_EXPERT_C_P1_seed42 TEMPLATE_EXPERT_G_P1_seed42 TEMPLATE_EXPERT_T_P1_seed42; do
  ckpt="$SAVE_ROOT/$config/checkpoints/TBSITrack_ep0003.pth.tar"
  echo "TEST $config START $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/template_expert_rpp_v1_serial.log"
  TBSI_CHECKPOINT_OVERRIDE="$ckpt" "$PYTHON_BIN" -u tracking/test.py tbsi_track "$config" \
    --dataset_name rpp_lasher_test --threads 0 --num_gpus 1 2>&1 | tee "$LOG_DIR/${config}_rpp_v1_test.log"
  "$PYTHON_BIN" -u tracking/analysis_results.py --tracker_name tbsi_track \
    --tracker_param "$config" --dataset_name rpp_lasher_test 2>&1 | tee "$LOG_DIR/${config}_rpp_v1_analysis.log"
  echo "TEST $config EXIT=0 $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/template_expert_rpp_v1_serial.log"
done
echo "TEMPLATE EXPERT RPP-v1 SERIAL FINISH $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/template_expert_rpp_v1_serial.log"
