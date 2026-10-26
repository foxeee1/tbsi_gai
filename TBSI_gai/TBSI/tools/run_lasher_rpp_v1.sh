#!/usr/bin/env bash
set -Eeuo pipefail

# The only supported quick train/test protocol for LasHeR.
ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
BASELINE_YAML="${ROOT_DIR}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
MANIFEST_DIR="${ROOT_DIR}/experiments/proxy/lasher_rpp_v1"
CONFIG="RPP_LasHeR_v1_seed42"
SEED=42
TEST_THREADS=6
DO_TRAIN=1
DO_TEST=1
DO_ANALYZE=1
MIN_FREE_GB="${RPP_MIN_FREE_GB:-100}"

usage() {
  echo "Usage: $0 [--config NAME] [--seed 42|3407] [--skip-train] [--skip-test] [--skip-analysis]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --seed) SEED="$2"; CONFIG="RPP_LasHeR_v1_seed${SEED}"; shift 2 ;;
    --skip-train) DO_TRAIN=0; shift ;;
    --skip-test) DO_TEST=0; shift ;;
    --skip-analysis) DO_ANALYZE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$CONFIG" != *"v1.0.0-单卡3090-48G-等效全局128基线"* ]] || { echo "canonical baseline config is forbidden" >&2; exit 3; }
[[ "$SEED" == 42 || "$SEED" == 3407 ]] || { echo "seed must be 42 or 3407" >&2; exit 3; }
case "${CONFIG}:${SEED}" in
  RPP_LasHeR_v1_seed42:42|RPP_LasHeR_v1_seed3407:3407) ;;
  *) echo "config and seed must be one of the two frozen RPP-v1 pairs" >&2; exit 3 ;;
esac
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 3; }
[[ -f "$BASELINE_YAML" ]] || { echo "missing protected baseline YAML" >&2; exit 3; }
git -C "$ROOT_DIR" diff --quiet -- "$BASELINE_YAML" || { echo "protected baseline YAML has a diff" >&2; exit 4; }
for f in proxy_train_sequences.txt proxy_train_manifest.json proxy_test_sequences.txt proxy_test_manifest.json; do
  [[ -s "${MANIFEST_DIR}/${f}" ]] || { echo "missing RPP manifest: ${MANIFEST_DIR}/${f}" >&2; exit 5; }
done
[[ -f "${ROOT_DIR}/experiments/tbsi_track/${CONFIG}.yaml" ]] || { echo "missing config: ${CONFIG}" >&2; exit 5; }

FREE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
(( FREE_KB >= MIN_FREE_GB * 1024 * 1024 )) || { echo "insufficient free disk: ${FREE_KB} KiB" >&2; exit 6; }
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "GPU check failed" >&2; exit 6; }
else
  echo "nvidia-smi is required" >&2; exit 6
fi
if pgrep -af '[t]racking/(train|test)\.py|[l]ib/train/run_training\.py' >/tmp/tbsi_rpp_processes.$$ 2>/dev/null; then
  cat /tmp/tbsi_rpp_processes.$$ >&2
  rm -f /tmp/tbsi_rpp_processes.$$
  echo "another TBSI train/test process is running" >&2
  exit 7
fi
rm -f /tmp/tbsi_rpp_processes.$$

LOCK_DIR="${ROOT_DIR}/output/.locks"
mkdir -p "$LOCK_DIR"
exec 9>"${LOCK_DIR}/lasher_rpp_v1.lock"
flock -n 9 || { echo "another LasHeR RPP-v1 campaign holds the lock" >&2; exit 7; }

TRAIN_LIST="${MANIFEST_DIR}/proxy_train_sequences.txt"
TEST_LIST="${MANIFEST_DIR}/proxy_test_sequences.txt"
RUN_DIR="${ROOT_DIR}/output/experiments/${CONFIG}"
RESULT_DIR="${ROOT_DIR}/output/test/tracking_results/tbsi_track/${CONFIG}"
CHECKPOINT="${RUN_DIR}/checkpoints/TBSITrack_ep0015.pth.tar"
LOG_FILE="${RUN_DIR}/logs/rpp_v1_pipeline.log"
mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TBSI_RPP_TRAIN_LIST="$TRAIN_LIST"
export TBSI_RPP_TEST_LIST="$TEST_LIST"
export TBSI_INFER_FP16=0
cd "$ROOT_DIR"
echo "RPP-v1 START $(date -u +%FT%TZ) config=${CONFIG} seed=${SEED}"
echo "GIT $(git rev-parse HEAD)"
echo "BASELINE_YAML_SHA256 $(sha256sum "$BASELINE_YAML" | awk '{print $1}')"
echo "TRAIN_MANIFEST_SHA256 $(sha256sum "${MANIFEST_DIR}/proxy_train_manifest.json" | awk '{print $1}')"
echo "TEST_MANIFEST_SHA256 $(sha256sum "${MANIFEST_DIR}/proxy_test_manifest.json" | awk '{print $1}')"
echo "RPP_TEST_THREADS ${TEST_THREADS}"

if (( DO_TRAIN )); then
  [[ ! -e "$CHECKPOINT" ]] || { echo "refusing to overwrite checkpoint: $CHECKPOINT" >&2; exit 8; }
  "$PYTHON_BIN" -u lib/train/run_training.py --script tbsi_track --config "$CONFIG" \
    --save_dir ./output --seed "$SEED" --local_rank -1 --use_lmdb 0 \
    --config_prv baseline --use_wandb 0
fi

if (( DO_TEST )); then
  [[ -s "$CHECKPOINT" ]] || { echo "explicit epoch-15 checkpoint missing: $CHECKPOINT" >&2; exit 9; }
  [[ ! -e "$RESULT_DIR" ]] || { echo "refusing to overwrite tracking results: $RESULT_DIR" >&2; exit 9; }
  "$PYTHON_BIN" -u tracking/test.py tbsi_track "$CONFIG" --dataset_name rpp_lasher_test \
    --threads "$TEST_THREADS" --num_gpus 1
fi

if (( DO_ANALYZE )); then
  "$PYTHON_BIN" -u tracking/analysis_results.py --tracker_name tbsi_track \
    --tracker_param "$CONFIG" --dataset_name rpp_lasher_test
fi
echo "RPP-v1 FINISH $(date -u +%FT%TZ)"
