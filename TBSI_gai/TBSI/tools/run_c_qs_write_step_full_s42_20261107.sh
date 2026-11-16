#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
CONFIG="C_QS_WRITE_STEP_S42_20261107"
RUN_DIR="${ROOT_DIR}/output/experiments/${CONFIG}"
RESULT_DIR="${ROOT_DIR}/output/test/tracking_results/tbsi_track/${CONFIG}"
CHECKPOINT="${RUN_DIR}/checkpoints/TBSITrack_ep0015.pth.tar"
LOG_FILE="${RUN_DIR}/logs/full_train_test.log"
BASELINE_CFG="${ROOT_DIR}/experiments/tbsi_track/EG_SEARCH_CONTEXT_C_SOT_S42_20261031.yaml"
CONFIG_YAML="${ROOT_DIR}/experiments/tbsi_track/${CONFIG}.yaml"
PRETRAIN="${ROOT_DIR}/pretrained_models/TBSITrack_SOT_Pretrained.pth.tar"

[[ "$ROOT_DIR" == /root/autodl-tmp/* ]] || { echo "invalid project root" >&2; exit 2; }
[[ -x "$PYTHON_BIN" && -f "$CONFIG_YAML" && -f "$PRETRAIN" ]] || { echo "missing runtime/config/pretrain" >&2; exit 3; }
[[ ! -e "$RUN_DIR" && ! -e "$RESULT_DIR" ]] || { echo "refusing to reuse output" >&2; exit 4; }
if pgrep -af '[t]racking/(train|test)\.py|[l]ib/train/run_training\.py' >/tmp/tbsi_qs_existing.$$ 2>/dev/null; then
  cat /tmp/tbsi_qs_existing.$$ >&2; rm -f /tmp/tbsi_qs_existing.$$; exit 5
fi
rm -f /tmp/tbsi_qs_existing.$$
mkdir -p "${RUN_DIR}/logs" "${ROOT_DIR}/output/.locks"
exec 9>"${ROOT_DIR}/output/.locks/c_qs_write_step_full_s42.lock"
flock -n 9 || { echo "another TBSI run holds the lock" >&2; exit 5; }
exec > >(tee -a "$LOG_FILE") 2>&1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg TBSI_INFER_FP16=0
cd "$ROOT_DIR"
echo "START $(date -u +%FT%TZ)"
echo "CONFIG ${CONFIG} SEED 42"
echo "ROOT ${ROOT_DIR}"
echo "GIT $(git rev-parse HEAD)"
echo "CONFIG_SHA256 $(sha256sum "$CONFIG_YAML" | awk '{print $1}')"
echo "BASELINE_C_SHA256 $(sha256sum "$BASELINE_CFG" | awk '{print $1}')"
echo "PRETRAIN_SHA256 $(sha256sum "$PRETRAIN" | awk '{print $1}')"
"$PYTHON_BIN" -c 'import sys,torch; print("PYTHON",sys.version.replace("\n"," ")); print("PYTORCH",torch.__version__); print("CUDA",torch.version.cuda); print("GPU",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")'

echo "=== TRAIN: public SOT pretrain, fresh single-stage, C-StepLR, seed42 ==="
"$PYTHON_BIN" -u tracking/train.py --script tbsi_track --config "$CONFIG" --save_dir ./output --mode single
[[ -s "$CHECKPOINT" ]] || { echo "missing epoch15 checkpoint" >&2; exit 6; }

echo "=== FULL TEST: LasHeR 244, FP32, threads0 ==="
"$PYTHON_BIN" -u tracking/test.py tbsi_track "$CONFIG" --dataset_name lasher_test --threads 0 --num_gpus 1
[[ -d "$RESULT_DIR" ]] || { echo "missing result directory" >&2; exit 7; }
echo "RESULT_FILES $(find "$RESULT_DIR" -maxdepth 1 -type f -name '*.txt' | wc -l)"

echo "=== ANALYSIS ==="
"$PYTHON_BIN" -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param "$CONFIG" --dataset_name lasher_test
echo "FINISH $(date -u +%FT%TZ)"
