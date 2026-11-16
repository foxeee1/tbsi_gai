#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
CONFIG="C_MASS_REDIST_EVDR_ALIGNED_U936_S3407_20261109"
RUN_DIR="${ROOT_DIR}/output/experiments/${CONFIG}"
RESULT_DIR="${ROOT_DIR}/output/test/tracking_results/tbsi_track/${CONFIG}"
CHECKPOINT="${RUN_DIR}/checkpoints/TBSITrack_ep0015.pth.tar"
LOG_FILE="${RUN_DIR}/logs/full_train_test.log"
CONFIG_YAML="${ROOT_DIR}/experiments/tbsi_track/${CONFIG}.yaml"
PRETRAIN="${ROOT_DIR}/pretrained_models/TBSITrack_SOT_Pretrained.pth.tar"

[[ "$ROOT_DIR" == /root/autodl-tmp/* ]] || { echo "invalid project root" >&2; exit 2; }
[[ -x "$PYTHON_BIN" && -f "$CONFIG_YAML" && -f "$PRETRAIN" ]] || { echo "missing runtime/config/pretrain" >&2; exit 3; }
[[ ! -e "$RUN_DIR" && ! -e "$RESULT_DIR" ]] || { echo "refusing to reuse output" >&2; exit 4; }

mkdir -p "${RUN_DIR}/logs" "${ROOT_DIR}/output/.locks"
exec 9>"${ROOT_DIR}/output/.locks/c_mass_redist_evdr_u936_s3407.lock"
flock -n 9 || { echo "another resource-intensive run holds the lock" >&2; exit 5; }
exec > >(tee -a "$LOG_FILE") 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg TBSI_INFER_FP16=0
cd "$ROOT_DIR"

if pgrep -af '[t]racking/(train|test)\.py|[l]ib/train/run_training\.py|[t]racking/analysis_results\.py' >/tmp/tbsi_evdr3407_active.$$ 2>/dev/null; then
  cat /tmp/tbsi_evdr3407_active.$$ >&2
  rm -f /tmp/tbsi_evdr3407_active.$$
  echo "resource-intensive process is active; refusing overlap" >&2
  exit 6
fi
rm -f /tmp/tbsi_evdr3407_active.$$

FREE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
(( FREE_KB >= 5 * 1024 * 1024 )) || { echo "less than 5 GiB free disk" >&2; exit 7; }

echo "START $(date -u +%FT%TZ)"
echo "CONFIG ${CONFIG} SEED 3407 SCHEDULER Mstep MILESTONES=[2] GAMMA=0.1"
echo "MODEL EVDR ENABLED; all model fields match seed42 candidate"
echo "ROOT ${ROOT_DIR}"
echo "TRAIN_ENTRY lib/train/run_training.py --seed 3407 (tracking/train.py drops --seed)"
echo "GIT $(git rev-parse HEAD)"
echo "CONFIG_SHA256 $(sha256sum "$CONFIG_YAML" | awk '{print $1}')"
echo "PRETRAIN_SHA256 $(sha256sum "$PRETRAIN" | awk '{print $1}')"
tar --exclude='__pycache__' --exclude='*.pyc' -czf "${RUN_DIR}/source_code_snapshot.tar.gz" \
  lib tracking tools/run_c_mass_redist_evdr_aligned_u936_s3407_20261109.sh \
  experiments/tbsi_track experiments/early_gate_closed_loop_20261026/p1_sot_v3
sha256sum "${RUN_DIR}/source_code_snapshot.tar.gz" | tee "${RUN_DIR}/source_code_snapshot.sha256"
"$PYTHON_BIN" -c 'import sys,torch; print("PYTHON",sys.version.replace("\n"," ")); print("PYTORCH",torch.__version__); print("CUDA",torch.version.cuda); print("GPU",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")'

echo "=== TRAIN: fresh public SOT, single-stage, 7020 updates, u936 decay, seed3407 ==="
"$PYTHON_BIN" -u lib/train/run_training.py --script tbsi_track --config "$CONFIG" --save_dir ./output --seed 3407
[[ -s "$CHECKPOINT" ]] || { echo "missing epoch15 checkpoint" >&2; exit 10; }

echo "=== FULL TEST: LasHeR 244, FP32, threads0 ==="
"$PYTHON_BIN" -u tracking/test.py tbsi_track "$CONFIG" --dataset_name lasher_test --threads 0 --num_gpus 1
[[ -d "$RESULT_DIR" ]] || { echo "missing result directory" >&2; exit 11; }
echo "RESULT_FILES $(find "$RESULT_DIR" -maxdepth 1 -type f -name '*.txt' | wc -l)"

echo "=== ANALYSIS ==="
"$PYTHON_BIN" -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param "$CONFIG" --dataset_name lasher_test
echo "FINISH $(date -u +%FT%TZ)"
