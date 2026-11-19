#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
CONFIG="TBSI_E_MHSA_FULL_4090D_LOC4710_seed42"
BASELINE_YAML="${ROOT_DIR}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
PROTECTED_M1_YAML="${ROOT_DIR}/experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml"
CONFIG_YAML="${ROOT_DIR}/experiments/tbsi_track/${CONFIG}.yaml"
RUN_DIR="${ROOT_DIR}/output/experiments/${CONFIG}"
RESULT_DIR="${ROOT_DIR}/output/test/tracking_results/tbsi_track/${CONFIG}"
CHECKPOINT="${RUN_DIR}/checkpoints/TBSITrack_ep0015.pth.tar"
LOG_FILE="${RUN_DIR}/logs/full_pipeline.log"
MIN_FREE_GB="${TBSI_E_MIN_FREE_GB:-60}"

[[ -x "$PYTHON_BIN" && -f "$CONFIG_YAML" && -f "$BASELINE_YAML" ]] || exit 3
git -C "$ROOT_DIR" diff --quiet -- "$BASELINE_YAML" || exit 4
[[ "$(sha256sum "$PROTECTED_M1_YAML" | awk '{print $1}')" == "fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e" ]] || exit 4
[[ ! -e "$RUN_DIR" && ! -e "$RESULT_DIR" ]] || exit 5
FREE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
(( FREE_KB >= MIN_FREE_GB * 1024 * 1024 )) || exit 6
if pgrep -af '[t]racking/(train|test)\.py|[l]ib/train/run_training\.py' >/tmp/tbsi_loc4710_processes.$$ 2>/dev/null; then
  cat /tmp/tbsi_loc4710_processes.$$ >&2; rm -f /tmp/tbsi_loc4710_processes.$$; exit 7
fi
rm -f /tmp/tbsi_loc4710_processes.$$
mkdir -p "${ROOT_DIR}/output/.locks"
exec 9>"${ROOT_DIR}/output/.locks/tbsi_e_loc4710_full.lock"
flock -n 9 || exit 7
mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "$LOG_FILE") 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg TBSI_INFER_FP16=0
cd "$ROOT_DIR"
echo "START $(date -u +%FT%TZ)"
echo "CONFIG ${CONFIG}"
echo "GIT $(git rev-parse HEAD)"
echo "CONFIG_YAML_SHA256 $(sha256sum "$CONFIG_YAML" | awk '{print $1}')"
echo "PROTECTED_M1_YAML_SHA256 $(sha256sum "$PROTECTED_M1_YAML" | awk '{print $1}')"
"$PYTHON_BIN" -c 'import sys, torch; print("PYTHON", sys.version.replace("\n", " ")); print("PYTORCH", torch.__version__); print("CUDA", torch.version.cuda); print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")'
"$PYTHON_BIN" -u tracking/train.py --script tbsi_track --config "$CONFIG" --save_dir ./output --mode single
[[ -s "$CHECKPOINT" ]] || exit 8
"$PYTHON_BIN" -u tracking/test.py tbsi_track "$CONFIG" --dataset_name lasher_test --threads 0 --num_gpus 1
[[ -d "$RESULT_DIR" ]] || exit 9
"$PYTHON_BIN" -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param "$CONFIG" --dataset_name lasher_test
echo "FINISH $(date -u +%FT%TZ)"
