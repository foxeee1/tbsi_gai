#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export TBSI_INFER_FP16=0

PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
CONFIG="M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023"
CONFIG_YAML="experiments/tbsi_track/${CONFIG}.yaml"
RUN_DIR="output/experiments/${CONFIG}"
RESULT_DIR="output/test/tracking_results/tbsi_track/${CONFIG}"
LOG_FILE="${RUN_DIR}/logs/full_pipeline.log"

[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 3; }
[[ -f "$CONFIG_YAML" ]] || { echo "missing config: $CONFIG_YAML" >&2; exit 3; }
[[ ! -e "$RUN_DIR" ]] || { echo "refusing to reuse run directory: $RUN_DIR" >&2; exit 5; }
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing to reuse result directory: $RESULT_DIR" >&2; exit 5; }
if pgrep -af '[t]racking/(train|test)\.py|[l]ib/train/run_training\.py' >/tmp/m1_early_gate_processes.$$ 2>/dev/null; then
  cat /tmp/m1_early_gate_processes.$$ >&2
  rm -f /tmp/m1_early_gate_processes.$$ 
  echo "another TBSI training or test process is running" >&2
  exit 7
fi
rm -f /tmp/m1_early_gate_processes.$$
mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "START $(date -u +%FT%TZ)"
echo "CONFIG ${CONFIG}"
echo "CONFIG_YAML_SHA256 $(sha256sum "$CONFIG_YAML" | awk '{print $1}')"
"$PYTHON_BIN" -c 'import sys, torch; print("PYTHON", sys.version.replace("\n", " ")); print("PYTORCH", torch.__version__); print("CUDA", torch.version.cuda); print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")'

"$PYTHON_BIN" -u tracking/train.py \
  --script tbsi_track \
  --config "$CONFIG" \
  --save_dir "output" \
  --mode single

CHECKPOINT="${RUN_DIR}/checkpoints/TBSITrack_ep0015.pth.tar"
[[ -s "$CHECKPOINT" ]] || { echo "missing epoch-15 checkpoint: $CHECKPOINT" >&2; exit 8; }
"$PYTHON_BIN" -u tracking/test.py tbsi_track "$CONFIG" \
  --dataset_name lasher_test --threads 0 --num_gpus 1
[[ -d "$RESULT_DIR" ]] || { echo "missing tracking result directory: $RESULT_DIR" >&2; exit 9; }
"$PYTHON_BIN" -u tracking/analysis_results.py --tracker_name tbsi_track \
  --tracker_param "$CONFIG" --dataset_name lasher_test
echo "FINISH $(date -u +%FT%TZ)"
