#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TBSI_gai/TBSI

LOG_DIR="output/diagnostics/reliability_aware_residual_late_layers_v211"
mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED=1
export TBSI_TEST_CPU_THREADS="${TBSI_TEST_CPU_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

run_one() {
  local subset="$1"
  local cfg="v2.1.1-reliability-aware-residual-late-layers-${subset}"
  local ckpt="output/experiments/${cfg}/checkpoints/TBSITrack_ep0015.pth.tar"
  local test_ckpt="output/checkpoints/train/tbsi_track/${cfg}/TBSITrack_ep0015.pth.tar"

  echo "[$(date '+%F %T')] START ${cfg}"

  if [[ -f "${ckpt}" ]]; then
    echo "[$(date '+%F %T')] ${cfg} checkpoint exists, skip train: ${ckpt}"
  else
    python tracking/train.py \
      --script tbsi_track \
      --config "${cfg}" \
      --save_dir ./output \
      --mode single \
      --use_lmdb 0 \
      --use_wandb 0 \
      > "${LOG_DIR}/train_${subset}.console.log" 2>&1
  fi

  if [[ ! -f "${ckpt}" ]]; then
    echo "[$(date '+%F %T')] ERROR ${cfg} final checkpoint missing: ${ckpt}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${test_ckpt}")"
  ln -sfn "$(pwd)/${ckpt}" "${test_ckpt}"

  echo "[$(date '+%F %T')] TEST ${cfg}"
  python tracking/test.py tbsi_track "${cfg}" \
    --dataset_name mini_lasher_test \
    --threads 4 \
    --num_gpus 1 \
    > "${LOG_DIR}/test_${subset}.console.log" 2>&1

  echo "[$(date '+%F %T')] ANALYZE ${cfg}"
  python tracking/analysis_results.py \
    --tracker_name tbsi_track \
    --tracker_param "${cfg}" \
    --dataset_name mini_lasher_test \
    > "${LOG_DIR}/analysis_${subset}.console.log" 2>&1

  echo "[$(date '+%F %T')] DONE ${cfg}"
}

run_one A
run_one B
run_one C

echo "[$(date '+%F %T')] SUMMARIZE v2.1.1 reliability-aware residual late-layers ABC"
python scripts/summarize_v211_reliability_aware_residual_late_layers.py \
  > "${LOG_DIR}/summarize.console.log" 2>&1

echo "[$(date '+%F %T')] ALL DONE v2.1.1 reliability-aware residual late-layers selective interaction ABC"
