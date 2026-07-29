#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TBSI_gai/TBSI

LOG_DIR="output/diagnostics/fcc_bc_v4110_v4111"
mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED=1
export TBSI_TEST_CPU_THREADS="${TBSI_TEST_CPU_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

run_one() {
  local subset="$1"
  local cfg="${2}-${subset}"
  local ckpt="output/experiments/${cfg}/checkpoints/TBSITrack_ep0015.pth.tar"
  local test_ckpt="output/checkpoints/train/tbsi_track/${cfg}/TBSITrack_ep0015.pth.tar"
  local analysis_log="${LOG_DIR}/analysis_${cfg}.console.log"

  echo "[$(date '+%F %T')] START ${cfg}"

  if [[ -f "${analysis_log}" ]] && grep -q "Reporting results over" "${analysis_log}"; then
    echo "[$(date '+%F %T')] ${cfg} analysis exists, skip completed seed: ${analysis_log}"
    return
  fi

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
      > "${LOG_DIR}/train_${cfg}.console.log" 2>&1
  fi

  if [[ ! -f "${ckpt}" ]]; then
    echo "[$(date '+%F %T')] ERROR ${cfg} final checkpoint missing: ${ckpt}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${test_ckpt}")"
  ln -sfn "$(pwd)/${ckpt}" "${test_ckpt}"

  echo "[$(date '+%F %T')] TEST ${cfg}"
  rm -rf "output/test/tracking_results/tbsi_track/${cfg}"
  python tracking/test.py tbsi_track "${cfg}" \
    --dataset_name mini_lasher_test \
    --threads 4 \
    --num_gpus 1 \
    > "${LOG_DIR}/test_${cfg}.console.log" 2>&1

  echo "[$(date '+%F %T')] ANALYZE ${cfg}"
  python tracking/analysis_results.py \
    --tracker_name tbsi_track \
    --tracker_param "${cfg}" \
    --dataset_name mini_lasher_test \
    > "${analysis_log}" 2>&1

  find "output/experiments/${cfg}/checkpoints" -type f -name 'TBSITrack_ep0015.pth.tar' -delete
  find "output/checkpoints/train/tbsi_track/${cfg}" -type l -delete 2>/dev/null || true

  echo "[$(date '+%F %T')] DONE ${cfg}"
}

for cfg_prefix in v4.1.10-fcc-lowmid-middle v4.1.11-fcc-layerwise; do
  run_one A "${cfg_prefix}"
  run_one B "${cfg_prefix}"
  run_one C "${cfg_prefix}"
done

echo "[$(date '+%F %T')] SUMMARIZE v4.1.10 and v4.1.11 FCC ABC"
python scripts/summarize_v4110_v4111_fcc_bc.py \
  > "${LOG_DIR}/summarize.console.log" 2>&1

echo "[$(date '+%F %T')] ALL DONE v4.1.10 and v4.1.11 FCC ABC"
