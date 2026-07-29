#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TBSI_gai/TBSI

ROOT_LOG_DIR="output/diagnostics/full_candidates_serial"
mkdir -p "${ROOT_LOG_DIR}" "output/test/tracking_results/tbsi_track"

export PYTHONUNBUFFERED=1
export TBSI_TEST_CPU_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TBSI_SEQUENCE_TIMEOUT="${TBSI_SEQUENCE_TIMEOUT:-1800}"

BASELINE_SRC="$(pwd)/output/experiments/v1.0.0-完美基线-全量/test_results"
BASELINE_DST="output/test/tracking_results/tbsi_track/v1.0.0-完美基线-全量"
ln -sfn "${BASELINE_SRC}" "${BASELINE_DST}"

run_one() {
  local cfg="$1"
  local tag="$2"
  local log_dir="${ROOT_LOG_DIR}/${tag}"
  local ckpt="output/experiments/${cfg}/checkpoints/TBSITrack_ep0015.pth.tar"
  local test_ckpt="output/checkpoints/train/tbsi_track/${cfg}/TBSITrack_ep0015.pth.tar"

  mkdir -p "${log_dir}" "$(dirname "${test_ckpt}")"
  echo "[$(date '+%F %T')] START ${cfg}"

  if [[ -f "${log_dir}/summary.json" ]]; then
    echo "[$(date '+%F %T')] summary exists, skip completed candidate: ${cfg}"
    return
  fi

  if [[ -f "${ckpt}" ]]; then
    echo "[$(date '+%F %T')] checkpoint exists, skip train: ${ckpt}"
  else
    python tracking/train.py \
      --script tbsi_track \
      --config "${cfg}" \
      --save_dir ./output \
      --mode single \
      --use_lmdb 0 \
      --use_wandb 0 \
      > "${log_dir}/train.console.log" 2>&1
  fi

  if [[ ! -f "${ckpt}" ]]; then
    echo "[$(date '+%F %T')] ERROR final checkpoint missing: ${ckpt}" >&2
    exit 2
  fi

  rm -f "${test_ckpt}"
  ln -s "$(pwd)/${ckpt}" "${test_ckpt}"

  echo "[$(date '+%F %T')] TEST ${cfg}"
  mkdir -p "output/test/tracking_results/tbsi_track/${cfg}"
  python tracking/test.py tbsi_track "${cfg}" \
    --dataset_name lasher_test \
    --threads 4 \
    --num_gpus 1 \
    > "${log_dir}/test.console.log" 2>&1

  echo "[$(date '+%F %T')] ANALYZE ${cfg}"
  python tracking/analysis_results.py \
    --tracker_name tbsi_track \
    --tracker_param "${cfg}" \
    --dataset_name lasher_test \
    > "${log_dir}/analysis.console.log" 2>&1

  echo "[$(date '+%F %T')] SUMMARIZE ${cfg}"
  python scripts/summarize_full_candidate.py \
    "${cfg}" \
    "${log_dir}" \
    "summary.json" \
    > "${log_dir}/summarize.console.log" 2>&1

  find "output/experiments/${cfg}/checkpoints" -type f -name 'TBSITrack_ep0015.pth.tar' -delete
  find "output/checkpoints/train/tbsi_track/${cfg}" -type l -delete 2>/dev/null || true
  echo "[$(date '+%F %T')] DELETED checkpoint for ${cfg}"
  echo "[$(date '+%F %T')] DONE ${cfg}"
}

run_one "v2.1.0-reliability-aware-residual-全量" "v210_full"
run_one "v4.2.0-fcc-residual-全量" "v420_full"
run_one "v3.1.0-cfs-reliability-bridge-全量" "v310_full"

echo "[$(date '+%F %T')] ALL DONE full candidates"
