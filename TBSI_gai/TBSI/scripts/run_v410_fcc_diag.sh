#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TBSI_gai/TBSI

LOG_DIR="output/experiments/v4.1.0-fcc-A/logs/stage1_attribute_sentinel"
DIAG_DIR="${LOG_DIR}/fcc_penalty_jsonl"
mkdir -p "${LOG_DIR}" "${DIAG_DIR}"

export PYTHONUNBUFFERED=1
export TBSI_TEST_CPU_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TBSI_FCC_DIAG_DIR="${DIAG_DIR}"

CFG="v4.1.0-fcc-A"
CKPT="output/experiments/${CFG}/checkpoints/TBSITrack_ep0015.pth.tar"
TEST_CKPT="output/checkpoints/train/tbsi_track/${CFG}/TBSITrack_ep0015.pth.tar"

if [[ ! -f "${CKPT}" ]]; then
  echo "missing checkpoint: ${CKPT}" >&2
  exit 2
fi
mkdir -p "$(dirname "${TEST_CKPT}")"
ln -sfn "$(pwd)/${CKPT}" "${TEST_CKPT}"

rm -f "${DIAG_DIR}"/fcc_stats_*.jsonl
rm -rf "output/test/tracking_results/tbsi_track/${CFG}"

python tracking/test.py tbsi_track "${CFG}" \
  --dataset_name mini_lasher_test \
  --threads 1 \
  --num_gpus 1 \
  > "${LOG_DIR}/test_fcc_diag_A.console.log" 2>&1

python scripts/summarize_fcc_diag.py \
  "${DIAG_DIR}" \
  "${LOG_DIR}/fcc_penalty_attribute_summary.json" \
  > "${LOG_DIR}/summarize_fcc_diag.console.log" 2>&1
