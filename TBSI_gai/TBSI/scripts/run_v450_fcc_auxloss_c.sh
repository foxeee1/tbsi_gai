#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TBSI_gai/TBSI

CFG="v4.5.0-fcc-auxloss-C"
LOG_DIR="output/diagnostics/fcc_auxloss_v450"
CKPT="output/experiments/${CFG}/checkpoints/TBSITrack_ep0015.pth.tar"
TEST_CKPT="output/checkpoints/train/tbsi_track/${CFG}/TBSITrack_ep0015.pth.tar"
ANALYSIS_LOG="${LOG_DIR}/analysis_${CFG}.console.log"

mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED=1
export TBSI_TEST_CPU_THREADS="${TBSI_TEST_CPU_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

echo "[$(date '+%F %T')] START ${CFG}"

python tracking/train.py \
  --script tbsi_track \
  --config "${CFG}" \
  --save_dir ./output \
  --mode single \
  --use_lmdb 0 \
  --use_wandb 0 \
  > "${LOG_DIR}/train_${CFG}.console.log" 2>&1

if [[ ! -f "${CKPT}" ]]; then
  echo "[$(date '+%F %T')] ERROR ${CFG} final checkpoint missing: ${CKPT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${TEST_CKPT}")"
ln -sfn "$(pwd)/${CKPT}" "${TEST_CKPT}"

echo "[$(date '+%F %T')] TEST ${CFG}"
rm -rf "output/test/tracking_results/tbsi_track/${CFG}"
python tracking/test.py tbsi_track "${CFG}" \
  --dataset_name mini_lasher_test \
  --threads 4 \
  --num_gpus 1 \
  > "${LOG_DIR}/test_${CFG}.console.log" 2>&1

echo "[$(date '+%F %T')] ANALYZE ${CFG}"
python tracking/analysis_results.py \
  --tracker_name tbsi_track \
  --tracker_param "${CFG}" \
  --dataset_name mini_lasher_test \
  > "${ANALYSIS_LOG}" 2>&1

find "output/experiments/${CFG}/checkpoints" -type f -name 'TBSITrack_ep0015.pth.tar' -delete
find "output/checkpoints/train/tbsi_track/${CFG}" -type l -delete 2>/dev/null || true

echo "[$(date '+%F %T')] DONE ${CFG}"
