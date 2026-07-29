#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TBSI_gai/TBSI

LOG_DIR="output/diagnostics/fcc_v416_full"
mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED=1

CFG="v4.1.6-fcc-middle-全量"
CKPT="output/experiments/${CFG}/checkpoints/TBSITrack_ep0015.pth.tar"
TEST_CKPT="output/checkpoints/train/tbsi_track/${CFG}/TBSITrack_ep0015.pth.tar"
BASELINE_SRC="$(pwd)/output/experiments/v1.0.0-完美基线-全量/test_results"
BASELINE_DST="output/test/tracking_results/tbsi_track/v1.0.0-完美基线-全量"

mkdir -p "$(dirname "${TEST_CKPT}")" "output/test/tracking_results/tbsi_track"
ln -sfn "${BASELINE_SRC}" "${BASELINE_DST}"

echo "[$(date '+%F %T')] START ${CFG}"
if [[ -f "${CKPT}" ]]; then
  echo "[$(date '+%F %T')] checkpoint exists, skip train: ${CKPT}"
else
  python tracking/train.py \
    --script tbsi_track \
    --config "${CFG}" \
    --save_dir ./output \
    --mode single \
    --use_lmdb 0 \
    --use_wandb 0 \
    > "${LOG_DIR}/train.console.log" 2>&1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[$(date '+%F %T')] ERROR final checkpoint missing: ${CKPT}" >&2
  exit 2
fi

ln -sfn "$(pwd)/${CKPT}" "${TEST_CKPT}"

export TBSI_TEST_CPU_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[$(date '+%F %T')] TEST ${CFG}"
rm -rf "output/test/tracking_results/tbsi_track/${CFG}"
python tracking/test.py tbsi_track "${CFG}" \
  --dataset_name lasher_test \
  --threads 4 \
  --num_gpus 1 \
  > "${LOG_DIR}/test.console.log" 2>&1

echo "[$(date '+%F %T')] ANALYZE ${CFG}"
python tracking/analysis_results.py \
  --tracker_name tbsi_track \
  --tracker_param "${CFG}" \
  --dataset_name lasher_test \
  > "${LOG_DIR}/analysis.console.log" 2>&1

echo "[$(date '+%F %T')] SUMMARIZE ${CFG}"
python scripts/summarize_v416_full.py \
  > "${LOG_DIR}/summarize.console.log" 2>&1

echo "[$(date '+%F %T')] ALL DONE ${CFG}"
