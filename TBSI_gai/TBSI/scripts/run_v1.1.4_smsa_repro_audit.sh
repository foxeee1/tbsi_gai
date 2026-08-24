#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
CONFIG=v1.1.4-smsa-only-repro-audit
RUN_DIR=output/experiments/${CONFIG}
LOG_DIR=${RUN_DIR}/logs
RESULT_DIR=output/test/tracking_results/tbsi_track/${CONFIG}
OLD_CKPT=output/experiments/v1.0.7-smsa-only-full-15ep/checkpoints/TBSITrack_ep0015.pth.tar
mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${RUN_DIR}/checkpoints"
if [[ ! -e "${RUN_DIR}/checkpoints/TBSITrack_ep0015.pth.tar" ]]; then
  ln -s "$(realpath "${OLD_CKPT}")" "${RUN_DIR}/checkpoints/TBSITrack_ep0015.pth.tar"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TBSI_INFER_FP16=0
echo "[$(date -Is)] START SMSA-only checkpoint reproduction audit" | tee -a "${LOG_DIR}/pipeline.log"
"${PYTHON}" -u tracking/test.py tbsi_track "${CONFIG}" \
  --dataset_name lasher_test --threads 0 --num_gpus 1 \
  >"${LOG_DIR}/test.stdout.log" 2>&1
echo "[$(date -Is)] DONE test" | tee -a "${LOG_DIR}/pipeline.log"
"${PYTHON}" -u tracking/analysis_results.py \
  --tracker_name tbsi_track --tracker_param "${CONFIG}" \
  --dataset_name lasher_test \
  >"${LOG_DIR}/analysis.stdout.log" 2>&1
echo "[$(date -Is)] DONE analysis" | tee -a "${LOG_DIR}/pipeline.log"
