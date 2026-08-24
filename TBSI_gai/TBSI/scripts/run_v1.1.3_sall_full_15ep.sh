#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
CONFIG=v1.1.3-sall-full-15ep
SAVE_DIR=./output
RUN_DIR=${SAVE_DIR}/experiments/${CONFIG}
LOG_DIR=${RUN_DIR}/logs
CHECKPOINT=${RUN_DIR}/checkpoints/TBSITrack_ep0015.pth.tar
RESULT_DIR=output/test/tracking_results/tbsi_track/${CONFIG}
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TBSI_DEBUG_FINITE=1

log(){ echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/pipeline.log"; }

run_stage(){
  local stage="$1"; shift
  local out="${LOG_DIR}/${stage}.stdout.log"
  log "START ${stage}"
  "$@" >"${out}" 2>&1 &
  local pid=$!
  local last=0
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 60
    if ! kill -0 "${pid}" 2>/dev/null; then break; fi
    local now; now=$(date +%s)
    if (( now - last >= 3600 )); then
      log "HEARTBEAT ${stage} pid=${pid}"
      nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | tee -a "${LOG_DIR}/pipeline.log" || true
      df -h /root/autodl-tmp | tail -1 | tee -a "${LOG_DIR}/pipeline.log"
      tail -n 3 "${out}" | tee -a "${LOG_DIR}/pipeline.log" || true
      last=${now}
    fi
  done
  wait "${pid}"
  log "DONE ${stage}"
}

if [[ -f "${CHECKPOINT}" ]]; then
  log "SKIP train; checkpoint exists: ${CHECKPOINT}"
else
  run_stage train "${PYTHON}" -u tracking/train.py \
    --script tbsi_track --config "${CONFIG}" --save_dir "${SAVE_DIR}" --mode single
fi

result_count=$(find "${RESULT_DIR}" -maxdepth 1 -type f -name '*.txt' ! -name '*_time.txt' 2>/dev/null | wc -l || true)
if [[ "${result_count}" -ge 244 ]]; then
  log "SKIP test; ${result_count}/244 sequence results already exist"
else
  run_stage test env TBSI_INFER_FP16=0 "${PYTHON}" -u tracking/test.py \
    tbsi_track "${CONFIG}" --dataset_name lasher_test --threads 0 --num_gpus 1
fi

run_stage analysis "${PYTHON}" -u tracking/analysis_results.py \
  --tracker_name tbsi_track --tracker_param "${CONFIG}" --dataset_name lasher_test

log "FULL S-ALL EXPERIMENT COMPLETE"
