#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TBSI_gai/TBSI

PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
CONFIG=v1.0.7-smsa-only-full-15ep
SAVE_DIR=./output
RUN_DIR=./output/experiments/${CONFIG}
LOG_DIR=${RUN_DIR}/logs
mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TBSI_DEBUG_FINITE=1

run_stage() {
    local name="$1"
    shift
    local log="${LOG_DIR}/${name}.stdout.log"
    echo "[$(date -Is)] START ${name}" | tee -a "${LOG_DIR}/pipeline.log"
    "$@" >"${log}" 2>&1 &
    local pid=$!
    while kill -0 "${pid}" 2>/dev/null; do
        sleep 3600
        if kill -0 "${pid}" 2>/dev/null; then
            {
                echo "[$(date -Is)] HEARTBEAT ${name} pid=${pid}"
                df -h /root/autodl-tmp | tail -1
                tail -n 3 "${log}" 2>/dev/null || true
            } | tee -a "${LOG_DIR}/pipeline.log"
        fi
    done
    wait "${pid}"
    echo "[$(date -Is)] DONE ${name}" | tee -a "${LOG_DIR}/pipeline.log"
}

run_stage train \
    "${PYTHON}" -u tracking/train.py \
    --script tbsi_track --config "${CONFIG}" --save_dir "${SAVE_DIR}" --mode single

run_stage test env TBSI_INFER_FP16=0 "${PYTHON}" -u tracking/test.py \
    tbsi_track "${CONFIG}" --dataset_name lasher_test --threads 0 --num_gpus 1

run_stage analysis "${PYTHON}" -u tracking/analysis_results.py \
    --tracker_name tbsi_track --tracker_param "${CONFIG}" \
    --dataset_name lasher_test

echo "[$(date -Is)] PIPELINE COMPLETE" | tee -a "${LOG_DIR}/pipeline.log"
