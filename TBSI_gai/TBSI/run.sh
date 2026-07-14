#!/bin/bash
# ============================================================================
# TBSI Training → Testing → Analysis Pipeline
# ============================================================================
# Usage: bash run.sh <config_name>
#   e.g. bash run.sh vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_reproduce
# ============================================================================
set -e

if [ $# -lt 1 ]; then
    echo "Usage: bash run.sh <config_name>"
    echo "  e.g. bash run.sh vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_reproduce"
    exit 1
fi

CONFIG="$1"
SCRIPT="tbsi_track"
DATASET="lasher_test"
ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
LOG_DIR="${ROOT}/output/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PIPELINE_LOG="${LOG_DIR}/full_pipeline_${CONFIG}_${TIMESTAMP}.log"

cd "${ROOT}"

echo "================================================================"
echo " TBSI Pipeline"
echo " Config: ${CONFIG}"
echo " Start: $(date)"
echo "================================================================"

# ===== Step 0: Kill old processes & clear GPU =====
echo "[$(date '+%H:%M:%S')] Cleaning up..." | tee -a "${PIPELINE_LOG}"
ps aux | grep -E "tracking/(train|test)\.py|run_training\.py" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 1
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# ===== Step 1: Training =====
echo "" | tee -a "${PIPELINE_LOG}"
echo ">>> PHASE 1: Training on ${DATASET}" | tee -a "${PIPELINE_LOG}"

unset OMP_NUM_THREADS

python tracking/train.py \
    --script "${SCRIPT}" \
    --config "${CONFIG}" \
    --save_dir ./output \
    --mode single 2>&1 | tee -a "${PIPELINE_LOG}"

echo "[$(date '+%H:%M:%S')] Training done" | tee -a "${PIPELINE_LOG}"

# ===== Step 2: Testing =====
echo "" | tee -a "${PIPELINE_LOG}"
echo ">>> PHASE 2: Testing on ${DATASET}" | tee -a "${PIPELINE_LOG}"

pkill -f "tracking/test.py" 2>/dev/null || true
sleep 2

python tracking/test.py \
    "${SCRIPT}" \
    "${CONFIG}" \
    --dataset_name "${DATASET}" \
    --threads 0 \
    --num_gpus 1 2>&1 | tee -a "${PIPELINE_LOG}"

echo "[$(date '+%H:%M:%S')] Testing done" | tee -a "${PIPELINE_LOG}"

# ===== Step 3: Analysis =====
echo "" | tee -a "${PIPELINE_LOG}"
echo ">>> PHASE 3: Results Analysis" | tee -a "${PIPELINE_LOG}"

python tracking/analysis_results.py \
    --tracker_name "${SCRIPT}" \
    --tracker_param "${CONFIG}" \
    --dataset_name "${DATASET}" 2>&1 | tee -a "${PIPELINE_LOG}"

echo "" | tee -a "${PIPELINE_LOG}"
echo "================================================================"
echo " ALL DONE  $(date)"
echo " Config: ${CONFIG}"
echo " Log: ${PIPELINE_LOG}"
echo "================================================================"
