#!/bin/bash
# =====================================================================
# TBSI Paper Baseline: Original BS=32 config
# Config: vitb_256_tbsi_32x1_1e4_lasher_15ep_sot (SOT pretrained)
# =====================================================================

set -e
cd /root/autodl-tmp/TBSI_gai/TBSI
unset OMP_NUM_THREADS

LOG_DIR="output/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONFIG="vitb_256_tbsi_32x1_1e4_lasher_15ep_sot"

echo "========================================"
echo " TBSI Paper Baseline Pipeline"
echo " Config: ${CONFIG} (BS=32, LR=1e-4, 15ep)"
echo " Started: $(date)"
echo "========================================"

# ===== Phase 1: Training =====
echo ""
echo "[Phase 1] Training..."
echo "----------------------------------------"

PYTHONUNBUFFERED=1 python tracking/train.py \
  --script tbsi_track \
  --config ${CONFIG} \
  --save_dir ./output \
  --mode single \
  2>&1 | tee "${LOG_DIR}/train_${CONFIG}.log"

echo "[$(date)] Training completed."

# ===== Phase 2: Testing =====
echo ""
echo "[Phase 2] Testing..."
echo "----------------------------------------"

PYTHONUNBUFFERED=1 python tracking/test.py \
  tbsi_track ${CONFIG} \
  --dataset_name lasher_test \
  --threads 0 \
  --num_gpus 1 \
  2>&1 | tee "${LOG_DIR}/test_${CONFIG}.log"

echo "[$(date)] Testing completed."

# ===== Phase 3: Analysis =====
echo ""
echo "[Phase 3] Analysis..."
echo "----------------------------------------"

PYTHONUNBUFFERED=1 python tracking/analysis_results.py \
  --tracker_name tbsi_track \
  --tracker_param ${CONFIG} \
  --dataset_name lasher_test \
  2>&1 | tee "${LOG_DIR}/analysis_${CONFIG}.log"

echo ""
echo "========================================"
echo " ALL DONE at $(date)"
echo " Results saved to:"
echo "   ${LOG_DIR}/train_${CONFIG}.log"
echo "   ${LOG_DIR}/test_${CONFIG}.log"
echo "   ${LOG_DIR}/analysis_${CONFIG}.log"
echo "========================================"
