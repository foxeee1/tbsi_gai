#!/bin/bash
# TBSI Paper Baseline - Environment Aligned
# PyTorch 1.9.0+cu111, timm 0.5.4 (matches paper environment)
# Config: vitb_256_tbsi_32x1_1e4_lasher_15ep_sot (BS=32, LR=1e-4, 15ep)

PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
BASE_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
LOG_DIR="${BASE_DIR}/output/logs"
CONFIG="vitb_256_tbsi_32x1_1e4_lasher_15ep_sot"
TS=$(date +"%Y%m%d_%H%M%S")

cd "${BASE_DIR}"
unset OMP_NUM_THREADS

echo "=========================================="
echo " TBSI Paper Baseline Pipeline"
echo " Config: ${CONFIG}"
echo " Python: $(${PYTHON} --version 2>&1)"
echo " PyTorch: $(${PYTHON} -c "import torch; print(torch.__version__)" 2>&1)"
echo " timm: $(${PYTHON} -c "import timm; print(timm.__version__)" 2>&1)"
echo " Started: $(date)"
echo "=========================================="

echo ""
echo ">>> Phase 1: Training (15 epochs)"
echo ""
${PYTHON} -u tracking/train.py \
  --script tbsi_track \
  --config ${CONFIG} \
  --save_dir ./output \
  --mode single
echo "[$(date)] Training done."

echo ""
echo ">>> Phase 2: Testing (244 sequences)"
echo ""
${PYTHON} -u tracking/test.py \
  tbsi_track ${CONFIG} \
  --dataset_name lasher_test \
  --threads 0 \
  --num_gpus 1
echo "[$(date)] Testing done."

echo ""
echo ">>> Phase 3: Analysis"
echo ""
${PYTHON} -u tracking/analysis_results.py \
  --tracker_name tbsi_track \
  --tracker_param ${CONFIG} \
  --dataset_name lasher_test
echo "[$(date)] ALL DONE."
