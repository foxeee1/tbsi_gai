#!/bin/bash
set -e

cd /root/autodl-tmp/TBSI_gai/TBSI
unset OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

LOG_DIR="output/logs"
CONFIG="vitb_256_tbsi_sot_opt"
TS=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo " TBSI Paper Baseline (Optimized)"
echo " Config: ${CONFIG} (BS=32, LR=1e-4, 15ep)"
echo " SOT: clean original weights"
echo " Started: $(date)"
echo "=========================================="

echo ""
echo ">>> Phase 1: Training (15 epochs) <<<"
python3 -u tracking/train.py \
  --script tbsi_track \
  --config ${CONFIG} \
  --save_dir ./output \
  --mode single \
  2>&1 | tee "${LOG_DIR}/train_opt.log"
echo "[$(date)] Training done."

echo ""
echo ">>> Phase 2: Testing (244 seqs) <<<"
python3 -u tracking/test.py \
  tbsi_track ${CONFIG} \
  --dataset_name lasher_test \
  --threads 0 \
  --num_gpus 1 \
  2>&1 | tee "${LOG_DIR}/test_opt.log"
echo "[$(date)] Testing done."

echo ""
echo ">>> Phase 3: Analysis <<<"
python3 -u tracking/analysis_results.py \
  --tracker_name tbsi_track \
  --tracker_param ${CONFIG} \
  --dataset_name lasher_test \
  2>&1 | tee "${LOG_DIR}/analysis_opt.log"
echo "[$(date)] ALL DONE."
