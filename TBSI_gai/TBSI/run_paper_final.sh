#!/bin/bash
# TBSI Paper Baseline - Final Run
# PyTorch 2.0 + timm 1.0.27 (verified working)
# vitb_256_tbsi_32x1_1e4_lasher_15ep_sot (BS=32, LR=1e-4)
# SOT weights: newly downloaded original

cd /root/autodl-tmp/TBSI_gai/TBSI
unset OMP_NUM_THREADS
LOG_DIR="output/logs"
CONFIG="vitb_256_tbsi_32x1_1e4_lasher_15ep_sot"

echo "=== TBSI Paper Baseline ==="
echo "Config: ${CONFIG} (BS=32, LR=1e-4, 15ep)"
python3 -c "import torch,timm; print(f'PyTorch: {torch.__version__}, timm: {timm.__version__}')"
echo "Started: $(date)"

echo ""
echo ">>> Phase 1: Training (15 epochs) <<<"
python3 -u tracking/train.py \
  --script tbsi_track \
  --config ${CONFIG} \
  --save_dir ./output \
  --mode single \
  2>&1 | tee "${LOG_DIR}/train_final.log"
echo "[$(date)] Training done."

echo ""
echo ">>> Phase 2: Testing (244 seqs) <<<"
python3 -u tracking/test.py \
  tbsi_track ${CONFIG} \
  --dataset_name lasher_test \
  --threads 0 \
  --num_gpus 1 \
  2>&1 | tee "${LOG_DIR}/test_final.log"
echo "[$(date)] Testing done."

echo ""
echo ">>> Phase 3: Analysis <<<"
python3 -u tracking/analysis_results.py \
  --tracker_name tbsi_track \
  --tracker_param ${CONFIG} \
  --dataset_name lasher_test \
  2>&1 | tee "${LOG_DIR}/analysis_final.log"
echo "[$(date)] ALL DONE."
