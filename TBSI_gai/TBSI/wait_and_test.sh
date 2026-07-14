#!/bin/bash
# Wait for training to complete, then run test + analysis
# Usage: bash wait_and_test.sh

PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
BASE_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
LOG_DIR="${BASE_DIR}/output/logs"
CONFIG="vitb_256_tbsi_32x1_1e4_lasher_15ep_sot"
CKPT_DIR="${BASE_DIR}/output/checkpoints/train/tbsi_track/${CONFIG}"

cd "${BASE_DIR}"
unset OMP_NUM_THREADS

echo "Waiting for training to complete..."
echo "Checkpoint dir: ${CKPT_DIR}"
echo ""

# Poll every 5 minutes
while true; do
    if [ -f "${CKPT_DIR}/TBSITrack_ep0015.pth.tar" ]; then
        echo "[$(date)] Checkpoint found! Training completed."
        break
    fi

    # Check if training process is still alive
    if ! pgrep -f "run_training.*${CONFIG}" > /dev/null; then
        echo "[$(date)] Training process not found. Checking checkpoint anyway..."
        if [ -f "${CKPT_DIR}/TBSITrack_ep0015.pth.tar" ]; then
            break
        fi
        echo "[$(date)] No checkpoint found. Training may have failed. Waiting..."
    fi

    sleep 300  # 5 minutes
done

echo ""
echo ">>> Phase 2: Testing (244 sequences)"
echo ""
${PYTHON} -u tracking/test.py \
  tbsi_track ${CONFIG} \
  --dataset_name lasher_test \
  --threads 0 --num_gpus 1 \
  2>&1 | tee "${LOG_DIR}/test_paper_ep0015.log"

echo "[$(date)] Testing completed."

echo ""
echo ">>> Phase 3: Analysis"
echo ""
${PYTHON} -u tracking/analysis_results.py \
  --tracker_name tbsi_track \
  --tracker_param ${CONFIG} \
  --dataset_name lasher_test \
  2>&1 | tee "${LOG_DIR}/analysis_paper_ep0015.log"

echo ""
echo "========================================"
echo " ALL DONE at $(date)"
grep -E "AUC|Precision|Norm" "${LOG_DIR}/analysis_paper_ep0015.log" 2>/dev/null
echo "========================================"
