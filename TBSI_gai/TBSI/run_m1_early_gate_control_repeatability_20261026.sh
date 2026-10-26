#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/TBSI_gai/TBSI

MANIFEST=experiments/manifests/m1_dcvr_quick7_legacy_20261025.txt
BASE=output/experiments

for ARM in A B; do
  if [[ "$ARM" == A ]]; then
    CONFIG=M1_EARLY_GATE_CONTROL_REPEAT_A_QUICK3_20261026
    RUN_ID=repeatability_A_quick7
  else
    CONFIG=M1_EARLY_GATE_CONTROL_REPEAT_B_QUICK3_20261026
    RUN_ID=repeatability_B_quick7
  fi
  OUT="$BASE/M1_EARLY_GATE_CONTROL_REPEAT_${ARM}_QUICK3_20261026"
  mkdir -p "$OUT/logs"
  test ! -e "$OUT/checkpoints/TBSITrack_ep0003.pth.tar"
  TBSI_EARLY_GATE_TOKEN_REL_MODE=off \
  TBSI_SAMPLE_TRACE_DIR="$OUT/sample_trace" \
    python -u lib/train/run_training.py --script tbsi_track --config "$CONFIG" \
      --save_dir output --seed 42 --use_lmdb 0 2>&1 | tee "$OUT/logs/console.log"
done

for ARM in A B; do
  if [[ "$ARM" == A ]]; then
    CONFIG=M1_EARLY_GATE_CONTROL_REPEAT_A_QUICK3_20261026
    RUN_ID=repeatability_A_quick7
  else
    CONFIG=M1_EARLY_GATE_CONTROL_REPEAT_B_QUICK3_20261026
    RUN_ID=repeatability_B_quick7
  fi
  CKPT="$BASE/M1_EARLY_GATE_CONTROL_REPEAT_${ARM}_QUICK3_20261026/checkpoints/TBSITrack_ep0003.pth.tar"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TBSI_INFER_FP16=0 TBSI_TEMPLATE_UPDATE_MODE=off \
  TBSI_EARLY_GATE_TOKEN_REL_MODE=off \
  TBSI_RPP_TEST_LIST="$MANIFEST" TBSI_CHECKPOINT_OVERRIDE="$CKPT" \
    python -u tracking/test.py tbsi_track "$CONFIG" --runid "$RUN_ID" \
      --dataset_name rpp_lasher_test --threads 1 --num_gpus 1 2>&1 \
      | tee "$BASE/M1_EARLY_GATE_CONTROL_REPEAT_${ARM}_QUICK3_20261026/logs/test_quick7.log"
done
