#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/root/autodl-tmp/TBSI_gai/TBSI"; PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
NAME="M1_QUERY_CONDITIONED_CROSS_VALUE_FROZEN_CONTROL_QUICK3_20261024"
TEST_LIST="$ROOT/experiments/protocols/m1_query_conditioned_cross_value_quick12_20261024.txt"
OUT="$ROOT/output/experiments/$NAME"
export TBSI_MSHA_QUERY_CONDITIONED_CROSS_VALUE=1 TBSI_MSHA_QUERY_CONDITIONED_DISABLE=1
export TBSI_INFER_FP16=0 TBSI_TEMPLATE_UPDATE_MODE=off TBSI_DISABLE_TRACKER_CACHE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
test ! -e "$OUT/checkpoints/TBSITrack_ep0003.pth.tar"; mkdir -p "$OUT"
cd "$ROOT"
"$PYTHON" -u lib/train/run_training.py --script tbsi_track --config "$NAME" --save_dir output --seed 42 --use_lmdb 0 > "$OUT/train.log" 2>&1
test -s "$OUT/checkpoints/TBSITrack_ep0003.pth.tar"
unset TBSI_MSHA_QUERY_CONDITIONED_DISABLE
test ! -e "$ROOT/output/test/tracking_results/tbsi_track/${NAME}_control_quick12"
TBSI_RPP_TEST_LIST="$TEST_LIST" TBSI_CHECKPOINT_OVERRIDE="$OUT/checkpoints/TBSITrack_ep0003.pth.tar" \
  "$PYTHON" -u tracking/test.py tbsi_track "$NAME" --runid control_quick12 --dataset_name rpp_lasher_test --threads 0 --num_gpus 1 \
  > "$OUT/test_quick12.log" 2>&1
test "$(find "$ROOT/output/test/tracking_results/tbsi_track/${NAME}_control_quick12" -maxdepth 1 -name '*.txt' ! -name '*_time.txt' | wc -l)" -eq 12
