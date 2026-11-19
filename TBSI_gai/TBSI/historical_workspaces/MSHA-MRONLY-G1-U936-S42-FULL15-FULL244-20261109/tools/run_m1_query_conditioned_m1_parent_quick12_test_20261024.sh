#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/root/autodl-tmp/TBSI_gai/TBSI"; PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
TEST_LIST="$ROOT/experiments/protocols/m1_query_conditioned_cross_value_quick12_20261024.txt"
CKPT="$ROOT/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
RESULT="$ROOT/output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_query12_parent20261024"
export TBSI_RPP_TEST_LIST="$TEST_LIST" TBSI_CHECKPOINT_OVERRIDE="$CKPT"
export TBSI_INFER_FP16=0 TBSI_TEMPLATE_UPDATE_MODE=off TBSI_DISABLE_TRACKER_CACHE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset TBSI_MSHA_QUERY_CONDITIONED_CROSS_VALUE TBSI_MSHA_QUERY_CONDITIONED_DISABLE || true
test -s "$CKPT"; test ! -e "$RESULT"; test "$(wc -l < "$TEST_LIST")" -eq 12
if ps -eo args= | grep -E '[p]ython.*(run_training\.py|tracking/test\.py|tracking/train\.py)' >/dev/null; then exit 75; fi
cd "$ROOT"
"$PYTHON" -u tracking/test.py tbsi_track TBSI_E_MHSA_FULL_4090D_seed42 --runid query12_parent20261024 --dataset_name rpp_lasher_test --threads 0 --num_gpus 1 \
  > "$ROOT/output/experiments/M1_QUERY_CONDITIONED_CROSS_VALUE_QUICK3_20261024/parent_quick12.log" 2>&1
test "$(find "$RESULT" -maxdepth 1 -name '*.txt' ! -name '*_time.txt' | wc -l)" -eq 12
