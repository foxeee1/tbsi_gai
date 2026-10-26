#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
CONTROL="M1_QUERY_CONDITIONED_CROSS_VALUE_CONTROL_QUICK3_20261024"
CANDIDATE="M1_QUERY_CONDITIONED_CROSS_VALUE_QUICK3_20261024"
TEST_LIST="$ROOT/experiments/protocols/m1_query_conditioned_cross_value_quick12_20261024.txt"
export TBSI_INFER_FP16=0 TBSI_TEMPLATE_UPDATE_MODE=off TBSI_DISABLE_TRACKER_CACHE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
test -s "$ROOT/output/experiments/$CONTROL/checkpoints/TBSITrack_ep0003.pth.tar"
test -s "$ROOT/output/experiments/$CANDIDATE/checkpoints/TBSITrack_ep0003.pth.tar"
test "$(wc -l < "$TEST_LIST")" -eq 12
if ps -eo args= | grep -E '[p]ython.*(run_training\.py|tracking/test\.py|tracking/train\.py)' >/dev/null; then
  echo 'resource gate: existing training/test process is active' >&2; exit 75
fi
cd "$ROOT"
run_test() {
  local name="$1"; local gate="$2"; local runid="$3"
  local ckpt="$ROOT/output/experiments/$name/checkpoints/TBSITrack_ep0003.pth.tar"
  local result="$ROOT/output/test/tracking_results/tbsi_track/${name}_${runid}"
  test ! -e "$result"
  if [[ "$gate" == 1 ]]; then export TBSI_MSHA_QUERY_CONDITIONED_CROSS_VALUE=1; else unset TBSI_MSHA_QUERY_CONDITIONED_CROSS_VALUE || true; fi
  TBSI_RPP_TEST_LIST="$TEST_LIST" TBSI_CHECKPOINT_OVERRIDE="$ckpt" \
    "$PYTHON" -u tracking/test.py tbsi_track "$name" --runid "$runid" --dataset_name rpp_lasher_test --threads 0 --num_gpus 1 \
    > "$ROOT/output/experiments/$name/test_quick12.log" 2>&1
  test "$(find "$result" -maxdepth 1 -name '*.txt' ! -name '*_time.txt' | wc -l)" -eq 12
}
run_test "$CONTROL" 0 control_quick12
run_test "$CANDIDATE" 1 candidate_quick12
