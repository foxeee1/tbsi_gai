#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
CONTROL="M1_EARLY_GATE_VALUE_RECON_QUICK3_CONTROL_20261026"
CANDIDATE="M1_EARLY_GATE_VALUE_RECON_QUICK3_CANDIDATE_20261026"
TEST_LIST="$ROOT/experiments/manifests/m1_dcvr_quick7_legacy_20261025.txt"
PRETRAIN="$ROOT/pretrained_models/TBSITrack_SOT_Pretrained.pth.tar"
SNAP="$ROOT/experiments/snapshots/m1_early_gate_value_recon_quick3_20261026"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TBSI_INFER_FP16=0 TBSI_TEMPLATE_UPDATE_MODE=off
unset TBSI_MSHA_QUERY_CONDITIONED_CROSS_VALUE TBSI_MSHA_QUERY_CONDITIONED_DISABLE
unset TBSI_RECEIVER_CENTERED_VALUE_MODE TBSI_FCSV_R_MODE

cd "$ROOT"
test -x "$PYTHON" && test -s "$PRETRAIN" && test -s "$TEST_LIST"
test "$(wc -l < "$TEST_LIST")" -eq 7
if ps -eo args= | grep -E '[p]ython.*(run_training\.py|tracking/test\.py|tracking/train\.py)' >/dev/null; then
  echo "resource gate: an existing training or tracking-test process is active" >&2
  exit 75
fi
AVAILABLE=$(df -PB 1 "$ROOT" | awk 'NR==2 {print $4}')
if (( AVAILABLE < 5000000000 )); then
  echo "resource gate: less than 5 GB free before paired run" >&2
  exit 76
fi
test ! -e "$ROOT/output/experiments/$CONTROL"
test ! -e "$ROOT/output/experiments/$CANDIDATE"
mkdir -p "$SNAP"
{
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "git_head=$(git rev-parse HEAD)"
  sha256sum "$PRETRAIN" "$TEST_LIST"
  sha256sum "experiments/tbsi_track/${CONTROL}.yaml" "experiments/tbsi_track/${CANDIDATE}.yaml"
  sha256sum lib/models/layers/tbsi_layer.py lib/models/tbsi_track/tbsi_track.py lib/config/tbsi_track/config.py
  "$PYTHON" -c 'import sys,torch; print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA")'
  nvidia-smi -L || true
} > "$SNAP/provenance.txt" 2>&1

run_train() {
  local name="$1"
  local out="$ROOT/output/experiments/$name"
  mkdir -p "$out/logs"
  echo "train_started_utc=$(date -u +%FT%TZ) arm=$name" >> "$SNAP/provenance.txt"
  "$PYTHON" -u lib/train/run_training.py --script tbsi_track --config "$name" \
    --save_dir output --seed 42 --use_lmdb 0 > "$out/logs/train.log" 2>&1
  test -s "$out/checkpoints/TBSITrack_ep0003.pth.tar"
  sha256sum "$out/checkpoints/TBSITrack_ep0003.pth.tar" >> "$SNAP/provenance.txt"
}

run_train "$CONTROL"
run_train "$CANDIDATE"

run_test() {
  local name="$1"; local runid="$2"
  local checkpoint="$ROOT/output/experiments/$name/checkpoints/TBSITrack_ep0003.pth.tar"
  local results="$ROOT/output/test/tracking_results/tbsi_track/${name}_${runid}"
  test -s "$checkpoint" && test ! -e "$results"
  TBSI_RPP_TEST_LIST="$TEST_LIST" TBSI_CHECKPOINT_OVERRIDE="$checkpoint" \
    "$PYTHON" -u tracking/test.py tbsi_track "$name" --runid "$runid" \
      --dataset_name rpp_lasher_test --threads 0 --num_gpus 1 \
      > "$ROOT/output/experiments/$name/logs/test_quick7.log" 2>&1
  test "$(find "$results" -maxdepth 1 -name '*.txt' ! -name '*_time.txt' | wc -l)" -eq 7
}

run_test "$CONTROL" control_ep3
run_test "$CANDIDATE" candidate_ep3
echo "completed_utc=$(date -u +%FT%TZ)" >> "$SNAP/provenance.txt"
