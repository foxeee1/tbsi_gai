#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
CONTROL="M1_EARLY_GATE_TOKEN_REL_QUICK3_CONTROL_RETRY1_20261026"
CANDIDATE="M1_EARLY_GATE_TOKEN_REL_QUICK3_CANDIDATE_RETRY1_20261026"
TEST_LIST="$ROOT/experiments/manifests/m1_dcvr_quick7_legacy_20261025.txt"
PRETRAIN="$ROOT/pretrained_models/TBSITrack_SOT_Pretrained.pth.tar"
SNAP="$ROOT/experiments/snapshots/m1_early_gate_token_reliability_quick3_20261026"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TBSI_DCVR_MODE=off TBSI_DCVR_ALPHA=0.25
export TBSI_EARLY_GATE_TOKEN_REL_MODE=off
export TBSI_INFER_FP16=0 TBSI_TEMPLATE_UPDATE_MODE=off
unset TBSI_MSHA_QUERY_CONDITIONED_CROSS_VALUE TBSI_MSHA_QUERY_CONDITIONED_DISABLE
unset TBSI_NATIVE_MSHA_MESSAGE_MODE TBSI_NATIVE_MSHA_FIXED_R2T TBSI_NATIVE_MSHA_FIXED_T2R
unset TBSI_DCVR_DIAGNOSTICS_PATH TBSI_MSHA_FG_CROSS_MODE
unset TBSI_T2S_CONDITIONED_MESSAGE_MODE TBSI_T2S_CONDITIONED_MESSAGE_ALPHA
unset TBSI_TEST_EARLY_GATE_STRENGTH TBSI_RPP_TRAIN_LIST

cd "$ROOT"
test -x "$PYTHON" && test -s "$PRETRAIN" && test -s "$TEST_LIST"
test "$(wc -l < "$TEST_LIST")" -eq 7
test ! -e "$ROOT/output/experiments/$CONTROL"
test ! -e "$ROOT/output/experiments/$CANDIDATE"
test ! -e "$ROOT/output/test/tracking_results/tbsi_track/${CONTROL}_early_gate_sot_ep3"
test ! -e "$ROOT/output/test/tracking_results/tbsi_track/${CANDIDATE}_reliability_sot_ep3"
if ps -eo args= | grep -E '[p]ython.*(run_training\.py|tracking/test\.py|tracking/train\.py)' >/dev/null; then
  echo "resource gate: an existing training or tracking-test process is active" >&2
  exit 75
fi
AVAILABLE=$(df -PB 1 "$ROOT" | awk 'NR==2 {print $4}')
if (( AVAILABLE < 5000000000 )); then
  echo "resource gate: less than 5 GB free before paired run" >&2
  exit 76
fi

mkdir -p "$SNAP"
{
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_worktree_status_begin"
  git status --short
  echo "git_worktree_status_end"
  sha256sum "$PRETRAIN" "$TEST_LIST"
  sha256sum "experiments/tbsi_track/${CONTROL}.yaml" "experiments/tbsi_track/${CANDIDATE}.yaml"
  sha256sum lib/models/layers/tbsi_layer.py
  sha256sum experiments/snapshots/m1_early_gate_token_reliability_quick3_20261026/prechange_tbsi_layer.patch
  "$PYTHON" -c 'import sys,torch; print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA")'
  nvidia-smi -L || true
  df -h "$ROOT"
} > "$SNAP/provenance.txt" 2>&1

run_train() {
  local name="$1"
  local rel_mode="$2"
  local out="$ROOT/output/experiments/$name"
  mkdir -p "$out/logs"
  echo "train_started_utc=$(date -u +%FT%TZ) arm=$name token_rel_mode=$rel_mode" >> "$SNAP/provenance.txt"
  TBSI_EARLY_GATE_TOKEN_REL_MODE="$rel_mode" \
  TBSI_EARLY_GATE_TOKEN_REL_DIAGNOSTICS_PATH="$out/logs/token_reliability_train.jsonl" \
  TBSI_EARLY_GATE_TOKEN_REL_DIAGNOSTICS_STRIDE=16 \
    "$PYTHON" -u lib/train/run_training.py --script tbsi_track --config "$name" \
      --save_dir output --seed 42 --use_lmdb 0 > "$out/logs/train.log" 2>&1
  test -s "$out/checkpoints/TBSITrack_ep0003.pth.tar"
  sha256sum "$out/checkpoints/TBSITrack_ep0003.pth.tar" >> "$SNAP/provenance.txt"
  echo "train_completed_utc=$(date -u +%FT%TZ) arm=$name" >> "$SNAP/provenance.txt"
}

run_test() {
  local name="$1"
  local runid="$2"
  local rel_mode="$3"
  local checkpoint="$ROOT/output/experiments/$name/checkpoints/TBSITrack_ep0003.pth.tar"
  local results="$ROOT/output/test/tracking_results/tbsi_track/${name}_${runid}"
  test -s "$checkpoint" && test ! -e "$results"
  echo "test_started_utc=$(date -u +%FT%TZ) arm=$name token_rel_mode=$rel_mode" >> "$SNAP/provenance.txt"
  TBSI_RPP_TEST_LIST="$TEST_LIST" TBSI_CHECKPOINT_OVERRIDE="$checkpoint" \
  TBSI_EARLY_GATE_TOKEN_REL_MODE="$rel_mode" \
  TBSI_EARLY_GATE_TOKEN_REL_DIAGNOSTICS_PATH="$ROOT/output/experiments/$name/logs/token_reliability_test.jsonl" \
  TBSI_EARLY_GATE_TOKEN_REL_DIAGNOSTICS_STRIDE=64 \
    "$PYTHON" -u tracking/test.py tbsi_track "$name" --runid "$runid" \
      --dataset_name rpp_lasher_test --threads 0 --num_gpus 1 \
      > "$ROOT/output/experiments/$name/logs/test_rpp7.log" 2>&1
  test "$(find "$results" -maxdepth 1 -name '*.txt' ! -name '*_time.txt' | wc -l)" -eq 7
  echo "test_completed_utc=$(date -u +%FT%TZ) arm=$name" >> "$SNAP/provenance.txt"
}

run_train "$CONTROL" off
run_train "$CANDIDATE" on
run_test "$CONTROL" early_gate_sot_ep3 off
run_test "$CANDIDATE" reliability_sot_ep3 on
TBSI_RPP_TEST_LIST="$TEST_LIST" \
  "$PYTHON" tools/analyze_m1_early_gate_token_rel_quick3_20261026.py \
    --sequence-list "$TEST_LIST" --output-dir "$SNAP/analysis_retry1" \
    > "$SNAP/paired_analysis_retry1.json" 2>&1
echo "experiment_completed_utc=$(date -u +%FT%TZ)" >> "$SNAP/provenance.txt"
