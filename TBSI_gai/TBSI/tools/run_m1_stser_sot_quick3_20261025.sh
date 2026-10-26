#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
TRAIN_LIST="$ROOT/experiments/protocols/m1_screening_v1_20261022/quick_train_sequences.txt"
TEST_LIST="$ROOT/output/experiments/M1_EARLY_SEARCH_RELATION_BIAS_QUICK3_20261024/rpp_v2_60_manifest.txt"
PRETRAIN="$ROOT/pretrained_models/TBSITrack_SOT_Pretrained.pth.tar"
SNAP="$ROOT/experiments/snapshots/m1_stser_sot_quick_20261025"
CONTROL="M1_STSER_SOT_QUICK3_PLAIN_20261025"
CANDIDATE="M1_STSER_SOT_QUICK3_COMPETITION_20261025"

export TBSI_RPP_TRAIN_LIST="$TRAIN_LIST"
export TBSI_INFER_FP16=0 TBSI_TEMPLATE_UPDATE_MODE=off TBSI_DISABLE_TRACKER_CACHE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

test -x "$PYTHON" && test -s "$PRETRAIN" && test -s "$TRAIN_LIST" && test -s "$TEST_LIST"
test "$(wc -l < "$TRAIN_LIST")" -eq 176
test "$(wc -l < "$TEST_LIST")" -eq 60
if ps -eo args= | grep -E '[p]ython.*(run_training\.py|tracking/test\.py|tracking/train\.py)' >/dev/null; then
  echo "resource gate: an existing training or tracking-test process is active" >&2
  exit 75
fi
if [[ -e "$ROOT/output/experiments/$CONTROL" || -e "$ROOT/output/experiments/$CANDIDATE" ]]; then
  echo "refusing to reuse an existing run directory" >&2
  exit 2
fi
AVAILABLE=$(df -PB 1 "$ROOT" | awk 'NR==2 {print $4}')
if (( AVAILABLE < 5000000000 )); then
  echo "resource gate: less than 5 GB free before two-arm run" >&2
  exit 76
fi

cd "$ROOT"
mkdir -p "$SNAP"
{
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_status_begin"
  git status --short
  echo "git_status_end"
  sha256sum "$PRETRAIN" "$TRAIN_LIST" "$TEST_LIST"
  sha256sum "experiments/tbsi_track/${CONTROL}.yaml" "experiments/tbsi_track/${CANDIDATE}.yaml"
  sha256sum lib/models/layers/search_conditioned_t2s_router.py lib/train/trainers/diagnostics.py
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
  echo "train_completed_utc=$(date -u +%FT%TZ) arm=$name" >> "$SNAP/provenance.txt"
}

run_test() {
  local name="$1"
  local runid="$2"
  local checkpoint="$ROOT/output/experiments/$name/checkpoints/TBSITrack_ep0003.pth.tar"
  local results="$ROOT/output/test/tracking_results/tbsi_track/${name}_${runid}"
  test -s "$checkpoint"
  test ! -e "$results"
  echo "test_started_utc=$(date -u +%FT%TZ) arm=$name" >> "$SNAP/provenance.txt"
  TBSI_RPP_TEST_LIST="$TEST_LIST" TBSI_CHECKPOINT_OVERRIDE="$checkpoint" \
    "$PYTHON" -u tracking/test.py tbsi_track "$name" --runid "$runid" \
    --dataset_name rpp_lasher_test --threads 0 --num_gpus 1 \
    > "$ROOT/output/experiments/$name/logs/test_rpp60.log" 2>&1
  test "$(find "$results" -maxdepth 1 -name '*.txt' ! -name '*_time.txt' | wc -l)" -eq 60
  echo "test_completed_utc=$(date -u +%FT%TZ) arm=$name" >> "$SNAP/provenance.txt"
}

run_train "$CONTROL"
run_train "$CANDIDATE"
run_test "$CONTROL" plain_sot_ep3
run_test "$CANDIDATE" competition_sot_ep3
"$PYTHON" tools/analyze_m1_stser_sot_quick3_20261025.py | tee "$SNAP/paired_summary.stdout.json"
echo "experiment_completed_utc=$(date -u +%FT%TZ)" >> "$SNAP/provenance.txt"
