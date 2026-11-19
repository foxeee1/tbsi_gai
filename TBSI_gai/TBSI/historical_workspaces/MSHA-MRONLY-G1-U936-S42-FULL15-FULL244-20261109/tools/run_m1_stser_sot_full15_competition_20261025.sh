#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
NAME="M1_STSER_SOT_FULL15_COMPETITION_20261025"
CONFIG="experiments/tbsi_track/${NAME}.yaml"
OUT="${ROOT}/output/experiments/${NAME}"
SNAP="${ROOT}/experiments/snapshots/m1_stser_sot_full15_competition_20261025"
PRETRAIN="${ROOT}/pretrained_models/TBSITrack_SOT_Pretrained.pth.tar"
CKPT="${OUT}/checkpoints/TBSITrack_ep0015.pth.tar"
RESULTS="${ROOT}/output/test/tracking_results/tbsi_track/${NAME}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export TBSI_INFER_FP16=0 TBSI_TEMPLATE_UPDATE_MODE=off TBSI_DISABLE_TRACKER_CACHE=1
cd "$ROOT"
test -x "$PYTHON"
test -s "$PRETRAIN"
test -f "$CONFIG"
test ! -e "$OUT"
test ! -e "$RESULTS"
if ps -eo args= | grep -E '[p]ython.*(run_training\.py|tracking/test\.py|tracking/train\.py)' >/tmp/tbsi_stser_full_processes.$$ 2>/dev/null; then
  cat /tmp/tbsi_stser_full_processes.$$ >&2
  rm -f /tmp/tbsi_stser_full_processes.$$
  exit 75
fi
rm -f /tmp/tbsi_stser_full_processes.$$
mkdir -p "$OUT/logs" "$SNAP"
{
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "name=$NAME"
  echo "git_head=$(git rev-parse HEAD)"
  echo "init_checkpoint=NONE"
  echo "pretrain=$(sha256sum "$PRETRAIN")"
  sha256sum "$CONFIG" lib/models/layers/search_conditioned_t2s_router.py lib/train/trainers/diagnostics.py
  "$PYTHON" -c 'import sys,torch; print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA")'
  nvidia-smi -L || true
  echo "objective=full LasHeR SOT-initialized STSER competition versus protected M1"
  echo "flow=search tokens -> non-neighbor top-k competition pool -> query-minus-competitor router -> shared/private/null T2S message -> CENTER head"
  echo "optimization=fresh ADAMW and StepLR; 15 epochs; effective batch 128; no M1 optimizer/checkpoint state"
  echo "evaluation=LasHeR test full244; FP32; threads=0; template update off"
} > "$SNAP/provenance.txt"

echo "train_started_utc=$(date -u +%FT%TZ)" >> "$SNAP/provenance.txt"
"$PYTHON" -u lib/train/run_training.py --script tbsi_track --config "$NAME" --save_dir output --seed 42 --use_lmdb 0 > "$OUT/logs/train.log" 2>&1
test -s "$CKPT"
sha256sum "$CKPT" >> "$SNAP/provenance.txt"
echo "train_completed_utc=$(date -u +%FT%TZ)" >> "$SNAP/provenance.txt"

echo "test_started_utc=$(date -u +%FT%TZ)" >> "$SNAP/provenance.txt"
"$PYTHON" -u tracking/test.py tbsi_track "$NAME" --dataset_name lasher_test --threads 0 --num_gpus 1 > "$OUT/logs/test_lasher244.log" 2>&1
test -d "$RESULTS"
echo "test_completed_utc=$(date -u +%FT%TZ)" >> "$SNAP/provenance.txt"
"$PYTHON" -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param "$NAME" --dataset_name lasher_test > "$OUT/logs/analysis_lasher244.log" 2>&1
echo "analysis_completed_utc=$(date -u +%FT%TZ)" >> "$SNAP/provenance.txt"
echo "experiment_completed_utc=$(date -u +%FT%TZ)" >> "$SNAP/provenance.txt"
