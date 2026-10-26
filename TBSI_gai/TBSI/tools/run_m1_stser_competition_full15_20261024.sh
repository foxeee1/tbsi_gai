#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
NAME="M1_STSER_COMPETITION_FULL15_FROM_M1_20261024"
CONFIG="experiments/tbsi_track/${NAME}.yaml"
OUT="${ROOT}/output/experiments/${NAME}"
SNAP="${ROOT}/experiments/snapshots/m1_stser_competition_full15_20261024"
CKPT="${OUT}/checkpoints/TBSITrack_ep0015.pth.tar"
RESULTS="${ROOT}/output/test/tracking_results/tbsi_track/${NAME}"
M1="${ROOT}/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export TBSI_INFER_FP16=0 TBSI_TEMPLATE_UPDATE_MODE=off TBSI_DISABLE_TRACKER_CACHE=1
cd "$ROOT"
test -x "$PYTHON"; test -s "$M1"; test -f "$CONFIG"
test ! -e "$OUT"; test ! -e "$RESULTS"
if pgrep -af '[l]ib/train/run_training.py|[t]racking/(train|test).py' >/tmp/tbsi_full_processes.$$ 2>/dev/null; then
  cat /tmp/tbsi_full_processes.$$; rm -f /tmp/tbsi_full_processes.$$; exit 75
fi
rm -f /tmp/tbsi_full_processes.$$
mkdir -p "$OUT/logs" "$SNAP"
{
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "name=$NAME"
  echo "git_head=$(git rev-parse HEAD)"
  sha256sum "$M1" "$CONFIG" lib/models/layers/search_conditioned_t2s_router.py
  "$PYTHON" -c 'import sys,torch; print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA")'
  nvidia-smi -L || true
  echo "objective=full LasHeR training/test of STSER competition descriptor from protected M1"
  echo "flow=search tokens -> non-neighbor top-k competition pool -> query-minus-competitor router -> shared/private/null T2S message -> CENTER head"
  echo "optimization=ADAMW; LR=1e-4; backbone_multiplier=0.1; GIOU=2.0; L1=5.0; AMP=True; effective_batch=128"
  echo "instrumentation=trainer gradient_flow, feature_changes, internal_signals, loss_components, amcm_gradient_flow"
} > "$SNAP/provenance.txt"
"$PYTHON" -u lib/train/run_training.py --script tbsi_track --config "$NAME" --save_dir output --seed 42 --use_lmdb 0 > "$OUT/logs/train.log" 2>&1
test -s "$CKPT"
sha256sum "$CKPT" >> "$SNAP/provenance.txt"
"$PYTHON" -u tracking/test.py tbsi_track "$NAME" --dataset_name lasher_test --threads 0 --num_gpus 1 > "$OUT/logs/test_lasher244.log" 2>&1
test -d "$RESULTS"
"$PYTHON" -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param "$NAME" --dataset_name lasher_test > "$OUT/logs/analysis_lasher244.log" 2>&1
echo "completed_utc=$(date -u +%FT%TZ)" >> "$SNAP/provenance.txt"
