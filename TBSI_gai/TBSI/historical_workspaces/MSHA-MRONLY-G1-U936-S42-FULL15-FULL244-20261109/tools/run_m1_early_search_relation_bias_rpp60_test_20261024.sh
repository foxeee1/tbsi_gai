#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
NAME="M1_EARLY_SEARCH_RELATION_BIAS_QUICK3_20261024"
RUNID="relation_bias_rpp60"
CHECKPOINT="$ROOT/output/experiments/$NAME/checkpoints/TBSITrack_ep0003.pth.tar"
SEQUENCES="$ROOT/output/experiments/$NAME/rpp_v2_60_manifest.txt"
OUT="$ROOT/output/experiments/$NAME"
RESULTS="$ROOT/output/test/tracking_results/tbsi_track/${NAME}_${RUNID}"
if [[ ! -s "$CHECKPOINT" || ! -s "$SEQUENCES" ]]; then echo "missing input" >&2; exit 2; fi
if [[ -e "$RESULTS" ]]; then echo "Refusing to reuse $RESULTS" >&2; exit 2; fi
mkdir -p "$OUT/logs"
cd "$ROOT"
{
  echo "test_started_utc=$(date -u +%FT%TZ)"
  echo "git_head=$(git rev-parse HEAD)"
  sha256sum "experiments/tbsi_track/${NAME}.yaml" "$CHECKPOINT" "$SEQUENCES"
  "$PYTHON" -c 'import sys,torch; print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA")'
  nvidia-smi -L || true
} > "$OUT/provenance_rpp60.txt" 2>&1
TBSI_RPP_TEST_LIST="$SEQUENCES" \
TBSI_CHECKPOINT_OVERRIDE="$CHECKPOINT" \
TBSI_TEMPLATE_UPDATE_MODE=off TBSI_INFER_FP16=0 TBSI_DISABLE_TRACKER_CACHE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  "$PYTHON" -u tracking/test.py tbsi_track "$NAME" --runid "$RUNID" \
  --dataset_name rpp_lasher_test --threads 0 --num_gpus 1 > "$OUT/logs/test_rpp60.log" 2>&1
count=$(find "$RESULTS" -maxdepth 1 -name '*.txt' ! -name '*_time.txt' | wc -l)
[[ "$count" -eq 60 ]] || { echo "Expected 60 bbox results, found $count" >&2; exit 3; }
echo "test_completed_utc=$(date -u +%FT%TZ)" >> "$OUT/provenance_rpp60.txt"
echo "completed: $count/60 sequences"
