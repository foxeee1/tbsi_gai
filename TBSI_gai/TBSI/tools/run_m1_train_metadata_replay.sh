#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
LIST="$ROOT_DIR/experiments/proxy/lasher_rpp_v1/g0_ffm_20261005/ffm_train_sequences.txt"
OUT="$ROOT_DIR/output/experiments/m1_train_metadata_replay_v1_20261007"
TRACE="$OUT/trace"
CHECKPOINT="$ROOT_DIR/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
BASELINE="$ROOT_DIR/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
mkdir -p "$OUT"
exec 9>"$OUT/.lock"; flock -n 9 || { echo 'duplicate metadata replay' >&2; exit 75; }
cd "$ROOT_DIR"
test -x "$PYTHON_BIN"; test -s "$LIST"; test -s "$CHECKPOINT"
git diff --quiet -- "$BASELINE" || { echo 'canonical baseline has a diff' >&2; exit 4; }
export TBSI_TRACE_TRAIN_LIST="$LIST"
export TBSI_M1_TRACE_DIR="$TRACE"
export TBSI_CHECKPOINT_OVERRIDE="$CHECKPOINT"
export TBSI_INFER_FP16=0
export TBSI_DISABLE_TRACKER_CACHE=1
export TBSI_TRACE_SAVE_RAW=0
export PYTHONUNBUFFERED=1
exec > >(tee "$OUT/replay.log") 2>&1
echo "M1_METADATA_REPLAY_START $(date -u +%FT%TZ)"
echo "SEQUENCE_COUNT $(wc -l < "$LIST")"
echo "CHECKPOINT_SHA256 $(sha256sum "$CHECKPOINT" | awk '{print $1}')"
"$PYTHON_BIN" tracking/test.py tbsi_track G0_CAUSAL_TAM_PILOT_seed42 \
  --dataset_name m1_trace_lasher_train --threads 0 --num_gpus 1
echo "M1_METADATA_REPLAY_FINISH $(date -u +%FT%TZ)"
