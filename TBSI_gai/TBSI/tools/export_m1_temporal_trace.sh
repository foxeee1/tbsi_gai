#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
OUT="${ROOT_DIR}/output/experiments/m1_temporal_trace_v1_20261006"
LIST="${OUT}/trace_sequences.txt"
CHECKPOINT="${ROOT_DIR}/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
BASELINE="${ROOT_DIR}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
mkdir -p "$OUT"
exec 9>"${OUT}/.trace.lock"
flock -n 9 || { echo "Another trace export is active" >&2; exit 75; }
cd "$ROOT_DIR"
[[ -x "$PYTHON_BIN" && -s "$CHECKPOINT" && -f "$BASELINE" ]] || exit 3
git diff --quiet -- "$BASELINE" || { echo "canonical baseline has a diff" >&2; exit 4; }
printf '%s\n' abreastinnerboy right2ndfarboytakinglight2left biketurndark > "$LIST"
export TBSI_TRACE_TRAIN_LIST="$LIST"
export TBSI_M1_TRACE_DIR="$OUT"
export TBSI_CHECKPOINT_OVERRIDE="$CHECKPOINT"
export TBSI_INFER_FP16=0
export TBSI_DISABLE_TRACKER_CACHE=1
export TBSI_ATTR_STATS=1
export TBSI_TRACE_SAVE_RAW=1
export PYTHONUNBUFFERED=1
exec > >(tee "${OUT}/export.log") 2>&1
echo "M1_TRACE_EXPORT_START $(date -u +%FT%TZ)"
echo "CHECKPOINT_SHA256 $(sha256sum "$CHECKPOINT" | awk '{print $1}')"
echo "BASELINE_SHA256 $(sha256sum "$BASELINE" | awk '{print $1}')"
"$PYTHON_BIN" tracking/test.py tbsi_track G0_CAUSAL_TAM_PILOT_seed42 \
  --dataset_name m1_trace_lasher_train --threads 0 --num_gpus 1
cat > "$OUT/manifest.json" <<EOF
{"schema_version":2,"kind":"frozen_m1_temporal_trace","checkpoint":"$CHECKPOINT","sequence_list":"$LIST","tam_enabled":true,"gamma_initial":0.0,"fp16":false,"raw_head_tokens":"float32_npy_per_frame"}
EOF
echo "M1_TRACE_EXPORT_FINISH $(date -u +%FT%TZ)"
