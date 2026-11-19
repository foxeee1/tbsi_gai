#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
TRACE="$ROOT_DIR/output/experiments/m1_temporal_trace_v1_20261006"
OUT="$ROOT_DIR/output/experiments/tam_online_replay_smoke_v1_20261007"
STEPS="${TAM_IMAGE_REPLAY_STEPS:-6}"
MANIFEST="$OUT/m1_crop_replay_manifest.json"
CHECKPOINT="$ROOT_DIR/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
CONFIG="$ROOT_DIR/experiments/tbsi_track/G0_CAUSAL_TAM_PILOT_seed42.yaml"
BASELINE="$ROOT_DIR/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
mkdir -p "$OUT"
exec 9>"$OUT/.lock"; flock -n 9 || { echo 'duplicate image replay smoke' >&2; exit 75; }
cd "$ROOT_DIR"
git diff --quiet -- "$BASELINE" || { echo 'canonical baseline has a diff' >&2; exit 4; }
test -x "$PYTHON_BIN"; test -s "$CHECKPOINT"; test -f "$TRACE/manifest.json"
exec > >(tee "$OUT/run.log") 2>&1
echo "TAM_IMAGE_REPLAY_SMOKE_START $(date -u +%FT%TZ)"
"$PYTHON_BIN" tools/generate_m1_crop_replay_manifest.py --trace-dir "$TRACE" --dataset-root "$ROOT_DIR/../data/lasher/trainingset" --checkpoint "$CHECKPOINT" --output "$MANIFEST"
"$PYTHON_BIN" tools/run_causal_tam_image_replay_smoke.py --manifest "$MANIFEST" --config "$CONFIG" --checkpoint "$CHECKPOINT" --output-dir "$OUT" --steps "$STEPS" --seed 42
echo "TAM_IMAGE_REPLAY_SMOKE_FINISH $(date -u +%FT%TZ)"
