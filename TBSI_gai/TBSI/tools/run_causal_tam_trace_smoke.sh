#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
OUT="${ROOT_DIR}/output/experiments/tam_trace_smoke_v1_20261007"
TRACE="${ROOT_DIR}/output/experiments/m1_temporal_trace_v1_20261006"
BASELINE="${ROOT_DIR}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
CHECKPOINT="${ROOT_DIR}/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"

mkdir -p "$OUT"
exec 9>"${OUT}/.lock"
flock -n 9 || { echo "another TAM smoke is active" >&2; exit 75; }
cd "$ROOT_DIR"
test -x "$PYTHON_BIN"
test -s "$CHECKPOINT"
test -f "$BASELINE"
test -f "$TRACE/trace_validation.json"
git diff --quiet -- "$BASELINE"
avail_kb=$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')
test "$avail_kb" -gt 5000000
export CUDA_VISIBLE_DEVICES=0
export TBSI_INFER_FP16=0
export PYTHONUNBUFFERED=1
exec > >(tee "${OUT}/smoke.log") 2>&1
echo "TAM_TRACE_SMOKE_START $(date -u +%FT%TZ)"
echo "BASELINE_SHA256 $(sha256sum "$BASELINE" | awk '{print $1}')"
echo "CHECKPOINT_SHA256 $(sha256sum "$CHECKPOINT" | awk '{print $1}')"
echo "TRACE_DIR $TRACE"
"$PYTHON_BIN" tools/run_causal_tam_trace_smoke.py \
  --trace-dir "$TRACE" \
  --output-dir "$OUT" \
  --device cuda:0 \
  --steps 100 \
  --batch-size 4 \
  --seed 42
echo "TAM_TRACE_SMOKE_FINISH $(date -u +%FT%TZ)"
