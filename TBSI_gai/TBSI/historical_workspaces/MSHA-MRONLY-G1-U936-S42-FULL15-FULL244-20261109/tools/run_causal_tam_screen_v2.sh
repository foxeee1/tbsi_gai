#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
OUT="${ROOT_DIR}/output/experiments/causal_tam_screen_v2_20261005"
BASELINE="${ROOT_DIR}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
mkdir -p "$OUT"
exec 9>"${OUT}/.run.lock"
flock -n 9 || { echo "Another TAM serial runner is active" >&2; exit 75; }
cd "$ROOT_DIR"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 3; }
[[ -f "$BASELINE" ]] || { echo "Missing canonical baseline" >&2; exit 3; }
git diff --quiet -- "$BASELINE" || { echo "Canonical baseline has a diff" >&2; exit 4; }
export TBSI_INFER_FP16=0
exec > >(tee -a "${OUT}/serial_runner.log") 2>&1
echo "CAUSAL_TAM_SERIAL_START $(date -u +%FT%TZ)"
echo "BASELINE_SHA256 $(sha256sum "$BASELINE" | awk '{print $1}')"

"$PYTHON_BIN" tools/p0_causal_tam_v2.py --device cpu --output "$OUT/p0_v2.json"
"$PYTHON_BIN" tools/audit_causal_tam_real_batch.py --output "$OUT/real_batch.json"

echo "PRE_G0_STOP: training history contract is not wired; no G0 or training launched" >&2
exit 20
