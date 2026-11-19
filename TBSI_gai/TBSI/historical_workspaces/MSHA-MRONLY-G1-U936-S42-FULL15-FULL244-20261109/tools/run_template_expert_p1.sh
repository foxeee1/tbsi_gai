#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
BASELINE="experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
CKPT="output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
LOG_DIR="output/experiments/template_expert_p1_20260928/logs"
SAVE_ROOT="./output/template_expert_p1_retry"
mkdir -p "$LOG_DIR"
exec 9>"output/experiments/template_expert_p1_20260928/.lock"
flock -n 9 || { echo "another template expert campaign is running" >&2; exit 2; }

if [[ -n "$(git diff -- "$BASELINE")" ]]; then
  echo "protected canonical YAML is modified" >&2
  exit 3
fi
[[ -f "$CKPT" ]] || { echo "M1 checkpoint missing: $CKPT" >&2; exit 4; }
free_kb=$(df --output=avail . | tail -n 1 | tr -d ' ')
(( free_kb >= 6000000 )) || { echo "less than 6 GiB free; refusing to start" >&2; exit 5; }
for config in TEMPLATE_EXPERT_C_P1_seed42 TEMPLATE_EXPERT_G_P1_seed42 TEMPLATE_EXPERT_T_P1_seed42; do
  [[ -f "experiments/tbsi_track/${config}.yaml" ]] || exit 6
  [[ ! -e "$SAVE_ROOT/experiments/${config}" ]] || { echo "run directory exists: ${SAVE_ROOT}/experiments/${config}" >&2; exit 7; }
done

export PYTHONPATH="$ROOT"
export TBSI_INFER_FP16=0
echo "checkpoint_sha256=$(sha256sum "$CKPT" | awk '{print $1}')" | tee "$LOG_DIR/launcher.log"
echo "canonical_yaml_sha256=$(sha256sum "$BASELINE" | awk '{print $1}')" | tee -a "$LOG_DIR/launcher.log"
echo "P0 start $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/launcher.log"
"$PYTHON_BIN" -u tools/p0_template_expert_audit.py --device cuda 2>&1 | tee "$LOG_DIR/p0.log"
echo "P0 PASS $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/launcher.log"

for config in TEMPLATE_EXPERT_C_P1_seed42 TEMPLATE_EXPERT_G_P1_seed42 TEMPLATE_EXPERT_T_P1_seed42; do
  echo "TRAIN ${config} start $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/launcher.log"
  "$PYTHON_BIN" -u tracking/train.py --script tbsi_track --config "$config" --save_dir "$SAVE_ROOT" --mode single 2>&1 | tee "$LOG_DIR/${config}_retry.log"
  echo "TRAIN ${config} exit=$? $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/launcher.log"
done
echo "campaign complete $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/launcher.log"
