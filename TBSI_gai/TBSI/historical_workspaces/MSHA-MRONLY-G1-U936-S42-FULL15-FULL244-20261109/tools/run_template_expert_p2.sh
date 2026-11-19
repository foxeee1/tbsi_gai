#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
BASELINE="experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
CKPT="output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
LOG_DIR="output/experiments/template_expert_p2_retry_20260928/logs"
SAVE_ROOT="./output/template_expert_p2_retry_20260928"
mkdir -p "$LOG_DIR"
exec 9>"output/experiments/template_expert_p2_retry_20260928/.lock"
flock -n 9 || { echo "another P2 campaign is running" >&2; exit 2; }

[[ -z "$(git diff -- "$BASELINE")" ]] || { echo "protected canonical YAML is modified" >&2; exit 3; }
[[ -f "$CKPT" ]] || { echo "M1 checkpoint missing: $CKPT" >&2; exit 4; }
free_kb=$(df --output=avail . | tail -n 1 | tr -d ' ')
(( free_kb >= 4000000 )) || { echo "less than 4 GiB free; refusing P2 start" >&2; exit 5; }
for config in TEMPLATE_EXPERT_DISPONLY_P2_seed42 TEMPLATE_EXPERT_SOFT_P2_seed42; do
  [[ -f "experiments/tbsi_track/${config}.yaml" ]] || exit 6
  [[ ! -e "$SAVE_ROOT/experiments/${config}" ]] || { echo "run directory exists: ${SAVE_ROOT}/experiments/${config}" >&2; exit 7; }
done
if pgrep -af '[t]racking/(train|test)\.py|[l]ib/train/run_training\.py' >/tmp/tbsi_template_p2_processes.$$ 2>/dev/null; then
  cat /tmp/tbsi_template_p2_processes.$$ >&2
  rm -f /tmp/tbsi_template_p2_processes.$$
  echo "another TBSI train/test process is running" >&2
  exit 8
fi
rm -f /tmp/tbsi_template_p2_processes.$$

export PYTHONPATH="$ROOT"
export TBSI_INFER_FP16=0
echo "P2 start $(date -u +%FT%TZ)" | tee "$LOG_DIR/p2_launcher.log"
echo "checkpoint_sha256=$(sha256sum "$CKPT" | awk '{print $1}')" | tee -a "$LOG_DIR/p2_launcher.log"
echo "canonical_yaml_sha256=$(sha256sum "$BASELINE" | awk '{print $1}')" | tee -a "$LOG_DIR/p2_launcher.log"
"$PYTHON_BIN" -u tools/p0_template_expert_p2_audit.py --device cuda 2>&1 | tee "$LOG_DIR/p2_p0.log"

for config in TEMPLATE_EXPERT_DISPONLY_P2_seed42 TEMPLATE_EXPERT_SOFT_P2_seed42; do
  echo "TRAIN $config start $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/p2_launcher.log"
  "$PYTHON_BIN" -u tracking/train.py --script tbsi_track --config "$config" --save_dir "$SAVE_ROOT" --mode single 2>&1 | tee "$LOG_DIR/${config}.log"
  ckpt_dir="$SAVE_ROOT/experiments/$config/checkpoints"
  [[ -s "$ckpt_dir/TBSITrack_ep0003.pth.tar" ]] || {
    echo "missing final checkpoint after training: $config" >&2
    exit 9
  }
  find "$ckpt_dir" -maxdepth 1 -type f ! -name 'TBSITrack_ep0003.pth.tar' -delete
  echo "TRAIN $config exit=0 $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/p2_launcher.log"
done
echo "P2 training complete $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/p2_launcher.log"
