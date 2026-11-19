#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
OUT="$ROOT/output/experiments/innovation1_path_effect"
LOCK="$ROOT/output/.locks/innovation1_path_effect.lock"
LOG="$OUT/launcher.log"

active_tbsi_pids() {
  pgrep -f '(^|/)(train|test)\.py' || true
}
mkdir -p "$OUT" "$(dirname "$LOCK")"

if [[ -e "$LOCK" ]]; then
  lock_pid="$(cat "$LOCK" 2>/dev/null || true)"
  if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
    rm -f "$LOCK"
  else
    echo "refusing to start: lock exists at $LOCK" >&2
    exit 2
  fi
fi
if [[ -n "$(git -C "$ROOT" diff -- "$ROOT/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml")" ]]; then
  echo "refusing to start: canonical baseline YAML is modified" >&2
  exit 2
fi
WAIT_PID="${TBSI_WAIT_PID:-}"
if [[ -z "$WAIT_PID" ]]; then
  WAIT_PID="$(pgrep -f 'tracking/test.py tbsi_track v1.0.3-单卡4090D-24G-等效全局128-seed3407' | head -n1 || true)"
fi
if [[ "${TBSI_PATH_EFFECT_ALLOW_RUNNING:-0}" != "1" ]]; then
  while read -r active_pid; do
    [[ -z "$active_pid" || "$active_pid" == "$$" || "$active_pid" == "$WAIT_PID" ]] && continue
    echo "refusing to start: another TBSI train/test process is active (PID $active_pid)" >&2
    exit 2
  done < <(active_tbsi_pids)
fi
printf '%s\n' "$$" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

if [[ -n "$WAIT_PID" ]]; then
  echo "$(date -u +%FT%TZ) waiting for test PID $WAIT_PID; polling every 1800s" | tee -a "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 1800
    echo "$(date -u +%FT%TZ) test PID $WAIT_PID still active: $(ps -p "$WAIT_PID" -o etime=,stat=,args= || true)" | tee -a "$LOG"
  done
  echo "$(date -u +%FT%TZ) test PID $WAIT_PID ended" | tee -a "$LOG"
else
  echo "$(date -u +%FT%TZ) no matching test PID; starting preparation immediately" | tee -a "$LOG"
fi

remaining_pid="$(active_tbsi_pids | grep -v "^${WAIT_PID}$" | head -n1 || true)"
if [[ -n "$remaining_pid" ]]; then
  echo "refusing to start: a TBSI train/test process remains active" | tee -a "$LOG" >&2
  exit 2
fi
cd "$ROOT"
export PYTHONPATH="$ROOT"
export TBSI_INFER_FP16=0
"$PYTHON" tools/run_path_effect_minimal.py \
  --config "${TBSI_PATH_EFFECT_CONFIG:-v1.0.3-单卡4090D-24G-等效全局128-seed3407}" \
  --output "$OUT" \
  --sequences "${TBSI_PATH_EFFECT_SEQUENCES:-12}" \
  --points "${TBSI_PATH_EFFECT_POINTS:-24}" \
  --horizon 3 --repeats 2 --layer "${TBSI_PATH_EFFECT_LAYER:-2}" \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
exit "$status"
