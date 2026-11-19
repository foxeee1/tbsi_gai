#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
OUT="$ROOT/output/experiments/innovation1_path_effect_audit"
LOCK="$ROOT/output/.locks/innovation1_path_effect_audit.lock"
LOG="$OUT/launcher.log"
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
if pgrep -f '(^|/)(train|test)\.py' >/dev/null; then
  echo "refusing to start: another TBSI train/test process is active" >&2
  exit 2
fi
printf '%s\n' "$$" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
cd "$ROOT"
export PYTHONPATH="$ROOT" TBSI_INFER_FP16=0
"$PYTHON" tools/audit_path_effect.py --output "$OUT" --sequences 8 --points 3 --layer 2 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
exit "$status"
