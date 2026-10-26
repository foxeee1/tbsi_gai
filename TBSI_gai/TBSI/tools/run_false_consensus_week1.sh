#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
OUT="$ROOT/output/experiments/false_consensus_week1"
LOCK="$ROOT/output/.locks/false_consensus_week1.lock"
LOG="$OUT/launcher.log"
mkdir -p "$OUT" "$(dirname "$LOCK")"
if [[ -e "$LOCK" ]]; then
  lock_pid="$(cat "$LOCK" 2>/dev/null || true)"
  if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then rm -f "$LOCK"; else echo "refusing to start: lock exists" >&2; exit 2; fi
fi
if [[ -n "$(git -C "$ROOT" diff -- "$ROOT/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml")" ]]; then echo "refusing to start: canonical baseline YAML modified" >&2; exit 2; fi
if pgrep -f '(^|/)(train|test)\.py' >/dev/null; then echo "refusing to start: TBSI train/test process active" >&2; exit 2; fi
printf '%s\n' "$$" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
cd "$ROOT"; export PYTHONPATH="$ROOT" TBSI_INFER_FP16=0
"$PYTHON" tools/false_consensus_week1_collect.py --output "$OUT" --sequence-count 30 --points 6 --layer 0 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}; exit "$status"
