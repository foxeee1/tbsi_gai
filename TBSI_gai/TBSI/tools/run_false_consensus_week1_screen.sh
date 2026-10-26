#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
SOURCE="$ROOT/output/experiments/false_consensus_week1"
OUT="$ROOT/output/experiments/false_consensus_week1/v1.1_pre_review_screen"
LOCK="$ROOT/output/.locks/false_consensus_week1_screen.lock"
mkdir -p "$OUT" "$(dirname "$LOCK")"
if [[ -e "$LOCK" ]]; then echo "refusing to start: lock exists" >&2; exit 2; fi
if [[ -e "$OUT/candidate_v1.csv" ]]; then echo "refusing to overwrite existing v1.1 output" >&2; exit 2; fi
if [[ ! -f "$SOURCE/candidate_events.csv" || ! -d "$SOURCE/review_images" ]]; then echo "missing original candidate artifacts" >&2; exit 2; fi
if [[ -n "$(git -C "$ROOT" diff -- "$ROOT/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml")" ]]; then echo "refusing to start: canonical baseline YAML modified" >&2; exit 2; fi
if pgrep -f '(^|/)(train|test)\.py' >/dev/null; then echo "refusing to start: TBSI train/test process active" >&2; exit 2; fi
printf '%s\n' "$$" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
cd "$ROOT"
"$PYTHON" tools/false_consensus_week1_screen.py \
  --input "$SOURCE/candidate_events.csv" \
  --output "$OUT" \
  --image-root "$SOURCE/review_images" \
  --expanded-ratio 0.10 --max-normalized-distance 0.50 \
  --random-rejected 40 --seed 3407 \
  2>&1 | tee "$OUT/screen.log"
status=${PIPESTATUS[0]}; exit "$status"
