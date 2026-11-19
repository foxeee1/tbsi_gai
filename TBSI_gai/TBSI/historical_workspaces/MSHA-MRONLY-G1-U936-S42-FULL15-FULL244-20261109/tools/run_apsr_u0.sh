#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
BASELINE_YAML="${ROOT_DIR}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
BASELINE_SHA="4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b"
OUT_ROOT="${ROOT_DIR}/output/experiments/apsr-u0"
LOCK_FILE="${OUT_ROOT}/.campaign.lock"
MODE=""
CONFIG="v1.0.0-单卡3090-48G-等效全局128基线"
CHECKPOINT="${ROOT_DIR}/output/experiments/v1.0.0-单卡3090-48G-等效全局128基线/checkpoints/TBSITrack_ep0015.pth.tar"
SEQUENCE_LIST=""
SPLIT="train"
RUN_ID=""
MAX_SEQUENCES=0
WRITE_INTERVAL=25
RESUME=0

usage() {
  echo "Usage: $0 --mode gt|pred --sequence-list LIST --run-id ID [options]"
  echo "  --config NAME --checkpoint PATH --split train|testingset --max-sequences N"
  echo "  --write-interval N (default 25)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --sequence-list) SEQUENCE_LIST="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --max-sequences) MAX_SEQUENCES="$2"; shift 2 ;;
    --write-interval) WRITE_INTERVAL="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$MODE" == "gt" || "$MODE" == "pred" ]] || { echo "--mode gt|pred is required" >&2; exit 2; }
[[ -n "$SEQUENCE_LIST" && -f "$SEQUENCE_LIST" ]] || { echo "sequence list missing" >&2; exit 2; }
[[ -n "$RUN_ID" ]] || { echo "--run-id is required" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "checkpoint not found: $CHECKPOINT" >&2; exit 1; }
[[ -f "$BASELINE_YAML" ]] || { echo "canonical baseline missing" >&2; exit 1; }

actual_sha="$(sha256sum "$BASELINE_YAML" | awk '{print $1}')"
[[ "$actual_sha" == "$BASELINE_SHA" ]] || { echo "canonical baseline SHA mismatch" >&2; exit 1; }

mkdir -p "$OUT_ROOT"
if [[ -e "$LOCK_FILE" ]]; then
  echo "APSR campaign lock exists: $LOCK_FILE" >&2
  exit 1
fi
if pgrep -af 'run_apsr_u0.py|tracking/test.py' >/dev/null; then
  echo "another TBSI experiment process is running" >&2
  exit 1
fi
trap 'rm -f "$LOCK_FILE"' EXIT
printf '%s\n' "$$ $(date -u +%FT%TZ) $MODE $RUN_ID" > "$LOCK_FILE"

export PYTHONUNBUFFERED=1
export TBSI_INFER_FP16=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ROOT_DIR"

"$PYTHON_BIN" tools/run_apsr_u0.py \
  --mode "$MODE" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --run-id "$RUN_ID" --sequence-list "$SEQUENCE_LIST" --split "$SPLIT" \
  --max-sequences "$MAX_SEQUENCES" --write-interval "$WRITE_INTERVAL" \
  $([[ "$RESUME" == 1 ]] && echo --resume) \
  --output-root "$OUT_ROOT"
