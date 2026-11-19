#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
BASELINE_YAML="$ROOT_DIR/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
CHECKPOINT="$ROOT_DIR/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
TRAIN_LIST="$ROOT_DIR/experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt"
OUT="$ROOT_DIR/output/experiments/frcb_admission_v1"
MANIFEST="$ROOT_DIR/experiments/diagnostics/frcb_admission_v1"
LOG="$OUT/serial.log"
MIN_FREE_GB="${FRCB_MIN_FREE_GB:-20}"
EVENTS_PER_SEQUENCE="${FRCB_EVENTS_PER_SEQUENCE:-2}"

mkdir -p "$ROOT_DIR/output/.locks"
exec 9>"$ROOT_DIR/output/.locks/frcb_admission_v1.lock"
flock -n 9 || { echo "another FRCB admission run is active" >&2; exit 7; }

[[ -x "$PYTHON_BIN" && -s "$CHECKPOINT" && -s "$TRAIN_LIST" ]] || { echo "missing Python/checkpoint/train list" >&2; exit 3; }
[[ -f "$BASELINE_YAML" ]] || { echo "missing protected baseline YAML" >&2; exit 3; }
git -C "$ROOT_DIR" diff --quiet -- "$BASELINE_YAML" || { echo "protected baseline YAML changed" >&2; exit 4; }
[[ ! -e "$OUT" ]] || { echo "refusing to reuse output directory: $OUT" >&2; exit 8; }
FREE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
(( FREE_KB >= MIN_FREE_GB * 1024 * 1024 )) || { echo "insufficient free disk: ${FREE_KB} KiB" >&2; exit 6; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || { echo "GPU check failed" >&2; exit 6; }
if pgrep -af '[p]ython[^ ]*.*(tracking/(train|test)\.py|lib/train/run_training\.py|tools/collect_frcb_admission_v1\.py)' >/tmp/frcb_admission_processes.$$ 2>/dev/null; then
  cat /tmp/frcb_admission_processes.$$ >&2; rm -f /tmp/frcb_admission_processes.$$; exit 7
fi
rm -f /tmp/frcb_admission_processes.$$

mkdir -p "$OUT" "$MANIFEST"
exec > >(tee -a "$LOG") 2>&1
export PYTHONPATH="$ROOT_DIR" PYTHONUNBUFFERED=1 MPLBACKEND=Agg TBSI_INFER_FP16=0
cd "$ROOT_DIR"
echo "FRCB_ADMISSION_START $(date -u +%FT%TZ)"
echo "GIT $(git rev-parse HEAD)"
echo "BASELINE_YAML_SHA256 $(sha256sum "$BASELINE_YAML" | awk '{print $1}')"
echo "CHECKPOINT_SHA256 $(sha256sum "$CHECKPOINT" | awk '{print $1}')"
echo "TRAIN_LIST_SHA256 $(sha256sum "$TRAIN_LIST" | awk '{print $1}')"
echo "EVENTS_PER_SEQUENCE $EVENTS_PER_SEQUENCE"

if [[ -s "$MANIFEST/events.csv" && -s "$MANIFEST/manifest.json" && -s "$MANIFEST/sequence_list.txt" ]]; then
  cmp -s "$TRAIN_LIST" "$MANIFEST/sequence_list.txt" || { echo "existing FRCB manifest does not match frozen RPP train list" >&2; exit 8; }
  echo "FRCB manifest already frozen; reusing $MANIFEST"
else
  [[ ! -e "$MANIFEST/events.csv" && ! -e "$MANIFEST/manifest.json" ]] || { echo "partial FRCB manifest; refusing repair in place" >&2; exit 8; }
  "$PYTHON_BIN" -u tools/build_frcb_admission_manifest.py \
    --sequence-list "$TRAIN_LIST" --output-dir "$MANIFEST" \
    --events-per-sequence "$EVENTS_PER_SEQUENCE"
fi
"$PYTHON_BIN" -u tools/collect_frcb_admission_v1.py \
  --config TBSI_E_MHSA_FULL_4090D_seed42 --checkpoint "$CHECKPOINT" \
  --events "$MANIFEST/events.csv" --output "$OUT/raw" --seed 42
"$PYTHON_BIN" -u tools/analyze_frcb_admission_v1.py \
  --input "$OUT/raw/event_frequency_rows.csv" --output "$OUT/admission_summary.json"
echo "FRCB_ADMISSION_FINISH $(date -u +%FT%TZ)"
