#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
RUN_ID="g0_information_source_20261004"
OUT="$ROOT_DIR/output/experiments/$RUN_ID"
LOG="$OUT/runner.log"
CHECKPOINT="$ROOT_DIR/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
SEQUENCE_LIST="$ROOT_DIR/experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt"
BASELINE_YAML="$ROOT_DIR/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
G0 frozen information-source experiment
  Run:     tools/run_g0_information_source.sh
  Monitor: tail -f output/experiments/g0_information_source_20261004/runner.log
  Result:  output/experiments/g0_information_source_20261004/decision.json
Exit codes: 0=EXPAND_DIAGNOSTIC, 2=STOP, other=execution/preflight failure.
USAGE
  exit 0
fi
if [[ "$#" -gt 0 ]]; then
  echo "unknown argument; use --help" >&2
  exit 64
fi

mkdir -p "$ROOT_DIR/output/.locks"
mkdir -p "$OUT"
exec 9>"$ROOT_DIR/output/.locks/$RUN_ID.lock"
flock -n 9 || { echo "another G0 run is active" >&2; exit 7; }
exec > >(tee -a "$LOG") 2>&1
export PYTHONPATH="$ROOT_DIR" PYTHONUNBUFFERED=1 MPLBACKEND=Agg TBSI_INFER_FP16=0

[[ -x "$PYTHON_BIN" && -s "$CHECKPOINT" && -s "$SEQUENCE_LIST" ]] || {
  echo "missing Python, M1 checkpoint, or frozen train sequence list" >&2; exit 3;
}
git -C "$ROOT_DIR" diff --quiet -- "$BASELINE_YAML" || {
  echo "protected baseline YAML changed; refusing G0" >&2; exit 4;
}
[[ ! -e "$OUT/export/events.jsonl" ]] || { echo "G0 output already exists; refusing overwrite" >&2; exit 8; }
FREE_KIB=$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')
[[ "${FREE_KIB:-0}" -ge 10485760 ]] || { echo "less than 10 GiB free; refusing feature export" >&2; exit 6; }
if pgrep -af 'python[^ ]* .*((tracking/(train|test)\.py)|run_training\.py)' >/tmp/g0_processes.$$ 2>/dev/null; then
  cat /tmp/g0_processes.$$ >&2; rm -f /tmp/g0_processes.$$; exit 7
fi
rm -f /tmp/g0_processes.$$

cd "$ROOT_DIR"
echo "G0_START $(date -u +%FT%TZ)"
echo "GIT $(git rev-parse HEAD)"
echo "CHECKPOINT_SHA256 $(sha256sum "$CHECKPOINT" | awk '{print $1}')"
echo "SEQUENCE_LIST_SHA256 $(sha256sum "$SEQUENCE_LIST" | awk '{print $1}')"
echo "BASELINE_YAML_SHA256 $(sha256sum "$BASELINE_YAML" | awk '{print $1}')"
echo "CONTRACT frozen M1 epoch15; train split; 24 sequences x 2 events; serial FP32; no SCSU training"

"$PYTHON_BIN" -m py_compile \
  tools/g0_export_frozen_features.py tools/g0_make_token_labels.py \
  tools/g0_conditional_probe.py tools/g0_decide.py \
  lib/models/tbsi_track/base_backbone.py lib/models/tbsi_track/vit_tbsi_care.py \
  lib/models/tbsi_track/tbsi_track.py lib/test/tracker/tbsi_track.py

"$PYTHON_BIN" -u tools/g0_export_frozen_features.py \
  --config TBSI_E_MHSA_FULL_4090D_seed42 --checkpoint "$CHECKPOINT" \
  --sequence-list "$SEQUENCE_LIST" --output "$OUT/export" \
  --sequences 24 --events-per-sequence 2 --parity-sequences 3 \
  --early-block 2 --late-block 9 --seed 42

"$PYTHON_BIN" -u tools/g0_make_token_labels.py \
  --export-dir "$OUT/export" --output "$OUT/labels"

"$PYTHON_BIN" -u tools/g0_conditional_probe.py \
  --export-dir "$OUT/export" --labels-dir "$OUT/labels" \
  --output "$OUT/probe" --epochs 80 --seed 42

set +e
"$PYTHON_BIN" -u tools/g0_decide.py \
  --export-dir "$OUT/export" --labels-dir "$OUT/labels" \
  --probe "$OUT/probe/probe_metrics.json" --output "$OUT/decision.json" \
  --min-delta-ap 0.03
DECISION_RC=$?
set -e
echo "G0_DECISION_EXIT_CODE $DECISION_RC"
echo "G0_FINISH $(date -u +%FT%TZ)"
exit "$DECISION_RC"
