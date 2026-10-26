#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
RUN_ID="g0_expand_20261004"
OUT="$ROOT_DIR/output/experiments/$RUN_ID"
CHECKPOINT="$ROOT_DIR/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
BASELINE_YAML="$ROOT_DIR/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
OLD_EXPORT="$ROOT_DIR/output/experiments/g0_information_source_20261004/export"
OLD_LABELS="$ROOT_DIR/output/experiments/g0_information_source_20261004/labels"
LOCK="$ROOT_DIR/output/.locks/$RUN_ID.lock"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Run: tools/run_g0_expand.sh"
  echo "Monitor: tail -f output/experiments/$RUN_ID/runner.log"
  exit 0
fi
[[ $# -eq 0 ]] || { echo "unknown argument; use --help" >&2; exit 64; }
mkdir -p "$ROOT_DIR/output/.locks" "$OUT"
exec 9>"$LOCK"; flock -n 9 || { echo "another G0-Expand run is active" >&2; exit 7; }
exec > >(tee -a "$OUT/runner.log") 2>&1
export PYTHONPATH="$ROOT_DIR" PYTHONUNBUFFERED=1 MPLBACKEND=Agg TBSI_INFER_FP16=0
[[ -x "$PYTHON_BIN" && -s "$CHECKPOINT" && -s "$OLD_EXPORT/events.jsonl" && -s "$OLD_LABELS/token_labels.jsonl" ]] || { echo "missing Python/checkpoint/original G0 cache" >&2; exit 3; }
git -C "$ROOT_DIR" diff --quiet -- "$BASELINE_YAML" || { echo "protected baseline YAML changed" >&2; exit 4; }
[[ ! -e "$OUT/decision.json" ]] || { echo "output already has a decision; refusing overwrite" >&2; exit 8; }
FREE_KIB=$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')
[[ "${FREE_KIB:-0}" -ge 10485760 ]] || { echo "less than 10 GiB free" >&2; exit 6; }
if pgrep -af 'python[^ ]* .*((tracking/(train|test)\.py)|run_training\.py|g0_expand)' >/tmp/g0_expand_processes.$$ 2>/dev/null; then cat /tmp/g0_expand_processes.$$; rm -f /tmp/g0_expand_processes.$$; exit 7; fi
rm -f /tmp/g0_expand_processes.$$
cd "$ROOT_DIR"
echo "G0_EXPAND_START $(date -u +%FT%TZ)"
echo "GIT $(git rev-parse HEAD)"
echo "CHECKPOINT_SHA256 $(sha256sum "$CHECKPOINT" | awk '{print $1}')"
echo "BASELINE_YAML_SHA256 $(sha256sum "$BASELINE_YAML" | awk '{print $1}')"
"$PYTHON_BIN" -m py_compile tools/g0_expand_manifest.py tools/g0_expand_export.py tools/g0_expand_probe.py tools/g0_expand_decide.py
"$PYTHON_BIN" -u tools/g0_expand_manifest.py --output "$OUT/manifest" --count 24 --seed 42
"$PYTHON_BIN" -u tools/g0_expand_export.py --checkpoint "$CHECKPOINT" --manifest "$OUT/manifest/manifest.json" --output "$OUT/export"
"$PYTHON_BIN" -u tools/g0_expand_probe.py --manifest "$OUT/manifest/manifest.json" --old-export "$OLD_EXPORT" --old-labels "$OLD_LABELS" --fresh-export "$OUT/export" --output "$OUT/probe" --steps 640
set +e
"$PYTHON_BIN" -u tools/g0_expand_decide.py --summary "$OUT/probe/probe_summary.json" --output "$OUT/decision.json"
RC=$?
set -e
echo "G0_EXPAND_DECISION_EXIT_CODE $RC"
echo "G0_EXPAND_FINISH $(date -u +%FT%TZ)"
exit "$RC"
