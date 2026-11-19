#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=/root/autodl-tmp
PROJECT_ROOT="$REPO_ROOT/TBSI_gai/TBSI"
PYTHON_BIN=/root/autodl-tmp/conda_envs/tbsi/bin/python
RUN_ID=M1_FUSION_FLOW_FULL244_DIAG_20261009
RUN_ROOT="$PROJECT_ROOT/output/experiments/$RUN_ID"
MANIFEST="$RUN_ROOT/manifest.json"
LOCK_FILE="$RUN_ROOT/.launcher.lock"
mkdir -p "$RUN_ROOT/logs"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another $RUN_ID launcher holds $LOCK_FILE" >&2
  exit 3
fi
cd "$REPO_ROOT"
echo "[$(date -Is)] PRELIGHT"
"$PYTHON_BIN" -u "$PROJECT_ROOT/tools/preflight_m1_full244_flow_20261009.py" \
  --manifest "$MANIFEST" --output-dir "$RUN_ROOT/closed_loop_trace"
echo "[$(date -Is)] CLOSED_LOOP_TRACE"
"$PYTHON_BIN" -u "$PROJECT_ROOT/tools/run_m1_full244_flow_trace_20261009.py" \
  --manifest "$MANIFEST" --output-dir "$RUN_ROOT/closed_loop_trace"
echo "[$(date -Is)] GRADIENT_BATCHES"
"$PYTHON_BIN" -u "$PROJECT_ROOT/tools/run_m1_full244_flow_gradients_20261009.py" \
  --root "$RUN_ROOT" \
  --checkpoint "$PROJECT_ROOT/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar" \
  --config "$PROJECT_ROOT/experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml"
echo "[$(date -Is)] FINAL_ANALYSIS"
"$PYTHON_BIN" -u "$PROJECT_ROOT/tools/analyze_m1_test_flow_20261009.py" \
  --root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/analysis" \
  --archive-dir "$PROJECT_ROOT/output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42" \
  --full-per-sequence-csv "$PROJECT_ROOT/output/experiments/M1_FULL244_SEQUENCE_AUDIT_20261009/per_sequence.csv"
echo "[$(date -Is)] ALL_STAGES_COMPLETE"
