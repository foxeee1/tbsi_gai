#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
CKPT=${CTVM_AUX_CHECKPOINT:-output/experiments/v1.0.9-rpp-anchor-ctvm-15ep/checkpoints/TBSITrack_ep0015.pth.tar}
OUT=${CTVM_DIAG_ROOT:-output/analysis/ctvm-final-diagnostic-aux}
PROXY_ROOT=${RPP_PROXY_ROOT:-output/proxy_runs/rpp-v1-run-20260819-remaining3}
[[ -s "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 2; }
[[ -s "$PROXY_ROOT/proxy_train_sequences.txt" ]] || { echo "Missing train proxy" >&2; exit 3; }
[[ -s "$PROXY_ROOT/proxy_test_sequences.txt" ]] || { echo "Missing test proxy" >&2; exit 4; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 5; }
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TBSI_INFER_FP16=0
export TBSI_RPP_TRAIN_LIST="$PROXY_ROOT/proxy_train_sequences.txt"
export TBSI_RPP_TEST_LIST="$PROXY_ROOT/proxy_test_sequences.txt"
"$PYTHON" tools/ctvm_final_diagnostic.py --checkpoint "$CKPT" --out "$OUT" \
  >"$OUT/diagnostic.stdout.log" 2>&1
echo "Diagnostic complete: $OUT/ctvm_final_diagnostic.json"
