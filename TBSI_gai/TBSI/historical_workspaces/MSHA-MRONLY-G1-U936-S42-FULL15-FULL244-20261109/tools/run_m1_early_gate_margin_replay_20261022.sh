#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
LIST="$ROOT/output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/manifests/sequence_list.txt"
CHECKPOINT="$ROOT/output/experiments/M1_EARLY_GATE_MARGIN_OPERATOR_20261022/checkpoints/early_gate_parent_plus_margin_ep0003.pth.tar"
RESULT="$ROOT/output/test/tracking_results/tbsi_track/M1_EARLY_GATE_TARGET_MARGIN_QUICK3_20261022_margin_ep3"
LOG="$ROOT/output/experiments/M1_EARLY_GATE_MARGIN_OPERATOR_20261022/logs/test_12seq.log"
cd "$ROOT"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Missing merged checkpoint: $CHECKPOINT" >&2
    exit 2
fi
if [ -e "$RESULT" ]; then
    echo "Refusing to reuse result path: $RESULT" >&2
    exit 2
fi
mkdir -p "$(dirname "$LOG")"
TBSI_TRACE_TRAIN_LIST="$LIST" \
TBSI_CHECKPOINT_OVERRIDE="$CHECKPOINT" \
TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 TBSI_DISABLE_TRACKER_CACHE=1 \
"$PYTHON" -u tracking/test.py tbsi_track M1_EARLY_GATE_TARGET_MARGIN_QUICK3_20261022 \
    --runid margin_ep3 --dataset_name m1_trace_lasher_train --threads 0 --num_gpus 1 \
    > "$LOG" 2>&1
count=$(find "$RESULT" -maxdepth 1 -name '*.txt' ! -name '*_time.txt' | wc -l)
if [ "$count" -ne 12 ]; then
    echo "Expected 12 bbox results, found $count" >&2
    exit 3
fi
echo "margin replay complete: $count/12 sequences"
