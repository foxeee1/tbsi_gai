#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
CHECKPOINT="$ROOT/output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/checkpoints/M1_parent_state_plus_gate_ep0003.pth.tar"
SEQUENCES="$ROOT/output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/manifests/sequence_list.txt"
mkdir -p "$ROOT/output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/logs"
cd "$ROOT"

for arm in lam0 lam05 lam1; do
    if [ -e "$ROOT/output/test/tracking_results/tbsi_track/M1_EARLY_GATE_STRENGTH_REPLAY_20261022_$arm" ]; then
        echo "Refusing to reuse existing result directory for $arm" >&2
        exit 2
    fi
done

run_arm() {
    local arm="$1"
    local strength="$2"
    local log="$ROOT/output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/logs/${arm}.log"
    TBSI_TRACE_TRAIN_LIST="$SEQUENCES" \
    TBSI_CHECKPOINT_OVERRIDE="$CHECKPOINT" \
    TBSI_TEST_EARLY_GATE_STRENGTH="$strength" \
    TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 TBSI_DISABLE_TRACKER_CACHE=1 \
    "$PYTHON" -u tracking/test.py tbsi_track M1_EARLY_GATE_STRENGTH_REPLAY_20261022 \
        --runid "$arm" --dataset_name m1_trace_lasher_train --threads 0 --num_gpus 1 \
        > "$log" 2>&1
    local count
    count=$(find "$ROOT/output/test/tracking_results/tbsi_track/M1_EARLY_GATE_STRENGTH_REPLAY_20261022_$arm" -maxdepth 1 -name '*.txt' ! -name '*_time.txt' | wc -l)
    if [ "$count" -ne 12 ]; then
        echo "Expected 12 bbox results for $arm, found $count" >&2
        exit 3
    fi
    echo "$arm complete: $count/12 sequences"
}

run_arm lam0 0
run_arm lam05 0.5
run_arm lam1 1
