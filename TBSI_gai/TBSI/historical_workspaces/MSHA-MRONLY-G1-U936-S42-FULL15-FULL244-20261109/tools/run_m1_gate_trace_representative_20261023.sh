#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
OUT="$ROOT/output/experiments/M1_EARLY_CONDITIONED_GATE_TRACE_REPRESENTATIVE_20261023"
PARENT_TRACE="$OUT/parent_trace"
CAND_TRACE="$OUT/candidate_trace"
LOG_DIR="$OUT/logs"
mkdir -p "$PARENT_TRACE" "$CAND_TRACE" "$LOG_DIR"

PARENT_CKPT="$ROOT/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
CAND_CKPT="$ROOT/output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023/checkpoints/TBSITrack_ep0015.pth.tar"
PARENT_CFG="TBSI_E_MHSA_FULL_4090D_seed42"
CAND_CFG="M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023_RETRY1"

# Fixed representative frontier: severe regressions, large gains, and controls.
SEQUENCES=(
  womanback2car
  shinycarcoming
  boydownplatform
  4thboywithwhite
  ab_bikeoccluded
  3rdfatboy
  leftfarboycomingpicktheball
)

run_one() {
  local label="$1"
  local cfg="$2"
  local ckpt="$3"
  local trace="$4"
  local seq="$5"
  local log="$LOG_DIR/${label}_${seq}.log"
  echo "[$(date -u +%FT%TZ)] START $label $seq"
  TBSI_CHECKPOINT_OVERRIDE="$ckpt" \
  TBSI_INTERNAL_DIAGNOSTIC=1 \
  TBSI_ATTR_STATS=1 \
  TBSI_DISABLE_TRACKER_CACHE=1 \
  TBSI_M1_TRACE_DIR="$trace" \
  TBSI_TEMPLATE_UPDATE_MODE=off \
  TBSI_TRACE_SAVE_RAW=0 \
    python tracking/test.py tbsi_track "$cfg" --dataset_name lasher_test \
      --sequence "$seq" --threads 0 --num_gpus 1 >"$log" 2>&1
  test -s "$trace/$seq/frames.jsonl"
  echo "[$(date -u +%FT%TZ)] DONE $label $seq $(wc -l < "$trace/$seq/frames.jsonl") frames"
}

for seq in "${SEQUENCES[@]}"; do
  run_one parent "$PARENT_CFG" "$PARENT_CKPT" "$PARENT_TRACE" "$seq"
  run_one candidate "$CAND_CFG" "$CAND_CKPT" "$CAND_TRACE" "$seq"
done

printf '%s\n' "${SEQUENCES[@]}" > "$OUT/manifest.txt"
echo "[$(date -u +%FT%TZ)] COMPLETE"
