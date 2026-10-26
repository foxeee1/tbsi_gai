#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
OUT="$ROOT/output/experiments/M1_EARLY_GATE_CANDIDATE_CHECKPOINT_AB_20261024"
TRACE="$OUT/trace"
LOGS="$OUT/logs"
CKPT="$ROOT/output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023/checkpoints/TBSITrack_ep0015.pth.tar"
CFG="M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023_RETRY1"
SEQUENCES=(womanback2car shinycarcoming boydownplatform 4thboywithwhite ab_bikeoccluded 3rdfatboy leftfarboycomingpicktheball)

mkdir -p "$TRACE" "$LOGS"
if [[ -e "$OUT/manifest.txt" ]]; then
  echo "Refusing to overwrite existing run manifest: $OUT/manifest.txt" >&2
  exit 2
fi
if [[ ! -s "$CKPT" ]]; then
  echo "Missing candidate checkpoint: $CKPT" >&2
  exit 2
fi

{
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "git_head=$(git -C "$ROOT" rev-parse HEAD)"
  echo "checkpoint=$CKPT"
  sha256sum "$CKPT"
  sha256sum "$ROOT/experiments/tbsi_track/${CFG}.yaml"
  python -c 'import sys, torch; print("python=" + sys.version.replace("\n", " ")); print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("gpu=" + str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA"))'
  nvidia-smi -L || true
} > "$OUT/provenance.txt" 2>&1

run_arm() {
  local strength="$1"
  local seq="$2"
  local arm
  if [[ "$strength" == "0" ]]; then arm="off"; else arm="on"; fi
  local trace_dir="$TRACE/$arm"
  local trace_file="$trace_dir/$seq/frames.jsonl"
  local result_dir="$ROOT/output/test/tracking_results/tbsi_track/${CFG}_gate${strength}_ab20261024/$seq.txt"
  mkdir -p "$trace_dir"
  if [[ -e "$trace_file" || -e "$result_dir" ]]; then
    echo "Refusing to overwrite existing output for $arm/$seq" >&2
    exit 2
  fi
  local log="$LOGS/${arm}_${seq}.log"
  echo "[$(date -u +%FT%TZ)] START arm=$arm strength=$strength sequence=$seq" | tee -a "$OUT/run.log"
  cd "$ROOT"
  TBSI_CHECKPOINT_OVERRIDE="$CKPT" \
  TBSI_TEST_EARLY_GATE_STRENGTH="$strength" \
  TBSI_INTERNAL_DIAGNOSTIC=1 \
  TBSI_ATTR_STATS=1 \
  TBSI_DISABLE_TRACKER_CACHE=1 \
  TBSI_M1_TRACE_DIR="$trace_dir" \
  TBSI_TEMPLATE_UPDATE_MODE=off \
  TBSI_TRACE_SAVE_RAW=0 \
    python -u tracking/test.py tbsi_track "$CFG" --runid "gate${strength}_ab20261024" \
      --dataset_name lasher_test --sequence "$seq" --threads 0 --num_gpus 1 >"$log" 2>&1
  test -s "$trace_file"
  echo "[$(date -u +%FT%TZ)] DONE arm=$arm sequence=$seq frames=$(wc -l < "$trace_file")" | tee -a "$OUT/run.log"
}

for seq in "${SEQUENCES[@]}"; do
  run_arm 0 "$seq"
  run_arm 1 "$seq"
done

printf '%s\n' "${SEQUENCES[@]}" > "$OUT/manifest.txt"
echo "completed_utc=$(date -u +%FT%TZ)" >> "$OUT/provenance.txt"
echo "[$(date -u +%FT%TZ)] COMPLETE" | tee -a "$OUT/run.log"
