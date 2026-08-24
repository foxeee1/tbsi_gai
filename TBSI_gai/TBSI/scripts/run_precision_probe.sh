#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON="/root/miniconda3/bin/python"
OUTPUT="$ROOT/output"
CKPT="$OUTPUT/experiments/完美基线全量/checkpoints/TBSITrack_ep0015.pth.tar"
RUN_ROOT="$ROOT/artifacts/baseline/precision-probe"

test -f "$CKPT"
export PYTHONUNBUFFERED=1
export TBSI_LASHER_PROTOCOL=legacy244
export TBSI_DISABLE_TRACKER_CACHE=1
cd "$ROOT"

printf 'checkpoint='; sha256sum "$CKPT"
printf 'python='; "$PYTHON" -c 'import torch,sys; print(sys.version.split()[0], torch.__version__, torch.version.cuda)'

run_one() {
  local mode="$1"
  local cfg="precision_probe_$mode"
  local log="$RUN_ROOT/${mode}.log"
  local exp="$OUTPUT/experiments/$cfg"
  mkdir -p "$exp/checkpoints"
  if [ -e "$exp/checkpoints/TBSITrack_ep0015.pth.tar" ]; then
    unlink "$exp/checkpoints/TBSITrack_ep0015.pth.tar"
  fi
  ln "$CKPT" "$exp/checkpoints/TBSITrack_ep0015.pth.tar"

  printf '%s\n' "[$(date -u +%FT%TZ)] start $mode" | tee "$log"
  if [ "$mode" = "fp16" ]; then
    export TBSI_INFER_FP16=1
  else
    export TBSI_INFER_FP16=0
  fi
  "$PYTHON" tracking/test.py tbsi_track "$cfg" \
    --dataset_name lasher_test --threads 0 --num_gpus 1 2>&1 | tee -a "$log"
  "$PYTHON" tracking/analysis_results.py \
    --tracker_name tbsi_track --tracker_param "$cfg" --dataset_name lasher_test 2>&1 | tee -a "$log"
  printf '%s\n' "[$(date -u +%FT%TZ)] complete $mode" | tee -a "$log"
}

run_one fp16
run_one fp32
