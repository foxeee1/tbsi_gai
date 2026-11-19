#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
LIST="$ROOT_DIR/experiments/protocols/m1_screening_v1_20261022/quick_train_sequences.txt"
CHECKPOINT="$ROOT_DIR/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
OUT_ROOT="$ROOT_DIR/output/experiments"
test -x "$PYTHON_BIN"
test -s "$LIST"
test -s "$CHECKPOINT"
if ps -eo pid=,args= | awk -v self="$$" '$1 != self && ($0 ~ /python.*lib\/train\/run_training\.py/ || $0 ~ /torch\.distributed\.launch/) {found=1} END {exit found ? 0 : 1}'; then
  echo 'resource gate: another training process is active' >&2
  exit 75
fi
export TBSI_RPP_TRAIN_LIST="$LIST"
export TBSI_SPARSE_CROSS_MODE="neck_union"
export TBSI_SPARSE_TOPK="8"
export TBSI_SPARSE_MASK_CONTROL="none"
export TBSI_INFER_FP16=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$ROOT_DIR"
run_one() {
  local name="$1"; local config="$2"; local mode="$3"
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"
  if [[ "$mode" == "parent" ]]; then
    unset TBSI_SPARSE_CROSS_MODE
  else
    export TBSI_SPARSE_CROSS_MODE="neck_union"
  fi
  "$PYTHON_BIN" -u lib/train/run_training.py --script tbsi_track --config "$config" \
    --save_dir "$out" --seed 42 --use_lmdb 0
}
PARENT_CKPT="$OUT_ROOT/M1_SPARSE_CROSS_NECK_PARENT_QUICK_20261022/experiments/M1_SPARSE_CROSS_NECK_PARENT_QUICK_20261022/checkpoints/TBSITrack_ep0016.pth.tar"
if [[ ! -s "$PARENT_CKPT" ]]; then
  run_one M1_SPARSE_CROSS_NECK_PARENT_QUICK_20261022 M1_SPARSE_CROSS_NECK_PARENT_QUICK_20261022 parent
fi
run_one M1_SPARSE_CROSS_NECK_QUICK_RETRY1_20261022 M1_SPARSE_CROSS_NECK_QUICK_20261022 candidate
