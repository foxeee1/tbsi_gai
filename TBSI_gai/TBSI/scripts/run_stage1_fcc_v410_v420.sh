#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TBSI_gai/TBSI

LOG_DIR="output/diagnostics/stage1_fcc_v410_v420"
mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED=1
export TBSI_TEST_CPU_THREADS="${TBSI_TEST_CPU_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

echo "[$(date '+%F %T')] PREPARE attribute sentinel"
python scripts/prepare_attribute_sentinel.py \
  > "${LOG_DIR}/prepare_attribute_sentinel.console.log" 2>&1

run_one() {
  local cfg="$1"
  local ckpt="output/experiments/${cfg}/checkpoints/TBSITrack_ep0015.pth.tar"
  local test_ckpt="output/checkpoints/train/tbsi_track/${cfg}/TBSITrack_ep0015.pth.tar"
  local exp_log_dir="output/experiments/${cfg}/logs/stage1_attribute_sentinel"

  mkdir -p "${exp_log_dir}"
  cp "output/diagnostics/attribute_sentinel/attribute_sentinel_manifest.json" \
    "${exp_log_dir}/attribute_sentinel_manifest.json"
  cp "experiments/tbsi_track/attribute_sentinel_sequences.txt" \
    "${exp_log_dir}/attribute_sentinel_sequences.txt"

  echo "[$(date '+%F %T')] START ${cfg}"
  if [[ -f "${ckpt}" ]]; then
    echo "[$(date '+%F %T')] ${cfg} checkpoint exists, skip train: ${ckpt}"
  else
    echo "[$(date '+%F %T')] TRAIN ${cfg}"
    python tracking/train.py \
      --script tbsi_track \
      --config "${cfg}" \
      --save_dir ./output \
      --mode single \
      --use_lmdb 0 \
      --use_wandb 0 \
      > "${exp_log_dir}/train_stage1.console.log" 2>&1
  fi

  if [[ ! -f "${ckpt}" ]]; then
    echo "[$(date '+%F %T')] ERROR ${cfg} final checkpoint missing: ${ckpt}" >&2
    exit 2
  fi

  mkdir -p "$(dirname "${test_ckpt}")"
  ln -sfn "$(pwd)/${ckpt}" "${test_ckpt}"

  echo "[$(date '+%F %T')] TEST attribute sentinel ${cfg}"
  python tracking/test.py tbsi_track "${cfg}" \
    --dataset_name attribute_sentinel_lasher \
    --threads 4 \
    --num_gpus 1 \
    > "${exp_log_dir}/test_attribute_sentinel.console.log" 2>&1

  echo "[$(date '+%F %T')] DONE ${cfg}"
}

for cfg in \
  v4.1.0-fcc-A \
  v4.1.0-fcc-B \
  v4.1.0-fcc-C \
  v4.2.0-fcc-residual-A \
  v4.2.0-fcc-residual-B \
  v4.2.0-fcc-residual-C
do
  run_one "${cfg}"
done

echo "[$(date '+%F %T')] SUMMARIZE Stage-1 FCC v4.1.0/v4.2.0"
python scripts/summarize_stage1_fcc_v410_v420.py \
  > "${LOG_DIR}/summarize.console.log" 2>&1

echo "[$(date '+%F %T')] ALL DONE Stage-1 FCC v4.1.0/v4.2.0"
