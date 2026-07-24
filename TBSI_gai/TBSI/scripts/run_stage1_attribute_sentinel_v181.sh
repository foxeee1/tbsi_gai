#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TBSI_gai/TBSI

SHARED_LOG_DIR="output/diagnostics/attribute_sentinel"
mkdir -p "${SHARED_LOG_DIR}"

export PYTHONUNBUFFERED=1
export TBSI_TEST_CPU_THREADS="${TBSI_TEST_CPU_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

python scripts/prepare_attribute_sentinel.py \
  > "${SHARED_LOG_DIR}/prepare_attribute_sentinel.console.log" 2>&1

for subset in A B C; do
  cfg="v1.8.1-reliability-guided-late-layers-${subset}"
  ckpt="output/experiments/${cfg}/checkpoints/TBSITrack_ep0015.pth.tar"
  test_ckpt="output/checkpoints/train/tbsi_track/${cfg}/TBSITrack_ep0015.pth.tar"
  exp_log_dir="output/experiments/${cfg}/logs/stage1_attribute_sentinel"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${test_ckpt}")"
  mkdir -p "${exp_log_dir}"
  ln -sfn "$(pwd)/${ckpt}" "${test_ckpt}"
  cp "${SHARED_LOG_DIR}/attribute_sentinel_manifest.json" "${exp_log_dir}/attribute_sentinel_manifest.json"
  cp "experiments/tbsi_track/attribute_sentinel_sequences.txt" "${exp_log_dir}/attribute_sentinel_sequences.txt"

  echo "[$(date '+%F %T')] TEST attribute sentinel ${cfg}"
  python tracking/test.py tbsi_track "${cfg}" \
    --dataset_name attribute_sentinel_lasher \
    --threads 4 \
    --num_gpus 1 \
    > "${exp_log_dir}/test.console.log" 2>&1
done

python scripts/summarize_attribute_sentinel_v181.py \
  > "${SHARED_LOG_DIR}/summarize_v181.console.log" 2>&1

echo "[$(date '+%F %T')] DONE Stage-1 attribute sentinel v181"
