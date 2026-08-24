#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/foxeee1_repro/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
CONFIG_NAME=repro_clean_worker4
OUTPUT=/root/autodl-tmp/TBSI_gai/TBSI/output
RUN_DIR=${OUTPUT}/experiments/${CONFIG_NAME}
LOG=${RUN_DIR}/run.log

cd "${BASE}"
mkdir -p "${RUN_DIR}"
echo "config_sha256=$(sha256sum experiments/tbsi_track/${CONFIG_NAME}.yaml | awk '{print $1}')" | tee "${LOG}"
echo "python=$(${PYTHON} --version 2>&1)" | tee -a "${LOG}"
set +e
torch_info=$(${PYTHON} -c 'import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())' 2>&1)
timm_info=$(${PYTHON} -c 'import timm; print(timm.__version__)' 2>&1)
set -e
echo "torch=${torch_info}" | tee -a "${LOG}"
echo "timm=${timm_info}" | tee -a "${LOG}"
echo "started=$(date -Is)" | tee -a "${LOG}"

unset OMP_NUM_THREADS
"${PYTHON}" -u tracking/train.py --script tbsi_track --config "${CONFIG_NAME}" \
  --save_dir "${OUTPUT}" --mode single 2>&1 | tee -a "${LOG}"
echo "training_finished=$(date -Is)" | tee -a "${LOG}"

"${PYTHON}" -u tracking/test.py tbsi_track "${CONFIG_NAME}" --dataset_name lasher_test \
  --threads 0 --num_gpus 1 2>&1 | tee -a "${LOG}"
echo "testing_finished=$(date -Is)" | tee -a "${LOG}"

"${PYTHON}" -u tracking/analysis_results.py --tracker_name tbsi_track \
  --tracker_param "${CONFIG_NAME}" --dataset_name lasher_test 2>&1 | tee -a "${LOG}"
echo "finished=$(date -Is)" | tee -a "${LOG}"
