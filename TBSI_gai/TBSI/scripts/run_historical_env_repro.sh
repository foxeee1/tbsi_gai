#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/foxeee1_repro/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
CONFIG_SOURCE=experiments/tbsi_track/完美基线全量.yaml
CONFIG_NAME=historical_env_repro
CONFIG_LINK=experiments/tbsi_track/${CONFIG_NAME}.yaml
OUTPUT=/root/autodl-tmp/TBSI_gai/TBSI/output
RUN_DIR=${OUTPUT}/experiments/${CONFIG_NAME}
LOG=${RUN_DIR}/historical_env_repro.log

cd "${BASE}"
mkdir -p "${RUN_DIR}"
if [[ -e "${CONFIG_LINK}" && ! -L "${CONFIG_LINK}" ]]; then
    echo "Refusing to overwrite ${CONFIG_LINK}" >&2
    exit 2
fi
ln -sfn "$(basename "${CONFIG_SOURCE}")" "${CONFIG_LINK}"

echo "source_config_sha256=$(sha256sum "${CONFIG_SOURCE}" | awk '{print $1}')" | tee "${LOG}"
echo "python=$(${PYTHON} --version 2>&1)" | tee -a "${LOG}"
echo "torch=$(${PYTHON} -c 'import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())')" | tee -a "${LOG}"
echo "timm=$(${PYTHON} -c 'import timm; print(timm.__version__)')" | tee -a "${LOG}"
echo "started=$(date -Is)" | tee -a "${LOG}"

unset OMP_NUM_THREADS
"${PYTHON}" -u tracking/train.py \
    --script tbsi_track --config "${CONFIG_NAME}" \
    --save_dir "${OUTPUT}" --mode single 2>&1 | tee -a "${LOG}"

"${PYTHON}" -u tracking/test.py \
    tbsi_track "${CONFIG_NAME}" --dataset_name lasher_test \
    --threads 0 --num_gpus 1 2>&1 | tee -a "${LOG}"

"${PYTHON}" -u tracking/analysis_results.py \
    --tracker_name tbsi_track --tracker_param "${CONFIG_NAME}" \
    --dataset_name lasher_test 2>&1 | tee -a "${LOG}"

echo "finished=$(date -Is)" | tee -a "${LOG}"
