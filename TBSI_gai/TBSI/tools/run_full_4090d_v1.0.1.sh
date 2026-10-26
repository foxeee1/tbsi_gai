#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
CONFIG="${TBSI_CONFIG:-v1.0.1-单卡4090D-24G-等效全局128加速}"
SEED="${TBSI_SEED:-42}"
TEST_THREADS="${TBSI_TEST_THREADS:-5}"
TEST_ONLY="${TBSI_TEST_ONLY:-0}"
DATASET="lasher_test"
OUTPUT_ROOT="${ROOT_DIR}/output"
RUN_DIR="${OUTPUT_ROOT}/experiments/${CONFIG}"
LOG_FILE="${RUN_DIR}/full_train_test_analysis.log"
LOCK_DIR="${OUTPUT_ROOT}/.locks"
BASELINE_YAML="${ROOT_DIR}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
CONFIG_YAML="${ROOT_DIR}/experiments/tbsi_track/${CONFIG}.yaml"
CHECKPOINT="${RUN_DIR}/checkpoints/TBSITrack_ep0015.pth.tar"
RESULT_DIR="${OUTPUT_ROOT}/test/tracking_results/tbsi_track/${CONFIG}"

mkdir -p "${LOCK_DIR}"
exec 9>"${LOCK_DIR}/tbsi_full_run.lock"
if ! flock -n 9; then
    echo "Another TBSI full pipeline is already running. Exiting." >&2
    exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 3
fi
if [[ ! -f "${CONFIG_YAML}" ]]; then
    echo "Experiment YAML not found: ${CONFIG_YAML}" >&2
    exit 3
fi
if [[ ! -f "${BASELINE_YAML}" ]]; then
    echo "Protected baseline YAML not found: ${BASELINE_YAML}" >&2
    exit 3
fi
if ! git -C "${ROOT_DIR}" diff --quiet -- "${BASELINE_YAML}"; then
    echo "Protected baseline YAML has a working-tree diff; refusing to start." >&2
    exit 4
fi
if [[ -e "${RESULT_DIR}" && "${TEST_ONLY}" != "1" ]]; then
    echo "Tracking results already exist; refusing to overwrite: ${RESULT_DIR}" >&2
    exit 5
fi
if [[ -e "${RUN_DIR}" && -s "${CHECKPOINT}" && "${TEST_ONLY}" != "1" ]]; then
    echo "Checkpoint already exists; refusing to overwrite: ${CHECKPOINT}" >&2
    exit 5
fi
if pgrep -af '[t]racking/(train|test)\.py' >/tmp/tbsi_existing_processes.$$ 2>/dev/null; then
    echo "Existing TBSI train/test process detected; refusing to start:" >&2
    cat /tmp/tbsi_existing_processes.$$ >&2
    rm -f /tmp/tbsi_existing_processes.$$
    exit 6
fi
rm -f /tmp/tbsi_existing_processes.$$

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'rc=$?; echo "EXIT rc=${rc} time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; exit "$rc"' EXIT

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "${ROOT_DIR}"
START_TIME="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "START ${START_TIME}"
echo "PID $$"
echo "CONFIG ${CONFIG}"
echo "SEED ${SEED}"
echo "TEST_THREADS ${TEST_THREADS}"
echo "TEST_ONLY ${TEST_ONLY}"
echo "ROOT ${ROOT_DIR}"
echo "HOST $(hostname)"
echo "GIT $(git rev-parse HEAD)"
echo "BASELINE_YAML_SHA256 $(sha256sum "${BASELINE_YAML}" | awk '{print $1}')"
echo "CONFIG_YAML_SHA256 $(sha256sum "${CONFIG_YAML}" | awk '{print $1}')"
"${PYTHON_BIN}" -c 'import sys, torch; print("PYTHON", sys.version.replace("\\n", " ")); print("PYTORCH", torch.__version__); print("CUDA", torch.version.cuda); print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo "=== TRAIN ==="
if [[ "${TEST_ONLY}" == "1" ]]; then
    echo "TEST_ONLY=1: skipping training and reusing ${CHECKPOINT}"
elif [[ "${SEED}" == "42" ]]; then
    "${PYTHON_BIN}" -u tracking/train.py \
        --script tbsi_track \
        --config "${CONFIG}" \
        --save_dir ./output \
        --mode single
else
    # tracking/train.py does not expose --seed; call its single-GPU backend
    # directly so alternate-seed runs use the requested reproducible seed.
    "${PYTHON_BIN}" -u lib/train/run_training.py \
        --script tbsi_track \
        --config "${CONFIG}" \
        --save_dir ./output \
        --seed "${SEED}" \
        --local_rank -1 \
        --use_lmdb 0 \
        --config_prv baseline \
        --use_wandb 0
fi

if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "Expected checkpoint missing after training: ${CHECKPOINT}" >&2
    exit 7
fi
echo "CHECKPOINT $(stat -c '%s bytes %y' "${CHECKPOINT}")"

echo "=== FULL TEST: LasHeR 244 sequences, FP32 tracker, one GPU ==="
export TBSI_INFER_FP16=0
"${PYTHON_BIN}" -u tracking/test.py \
    tbsi_track "${CONFIG}" \
    --dataset_name "${DATASET}" \
    --threads "${TEST_THREADS}" \
    --num_gpus 1

if [[ ! -d "${RESULT_DIR}" ]]; then
    echo "Tracking result directory missing: ${RESULT_DIR}" >&2
    exit 8
fi
RESULT_COUNT="$(find "${RESULT_DIR}" -maxdepth 1 -type f -name '*.txt' | wc -l)"
echo "TRACKING_RESULT_TXT_COUNT ${RESULT_COUNT}"

echo "=== ANALYSIS ==="
"${PYTHON_BIN}" -u tracking/analysis_results.py \
    --tracker_name tbsi_track \
    --tracker_param "${CONFIG}" \
    --dataset_name "${DATASET}"

echo "FINISH $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
