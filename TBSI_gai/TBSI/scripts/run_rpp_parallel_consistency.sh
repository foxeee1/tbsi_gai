#!/usr/bin/env bash
set -Eeuo pipefail

# Compare serial and parallel test artifacts for one already-trained RPP
# anchor.  Only sequence scheduling changes.  The script uses an isolated
# non-canonical config and preserves both snapshots for audit.

cd /root/autodl-tmp/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
PROXY_ROOT=${RPP_PROXY_ROOT:-output/proxy_runs/rpp-v1}
TEST_DATASET=${RPP_TEST_DATASET:-rpp_lasher_test}
CONFIG=${1:?usage: run_rpp_parallel_consistency.sh <config>}
RUN_ROOT=${PROXY_ROOT}/parallel_consistency/${CONFIG}
RUN_ROOT=${RPP_CONSISTENCY_ROOT:-${RUN_ROOT}}
RESULT_ROOT=output/test/tracking_results/tbsi_track/${CONFIG}
SERIAL_ROOT=${RUN_ROOT}/serial
PARALLEL_ROOT=${RUN_ROOT}/parallel

[[ "${CONFIG}" != *"v1.0.0-单卡3090-48G-等效全局128基线"* ]] || exit 2
[[ -s "${PROXY_ROOT}/proxy_test_sequences.txt" ]] || { echo "Missing frozen test list" >&2; exit 3; }
[[ -s "${PROXY_ROOT}/proxy_test_manifest.json" ]] || { echo "Missing test manifest" >&2; exit 4; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Consistency directory already exists: ${RUN_ROOT}" >&2; exit 5; }

mkdir -p "${RUN_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TBSI_RPP_IMAGE_CACHE_MAX_SIZE=${TBSI_RPP_IMAGE_CACHE_MAX_SIZE:-256}
export TBSI_INFER_FP16=0
export TBSI_RPP_TEST_LIST="${PROXY_ROOT}/proxy_test_sequences.txt"

# The backtest already produced one result directory. Preserve it before
# making the serial/parallel pair; never delete or overwrite it.
if [[ -e "${RESULT_ROOT}" ]]; then
  mv "${RESULT_ROOT}" "${RUN_ROOT}/preexisting"
fi

"${PYTHON}" -u tracking/test.py tbsi_track "${CONFIG}" \
  --dataset_name "${TEST_DATASET}" --threads 0 --num_gpus 1 \
  >"${RUN_ROOT}/serial.stdout.log" 2>&1
mv "${RESULT_ROOT}" "${SERIAL_ROOT}"

"${PYTHON}" -u tracking/test.py tbsi_track "${CONFIG}" \
  --dataset_name "${TEST_DATASET}" --threads "${RPP_PARALLEL_THREADS:-6}" --num_gpus 1 \
  >"${RUN_ROOT}/parallel.stdout.log" 2>&1
mv "${RESULT_ROOT}" "${PARALLEL_ROOT}"

diff -qr "${SERIAL_ROOT}" "${PARALLEL_ROOT}" >"${RUN_ROOT}/diff.txt" || {
  echo "Serial/parallel tracking artifacts differ; RPP gate failed" >&2
  exit 7
}
find "${SERIAL_ROOT}" -type f -printf '%P\n' | sort >"${RUN_ROOT}/serial.files"
find "${PARALLEL_ROOT}" -type f -printf '%P\n' | sort >"${RUN_ROOT}/parallel.files"
diff -u "${RUN_ROOT}/serial.files" "${RUN_ROOT}/parallel.files"
echo "[$(date -Is)] serial/parallel consistency passed for ${CONFIG}"
