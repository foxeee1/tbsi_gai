#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   run_rpp_proxy_backtest.sh <proxy-config> [<proxy-config> ...]
# This performs the mandatory historical RPP-v1 backtest only.  It never
# launches the canonical baseline and refuses configs that still point at the
# full LasHeR train/test datasets.

cd /root/autodl-tmp/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
PROXY_ROOT=${RPP_PROXY_ROOT:-output/proxy_runs/rpp-v1}
TRAIN_DATASET=${RPP_TRAIN_DATASET:-RPP_LasHeR_train}
TEST_DATASET=${RPP_TEST_DATASET:-rpp_lasher_test}
MIN_FREE_GB=${RPP_MIN_FREE_GB:-8}
RUN_ROOT=${PROXY_ROOT}/historical_backtest
LOCK_FILE=/tmp/tbsi_rpp_v1.lock
CONSISTENCY_SCRIPT=scripts/run_rpp_parallel_consistency.sh

if [[ "${RPP_ALLOW_PARTIAL:-0}" == "1" ]]; then
  (( $# >= 1 )) || { echo "Provide >=1 explicit RPP config name for resume." >&2; exit 2; }
else
  (( $# >= 4 )) || { echo "Provide >=4 explicit RPP config names." >&2; exit 2; }
fi
exec 9>"${LOCK_FILE}"
flock -n 9 || { echo "RPP campaign lock is held: ${LOCK_FILE}" >&2; exit 3; }

free_gb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print int($4/1024/1024)}')
(( free_gb >= MIN_FREE_GB )) || { echo "Insufficient disk: ${free_gb} GiB free" >&2; exit 4; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi unavailable" >&2; exit 12; }
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
(( gpu_count >= 1 )) || { echo "No NVIDIA GPU detected" >&2; exit 13; }

baseline_yaml=experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml
git diff --quiet -- "${baseline_yaml}" || { echo "Protected baseline YAML changed" >&2; exit 5; }
for required in proxy_train_sequences.txt proxy_test_sequences.txt proxy_train_manifest.json proxy_test_manifest.json; do
  [[ -s "${PROXY_ROOT}/${required}" ]] || { echo "Missing ${PROXY_ROOT}/${required}" >&2; exit 6; }
done
pgrep -af 'tracking/(train|test|analysis_results)\.py' | grep -v grep >/dev/null && {
  echo "Another TBSI process is active" >&2; exit 7;
} || true

for config in "$@"; do
  [[ "${config}" != *"v1.0.0-单卡3090-48G-等效全局128基线"* ]] || { echo "Canonical baseline is forbidden" >&2; exit 8; }
  file="experiments/tbsi_track/${config}.yaml"
  [[ -s "${file}" ]] || { echo "Missing config: ${file}" >&2; exit 9; }
  grep -q "${TRAIN_DATASET}" "${file}" || { echo "${file} does not target ${TRAIN_DATASET}" >&2; exit 10; }
  grep -q "${TEST_DATASET}" "${file}" || { echo "${file} does not declare ${TEST_DATASET}" >&2; exit 11; }
done

[[ ! -e "${RUN_ROOT}" ]] || { echo "Independent output directory already exists: ${RUN_ROOT}" >&2; exit 14; }
mkdir -p "${RUN_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TBSI_INFER_FP16=0
export TBSI_RPP_IMAGE_CACHE_MAX_SIZE=${TBSI_RPP_IMAGE_CACHE_MAX_SIZE:-256}
export TBSI_RPP_TRAIN_LIST="${PROXY_ROOT}/proxy_train_sequences.txt"
export TBSI_RPP_TEST_LIST="${PROXY_ROOT}/proxy_test_sequences.txt"
for config in "$@"; do
  log_dir="${RUN_ROOT}/${config}/logs"; mkdir -p "${log_dir}"
  echo "[$(date -Is)] START ${config} train" | tee -a "${RUN_ROOT}/campaign.log"
  "${PYTHON}" -u tracking/train.py --script tbsi_track --config "${config}" --save_dir ./output --mode single >"${log_dir}/train.stdout.log" 2>&1
  echo "[$(date -Is)] DONE ${config} train" | tee -a "${RUN_ROOT}/campaign.log"
  "${PYTHON}" -u tracking/test.py tbsi_track "${config}" --dataset_name "${TEST_DATASET}" --threads "${RPP_TEST_THREADS:-6}" --num_gpus 1 >"${log_dir}/test.stdout.log" 2>&1
  "${PYTHON}" -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param "${config}" --dataset_name "${TEST_DATASET}" >"${log_dir}/analysis.stdout.log" 2>&1
  echo "[$(date -Is)] DONE ${config} test+analysis" | tee -a "${RUN_ROOT}/campaign.log"
done

# The first completed anchor is used only for the mandatory serial/parallel
# evaluator check.  This does not compare metrics yet; it verifies that
# parallel scheduling has not changed the per-sequence tracking artifacts.
first_config="$1"
"${CONSISTENCY_SCRIPT}" "${first_config}"

cat >"${RUN_ROOT}/backtest_complete.json" <<EOF
{
  "status": "complete_pending_gate",
  "configs": [$(printf '"%s",' "$@" | sed 's/,$//')],
  "train_dataset": "${TRAIN_DATASET}",
  "test_dataset": "${TEST_DATASET}",
  "serial_parallel_check": "passed",
  "next_required": "compute_proxy_full_correlations_and_write_backtest_gate.json"
}
EOF
echo "[$(date -Is)] RPP historical runs complete; correlation gate is still required" | tee -a "${RUN_ROOT}/campaign.log"
