#!/usr/bin/env bash
set -Eeuo pipefail

# RPP-v1 CTVM orchestrator. Training uses the fixed stratified train subset
# for the full 15-epoch protocol; testing uses the fixed stratified test subset.

cd /root/autodl-tmp/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
PROXY_ROOT=${RPP_PROXY_ROOT:-output/proxy_runs/rpp-v1}
TRAIN_DATASET=${RPP_TRAIN_DATASET:-RPP_LasHeR_train}
TEST_DATASET=${RPP_TEST_DATASET:-rpp_lasher_test}
MIN_FREE_GB=${RPP_MIN_FREE_GB:-8}
RUN_ROOT=${PROXY_ROOT}/ctvm_screen
LOCK_FILE=/tmp/tbsi_rpp_v1.lock
BACKTEST_GATE=${RPP_BACKTEST_GATE:-${PROXY_ROOT}/historical_backtest/backtest_gate.json}
P0_GATE=${RPP_CTVM_P0_GATE:-${PROXY_ROOT}/ctvm_screen/p0_gate.json}

# The 3-epoch stage is intentionally CTVM-only pre-screening.  It is not the
# general RPP protocol and cannot be used as final evidence.
PREFILTER_CONFIGS=(
  ${RPP_CTVM_PREFILTER_CONFIGS:-v1.2.0-ctvm-c0-rpp-3ep v1.2.1-ctvm-c1-rpp-3ep v1.2.2-ctvm-c2-rpp-3ep}
)

CONFIGS=(
  ${RPP_CTVM_CONFIGS:-v1.2.0-ctvm-c0-rpp-15ep v1.2.1-ctvm-c1-rpp-15ep v1.2.2-ctvm-c2-rpp-15ep}
)

run_stage() {
  local config="$1" stage="$2"; shift 2
  local log_dir="${RUN_ROOT}/${config}/logs"
  mkdir -p "${log_dir}"
  echo "[$(date -Is)] START ${config} ${stage}" | tee -a "${RUN_ROOT}/campaign.log"
  "$@" >"${log_dir}/${stage}.stdout.log" 2>&1
  echo "[$(date -Is)] DONE ${config} ${stage}" | tee -a "${RUN_ROOT}/campaign.log"
}

exec 9>"${LOCK_FILE}"
flock -n 9 || { echo "RPP campaign lock is held: ${LOCK_FILE}" >&2; exit 2; }

free_gb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print int($4/1024/1024)}')
(( free_gb >= MIN_FREE_GB )) || {
  echo "Insufficient disk: ${free_gb} GiB free, need >= ${MIN_FREE_GB} GiB" >&2
  exit 3
}
command -v nvidia-smi >/dev/null || { echo "nvidia-smi unavailable" >&2; exit 14; }
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
(( gpu_count >= 1 )) || { echo "No NVIDIA GPU detected" >&2; exit 15; }

baseline_yaml=experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml
git diff --quiet -- "${baseline_yaml}" || {
  echo "Protected baseline YAML has local changes; refusing to start." >&2
  exit 4
}

for required in \
  "${PROXY_ROOT}/proxy_train_sequences.txt" \
  "${PROXY_ROOT}/proxy_test_sequences.txt" \
  "${PROXY_ROOT}/proxy_train_manifest.json" \
  "${PROXY_ROOT}/proxy_test_manifest.json"; do
  [[ -s "${required}" ]] || { echo "Missing proxy artifact: ${required}" >&2; exit 5; }
done

[[ -s "${BACKTEST_GATE}" ]] || {
  echo "Missing passed historical RPP gate: ${BACKTEST_GATE}" >&2
  exit 8
}
[[ -s "${P0_GATE}" ]] || {
  echo "Missing passed CTVM P0 gate: ${P0_GATE}" >&2
  exit 9
}
"${PYTHON}" - "${BACKTEST_GATE}" "${P0_GATE}" <<'PY'
import json, sys
for path in sys.argv[1:]:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if data.get('status') not in ('pass', 'passed', 'Strong Go', 'Weak Go'):
        raise SystemExit('gate is not passed: ' + path)
PY

for config in "${PREFILTER_CONFIGS[@]}" "${CONFIGS[@]}"; do
  [[ "${config}" != *"v1.0.0-单卡3090-48G-等效全局128基线"* ]] || exit 10
  [[ -s "experiments/tbsi_track/${config}.yaml" ]] || {
    echo "Missing explicit CTVM config: experiments/tbsi_track/${config}.yaml" >&2
    exit 11
  }
  grep -q "${TRAIN_DATASET}" "experiments/tbsi_track/${config}.yaml" || {
    echo "${config} is not bound to ${TRAIN_DATASET}" >&2
    exit 12
  }
  grep -q "${TEST_DATASET}" "experiments/tbsi_track/${config}.yaml" || {
    echo "${config} is not bound to ${TEST_DATASET}" >&2
    exit 13
  }
done

if pgrep -af 'tracking/(train|test|analysis_results)\.py' | grep -v grep >/dev/null; then
  echo "Another TBSI train/test/analysis process is active; refusing to start." >&2
  exit 6
fi

[[ ! -e "${RUN_ROOT}" ]] || { echo "Independent output directory already exists: ${RUN_ROOT}" >&2; exit 16; }
mkdir -p "${RUN_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TBSI_INFER_FP16=0
export TBSI_RPP_IMAGE_CACHE_MAX_SIZE=${TBSI_RPP_IMAGE_CACHE_MAX_SIZE:-256}
export TBSI_RPP_TRAIN_LIST="${PROXY_ROOT}/proxy_train_sequences.txt"
export TBSI_RPP_TEST_LIST="${PROXY_ROOT}/proxy_test_sequences.txt"

for config in "${PREFILTER_CONFIGS[@]}"; do
  run_stage "${config}" train_3ep "${PYTHON}" -u tracking/train.py \
    --script tbsi_track --config "${config}" --save_dir ./output --mode single
  run_stage "${config}" test_3ep "${PYTHON}" -u tracking/test.py \
    tbsi_track "${config}" --dataset_name "${TEST_DATASET}" --threads "${RPP_TEST_THREADS:-6}" --num_gpus 1
  run_stage "${config}" analysis_3ep "${PYTHON}" -u tracking/analysis_results.py \
    --tracker_name tbsi_track --tracker_param "${config}" --dataset_name "${TEST_DATASET}"
done

echo "[$(date -Is)] CTVM 3-epoch pre-screen complete; proceeding to 15-epoch proxy only for explicitly retained configs" | tee -a "${RUN_ROOT}/campaign.log"

for config in "${CONFIGS[@]}"; do
  [[ -s "experiments/tbsi_track/${config}.yaml" ]] || {
    echo "Missing explicit RPP config: experiments/tbsi_track/${config}.yaml" >&2
    exit 7
  }
done

for config in "${CONFIGS[@]}"; do
  run_stage "${config}" train "${PYTHON}" -u tracking/train.py \
    --script tbsi_track --config "${config}" --save_dir ./output --mode single
  run_stage "${config}" test "${PYTHON}" -u tracking/test.py \
    tbsi_track "${config}" --dataset_name "${TEST_DATASET}" --threads "${RPP_TEST_THREADS:-6}" --num_gpus 1
  run_stage "${config}" analysis "${PYTHON}" -u tracking/analysis_results.py \
    --tracker_name tbsi_track --tracker_param "${config}" --dataset_name "${TEST_DATASET}"
done

echo "[$(date -Is)] RPP CTVM proxy campaign complete" | tee -a "${RUN_ROOT}/campaign.log"
