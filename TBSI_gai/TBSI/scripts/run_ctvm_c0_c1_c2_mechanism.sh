#!/usr/bin/env bash
set -Eeuo pipefail

# CTVM model-side mechanism campaign.
# C0 is the existing implementation; C1 removes only value detach; C2 adds
# only per-head relation on top of C1. This campaign intentionally does not
# run proxy testing or make an AUC decision because the current RPP gate is
# not valid for ranking candidates.

cd /root/autodl-tmp/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
RUN_ROOT=${CTVM_MECH_ROOT:-output/experiments/ctvm-mechanism-campaign}
PROXY_ROOT=${RPP_PROXY_ROOT:-output/proxy_runs/rpp-v1-run-20260819-remaining3}
LOCK_FILE=/tmp/tbsi_ctvm_mechanism.lock
MIN_FREE_GB=${CTVM_MIN_FREE_GB:-4}
CONFIGS=(v1.2.0-ctvm-c0-rpp-3ep v1.2.1-ctvm-c1-rpp-3ep v1.2.2-ctvm-c2-rpp-3ep)

exec 9>"${LOCK_FILE}"
flock -n 9 || { echo "CTVM mechanism lock is held" >&2; exit 2; }
if pgrep -af 'tracking/(train|test|analysis_results)\.py' | grep -v grep >/dev/null; then
  echo "Another TBSI process is active" >&2
  exit 3
fi
command -v nvidia-smi >/dev/null || { echo "nvidia-smi unavailable" >&2; exit 4; }
free_gb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print int($4/1024/1024)}')
(( free_gb >= MIN_FREE_GB )) || { echo "Insufficient disk: ${free_gb} GiB" >&2; exit 5; }
[[ -s "${PROXY_ROOT}/proxy_train_sequences.txt" ]] || { echo "Missing frozen train list" >&2; exit 6; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Output exists: ${RUN_ROOT}" >&2; exit 7; }

mkdir -p "${RUN_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TBSI_INFER_FP16=0
export TBSI_RPP_TRAIN_LIST="${PROXY_ROOT}/proxy_train_sequences.txt"
export TBSI_RPP_TEST_LIST="${PROXY_ROOT}/proxy_test_sequences.txt"

for config in "${CONFIGS[@]}"; do
  yaml="experiments/tbsi_track/${config}.yaml"
  log_dir="${RUN_ROOT}/${config}/logs"
  [[ -s "$yaml" ]] || { echo "Missing config: $yaml" >&2; exit 8; }
  mkdir -p "$log_dir"
  echo "[$(date -Is)] START ${config}" | tee -a "${RUN_ROOT}/campaign.log"
  "$PYTHON" -u tracking/train.py --script tbsi_track --config "$config" --save_dir ./output --mode single \
    >"${log_dir}/train.stdout.log" 2>&1
  if grep -q 'Traceback (most recent call last)' "${log_dir}/train.stdout.log"; then
    echo "${config} training log contains Traceback; stopping campaign" >&2
    exit 10
  fi
  echo "[$(date -Is)] DONE ${config}" | tee -a "${RUN_ROOT}/campaign.log"
  # Mechanism screening does not need a 2.3 GiB checkpoint. Keep logs and
  # diagnostics, remove only this newly-created campaign checkpoint.
  checkpoint_dir="output/experiments/${config}/checkpoints"
  if [[ -d "$checkpoint_dir" ]]; then
    rm -rf -- "$checkpoint_dir"
  fi
  free_gb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print int($4/1024/1024)}')
  (( free_gb >= MIN_FREE_GB )) || { echo "Disk fell below safety floor after ${config}" >&2; exit 9; }
done

echo "[$(date -Is)] CTVM mechanism campaign complete" | tee -a "${RUN_ROOT}/campaign.log"
