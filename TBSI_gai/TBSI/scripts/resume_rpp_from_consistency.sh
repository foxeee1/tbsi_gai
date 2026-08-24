#!/usr/bin/env bash
set -Eeuo pipefail

# Resume after the four historical anchors. No retraining is performed.
# The script reruns only the parallel consistency check with the selected
# six-worker setting, then computes the RPP gate and stops on failure.

cd /root/autodl-tmp/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
BASE_ROOT=${RPP_BASE_ROOT:-output/proxy_runs/rpp-v1-run-20260818b}
HIST_ROOT=${RPP_HIST_ROOT:-output/proxy_runs/rpp-v1-run-20260819-remaining3}
GATE_ROOT=${RPP_GATE_ROOT:-${HIST_ROOT}/rpp_gate_threads6}
CONFIG=v1.0.7-rpp-anchor-smsa-15ep

exec 9>/tmp/tbsi_rpp_resume.lock
flock -n 9 || { echo "RPP resume lock is held" >&2; exit 2; }
pgrep -af 'tracking/(train|test|analysis_results)\.py' | grep -v grep >/dev/null && { echo "TBSI process active" >&2; exit 3; } || true
free_gb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print int($4/1024/1024)}')
(( free_gb >= 4 )) || { echo "Insufficient disk: ${free_gb} GiB free" >&2; exit 4; }
mkdir -p "${GATE_ROOT}"
export RPP_PROXY_ROOT="${HIST_ROOT}"
export RPP_CONSISTENCY_ROOT="${GATE_ROOT}/parallel_consistency/${CONFIG}"
export RPP_PARALLEL_THREADS=6 TBSI_INFER_FP16=0 TBSI_RPP_IMAGE_CACHE_MAX_SIZE=256
export TBSI_RPP_TEST_LIST="${HIST_ROOT}/proxy_test_sequences.txt"
scripts/run_rpp_parallel_consistency.sh "${CONFIG}" | tee "${GATE_ROOT}/consistency.stdout.log"

set +e
"${PYTHON}" tools/compute_rpp_gate.py \
  --records \
  "v1.0.0-rpp-anchor-baseline-15ep=${BASE_ROOT}/historical_backtest/v1.0.0-rpp-anchor-baseline-15ep/logs/analysis.stdout.log=54.82" \
  "v1.0.7-rpp-anchor-smsa-15ep=${HIST_ROOT}/historical_backtest/v1.0.7-rpp-anchor-smsa-15ep/logs/analysis.stdout.log=55.40" \
  "v1.0.9-rpp-anchor-ctvm-15ep=${HIST_ROOT}/historical_backtest/v1.0.9-rpp-anchor-ctvm-15ep/logs/analysis.stdout.log=54.22" \
  "v1.1.3-rpp-anchor-sall-15ep=${HIST_ROOT}/historical_backtest/v1.1.3-rpp-anchor-sall-15ep/logs/analysis.stdout.log=54.62" \
  --out "${GATE_ROOT}/backtest_gate.json" | tee "${GATE_ROOT}/gate.stdout.log"
rc=${PIPESTATUS[0]}
set -e
if (( rc != 0 )); then
  echo "RPP gate failed; CTVM P0 and pre-screen are blocked" | tee "${GATE_ROOT}/STOP_CTVM.txt"
  exit "$rc"
fi
echo "RPP gate passed; CTVM P0 is the next explicit stage" | tee "${GATE_ROOT}/READY_FOR_CTVM_P0.txt"
