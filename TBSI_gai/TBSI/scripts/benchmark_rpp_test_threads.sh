#!/usr/bin/env bash
set -Eeuo pipefail

# Benchmark test scheduler workers, not PyTorch intra-op threads.
# The benchmark uses normal inference with TBSI_BENCHMARK_NO_SAVE=1 so no
# formal tracking results are overwritten and tracker debug mode is unchanged.

cd /root/autodl-tmp/TBSI_gai/TBSI
PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python
CONFIG=${1:?usage: benchmark_rpp_test_threads.sh <config> <sequence-list> <output-dir>}
SEQ_LIST=${2:?usage: benchmark_rpp_test_threads.sh <config> <sequence-list> <output-dir>}
OUT=${3:?usage: benchmark_rpp_test_threads.sh <config> <sequence-list> <output-dir>}
TIMEOUT_SEC=${RPP_BENCHMARK_TIMEOUT_SEC:-1200}

[[ -s "${SEQ_LIST}" ]] || { echo "Missing benchmark sequence list" >&2; exit 2; }
[[ "${CONFIG}" != *"v1.0.0-单卡3090-48G-等效全局128基线"* ]] || exit 3
command -v nvidia-smi >/dev/null || exit 4
mkdir -p "${OUT}"
printf 'threads,return_code,wall_seconds,mean_fps,median_fps\n' >"${OUT}/summary.csv"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TBSI_INFER_FP16=0
export TBSI_RPP_TEST_LIST="${SEQ_LIST}" TBSI_RPP_IMAGE_CACHE_MAX_SIZE=0 TBSI_BENCHMARK_NO_SAVE=1

for workers in 6 8 12; do
  log="${OUT}/threads_${workers}.log"
  start=$(date +%s)
  set +e
  timeout --signal=TERM --kill-after=30 "${TIMEOUT_SEC}" \
    "${PYTHON}" -u tracking/test.py tbsi_track "${CONFIG}" \
    --dataset_name rpp_lasher_test --threads "${workers}" --num_gpus 1 --debug 0 \
    >"${log}" 2>&1
  rc=$?
  set -e
  end=$(date +%s)
  wall=$((end-start))
  stats=$(
    awk -F'FPS: ' '/FPS: / {v=$2+0; if (v>0) {a[++n]=v; sum+=v}}
      END {if (n==0) {print "nan,nan"; exit} for(i=1;i<=n;i++) for(j=i+1;j<=n;j++) if(a[j]<a[i]){t=a[i];a[i]=a[j];a[j]=t} m=(n%2?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2); printf "%.4f,%.4f",sum/n,m}' "${log}"
  )
  echo "${workers},${rc},${wall},${stats}" >>"${OUT}/summary.csv"
  echo "threads=${workers} rc=${rc} wall=${wall}s mean/median_fps=${stats}"
done
