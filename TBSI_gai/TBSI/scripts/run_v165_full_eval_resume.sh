#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TBSI_gai/TBSI

export TBSI_TEST_CPU_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python tracking/test.py tbsi_track v1.6.5-signal-decouple-learnscale-all-smax05-init01-全量 \
  --dataset_name lasher_test \
  --threads 4 \
  --num_gpus 1 \
  > output/diagnostics/full_signal_decouple/test_v165_full_threads4_resume2.console.log 2>&1

python tracking/analysis_results.py \
  --tracker_name tbsi_track \
  --tracker_param v1.6.5-signal-decouple-learnscale-all-smax05-init01-全量 \
  --dataset_name lasher_test \
  > output/diagnostics/full_signal_decouple/analysis_v165_full.console.log 2>&1
