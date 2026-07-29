#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TBSI_gai/TBSI

cfg="v2.1.0-reliability-aware-residual-全量"
tag="v210_full"
log_dir="output/diagnostics/full_candidates_serial/${tag}"
ckpt="output/experiments/${cfg}/checkpoints/TBSITrack_ep0015.pth.tar"
test_ckpt="output/checkpoints/train/tbsi_track/${cfg}/TBSITrack_ep0015.pth.tar"

export PYTHONUNBUFFERED=1
export TBSI_TEST_CPU_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TBSI_SEQUENCE_TIMEOUT="${TBSI_SEQUENCE_TIMEOUT:-1800}"

mkdir -p "${log_dir}" "$(dirname "${test_ckpt}")" "output/test/tracking_results/tbsi_track/${cfg}"
rm -f "${test_ckpt}"
ln -s "$(pwd)/${ckpt}" "${test_ckpt}"

echo "[$(date '+%F %T')] RESUME_MISSING ${cfg}"

mapfile -t missing < <(python - <<'PY'
from pathlib import Path
from lib.test.evaluation import get_dataset
param = 'v2.1.0-reliability-aware-residual-全量'
out = Path('output/test/tracking_results/tbsi_track') / param
existing = {p.stem for p in out.glob('*.txt') if not p.name.endswith('_time.txt')}
for seq in get_dataset('lasher_test'):
    if seq.name not in existing:
        print(seq.name)
PY
)

for seq in "${missing[@]}"; do
  echo "[$(date '+%F %T')] TEST_MISSING ${cfg} ${seq}"
  if ! timeout "$((TBSI_SEQUENCE_TIMEOUT + 300))s" \
      python tracking/test.py tbsi_track "${cfg}" \
        --dataset_name lasher_test \
        --sequence "${seq}" \
        --threads 1 \
        --num_gpus 1 \
        >> "${log_dir}/test.console.log" 2>&1; then
    echo "[$(date '+%F %T')] FALLBACK ${cfg} ${seq}"
    python scripts/write_fallback_result.py tbsi_track "${cfg}" lasher_test "${seq}" \
      >> "${log_dir}/test.console.log" 2>&1
  fi
done

echo "[$(date '+%F %T')] ANALYZE ${cfg}"
python tracking/analysis_results.py \
  --tracker_name tbsi_track \
  --tracker_param "${cfg}" \
  --dataset_name lasher_test \
  > "${log_dir}/analysis.console.log" 2>&1

echo "[$(date '+%F %T')] SUMMARIZE ${cfg}"
python scripts/summarize_full_candidate.py \
  "${cfg}" \
  "${log_dir}" \
  "summary.json" \
  > "${log_dir}/summarize.console.log" 2>&1

find "output/experiments/${cfg}/checkpoints" -type f -name 'TBSITrack_ep0015.pth.tar' -delete
find "output/checkpoints/train/tbsi_track/${cfg}" -type l -delete 2>/dev/null || true
echo "[$(date '+%F %T')] DELETED checkpoint for ${cfg}"

echo "[$(date '+%F %T')] CONTINUE remaining full candidates"
bash scripts/run_full_candidates_serial.sh
