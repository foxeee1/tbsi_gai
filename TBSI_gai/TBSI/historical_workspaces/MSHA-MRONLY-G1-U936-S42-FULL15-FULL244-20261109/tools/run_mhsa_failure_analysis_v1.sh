#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
EXPERIMENT="${ROOT}/output/experiments/mhsa_failure_analysis_v1"
LOG_DIR="${EXPERIMENT}/reports"
LOCK_DIR="${EXPERIMENT}/.serial.lock"
BASELINE_YAML="${ROOT}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
BASELINE_SHA256="4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b"
CONTROL_DIR="${ROOT}/output/test/tracking_results/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线"
M1_DIR="${ROOT}/output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42"
M1_CHECKPOINT="${ROOT}/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
GT_DIR="/root/autodl-tmp/TBSI_gai/data/lasher/testingset"
PAIR_DIR="${EXPERIMENT}/tracking_pairs_v2"
ATTN_DIR="${EXPERIMENT}/attention_stats_bounded"
QUALITY_DIR="${EXPERIMENT}/image_quality"
FAILURE_DIR="${EXPERIMENT}/failure_cases"

mkdir -p "${LOG_DIR}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: another MHSA failure-analysis campaign holds ${LOCK_DIR}" >&2
  exit 2
fi
trap 'rmdir "${LOCK_DIR}"' EXIT

exec > >(tee -a "${LOG_DIR}/serial_runner.log") 2>&1
export PYTHONUNBUFFERED=1
export TBSI_INFER_FP16=0

echo "[START] $(date -u +%FT%TZ) mhsa_failure_analysis_v1 serial runner"
test -x "${PYTHON_BIN}"
test -f "${BASELINE_YAML}"
test "$(sha256sum "${BASELINE_YAML}" | awk '{print $1}')" = "${BASELINE_SHA256}"
test -d "${CONTROL_DIR}"; test -d "${M1_DIR}"; test -f "${M1_CHECKPOINT}"; test -d "${GT_DIR}"
available_kb="$(df -Pk "${ROOT}" | awk 'NR==2 {print $4}')"
test "${available_kb}" -ge 8388608
echo "[CHECK] baseline hash, inputs, FP32 mode, and >=8GiB free space passed"

if test -s "${PAIR_DIR}/frame_level_pairs.csv" && test -s "${PAIR_DIR}/sequence_level_pairs.csv" && test -s "${EXPERIMENT}/manifest/pair_collection_manifest.json"; then
  echo "[SKIP] stage A pair collection already complete"
else
  mkdir -p "${PAIR_DIR}"
  "${PYTHON_BIN}" -u "${ROOT}/tools/collect_mhsa_tracking_pairs.py" \
    --control-dir "${CONTROL_DIR}" --m1-dir "${M1_DIR}" --gt-dir "${GT_DIR}" \
    --output-dir "${PAIR_DIR}"
fi

if test -s "${ATTN_DIR}/mhsa_layer_stats.csv" && test -s "${ATTN_DIR}/collection_manifest.json"; then
  echo "[SKIP] stage B attention collection already complete"
else
  mkdir -p "${ATTN_DIR}"
  "${PYTHON_BIN}" -u "${ROOT}/tools/collect_mhsa_attention_stats.py" \
    --max-frames-per-sequence 64 --output "${ATTN_DIR}"
fi

if test -s "${QUALITY_DIR}/image_quality_stats.csv" && test -s "${QUALITY_DIR}/image_quality_manifest.json"; then
  echo "[SKIP] stage C image-quality collection already complete"
else
  mkdir -p "${QUALITY_DIR}"
  "${PYTHON_BIN}" -u "${ROOT}/tools/collect_mhsa_image_quality_stats.py" \
    --pairs "${PAIR_DIR}/frame_level_pairs.csv" --dataset-dir "${GT_DIR}" \
    --max-frames-per-sequence 64 --output-dir "${QUALITY_DIR}"
fi

mkdir -p "${FAILURE_DIR}"
"${PYTHON_BIN}" -u "${ROOT}/tools/analyze_mhsa_failure_statistics.py" \
  --pairs "${PAIR_DIR}/frame_level_pairs.csv" \
  --attention "${ATTN_DIR}/mhsa_layer_stats.csv" --output-dir "${FAILURE_DIR}"

test -s "${FAILURE_DIR}/failure_statistics_summary.json"
test -s "${FAILURE_DIR}/hard_regression.csv"
test -s "${FAILURE_DIR}/gain_cases.csv"
echo "[DONE] $(date -u +%FT%TZ) serial stages A-C and failure summaries passed"
