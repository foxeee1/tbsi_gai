#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
RUN_ROOT="${MHSA_CAUSAL_RUN_ROOT:-${ROOT}/output/experiments/mhsa_causal_probe_v2}"
MANIFEST="${RUN_ROOT}/manifest"
PROBE="${RUN_ROOT}/probe"
REPORTS="${RUN_ROOT}/reports"
LOCK="${RUN_ROOT}/.serial.lock"
BASELINE="${ROOT}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
BASELINE_SHA256="4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b"
PAIRS="${ROOT}/output/experiments/mhsa_failure_analysis_v1/tracking_pairs_v2/frame_level_pairs.csv"
CHECKPOINT="${ROOT}/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"

mkdir -p "${MANIFEST}" "${REPORTS}"
if ! mkdir "${LOCK}" 2>/dev/null; then echo "ERROR: causal probe lock is held" >&2; exit 2; fi
trap 'rmdir "${LOCK}"' EXIT
exec > >(tee -a "${REPORTS}/serial_runner.log") 2>&1
export PYTHONUNBUFFERED=1 TBSI_INFER_FP16=0

echo "[START] $(date -u +%FT%TZ) MHSA-CAUSAL-PROBE-V2"
test -x "${PYTHON_BIN}"; test -f "${BASELINE}"; test -f "${PAIRS}"; test -f "${CHECKPOINT}"
test "$(sha256sum "${BASELINE}" | awk '{print $1}')" = "${BASELINE_SHA256}"
available_kb="$(df -Pk "${ROOT}" | awk 'NR==2 {print $4}')"; test "${available_kb}" -ge 8388608
"${PYTHON_BIN}" -m py_compile "${ROOT}/lib/models/layers/tbsi_layer.py" \
  "${ROOT}/tools/select_mhsa_causal_events_v2.py" "${ROOT}/tools/probe_mhsa_effective_message_v2.py"
echo "[CHECK] baseline, inputs, FP32, disk, and syntax passed"

"${PYTHON_BIN}" -u "${ROOT}/tools/select_mhsa_causal_events_v2.py" \
  --pairs "${PAIRS}" --events-per-group 24 --output-dir "${MANIFEST}"
test "$(($(wc -l < "${MANIFEST}/regression_events.csv") - 1))" -eq 24
test "$(($(wc -l < "${MANIFEST}/gain_events.csv") - 1))" -eq 24
test "$(($(wc -l < "${MANIFEST}/matched_events.csv") - 1))" -eq 24

if test -s "${PROBE}/event_results.csv" && test -s "${PROBE}/probe_manifest.json"; then
  echo "[SKIP] probe output already complete"
else
  mkdir -p "${PROBE}"
  "${PYTHON_BIN}" -u "${ROOT}/tools/probe_mhsa_effective_message_v2.py" \
    --events-dir "${MANIFEST}" --output "${PROBE}" \
    2>&1 | tee "${REPORTS}/probe.log"
fi
test -s "${PROBE}/event_results.csv"; test -s "${PROBE}/probe_manifest.json"
echo "[DONE] $(date -u +%FT%TZ) causal probe completed"
