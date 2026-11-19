#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
RUN_ROOT="${MHSA_REPLAY_AUDIT_ROOT:-${ROOT}/output/experiments/mhsa_replay_audit_v1_20260929}"
REPORTS="${RUN_ROOT}/reports"
STORED="${ROOT}/output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42"
CHECKPOINT="${ROOT}/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
BASELINE="${ROOT}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
BASELINE_SHA256="4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b"

mkdir -p "${REPORTS}"
if ! mkdir "${RUN_ROOT}/.lock" 2>/dev/null; then echo "ERROR: audit lock is held" >&2; exit 2; fi
trap 'rmdir "${RUN_ROOT}/.lock"' EXIT
exec > >(tee -a "${REPORTS}/serial_runner.log") 2>&1
export PYTHONUNBUFFERED=1 TBSI_INFER_FP16=0

test -x "${PYTHON_BIN}"
test -f "${CHECKPOINT}"
test -d "${STORED}"
test "$(sha256sum "${BASELINE}" | awk '{print $1}')" = "${BASELINE_SHA256}"
"${PYTHON_BIN}" -m py_compile "${ROOT}/tools/audit_mhsa_replay_first_divergence.py"
echo "[START] $(date -u +%FT%TZ) MHSA replay audit"

run_one() {
  local seq="$1"; local target="$2"
  local out="${RUN_ROOT}/${seq}_to_${target}"
  if test -s "${out}.csv" && test -s "${out}.json"; then
    echo "[SKIP] ${seq} target=${target} existing complete output"
    return
  fi
  echo "[RUN] ${seq} target=${target} independent process"
  env TBSI_INFER_FP16=0 TBSI_TEMPLATE_CROSS_SCALE= TBSI_TEMPLATE_ROUTE_R2T= TBSI_TEMPLATE_ROUTE_T2R= \
    "${PYTHON_BIN}" -u "${ROOT}/tools/audit_mhsa_replay_first_divergence.py" \
      --checkpoint "${CHECKPOINT}" --sequence "${seq}" --target-frame "${target}" \
      --stored-results "${STORED}" --output "${out}"
}

# Positive control, largest event-frame mismatch, and an early divergence case.
run_one besom3 635
run_one lefthyalinepaperfrontpants 702
run_one blkboyback 103

echo "[DONE] $(date -u +%FT%TZ) MHSA replay audit"
