#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/conda_envs/tbsi/bin/python}"
BASE="${ROOT}/TBSI_gai/TBSI"
OUT="${BASE}/output/experiments/false_consensus_week1"
SCREEN="${OUT}/v1.1_pre_review_screen"
TOOLS="${BASE}/experiments/false_consensus_week1/review_tools"

[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${SCREEN}/candidate_v1.csv" ]] || { echo "missing candidate_v1.csv" >&2; exit 1; }
[[ -f "${SCREEN}/review_queue_v1.1.csv" ]] || { echo "missing frozen review queue" >&2; exit 1; }
[[ ! -e "${SCREEN}/reviewer_manifest.csv" || "${SCREEN}/reviewer_manifest.csv" != "${SCREEN}/review_queue_v1.1.csv" ]] || exit 1

"${PYTHON}" "${TOOLS}/export_review_manifest.py" \
  --input "${SCREEN}/review_queue_v1.1.csv" \
  --output "${SCREEN}/reviewer_manifest.csv"
"${PYTHON}" "${TOOLS}/review_audit.py" \
  --candidates "${SCREEN}/candidate_v1.csv" \
  --queue "${SCREEN}/review_queue_v1.1.csv" \
  --output "${OUT}/review_audit.json"
"${PYTHON}" "${TOOLS}/mask_geometry.py"

printf 'blind review tools completed; no intervention outputs were read\n'
