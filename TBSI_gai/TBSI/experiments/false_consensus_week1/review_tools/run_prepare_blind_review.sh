#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/conda_envs/tbsi/bin/python}"
BASE="${ROOT}/TBSI_gai/TBSI"
OUT="${BASE}/output/experiments/false_consensus_week1"
SCREEN="${OUT}/v1.1_pre_review_screen"
TOOLS="${BASE}/experiments/false_consensus_week1/review_tools"
CLEAN_IMAGES="${SCREEN}/review_images_clean_v1"

[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${SCREEN}/reviewer_manifest.csv" ]] || { echo "missing reviewer manifest" >&2; exit 1; }
[[ ! -e "${OUT}/review_draft_r2.csv" ]] || { echo "review_draft_r2.csv exists; refuse overwrite" >&2; exit 1; }
[[ ! -e "${OUT}/review_gallery_r2.html" ]] || { echo "review_gallery_r2.html exists; refuse overwrite" >&2; exit 1; }
[[ ! -e "${CLEAN_IMAGES}" ]] || { echo "clean image directory exists; refuse overwrite" >&2; exit 1; }

"${PYTHON}" "${TOOLS}/build_clean_review_images.py" \
  --queue "${SCREEN}/review_queue_v1.1.csv" \
  --sequence-list "${BASE}/experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt" \
  --out "${CLEAN_IMAGES}"
"${PYTHON}" "${TOOLS}/prepare_review_sheet.py" \
  --manifest "${SCREEN}/reviewer_manifest.csv" \
  --out "${OUT}/review_draft_r2.csv"
"${PYTHON}" "${TOOLS}/build_review_gallery.py" \
  --review "${OUT}/review_draft_r2.csv" \
  --images-dir "${CLEAN_IMAGES}" \
  --out "${OUT}/review_gallery_r2.html"
printf 'clean blind-review package prepared; manually inspect representative images before labeling\n'
