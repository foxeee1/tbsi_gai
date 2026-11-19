#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
RUN_ROOT="${MHSA_F_PARITY_RUN_ROOT:-${ROOT}/output/experiments/mhsa_f_parity_v2_20260929}"
MANIFEST="${ROOT}/output/experiments/mhsa_causal_probe_v2_rerun_20260928/manifest"
PROBE="${RUN_ROOT}/probe"
REPORTS="${RUN_ROOT}/reports"
BASELINE="${ROOT}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
BASELINE_SHA256="4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b"
CHECKPOINT="${ROOT}/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
STORED="${ROOT}/output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42"

mkdir -p "${REPORTS}"
if ! mkdir "${RUN_ROOT}/.lock" 2>/dev/null; then echo "ERROR: F parity lock is held" >&2; exit 2; fi
trap 'rmdir "${RUN_ROOT}/.lock"' EXIT
exec > >(tee -a "${REPORTS}/serial_runner.log") 2>&1
export PYTHONUNBUFFERED=1 TBSI_INFER_FP16=0

test -x "${PYTHON_BIN}"; test -f "${CHECKPOINT}"; test -d "${MANIFEST}"; test -d "${STORED}"
test "$(sha256sum "${BASELINE}" | awk '{print $1}')" = "${BASELINE_SHA256}"
"${PYTHON_BIN}" -m py_compile "${ROOT}/tools/probe_mhsa_effective_message_v2.py"
echo "[START] $(date -u +%FT%TZ) MHSA F-only parity"

if ! test -s "${PROBE}/event_results.csv"; then
  mkdir -p "${PROBE}"
  "${PYTHON_BIN}" -u "${ROOT}/tools/probe_mhsa_effective_message_v2.py" \
    --events-dir "${MANIFEST}" --output "${PROBE}" --conditions F \
    2>&1 | tee "${REPORTS}/probe_f.log"
fi

"${PYTHON_BIN}" - "${PROBE}/event_results.csv" "${MANIFEST}" "${STORED}" > "${RUN_ROOT}/parity_summary.json" <<'PY'
import csv, json, sys
from pathlib import Path
results, manifest, stored_dir = map(Path, sys.argv[1:])
events = {}
for fn in ('regression_events.csv', 'gain_events.csv'):
    for row in csv.DictReader((manifest / fn).open()):
        events[row['event_id']] = row
rows = [r for r in csv.DictReader(results.open()) if r['condition'] == 'F']
by = {r['event_id']: r for r in rows}
checks = []
for event_id, event in sorted(events.items()):
    stored = [float(x) for x in (stored_dir / (event['seq'] + '.txt')).read_text().splitlines()[int(event['frame'])].split()[:4]]
    row = by[event_id]
    actual = [float(row[k]) for k in ('pred_x', 'pred_y', 'pred_w', 'pred_h')]
    checks.append({'event_id': event_id, 'seq': event['seq'], 'frame': int(event['frame']),
                   'max_abs_box_diff': max(abs(a-b) for a,b in zip(actual, stored)),
                   'trunc_box_equal': [int(x) for x in actual] == [int(x) for x in stored],
                   'stored': stored, 'replay': actual})
out = {'events': len(checks), 'max_abs_box_diff': max(x['max_abs_box_diff'] for x in checks),
       'mean_max_abs_box_diff': sum(x['max_abs_box_diff'] for x in checks)/len(checks),
       'over_1px': sum(x['max_abs_box_diff'] > 1.0 for x in checks),
       'trunc_box_equal': sum(x['trunc_box_equal'] for x in checks), 'checks': checks}
print(json.dumps(out, indent=2))
PY
cat "${RUN_ROOT}/parity_summary.json"
echo "[DONE] $(date -u +%FT%TZ) MHSA F-only parity"
