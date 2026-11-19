#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
RUN_ROOT="${MHSA_P2_RUN_ROOT:-${ROOT}/output/experiments/mhsa_causal_probe_v2_p2_20260929}"
MANIFEST="${ROOT}/output/experiments/mhsa_causal_probe_v2_rerun_20260928/manifest"
PROBE="${RUN_ROOT}/probe"
REPORTS="${RUN_ROOT}/reports"
BASELINE="${ROOT}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
BASELINE_SHA256="4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b"
CHECKPOINT="${ROOT}/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"

mkdir -p "${REPORTS}"
if ! mkdir "${RUN_ROOT}/.lock" 2>/dev/null; then echo "ERROR: P2 lock is held" >&2; exit 2; fi
trap 'rmdir "${RUN_ROOT}/.lock"' EXIT
exec > >(tee -a "${REPORTS}/serial_runner.log") 2>&1
export PYTHONUNBUFFERED=1 TBSI_INFER_FP16=0
test -x "${PYTHON_BIN}"; test -f "${CHECKPOINT}"; test -d "${MANIFEST}"
test "$(sha256sum "${BASELINE}" | awk '{print $1}')" = "${BASELINE_SHA256}"
"${PYTHON_BIN}" -m py_compile "${ROOT}/tools/probe_mhsa_effective_message_v2.py"
echo "[START] $(date -u +%FT%TZ) MHSA P2 same-state split"

if ! test -s "${PROBE}/event_results.csv"; then
  mkdir -p "${PROBE}"
  "${PYTHON_BIN}" -u "${ROOT}/tools/probe_mhsa_effective_message_v2.py" \
    --events-dir "${MANIFEST}" --output "${PROBE}" \
    --conditions F,C_OFF,C_HALF,C_1.5,F_POST \
    2>&1 | tee "${REPORTS}/probe_p2.log"
fi

"${PYTHON_BIN}" - "${PROBE}/event_results.csv" "${PROBE}/p2_summary.json" <<'PY'
import csv, json, sys
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1])))
groups=defaultdict(list)
for r in rows:
    if r['condition'] in ('F','F_POST'):
        groups[(r['event_id'],r['layer'])].append(r)
checks=[]
for key, vals in groups.items():
    if len(vals)!=2: continue
    a,b=vals
    box_a=[float(a[k]) for k in ('pred_x','pred_y','pred_w','pred_h')]
    box_b=[float(b[k]) for k in ('pred_x','pred_y','pred_w','pred_h')]
    checks.append({'event_id':key[0],'layer':int(key[1]),'max_abs_f_pre_post_diff':max(abs(x-y) for x,y in zip(box_a,box_b)),
                   'state_hash_equal':a['state_hash']==b['state_hash'],'f_pre':box_a,'f_post':box_b})
out={'events':len(set(x['event_id'] for x in checks)),'layer_checks':len(checks),
     'max_abs_f_pre_post_diff':max(x['max_abs_f_pre_post_diff'] for x in checks),
     'state_hash_all_equal':all(x['state_hash_equal'] for x in checks),'checks':checks}
open(sys.argv[2],'w').write(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
PY
cat "${PROBE}/p2_summary.json"
echo "[DONE] $(date -u +%FT%TZ) MHSA P2 same-state split"
