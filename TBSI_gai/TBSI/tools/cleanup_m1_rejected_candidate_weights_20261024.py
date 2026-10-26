#!/usr/bin/env python3
"""Remove reviewed rejected-candidate checkpoints while retaining all reports/results."""
import hashlib, json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'experiments/protocols/m1_screening_v1_20261022/rejected_candidate_weight_cleanup_20261024.json'
DIRS = [
    'M1_EARLY_SEARCH_RELATION_BIAS_QUICK3_20261024',
    'M1_EARLY_TEMPLATE_VALUE_RECON_SCREEN_20261023',
    'M1_LOCAL_CROSS_MODAL_CORRESPONDENCE_QUICK3_20261024',
    'M1_LOCAL_CROSS_MODAL_CORRESPONDENCE_CONTROL_QUICK3_20261024',
    'M1_QUERY_CONDITIONED_CROSS_VALUE_QUICK3_20261024',
    'M1_QUERY_CONDITIONED_CROSS_VALUE_CONTROL_QUICK3_20261024',
    'M1_QUERY_CONDITIONED_CROSS_VALUE_FROZEN_CONTROL_QUICK3_20261024',
    'M1_SPATIAL_TOKEN_BRIDGE_SCREEN_20261023',
    'M1_SPATIAL_TOKEN_BRIDGE_LEARNABILITY_10EP_20261024',
]
EXCLUDE = {
    'TBSI_E_MHSA_FULL_4090D_seed42',
    'v1.0.1-单卡4090D-24G-等效全局128加速',
    'M1_STSER_COMPETITION_QUICK3_20261024',
    'M1_STSER_PLAIN_T2S_QUICK3_20261024',
    'M1_STSER_QUICK3_20261024',
}

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def main():
    if RECORD.exists():
        raise SystemExit('cleanup record already exists; refusing a second run')
    rows = []
    for name in DIRS:
        if name in EXCLUDE:
            raise SystemExit('protected name in cleanup list: ' + name)
        d = ROOT / 'output/experiments' / name / 'checkpoints'
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix in {'.tar', '.pth', '.pt', '.ckpt'}:
                rows.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                             'sha256_before_delete': sha256(p), 'status': 'DELETED'})
                p.unlink()
    if not rows:
        raise SystemExit('no reviewed checkpoint files found')
    record = {'id': 'M1-REJECTED-CANDIDATE-WEIGHT-CLEANUP-20261024',
              'created_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
              'mode': 'delete', 'count': len(rows), 'bytes': sum(x['bytes'] for x in rows),
              'reports_and_tracking_results_retained': True, 'files': rows}
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'record': str(RECORD.relative_to(ROOT)), 'count': len(rows),
                      'bytes': record['bytes']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
