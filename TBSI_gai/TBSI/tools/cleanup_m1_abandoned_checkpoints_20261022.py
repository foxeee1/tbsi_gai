#!/usr/bin/env python3
"""Remove only the reviewed, abandoned M1 screening checkpoints."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'experiments/protocols/m1_screening_v1_20261022/abandoned_checkpoint_cleanup_20261022.json'
FILES = [
    ('M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021/checkpoints/TBSITrack_ep0003.pth.tar', '15e0bf9b2a489cecd24f1d4994a8b44a535aae2c689eca2ba3e2be237981a1b2', 452186976, 'DR variant; current branch stopped after no supported sequence-level gain.'),
    ('M1_AMCM_DR_FG_MATCH_NOSUP_QUICK15_3EP_20261021/checkpoints/TBSITrack_ep0003.pth.tar', 'b0799ebccae9ef76d4c46f4ada8aaf3f6177655376c892d187bcfba303600b3e', 452186976, 'DR variant; current branch stopped after no supported sequence-level gain.'),
    ('M1_AMCM_DR_MIX50_NOSUP_QUICK15_3EP_20261021/checkpoints/TBSITrack_ep0003.pth.tar', '6219047e5da0a8a6af5f1abd02e8b528df9f790ac88b08a3e0fd8e1bc15b03d8', 452186976, 'DR variant; current branch stopped after no supported sequence-level gain.'),
    ('M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021/checkpoints/TBSITrack_ep0003.pth.tar', '1704b42f0cf0bb8dbb598b85c55a3eec0c303516c9fe7f055d4062a540bfb8c5', 441074961, 'Early gate candidate not promoted; follow-up strength and margin branches are not supported.'),
    ('M1_EARLY_GATE_MARGIN_OPERATOR_20261022/checkpoints/early_gate_parent_plus_margin_ep0003.pth.tar', 'f9bfd5cb69d3d2b580fb1bec26a762875d6b5ce0d3bf1cab57a93a6d07ab9596', 441078259, 'Learned margin operator not supported; correction barely changed gate values or CENTER rank.'),
    ('M1_EARLY_GATE_STRENGTH_REPLAY_20261022/checkpoints/M1_parent_state_plus_gate_ep0003.pth.tar', '40b69b33a52b5c7522c5d936aefd474a28048f50a12ee0ef88e1f0fd0f789861', 441070609, 'Exploratory strength replay; gains concentrated in one sequence and not promoted.'),
    ('M1_EARLY_GATE_TARGET_MARGIN_QUICK3_20261022/checkpoints/TBSITrack_ep0001.pth.tar', '8920725af95ef05bff35209e99893652c58241bb30dbd948f57f030d253863d6', 441073203, 'Rejected early target-margin branch; retain report and per-sequence results.'),
    ('M1_EARLY_GATE_TARGET_MARGIN_QUICK3_20261022/checkpoints/TBSITrack_ep0002.pth.tar', 'c16d7ef0420269fa00b916ee4953996458e91718826b0d33c8c1394a225be684', 441078067, 'Rejected early target-margin branch; retain report and per-sequence results.'),
    ('M1_EARLY_GATE_TARGET_MARGIN_QUICK3_20261022/checkpoints/TBSITrack_ep0003.pth.tar', '26f8679f8571462973fc49c2de8f25a432b609397fdccfc29a6b65d8a41ca1f9', 441082867, 'Rejected early target-margin branch; retain report and per-sequence results.'),
    ('M1_EARLY_MARGIN_FIXED_STATE_EVIDENCE_20261022/noop_margin_control.pth.tar', '4c676534ec832675da28721478c83619bf12a825b329b1dc1588f82a19cd1beb', 441078259, 'Redundant diagnostic no-op control; M1 parent checkpoint is separately protected.'),
    ('M1_MSHA_DER_OVERFIT16_20261022/checkpoints/TBSITrack_ep0005.pth.tar', 'ee82ee6e95dfbafe0aba9a514017e01793d9d53386e411c0bb822b590976b89a', 444611087, 'DER learnability screen inconclusive; the subsequent quick screen rejected this branch.'),
    ('M1_MSHA_DER_QUICK3_20261022/checkpoints/TBSITrack_ep0003.pth.tar', 'a8172e9a13fb8c2d90179bc1cb87b3f9868c5ec11eb9507ea42be29e4acac139', 444623823, 'DER quick branch not supported; report states stop current DER training branch.'),
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete', action='store_true', help='verify the fixed whitelist and unlink those checkpoint files')
    args = parser.parse_args()
    if RECORD.exists():
        previous = json.loads(RECORD.read_text())
        if not (args.delete and previous.get('mode') == 'dry_run'):
            raise RuntimeError('cleanup record already exists; refusing a second run')
    rows = []
    for relative, expected_hash, expected_size, reason in FILES:
        path = ROOT / 'output/experiments' / relative
        if not path.is_file():
            raise RuntimeError('expected checkpoint missing: {}'.format(path))
        if 'TBSI_E_MHSA_FULL_4090D_seed42' in str(path) or 'v1.0.1-' in str(path):
            raise RuntimeError('protected anchor path refused: {}'.format(path))
        actual_size = path.stat().st_size
        actual_hash = digest(path)
        if actual_size != expected_size or actual_hash != expected_hash:
            raise RuntimeError('checkpoint provenance changed: {}'.format(path))
        rows.append({'path': str(path.relative_to(ROOT)), 'size_bytes': actual_size,
                     'sha256_before_delete': actual_hash, 'reason': reason,
                     'status': 'PENDING' if args.delete else 'DRY_RUN'})
    record = {
        'id': 'M1-ABANDONED-CHECKPOINT-CLEANUP-20261022',
        'created_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'mode': 'delete' if args.delete else 'dry_run',
        'count': len(rows), 'bytes': sum(row['size_bytes'] for row in rows),
        'protected_assets_excluded': [
            'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar',
            'output/experiments/v1.0.1-单卡4090D-24G-等效全局128加速/checkpoints/TBSITrack_ep0015.pth.tar'],
        'experiment_reports_and_tracking_results_retained': True,
        'files': rows,
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    if args.delete:
        for row in record['files']:
            path = ROOT / row['path']
            path.unlink()
            row['status'] = 'DELETED'
            RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'record': str(RECORD.relative_to(ROOT)), 'mode': record['mode'],
                      'count': record['count'], 'bytes': record['bytes'],
                      'statuses': sorted(set(row['status'] for row in record['files']))},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
