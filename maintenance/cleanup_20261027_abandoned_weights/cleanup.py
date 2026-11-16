"""One-shot, explicit-list checkpoint cleanup authorized by the user.

No training, logs, reports, metrics or tracking outputs are removed.
"""
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/autodl-tmp/TBSI_gai/TBSI')
HERE = Path(__file__).resolve().parent
RELATIVE = [
    'EG_EVDR_P1_SOT_V3_CANDIDATE_S42_20261026/checkpoint_u0234.pth.tar',
    'EG_EVDR_P1_SOT_V3_CANDIDATE_S42_20261026/checkpoint_u0468.pth.tar',
    'EG_EVDR_TGRAD_P1_SOT_S42_20261027/checkpoint_u0234.pth.tar',
    'EG_EVDR_TGRAD_P1_SOT_S42_20261027/checkpoint_u0468.pth.tar',
    'EG_EARLY_SOURCE_EVIDENCE_SMOKE_S42_20261027/checkpoint_u0002.pth.tar',
    'EG_EARLY_SOURCE_EVIDENCE_P1_SOT_S42_20261027/checkpoint_u0234.pth.tar',
    'EG_EARLY_SOURCE_EVIDENCE_P1_SOT_S42_20261027/checkpoint_u0468.pth.tar',
    'EG_EARLY_SOURCE_EVIDENCE_P1_PAIRED_SOT_S42_20261027/checkpoint_u0234.pth.tar',
    'EG_EARLY_SOURCE_EVIDENCE_P1_PAIRED_SOT_S42_20261027/checkpoint_u0468.pth.tar',
    'EG_EVDR_ALIGNED_P1_SOT_S42_20261026/checkpoint_u0234.pth.tar',
    'EG_EVDR_ALIGNED_FULL15_SOT_S42_20261027/checkpoint_u3510.pth.tar',
    'EG_EVDR_P1_SOT_V3_CONTROL_S42_20261026/checkpoint_u0234.pth.tar',
]
TARGETS = [ROOT / 'output/experiments' / p for p in RELATIVE]
TABLES = [ROOT / 'experiments' / p for p in
          ('experiment_asset_registry.csv', '实验资产数据库.md')]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def is_weight(path):
    return path.name.endswith(('.pth.tar', '.pth', '.pt', '.ckpt', '.safetensors'))


def evidence_inventory():
    result = {}
    for folder in ('experiments', 'output'):
        for path in sorted((ROOT / folder).rglob('*')):
            if not path.is_file() or path in TABLES or is_weight(path):
                continue
            result[str(path.relative_to(ROOT))] = {
                'bytes': path.stat().st_size, 'sha256': digest(path)}
    return result


def weight_inventory():
    return {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size,
            'mtime_ns': p.stat().st_mtime_ns}
            for folder in ('output', 'pretrained_models')
            for p in (ROOT / folder).rglob('*') if p.is_file() and is_weight(p)}


def main():
    if (HERE / 'before.json').exists():
        raise RuntimeError('Cleanup manifest already exists; do not repeat blindly')
    for path in TARGETS:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Missing file or symlink: ' + str(path))
    rows = list(csv.DictReader(TABLES[0].open()))
    records = []
    for path in TARGETS:
        rel = str(path.relative_to(ROOT))
        records.append({'path': rel, 'bytes': path.stat().st_size,
                        'mtime_ns': path.stat().st_mtime_ns, 'sha256': digest(path),
                        'registry_references': [r['ID'] for r in rows
                            if rel in json.dumps(r, ensure_ascii=False)],
                        'reason': 'abandoned candidate' if 'ALIGNED' not in rel and
                            'CONTROL' not in rel else 'redundant intermediate; final retained'})
    protected = list((ROOT / 'output/主实验/实验').rglob('*.pth.tar'))
    protected += [ROOT / 'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar']
    protected += [ROOT / 'output/experiments' / p for p in (
        'EG_EVDR_P1_SOT_V3_CONTROL_S42_20261026/checkpoint_u0468.pth.tar',
        'EG_EVDR_ALIGNED_P1_SOT_S42_20261026/checkpoint_u0468.pth.tar',
        'EG_EVDR_ALIGNED_FULL15_SOT_S42_20261027/checkpoint_u7020.pth.tar')]
    protection = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    before = {'utc': datetime.now(timezone.utc).isoformat(), 'targets': records,
              'protected_sha256': protection, 'weights': weight_inventory(),
              'nonweight_evidence': evidence_inventory(),
              'free_bytes': shutil.disk_usage(ROOT).free}
    for path in TABLES:
        shutil.copy2(path, HERE / (path.name + '.before'))
    (HERE / 'before.json').write_text(json.dumps(before, ensure_ascii=False, indent=2))
    print('Manifest saved; deleting exactly', len(TARGETS), 'files', flush=True)
    for path in TARGETS:
        path.unlink()
    after_evidence = evidence_inventory()
    if after_evidence != before['nonweight_evidence']:
        raise RuntimeError('Nonweight evidence changed; inspect before.json')
    after_weights = weight_inventory()
    expected = {k: v for k, v in before['weights'].items()
                if k not in {str(p.relative_to(ROOT)) for p in TARGETS}}
    if after_weights != expected:
        raise RuntimeError('Unexpected weight inventory change')
    if any(digest(ROOT / k) != v for k, v in protection.items()):
        raise RuntimeError('Protected/final weight checksum mismatch')
    summary = {'utc': datetime.now(timezone.utc).isoformat(),
               'deleted_files': len(TARGETS),
               'deleted_bytes': sum(r['bytes'] for r in records),
               'free_bytes_before': before['free_bytes'],
               'free_bytes_after': shutil.disk_usage(ROOT).free,
               'nonweight_files_verified_unchanged': len(after_evidence),
               'protected_and_final_weights_verified': len(protection),
               'other_weights_inventory_unchanged': True,
               'training_or_evaluation_started': False}
    (HERE / 'after.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.path.insert(0, str(ROOT / 'tools'))
    from record_early_gate_p1_v2 import register
    for row in rows:
        referenced = [r for r in records if row['ID'] in r['registry_references']]
        if referenced:
            note = ('2026-10-27 user-authorized weight-only cleanup: ' +
                    ', '.join(r['path'] for r in referenced) +
                    '; original paths/hashes, metrics, logs, analyses and negative evidence retained. '
                    'Deleted checkpoints cannot be re-evaluated without retraining. '
                    'Manifest: ' + str(HERE / 'before.json'))
            register(row['ID'], {'Notes': row['Notes'] + ' | ' + note})
    register('EG-WEIGHT-CLEANUP-20261027', {
        'Date': '20261027', 'Parent': 'Early gate research asset maintenance',
        'CommitOrSnapshot': str(HERE / 'before.json'),
        'Dataset': 'NA; no data modified', 'EvalScope': 'File integrity only',
        'GPU': 'NA; no GPU job', 'HardwareConfig': 'CPU SHA256 and explicit file unlink only',
        'GPUHours': '0', 'AssumptionOrHypothesis': 'Release abandoned/intermediate checkpoints while preserving scientific evidence and protected/final references',
        'Conclusion': json.dumps(summary, ensure_ascii=False),
        'Status': 'MAINTENANCE-COMPLETE / WEIGHTS-ONLY / EVIDENCE-PRESERVED',
        'Evidence': str(HERE),
        'Notes': 'Original scientific statuses and metrics unchanged; two registry snapshots retained. No new efficacy conclusion.'})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
