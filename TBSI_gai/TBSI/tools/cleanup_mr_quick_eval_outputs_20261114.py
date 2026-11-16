#!/usr/bin/env python3
"""Remove retained raw evaluation files from completed MR quick screens."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/root/autodl-tmp/TBSI_gai/TBSI')
RUNS = [
    'MSHA-MR-LOCAL-RESIDUAL-QUICK-S42-20261113',
    'MSHA-MR-LOCAL-RESIDUAL-QUICK-S42-20261113-RETRY1',
    'MSHA-MR-LOCAL-RESIDUAL-QUICK-S42-20261113-RETRY2',
    'MSHA-MR-RADIAL-AMPLITUDE-QUICK-S42-20261113',
    'MSHA-MR-RADIAL-AMPLITUDE-QUICK-S42-20261113-RETRY1',
    'MSHA-MR-SELECTIVE-GATE-QUICK-S42-20261113',
    'MSHA-MR-SELECTIVE-GATE-QUICK-RETRY1-S42-20261113',
    'MSHA-MR-AGREEMENT-DELTA-GATE-QUICK-S42-20261113',
    'MSHA-MR-ORTHO-DIRECTION-QUICK-S42-20261113',
]
MANIFEST = ROOT / 'experiments/reports/mr_quick_eval_outputs_cleanup_20261114.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    experiments = ROOT / 'output/experiments'
    files = []
    for run in RUNS:
        run_root = experiments / run
        for folder in ('eval', 'eval_control', 'eval_candidate'):
            eval_root = run_root / folder
            if not eval_root.exists():
                continue
            for path in sorted(eval_root.rglob('*.txt')):
                if path.is_file():
                    files.append({
                        'run': run,
                        'path': str(path.relative_to(ROOT)),
                        'size_bytes': path.stat().st_size,
                        'sha256': sha256(path),
                    })

    manifest = {
        'cleanup_id': 'MR-QUICK-EVAL-OUTPUTS-CLEANUP-20261114',
        'authorized_by_user': True,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'Remove raw, reproducible per-sequence predictions and timing files from completed MR quick train/test screens; retain scalar/per-sequence analyses, reports, contracts, logs, and failure evidence.',
        'runs': RUNS,
        'deleted_count': len(files),
        'deleted_bytes': sum(item['size_bytes'] for item in files),
        'deleted': files,
        'retained': [
            'analysis/REPORT.md', 'analysis/per_sequence.csv',
            'analysis/paired_metrics.json', 'analysis/*summary*.json',
            'contracts/', 'train/*/completed.json', 'train/*/updates.jsonl',
            'train/*/batches.jsonl', 'failure and preflight records',
            'experiment asset registry rows and original metrics',
        ],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + '\n')

    for item in files:
        path = ROOT / item['path']
        if sha256(path) != item['sha256']:
            raise RuntimeError('file changed after manifest capture: ' + item['path'])
        path.unlink()

    print(json.dumps({
        'manifest': str(MANIFEST),
        'deleted_count': len(files),
        'deleted_bytes': manifest['deleted_bytes'],
    }, indent=2))


if __name__ == '__main__':
    main()
