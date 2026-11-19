"""Offline audit of existing MR tables; never loads a model or runs a tracker."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'output/experiments/MSHA-B0-U936-S43-FULL15-FULL244-20261112/analysis/two_seed_failure_20261113/per_sequence_two_seed.csv'
OUT = ROOT / 'experiments/reports/mr_shared_bottleneck_audit_20261114'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 244 or len({r['sequence'] for r in rows}) != 244:
        raise ValueError('Expected the existing unique 244-sequence table')
    summary = []
    for attr in (k for k in rows[0] if k.startswith('attr_')):
        selected = [r for r in rows if int(r[attr]) == 1]
        record = {'attribute': attr[5:], 'n': len(selected)}
        for seed in (42, 43):
            delta = np.array([float(r['seed%d_delta_auc' % seed]) for r in selected])
            ordered = np.sort(delta)
            trim = len(delta) // 10
            prefix = 'seed%d_' % seed
            record.update({
                prefix + 'mean_auc_pp': float(delta.mean()),
                prefix + 'median_auc_pp': float(np.median(delta)),
                prefix + 'trim10_mean_auc_pp': float(ordered[trim:-trim].mean() if trim else delta.mean()),
                prefix + 'mean_without_best_sequence_pp': float(ordered[:-1].mean()),
                prefix + 'wins': int((delta > 1e-9).sum()),
                prefix + 'ties': int((np.abs(delta) <= 1e-9).sum()),
                prefix + 'losses': int((delta < -1e-9).sum()),
                prefix + 'best_sequence': selected[int(delta.argmax())]['sequence'],
            })
        summary.append(record)
    with (OUT / 'attribute_existing_seed42_seed43.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    provenance = {
        'scope': 'CPU offline reaggregation; no training, tracker inference, checkpoint loading or new benchmark',
        'source': str(SOURCE),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sequences': len(rows),
        'attributes': len(summary),
        'units': 'AUC percentage points; unweighted sequence means',
        'trim_definition': 'remove floor(0.1*n) sequences from each tail separately for each seed',
        'limitations': [
            'Seed43 MR/B0 common initialization is unmatched; descriptive comparison only.',
            'Attribute slices overlap and were inspected after outcomes; no independent confirmation.',
            'Removing best sequence and trimming are influence diagnostics, not replacement benchmark scores.',
            'No new bootstrap or hypothesis tests; no training-seed uncertainty estimated.',
        ],
    }
    (OUT / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'sequences': len(rows), 'attributes': len(summary), 'output': str(OUT)}))


if __name__ == '__main__':
    main()
