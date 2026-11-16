#!/usr/bin/env python3
"""Lock a balanced 30-sequence MR delta-strength development cohort."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired-csv', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    source = Path(args.paired_csv).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(source.open(newline='')))
    if len(rows) != 244:
        raise ValueError('Expected the locked full244 paired table, got %d rows' % len(rows))
    for row in rows:
        row['delta_auc_value'] = float(row['delta_auc'])

    groups = {
        'improved': [r for r in rows if r['delta_auc_value'] > 0.25],
        'regressed': [r for r in rows if r['delta_auc_value'] < -0.25],
        'neutral': [r for r in rows if abs(r['delta_auc_value']) <= 0.25],
    }
    selected = []
    summaries = {}
    for name, members in groups.items():
        if len(members) < 10:
            raise ValueError('Insufficient members in %s: %d' % (name, len(members)))
        ordered = sorted(members, key=lambda r: (r['delta_auc_value'], r['sequence']))
        median = (ordered[(len(ordered) - 1) // 2]['delta_auc_value'] +
                  ordered[len(ordered) // 2]['delta_auc_value']) / 2.0
        chosen = sorted(members, key=lambda r: (abs(r['delta_auc_value'] - median), r['sequence']))[:10]
        chosen = sorted(chosen, key=lambda r: r['sequence'])
        for row in chosen:
            selected.append((name, row))
        summaries[name] = {
            'available': len(members), 'selected': len(chosen),
            'full_group_median_delta_auc_pp': median,
            'selected_sequences': [r['sequence'] for r in chosen],
            'selected_delta_auc_pp': [r['delta_auc_value'] for r in chosen],
        }

    selected.sort(key=lambda item: (item[0], item[1]['sequence']))
    manifest = out / 'development_30_sequences.txt'
    manifest.write_text(''.join(row['sequence'] + '\n' for _, row in selected))
    metadata = {
        'status': 'LOCKED DEVELOPMENT SCREEN; NOT INDEPENDENT CONFIRMATION',
        'source_run': 'MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109',
        'source_pair_table': str(source),
        'source_pair_table_sha256': sha(source),
        'selection': {
            'unit': 'sequence', 'metric': 'per-sequence AUC difference, MR minus matched B0, in percentage points',
            'improved': 'delta_auc > +0.25 pp', 'regressed': 'delta_auc < -0.25 pp',
            'neutral': 'abs(delta_auc) <= 0.25 pp',
            'sampling': '10 per stratum nearest the full-stratum median delta; ties by sequence name',
            'seed': 'deterministic; no RNG',
        },
        'group_counts': {name: len(members) for name, members in groups.items()},
        'groups': summaries,
        'manifest': str(manifest), 'manifest_sha256': sha(manifest),
        'warning': 'The same historical outcomes define this development cohort; do not report its mean as full-dataset performance or independent confirmation.',
    }
    (out / 'cohort_manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
