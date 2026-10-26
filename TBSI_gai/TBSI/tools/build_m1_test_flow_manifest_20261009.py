#!/usr/bin/env python3
"""Select a deterministic, attribute-diverse test subset without using outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-sequence-csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=30)
    args = parser.parse_args()

    source = Path(args.per_sequence_csv)
    rows = {}
    with source.open(newline='') as stream:
        for row in csv.DictReader(stream):
            name = row['sequence']
            attrs = set(filter(None, row['attributes'].split(';')))
            rows[name] = attrs
    if args.count < 1 or args.count > len(rows):
        raise ValueError('count must be in [1, {}]'.format(len(rows)))

    # Greedy weighted set cover: rare labels receive priority; ties use sequence name.
    selected = []
    covered = set()
    remaining = set(rows)
    while len(selected) < args.count:
        def score(name):
            return sum(1.0 / (1.0 + sum(attr in rows[item] for item in selected))
                       for attr in rows[name] - covered)

        best = min(remaining, key=lambda name: (-score(name), name))
        selected.append(best)
        covered.update(rows[best])
        remaining.remove(best)

    coverage = {attr: sum(attr in rows[name] for name in selected)
                for attr in sorted(covered)}
    canonical = '\n'.join('{}\t{}'.format(name, ';'.join(sorted(rows[name])))
                          for name in selected) + '\n'
    payload = {
        'selection_id': 'M1-FUSION-FLOW-TESTSET-MANIFEST-20261009',
        'parent': 'TBSI_E_MHSA_FULL_4090D_seed42',
        'split': 'LasHeR testingset',
        'selection_rule': 'Greedy weighted attribute coverage; only sequence names and official attributes read; ties by sequence name; no predictions or performance metrics used.',
        'requested_sequences': args.count,
        'selected_sequences': selected,
        'selected_sequence_attributes': {name: sorted(rows[name]) for name in selected},
        'attribute_sequence_counts': coverage,
        'source_per_sequence_csv': str(source.resolve()),
        'source_per_sequence_csv_sha256': sha256(source),
        'sequence_attribute_manifest_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: payload[k] for k in ('selection_id', 'requested_sequences',
                                               'selected_sequences', 'attribute_sequence_counts',
                                               'sequence_attribute_manifest_sha256')},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
