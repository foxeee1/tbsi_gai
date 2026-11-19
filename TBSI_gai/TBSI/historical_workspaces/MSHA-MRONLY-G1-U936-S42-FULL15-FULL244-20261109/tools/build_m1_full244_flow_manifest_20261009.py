#!/usr/bin/env python3
"""Freeze the complete valid LasHeR test split in official dataset order."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.test.evaluation.lasherdataset import LasHeRDataset


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
    args = parser.parse_args()

    source = Path(args.per_sequence_csv)
    rows = {row['sequence']: set(filter(None, row['attributes'].split(';')))
            for row in csv.DictReader(source.open(newline=''))}
    sequences = LasHeRDataset(split='testingset').get_sequence_list()
    names = [sequence.name for sequence in sequences]
    if len(names) != 244 or len(set(names)) != 244:
        raise RuntimeError('expected 244 unique valid LasHeR test sequences; got {}'.format(len(names)))
    missing = sorted(set(names) - set(rows))
    extra = sorted(set(rows) - set(names))
    if missing or extra:
        raise RuntimeError('attribute manifest does not match test split: missing={}, extra={}'.format(
            missing, extra))

    attributes = {name: sorted(rows[name]) for name in names}
    counts = Counter(attr for name in names for attr in rows[name])
    canonical = '\n'.join('{}\t{}'.format(name, ';'.join(attributes[name])) for name in names) + '\n'
    payload = {
        'experiment_id': 'M1-FUSION-FLOW-FULL244-DIAG-20261009',
        'parent': 'TBSI_E_MHSA_FULL_4090D_seed42',
        'split': 'LasHeR testingset',
        'selection_rule': 'All 244 valid sequences returned by LasHeRDataset(testingset), in official dataset order; attributes attached for descriptive stratification; no outcomes used.',
        'sequence_count': len(names),
        'sequences': names,
        'sequence_attributes': attributes,
        'attribute_sequence_counts': dict(sorted(counts.items())),
        'source_attribute_table': str(source.resolve()),
        'source_attribute_table_sha256': sha256(source),
        'sequence_attribute_manifest_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    sequence_list = output.with_name('sequence_list.txt')
    sequence_list.write_text('\n'.join(names) + '\n')
    print(json.dumps({
        'experiment_id': payload['experiment_id'],
        'sequence_count': len(names),
        'attribute_counts': payload['attribute_sequence_counts'],
        'sequence_attribute_manifest_sha256': payload['sequence_attribute_manifest_sha256'],
        'manifest_sha256': sha256(output),
        'sequence_list': str(sequence_list.resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
