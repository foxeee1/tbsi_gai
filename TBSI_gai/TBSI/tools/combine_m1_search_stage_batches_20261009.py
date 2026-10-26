#!/usr/bin/env python3
"""Combine disjoint train-side diagnostic cohorts before held-out probing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', action='append', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if len(args.batch) < 2:
        raise RuntimeError('at least two batches required')
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing nonempty output directory')
    output.mkdir(parents=True)
    arrays, rows, seen = [], [], set()
    provenance = []
    for batch_name in args.batch:
        batch = Path(batch_name).resolve()
        summary = json.loads((batch / 'summary.json').read_text())
        if summary['status'] != 'STAGE-CAPTURE-COMPLETE':
            raise RuntimeError('incomplete stage batch: ' + str(batch))
        features = np.load(str(batch / 'selected_features.npz'), allow_pickle=False)
        names = set(features['sequence'].astype(str).tolist())
        if names & seen:
            raise RuntimeError('overlapping sequence members: ' + repr(names & seen))
        seen.update(names)
        arrays.append(features)
        with (batch / 'per_arm.csv').open(newline='') as stream:
            rows.extend(csv.DictReader(stream))
        provenance.append({'batch': str(batch), 'summary_sha256': sha256(batch / 'summary.json'),
                           'features_sha256': sha256(batch / 'selected_features.npz'),
                           'sequences': sorted(names)})
    keys = ('rgb', 'tir', 'concat', 'fused', 'target', 'background',
            'gt_xy', 'sequence', 'frame', 'label')
    np.savez_compressed(output / 'selected_features.npz', **{
        key: np.concatenate([part[key] for part in arrays], axis=0) for key in keys})
    fields = sorted({key for row in rows for key in row})
    with (output / 'per_arm.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {'status': 'COMBINED', 'sequences': len(seen),
               'baseline_frames': sum(row['arm'] == 'baseline' for row in rows),
               'severe_onset_frames': sum(row['arm'] == 'baseline' and
                                          row['label'] == 'severe_onset' for row in rows),
               'members': sorted(seen), 'provenance': provenance}
    (output / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
