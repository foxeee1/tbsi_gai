#!/usr/bin/env python
"""Offline analysis of the message-recovery attribution probe."""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict

import numpy as np


CONDS = ('F', 'B', 'B_RGB_MESSAGE_RECOVERY', 'B_RAW_MESSAGE_HOOK',
         'B_TIR_PATH_CONTROL', 'B_RANDOM_MESSAGE_CONTROL')


def read_rows(path):
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['offset'] = int(row['offset']); row['iou'] = float(row['iou'])
    return rows


def mean(x): return float(np.mean(x)) if x else None
def median(x): return float(np.median(x)) if x else None


def bootstrap(seq_rows, field, replicates=5000):
    values = np.asarray([x[field] for x in seq_rows if np.isfinite(x[field])], dtype=float)
    if len(values) == 0:
        return {'n_sequences': 0, 'estimate': None, 'ci95': {'low': None, 'high': None}}
    rng = np.random.RandomState(42)
    sample = values[rng.randint(0, len(values), size=(replicates, len(values)))]
    out = sample.mean(axis=1)
    return {'n_sequences': int(len(values)), 'estimate': float(values.mean()),
            'ci95': {'low': float(np.quantile(out, .025)), 'high': float(np.quantile(out, .975))},
            'seed': 42, 'replicates': replicates}


def summarize(event_rows, field, replicates=5000):
    values = [x[field] for x in event_rows if np.isfinite(x[field])]
    order = np.argsort(np.asarray(values))[::-1]
    drop3 = [value for rank, value in enumerate(values) if rank >= min(3, len(values))]
    dropthird = [value for rank, value in enumerate(values) if rank >= int(math.ceil(len(values) / 3.0))]
    by_seq = defaultdict(list)
    for item in event_rows: by_seq[item['seq']].append(item)
    seq_rows = [{'seq': seq, field: mean([x[field] for x in vals])}
                for seq, vals in sorted(by_seq.items())]
    return {'n_events': len(values), 'n_sequences': len(seq_rows), 'event_mean': mean(values),
            'event_median': median(values), 'event_mean_drop_top_3': mean(drop3),
            'event_mean_drop_top_third': mean(dropthird),
            'sequence_cluster': bootstrap(seq_rows, field, replicates)}


def analyze(input_path, output_dir, replicates):
    rows = read_rows(input_path)
    if len(rows) != 72 * len(CONDS) * 6:
        raise ValueError('expected {} rows, got {}'.format(72 * len(CONDS) * 6, len(rows)))
    cells = {(x['event_id'], x['offset'], x['condition']): x for x in rows}
    derived = []
    for event_id in sorted(set(x['event_id'] for x in rows)):
        first = next(x for x in rows if x['event_id'] == event_id)
        for offset in range(6):
            get = lambda condition: cells[(event_id, offset, condition)]['iou']
            b = get('B')
            clean = get('F')
            item = {'event_id': event_id, 'seq': first['seq'], 'frame': int(first['frame']), 'offset': offset,
                    'iou_F': clean, 'iou_B': b,
                    'blur_loss': clean - b,
                    'R_recovery': get('B_RGB_MESSAGE_RECOVERY') - b,
                    'R_raw': get('B_RAW_MESSAGE_HOOK') - b,
                    'R_tir_control': get('B_TIR_PATH_CONTROL') - b,
                    'R_random': get('B_RANDOM_MESSAGE_CONTROL') - b,
                    'recovery_minus_random': get('B_RGB_MESSAGE_RECOVERY') - get('B_RANDOM_MESSAGE_CONTROL'),
                    'recovery_minus_tir': get('B_RGB_MESSAGE_RECOVERY') - get('B_TIR_PATH_CONTROL')}
            item['recovery_fraction'] = (item['R_recovery'] / item['blur_loss']
                                         if abs(item['blur_loss']) > 1e-12 else float('nan'))
            derived.append(item)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'event_summary.csv'), 'w', newline='') as handle:
        fields = list(derived[0].keys()); writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(derived)
    report = {'experiment': 'tbsi_message_recovery_v1',
              'input_csv': os.path.abspath(input_path),
              'input_sha256': hashlib.sha256(open(input_path, 'rb').read()).hexdigest(),
              'bootstrap': {'unit': 'sequence', 'seed': 42, 'replicates': replicates},
              'by_offset': {}, 'post_only': {}}
    fields = ('blur_loss', 'R_recovery', 'R_raw', 'R_tir_control', 'R_random',
              'recovery_minus_random', 'recovery_minus_tir', 'recovery_fraction')
    for offset in range(6):
        subset = [x for x in derived if x['offset'] == offset]
        report['by_offset'][str(offset)] = {field: summarize(subset, field, replicates) for field in fields}
    subset = [x for x in derived if x['offset'] in range(1, 6)]
    report['post_only'] = {field: summarize(subset, field, replicates) for field in fields}
    with open(os.path.join(output_dir, 'analysis.json'), 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    print('COMPLETED analysis events={} offsets=0..5'.format(len(set(x['event_id'] for x in derived))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True); ap.add_argument('--output-dir', required=True)
    ap.add_argument('--bootstrap-replicates', type=int, default=5000)
    args = ap.parse_args(); analyze(args.input, args.output_dir, args.bootstrap_replicates)


if __name__ == '__main__': main()
