#!/usr/bin/env python
"""Offline analysis for the frozen new-sequence RGB-blur revalidation."""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict

import numpy as np


def read_rows(path):
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['offset'] = int(row['offset'])
        row['iou'] = float(row['iou'])
    return rows


def mean(values):
    return float(np.mean(values)) if values else None


def median(values):
    return float(np.median(values)) if values else None


def bootstrap(sequence_rows, field, seed=42, replicates=5000):
    values = np.asarray([row[field] for row in sequence_rows], dtype=float)
    if len(values) == 0:
        return {'n_sequences': 0, 'estimate': None, 'ci95': {'low': None, 'high': None}}
    rng = np.random.RandomState(seed)
    sample = values[rng.randint(0, len(values), size=(replicates, len(values)))]
    averages = sample.mean(axis=1)
    return {'n_sequences': int(len(values)), 'estimate': float(values.mean()),
            'ci95': {'low': float(np.quantile(averages, 0.025)),
                     'high': float(np.quantile(averages, 0.975))},
            'seed': seed, 'replicates': replicates}


def summarize(event_rows, field='drop'):
    values = [row[field] for row in event_rows]
    order = np.argsort(np.asarray(values))[::-1]
    top3_removed = [value for rank, value in enumerate(values) if rank not in set(order[:3])]
    top_third_removed = [value for rank, value in enumerate(values)
                         if rank not in set(order[:int(math.ceil(len(values) / 3.0))])]
    grouped = defaultdict(list)
    for row in event_rows:
        grouped[row['seq']].append(row)
    seq_rows = []
    for seq, values_by_seq in sorted(grouped.items()):
        seq_rows.append({'seq': seq, field: mean([x[field] for x in values_by_seq]),
                         'n_events': len(values_by_seq)})
    return {'n_events': len(event_rows), 'n_sequences': len(seq_rows),
            'event_mean': mean(values), 'event_median': median(values),
            'event_mean_drop_top_3': mean(top3_removed),
            'event_mean_drop_top_third': mean(top_third_removed),
            'sequence_cluster': bootstrap(seq_rows, field)}


def analyze(input_path, output_dir, replicates):
    rows = read_rows(input_path)
    if len(rows) != 72 * 2 * 6:
        raise ValueError('expected 864 rows, got {}'.format(len(rows)))
    keys = {(r['event_id'], r['offset']) for r in rows}
    if len(keys) != 72 * 6 or {r['condition'] for r in rows} != {'00', '10'}:
        raise ValueError('invalid event/condition coverage')
    cells = {(r['event_id'], r['offset']): r for r in rows}
    events = []
    for event_id in sorted({r['event_id'] for r in rows}):
        event_rows = [r for r in rows if r['event_id'] == event_id]
        first = event_rows[0]
        for offset in range(6):
            a, b = cells[(event_id, offset)], None
            for candidate in event_rows:
                if candidate['offset'] == offset and candidate['condition'] == '10':
                    b = candidate
                    break
            clean = next(candidate for candidate in event_rows
                         if candidate['offset'] == offset and candidate['condition'] == '00')
            events.append({'event_id': event_id, 'seq': first['seq'], 'frame': int(first['frame']),
                           'offset': offset, 'size_bin': first['size_bin'],
                           'gt_area_frac': float(first['gt_area_frac']),
                           'iou_00': clean['iou'], 'iou_10': b['iou'],
                           'drop': clean['iou'] - b['iou']})
    os.makedirs(output_dir, exist_ok=True)
    event_fields = list(events[0].keys())
    with open(os.path.join(output_dir, 'event_summary.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields); writer.writeheader(); writer.writerows(events)
    sequence_rows = []
    grouped = defaultdict(list)
    for row in events:
        grouped[(row['offset'], row['seq'])].append(row)
    for (offset, seq), values in sorted(grouped.items()):
        sequence_rows.append({'offset': offset, 'seq': seq, 'n_events': len(values),
                              'drop': mean([x['drop'] for x in values]),
                              'iou_00': mean([x['iou_00'] for x in values]),
                              'iou_10': mean([x['iou_10'] for x in values])})
    with open(os.path.join(output_dir, 'sequence_summary.csv'), 'w', newline='') as handle:
        fields = list(sequence_rows[0].keys()); writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(sequence_rows)
    report = {'experiment': 'single_modality_blur_v2',
              'input_csv': os.path.abspath(input_path),
              'input_sha256': hashlib.sha256(open(input_path, 'rb').read()).hexdigest(),
              'bootstrap': {'unit': 'sequence', 'seed': 42, 'replicates': replicates},
              'all_events': {}, 'by_offset': {}, 'by_size': {}}
    for offset in range(6):
        subset = [x for x in events if x['offset'] == offset]
        report['by_offset'][str(offset)] = summarize(subset)
    for label in ('all_events', 'small', 'medium', 'large'):
        subset = events if label == 'all_events' else [x for x in events if x['size_bin'] == label]
        report['by_size'][label] = {'offset_0': summarize([x for x in subset if x['offset'] == 0]),
                                    'post_only': summarize([x for x in subset if x['offset'] in range(1, 6)])}
    with open(os.path.join(output_dir, 'analysis.json'), 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print('COMPLETED analysis events={} sequences={}'.format(len(set(x['event_id'] for x in events)),
                                                              len(set(x['seq'] for x in events))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--bootstrap-replicates', type=int, default=5000)
    args = ap.parse_args()
    analyze(args.input, args.output_dir, args.bootstrap_replicates)


if __name__ == '__main__':
    main()
