#!/usr/bin/env python
"""Offline P1 gates for exact TCSI support records."""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict

import numpy as np


FIELDS = ('shared_ratio', 'rgb_complementary_ratio', 'tir_complementary_ratio',
          'union_ratio', 'rgb_fg_enrichment', 'tir_fg_enrichment',
          'shared_fg_enrichment', 'rgb_comp_fg_enrichment', 'tir_comp_fg_enrichment',
          'random_fg_enrichment')


def read_rows(path):
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in FIELDS + ('layer', 'topk', 'frame'):
            row[field] = float(row[field])
        row['layer'] = int(row['layer'])
        row['topk'] = int(row['topk'])
    return rows


def mean(rows, field):
    return float(np.nanmean([row[field] for row in rows]))


def bootstrap_difference(rows, field, random_field='random_fg_enrichment', seed=42, reps=2000):
    values = np.asarray([row[field] - row[random_field] for row in rows], dtype=np.float64)
    groups = np.asarray([row['event_id'] for row in rows])
    unique = np.unique(groups)
    per_event = np.asarray([values[groups == item].mean() for item in unique])
    rng = np.random.RandomState(seed)
    draws = per_event[rng.randint(0, len(per_event), size=(reps, len(per_event)))].mean(axis=1)
    return {'mean_difference': float(per_event.mean()), 'ci95_low': float(np.quantile(draws, 0.025)),
            'ci95_high': float(np.quantile(draws, 0.975)), 'n_events': int(len(unique))}


def trimmed_mean(rows, field, fraction=0.10):
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    if len(values) < 4:
        return float(values.mean())
    keep = max(1, int(np.floor(len(values) * (1.0 - fraction))))
    order = np.argsort(np.abs(values - np.median(values)))
    return float(values[order[:keep]].mean())


def run(args):
    rows = read_rows(args.summary)
    if not rows:
        raise ValueError('empty support summary')
    meta = json.load(open(args.metadata))
    if meta.get('coordinate_space') != 'common 16x16 search grid; both directions are s2t with 256 search keys':
        raise ValueError('coordinate-space contract is not valid')
    report = {'experiment': 'tcsi_support_probe_v1_analysis',
              'summary': os.path.abspath(args.summary),
              'summary_sha256': hashlib.sha256(open(args.summary, 'rb').read()).hexdigest(),
              'metadata': os.path.abspath(args.metadata), 'coordinate_space_valid': True,
              'groups': {}, 'gates': {}}
    for topk in sorted(set(row['topk'] for row in rows)):
        for layer in sorted(set(row['layer'] for row in rows)):
            subset = [row for row in rows if row['topk'] == topk and row['layer'] == layer]
            key = 'k{}_layer{}'.format(topk, layer)
            report['groups'][key] = {'n_rows': len(subset), 'n_events': len(set(row['event_id'] for row in subset)),
                                     'shared_ratio': mean(subset, 'shared_ratio'),
                                     'rgb_complementary_ratio': mean(subset, 'rgb_complementary_ratio'),
                                     'tir_complementary_ratio': mean(subset, 'tir_complementary_ratio'),
                                     'union_ratio': mean(subset, 'union_ratio'),
                                     'rgb_fg_enrichment': mean(subset, 'rgb_fg_enrichment'),
                                     'tir_fg_enrichment': mean(subset, 'tir_fg_enrichment'),
                                     'shared_fg_enrichment': mean(subset, 'shared_fg_enrichment'),
                                     'rgb_comp_fg_enrichment': mean(subset, 'rgb_comp_fg_enrichment'),
                                     'tir_comp_fg_enrichment': mean(subset, 'tir_comp_fg_enrichment'),
                                     'random_fg_enrichment': mean(subset, 'random_fg_enrichment'),
                                     'rgb_random_difference': bootstrap_difference(subset, 'rgb_fg_enrichment'),
                                     'tir_random_difference': bootstrap_difference(subset, 'tir_fg_enrichment'),
                                     'rgb_comp_random_difference': bootstrap_difference(subset, 'rgb_comp_fg_enrichment'),
                                     'tir_comp_random_difference': bootstrap_difference(subset, 'tir_comp_fg_enrichment'),
                                     'trimmed_rgb_fg_enrichment': trimmed_mean(subset, 'rgb_fg_enrichment'),
                                     'trimmed_tir_fg_enrichment': trimmed_mean(subset, 'tir_fg_enrichment'),
                                     'trimmed_rgb_comp_fg_enrichment': trimmed_mean(subset, 'rgb_comp_fg_enrichment'),
                                     'trimmed_tir_comp_fg_enrichment': trimmed_mean(subset, 'tir_comp_fg_enrichment')}
    layers = sorted(set(row['layer'] for row in rows))
    topks = sorted(set(row['topk'] for row in rows))
    valid_enrichment_layers = []
    valid_comp_layers = []
    nondegenerate = []
    random_beats = []
    for layer in layers:
        groups = [report['groups']['k{}_layer{}'.format(k, layer)] for k in topks]
        best = max(groups, key=lambda item: item['rgb_fg_enrichment'] + item['tir_fg_enrichment'])
        if best['rgb_fg_enrichment'] > 1.2 and best['tir_fg_enrichment'] > 1.2:
            valid_enrichment_layers.append(layer)
        if best['rgb_comp_fg_enrichment'] > 1.0 and best['tir_comp_fg_enrichment'] > 1.0:
            valid_comp_layers.append(layer)
        for item in groups:
            if 0.05 < item['shared_ratio'] < 0.95:
                nondegenerate.append('k{}_layer{}'.format(topks[groups.index(item)], layer))
            if item['rgb_random_difference']['ci95_low'] > 0 and item['tir_random_difference']['ci95_low'] > 0:
                random_beats.append('k{}_layer{}'.format(topks[groups.index(item)], layer))
    report['gates'] = {
        'coordinate_space': True,
        'foreground_enrichment_two_layers': len(valid_enrichment_layers) >= 2,
        'foreground_enrichment_layers': valid_enrichment_layers,
        'complementary_foreground_enrichment': len(valid_comp_layers) >= 1,
        'complementary_enrichment_layers': valid_comp_layers,
        'shared_ratio_nondegenerate': len(nondegenerate) > 0,
        'nondegenerate_groups': nondegenerate,
        'beats_random_support': len(random_beats) > 0,
        'random_support_groups': random_beats,
    }
    report['decision'] = 'P1_PASS_TRAINING_ELIGIBLE' if all((report['gates']['coordinate_space'],
        report['gates']['foreground_enrichment_two_layers'], report['gates']['complementary_foreground_enrichment'],
        report['gates']['shared_ratio_nondegenerate'], report['gates']['beats_random_support'])) else 'P1_FAIL_STOP'
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'support_analysis.json'), 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, 'support_sequence_summary.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['event_id', 'seq', 'layer', 'topk', 'shared_ratio', 'rgb_fg_enrichment', 'tir_fg_enrichment', 'rgb_comp_fg_enrichment', 'tir_comp_fg_enrichment', 'random_fg_enrichment'])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
