#!/usr/bin/env python3
"""Analyze event-level cross-modal intervention effects."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean(values):
    return float(np.mean(values)) if values else None


def bootstrap(values, seed=42, n=5000):
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=float)
    samples = [float(np.mean(values[rng.randint(0, len(values), len(values))])) for _ in range(n)]
    return {'mean': float(np.mean(samples)), 'lower': float(np.percentile(samples, 2.5)),
            'upper': float(np.percentile(samples, 97.5))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--events-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.results)))
    event_meta = {}
    for filename in ('regression_events.csv', 'gain_events.csv'):
        for row in csv.DictReader(open(Path(args.events_dir) / filename)):
            event_meta[row['event_id']] = row
    event_condition = {}
    for row in rows:
        key = (row['event_id'], row['condition'])
        event_condition.setdefault(key, row)
    groups = defaultdict(list)
    for (event_id, condition), row in event_condition.items():
        groups[(event_meta[event_id]['group'], condition)].append(float(row['iou']))
    summary = []
    for (group, condition), values in sorted(groups.items()):
        summary.append({'group': group, 'condition': condition, 'events': len(values), 'mean_iou': mean(values)})
    effects = []
    for group in ('regression', 'gain'):
        event_ids = [event_id for event_id, meta in event_meta.items() if meta['group'] == group]
        for intervention in ('C_OFF', 'C_HALF', 'C_1.5'):
            values = [float(event_condition[(event_id, intervention)]['iou']) - float(event_condition[(event_id, 'F')]['iou'])
                      for event_id in event_ids]
            effects.append({'group': group, 'condition': intervention, 'events': len(values),
                            'mean_effect': mean(values), 'median_effect': float(np.median(values)),
                            'bootstrap': bootstrap(values)})
    differences = []
    for condition in ('C_OFF', 'C_HALF', 'C_1.5'):
        reg = next(item for item in effects if item['group'] == 'regression' and item['condition'] == condition)
        gain = next(item for item in effects if item['group'] == 'gain' and item['condition'] == condition)
        differences.append({'condition': condition, 'regression_minus_gain': reg['mean_effect'] - gain['mean_effect']})
    message = defaultdict(list)
    for row in rows:
        group = event_meta[row['event_id']]['group']
        for field in ('rgb_self_norm', 'rgb_cross_norm', 'tir_self_norm', 'tir_cross_norm',
                      'rgb_cross_self_ratio', 'tir_cross_self_ratio', 'output_change_norm'):
            message[(group, row['condition'], int(row['layer']), field)].append(float(row[field]))
    message_summary = []
    for (group, condition, layer, field), values in sorted(message.items()):
        message_summary.append({'group': group, 'condition': condition, 'layer': layer,
                                'metric': field, 'mean': mean(values)})
    result = {'experiment': 'MHSA-CAUSAL-PROBE-V2', 'rows': len(rows),
              'events': len(event_meta), 'conditions': sorted(set(row['condition'] for row in rows)),
              'summary': summary, 'effects': effects, 'group_effect_differences': differences,
              'message_summary': message_summary}
    # F must be one state-consistent observation per event; all layers share the same input hash.
    state_hashes = defaultdict(set)
    for row in rows: state_hashes[row['event_id']].add(row['state_hash'])
    result['state_hash_unique_per_event'] = all(len(values) == 1 for values in state_hashes.values())
    result['finite_values'] = all(np.isfinite(float(row[field])) for row in rows
                                  for field in ('iou', 'rgb_self_norm', 'rgb_cross_norm', 'tir_self_norm',
                                                'tir_cross_norm', 'rgb_cross_self_ratio', 'tir_cross_self_ratio',
                                                'output_change_norm'))
    with open(output / 'causal_probe_summary.json', 'w') as handle:
        json.dump(result, handle, indent=2)
    with open(output / 'condition_effects.csv', 'w', newline='') as handle:
        fields = list(effects[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(effects)
    print(json.dumps({'rows': len(rows), 'events': len(event_meta), 'finite': result['finite_values'],
                      'state_hash_unique': result['state_hash_unique_per_event'],
                      'effects': differences}, ensure_ascii=False))


if __name__ == '__main__': main()
