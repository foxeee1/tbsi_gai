#!/usr/bin/env python3
"""Select matched M1 regression/gain events for causal message probing."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def numeric(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def enrich(rows):
    previous = {}
    for row in rows:
        seq = row['seq']
        frame = int(row['frame'])
        center = (numeric(row, 'gt_x') + numeric(row, 'gt_w') / 2.0,
                  numeric(row, 'gt_y') + numeric(row, 'gt_h') / 2.0)
        area = numeric(row, 'gt_w') * numeric(row, 'gt_h')
        prior = previous.get(seq)
        row['motion_speed'] = 0.0 if prior is None else float(np.hypot(center[0] - prior[0][0], center[1] - prior[0][1]))
        row['target_area'] = area
        row['baseline_iou'] = numeric(row, 'iou_c')
        row['abs_delta_iou'] = abs(numeric(row, 'delta_iou'))
        row['_frame'] = frame
        previous[seq] = (center, area)
    return rows


def choose(rows, label, count):
    pool = [row for row in rows if row['failure_label'] == label]
    if label == 'regression':
        pool = [row for row in pool if row['baseline_iou'] >= 0.35]
    else:
        pool = [row for row in pool if numeric(row, 'iou_m') >= 0.35 and row['baseline_iou'] >= 0.35]
    pool = [row for row in pool if not (numeric(row, 'iou_c') < 0.30 and numeric(row, 'iou_m') < 0.30)]
    if not pool:
        raise RuntimeError('no eligible {} events'.format(label))
    # Stratify by baseline IoU, target area, and motion while limiting each sequence.
    values = np.array([[row['baseline_iou'], row['target_area'], row['motion_speed']] for row in pool], dtype=float)
    scale = np.std(values, axis=0); scale[scale < 1e-8] = 1.0
    center = np.mean(values, axis=0)
    for row, value in zip(pool, values):
        row['_distance'] = float(np.linalg.norm((value - center) / scale))
    pool.sort(key=lambda row: (-row['abs_delta_iou'], row['seq'], row['_frame']))
    selected, per_sequence = [], defaultdict(int)
    # First preserve high-effect diversity, then fill from remaining strata.
    for row in pool:
        if per_sequence[row['seq']] >= 2:
            continue
        selected.append(row); per_sequence[row['seq']] += 1
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError('only {} eligible {} events after sequence cap'.format(len(selected), label))
    return selected


def write_csv(path, rows):
    fields = list(rows[0])
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--events-per-group', type=int, default=24)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    with open(args.pairs, newline='') as handle:
        rows = enrich(list(csv.DictReader(handle)))
    regressions = choose(rows, 'hard_regression', args.events_per_group)
    # The requested hard-regression label is represented by the strict threshold.
    gains = choose(rows, 'gain', args.events_per_group)
    for index, row in enumerate(regressions): row['event_id'] = 'R{:03d}'.format(index + 1); row['group'] = 'regression'
    # Rebuild gain choices by nearest standardized baseline/area/motion match
    # against the selected regressions, rather than pairing two preselected extremes.
    gain_pool = [row for row in rows if row['failure_label'] == 'gain' and row['baseline_iou'] >= 0.35
                 and numeric(row, 'iou_m') >= 0.35 and not (numeric(row, 'iou_c') < 0.30 and numeric(row, 'iou_m') < 0.30)]
    all_values = np.array([[row['baseline_iou'], row['target_area'], row['motion_speed']]
                           for row in regressions + gain_pool], dtype=float)
    scales = np.std(all_values, axis=0); scales[scales < 1e-8] = 1.0
    matched = []; selected_gains = []; used = set(); per_sequence = defaultdict(int)
    for reg in regressions:
        candidates = [row for row in gain_pool if id(row) not in used and per_sequence[row['seq']] < 2]
        if not candidates: break
        def distance(gain):
            return float(np.linalg.norm(np.array([reg['baseline_iou'] - gain['baseline_iou'],
                                                   reg['target_area'] - gain['target_area'],
                                                   reg['motion_speed'] - gain['motion_speed']]) / scales))
        gain = min(candidates, key=distance); used.add(id(gain)); per_sequence[gain['seq']] += 1
        gain['event_id'] = 'G{:03d}'.format(len(selected_gains) + 1); gain['group'] = 'gain'
        selected_gains.append(gain)
        matched.append({'pair_id': 'P{:03d}'.format(len(matched) + 1),
                        'regression_event_id': reg['event_id'], 'gain_event_id': gain['event_id'],
                        'baseline_iou_abs_diff': abs(reg['baseline_iou'] - gain['baseline_iou']),
                        'target_area_abs_diff': abs(reg['target_area'] - gain['target_area']),
                        'motion_abs_diff': abs(reg['motion_speed'] - gain['motion_speed'])})
    if len(matched) != args.events_per_group:
        raise RuntimeError('only {} matched event pairs available'.format(len(matched)))
    gains = selected_gains
    for index, row in enumerate(gains): row['event_id'] = 'G{:03d}'.format(index + 1); row['group'] = 'gain'
    combined = regressions + gains
    for row in combined:
        row.pop('_frame', None); row.pop('_distance', None)
    write_csv(output / 'regression_events.csv', regressions)
    write_csv(output / 'gain_events.csv', gains)
    write_csv(output / 'matched_events.csv', matched)
    with open(output / 'selection_rules.json', 'w') as handle:
        json.dump({'experiment': 'MHSA-CAUSAL-PROBE-V2', 'events_per_group': args.events_per_group,
                   'regression_candidate': 'delta_iou <= -0.10 and iou_c >= 0.35',
                   'gain_candidate': 'delta_iou >= 0.05 and iou_m >= 0.35 and iou_c >= 0.35',
                   'common_failure_excluded': True, 'max_events_per_sequence': 2,
                   'pairing_features': ['baseline_iou', 'target_area', 'motion_speed'],
                   'source_pairs': str(Path(args.pairs).resolve())}, handle, indent=2)
    print(json.dumps({'regression': len(regressions), 'gain': len(gains), 'matched': len(matched)}))


if __name__ == '__main__':
    main()
