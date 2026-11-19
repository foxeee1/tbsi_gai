#!/usr/bin/env python3
"""Summarize paired T/S fixed-state interventions against M1 baseline."""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output/experiments/M1_MESSAGE_POSITION_LEVERAGE_20261012'
INPUTS = [RUN / 'fixed_state/per_state.csv',
          RUN / 'fixed_state_matched/per_state.csv']
OUTPUT = RUN / 'analysis/summary.json'
METRICS = ('window_target_bg_margin', 'window_gt_rank',
           'gt_cell_decoded_center_distance_tokens',
           'gt_cell_size_l1_tokens', 'iou', 'center_error_px')


def median(values):
    return statistics.median(values) if values else None


def main():
    rows = []
    for path in INPUTS:
        with path.open(newline='') as stream:
            rows.extend(csv.DictReader(stream))
    baseline = {}
    for row in rows:
        if row['site'] == 'baseline':
            baseline.setdefault((row['sequence'], row['frame']), row)
    # The matched run repeats baseline/T arms; retain one row per condition.
    conditions = {}
    for row in rows:
        key = (row['sequence'], row['frame'], row['site'], row['scale'])
        conditions.setdefault(key, row)

    labels = Counter(row['state'] for row in baseline.values())
    summaries = []
    for site, scale in (('T', '0.0'), ('T', '0.95'), ('T', '1.05'),
                        ('S', '0.0'), ('S', '0.9'), ('S', '1.1')):
        arm = [row for row in conditions.values()
               if row['site'] == site and row['scale'] == scale]
        deltas = []
        for row in arm:
            parent = baseline[(row['sequence'], row['frame'])]
            delta = {metric: float(row[metric]) - float(parent[metric])
                     for metric in METRICS}
            peak_changed = ((row['window_peak_x'], row['window_peak_y']) !=
                            (parent['window_peak_x'], parent['window_peak_y']))
            deltas.append({
                'sequence': row['sequence'], 'state': row['state'],
                'delta': delta, 'peak_changed': peak_changed,
                'message_delta_ratio': float(row['message_delta_ratio']),
            })
        if not deltas:
            continue
        metric_summary = {}
        for metric in METRICS:
            values = [item['delta'][metric] for item in deltas]
            metric_summary[metric] = {
                'median_delta': median(values),
                'improved_count': sum(v > 0 for v in values) if metric in
                    ('window_target_bg_margin', 'iou') else sum(v < 0 for v in values),
                'worsened_count': sum(v < 0 for v in values) if metric in
                    ('window_target_bg_margin', 'iou') else sum(v > 0 for v in values),
            }
        per_state = {}
        for label in sorted(labels):
            subset = [item for item in deltas if item['state'] == label]
            if subset:
                per_state[label] = {
                    'n': len(subset),
                    'margin_median_delta': median([x['delta']['window_target_bg_margin'] for x in subset]),
                    'iou_median_delta': median([x['delta']['iou'] for x in subset]),
                    'center_error_px_median_delta': median([x['delta']['center_error_px'] for x in subset]),
                }
        summaries.append({
            'site': site, 'scale': float(scale), 'n': len(deltas),
            'median_projected_message_delta_ratio': median(
                [item['message_delta_ratio'] for item in deltas]),
            'window_top1_peak_changes': sum(item['peak_changed'] for item in deltas),
            'severe_onset_top1_peak_changes': sum(
                item['peak_changed'] for item in deltas if item['state'] == 'severe_onset'),
            'metrics': metric_summary, 'by_state': per_state,
        })
    result = {
        'id': 'M1-MESSAGE-POSITION-LEVERAGE-20261012',
        'state_counts': dict(labels),
        'paired_arm_summaries': summaries,
        'matched_small_perturbations': {
            'T': {'scale': 0.95, 'actual_delta_ratio': summaries[1]['median_projected_message_delta_ratio']},
            'S': {'scale': 0.9, 'actual_delta_ratio': next(x for x in summaries if x['site'] == 'S' and x['scale'] == 0.9)['median_projected_message_delta_ratio']},
            'interpretation': 'T 0.95 versus S 0.90 has matched post-projection message-output perturbation magnitude; do not interpret fixed-state deltas as trained gains.',
        },
        'limitations': [
            'Only four severe-onset states and four pre-onset states occurred in this 12-sequence cohort.',
            'A small local response to post-softmax scaling does not bound what training can learn.',
            'Train-side sequence holdout is not model-training holdout.',
            'Current source replay is not archived M1 frame equivalence.',
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
