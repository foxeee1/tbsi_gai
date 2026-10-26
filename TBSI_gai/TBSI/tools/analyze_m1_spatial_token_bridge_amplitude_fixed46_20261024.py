#!/usr/bin/env python3
"""Summarize paired state and sequence effects from the bridge amplitude replay."""
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'output/experiments/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_RETRY1_20261024/runs'
OUT = ROOT / 'output/experiments/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_RETRY1_20261024'
METRICS = ('center_gt_hard_bg_margin', 'postfusion_query_gt_hard_bg_margin',
           'fixed_replay_iou')
RANKS = {'center_gt_hard_bg_margin': 'center_gt_rank',
         'postfusion_query_gt_hard_bg_margin': 'postfusion_query_gt_rank'}


def read_arm(name):
    with (RUNS / name / 'per_event.csv').open(newline='') as stream:
        rows = {row['event_id']: row for row in csv.DictReader(stream)}
    if len(rows) != 46:
        raise RuntimeError('{} has {} states, expected 46'.format(name, len(rows)))
    return rows


def mean(values):
    return statistics.mean(values) if values else float('nan')


def main():
    arms = {name: read_arm(name) for name in (
        'off_x1', 'target_x1', 'shifted_x1', 'target_x32', 'shifted_x32',
        'target_x128', 'shifted_x128')}
    old_arms = {}
    for short, label in (('off_x1', 'OFF'), ('target_x1', 'TARGET'),
                         ('shifted_x1', 'SHIFTED')):
        path = ROOT / ('output/experiments/M1_SPATIAL_TOKEN_BRIDGE_LEARNABILITY_E10_{}_FIXED46_20261024/per_event.csv'.format(label))
        with path.open(newline='') as stream:
            old_arms[short] = {row['event_id']: row for row in csv.DictReader(stream)}
    parity = {}
    for name, old in old_arms.items():
        parity[name] = {'same_event_ids': set(arms[name]) == set(old)}
        for metric in METRICS:
            diffs = [float(arms[name][key][metric]) - float(old[key][metric])
                     for key in old]
            parity[name][metric + '_mean_abs_error'] = mean([abs(v) for v in diffs])
            parity[name][metric + '_max_abs_error'] = max(abs(v) for v in diffs)

    state_rows, sequence_rows, contrasts = [], [], {}
    for scale in (1, 32, 128):
        target = arms['target_x{}'.format(scale)]
        shifted = arms['shifted_x{}'.format(scale)]
        off = arms['off_x1']
        contrasts[str(scale)] = {}
        for compare_name, left, right in (
                ('target_minus_off', target, off),
                ('shifted_minus_off', shifted, off),
                ('target_minus_shifted', target, shifted)):
            contrasts[str(scale)][compare_name] = {}
            sequence_metric_values = {}
            for metric in METRICS:
                deltas = {key: float(left[key][metric]) - float(right[key][metric])
                          for key in left}
                seq_values = {}
                for key, delta in deltas.items():
                    seq = left[key]['sequence']
                    seq_values.setdefault(seq, []).append(delta)
                    state_rows.append({
                        'scale': scale, 'contrast': compare_name,
                        'event_id': key, 'sequence': seq,
                        'phase': left[key]['phase'], 'metric': metric,
                        'delta': delta,
                    })
                seq_means = {seq: mean(vals) for seq, vals in seq_values.items()}
                sequence_metric_values[metric] = seq_means
                contrasts[str(scale)][compare_name][metric] = {
                    'mean_delta': mean(list(deltas.values())),
                    'state_positive': sum(value > 1e-8 for value in deltas.values()),
                    'state_negative': sum(value < -1e-8 for value in deltas.values()),
                    'state_tie': sum(abs(value) <= 1e-8 for value in deltas.values()),
                    'sequence_mean_positive': sum(value > 1e-8 for value in seq_means.values()),
                    'sequence_mean_negative': sum(value < -1e-8 for value in seq_means.values()),
                    'sequence_mean_tie': sum(abs(value) <= 1e-8 for value in seq_means.values()),
                }
                rank = RANKS.get(metric)
                if rank:
                    changes = [key for key in left
                               if int(left[key][rank]) != int(right[key][rank])]
                    contrasts[str(scale)][compare_name][metric]['rank_changed_states'] = len(changes)
                    contrasts[str(scale)][compare_name][metric]['rank_better_states'] = sum(
                        int(left[key][rank]) < int(right[key][rank]) for key in changes)
                    contrasts[str(scale)][compare_name][metric]['rank_worse_states'] = sum(
                        int(left[key][rank]) > int(right[key][rank]) for key in changes)
                for seq, value in seq_means.items():
                    sequence_rows.append({
                        'scale': scale, 'contrast': compare_name,
                        'sequence': seq, 'metric': metric, 'mean_delta': value,
                    })
            for phase in ('pre_onset', 'severe_onset'):
                keys = [key for key in left if left[key]['phase'] == phase]
                for metric in METRICS:
                    contrasts[str(scale)][compare_name].setdefault('by_phase', {})
                    contrasts[str(scale)][compare_name]['by_phase'].setdefault(phase, {})
                    contrasts[str(scale)][compare_name]['by_phase'][phase][metric] = mean([
                        float(left[key][metric]) - float(right[key][metric]) for key in keys])

    for filename, rows in (('per_state_contrasts.csv', state_rows),
                           ('per_sequence_contrasts.csv', sequence_rows)):
        with (OUT / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        'experiment': 'M1-SPATIAL-TOKEN-BRIDGE-AMPLITUDE-FIXED46-RETRY1-20261024',
        'states': 46, 'sequences': 23, 'metric_definitions': list(METRICS),
        'scale1_parity_to_E10_replay': parity,
        'contrasts': contrasts,
        'interpretation_boundary': 'Train-side fixed-state mechanism diagnostic only; no full tracking metrics or held-out claim.',
    }
    (OUT / 'paired_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'summary': str(OUT / 'paired_summary.json'),
                      'state_rows': len(state_rows), 'sequence_rows': len(sequence_rows),
                      'parity': parity}, indent=2))


if __name__ == '__main__':
    main()
