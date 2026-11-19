#!/usr/bin/env python3
"""Paired offline analysis of fixed directional MSHA interventions."""
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/experiments'
ARMS = {
    'parent': 'M1_T2S_CONDITIONED_PARENT_CALIB46_20261023',
    'r2t_075': 'M1_NATIVE_MSHA_FIXED_R2T075_CALIB46_20261023',
    'r2t_125': 'M1_NATIVE_MSHA_FIXED_R2T125_CALIB46_20261023',
    't2r_075': 'M1_NATIVE_MSHA_FIXED_T2R075_CALIB46_20261023',
    't2r_125': 'M1_NATIVE_MSHA_FIXED_T2R125_CALIB46_20261023',
}
METRICS = ('fixed_replay_iou', 'center_gt_hard_bg_margin',
           'center_gt_rank', 'center_top1_to_gt_center_cells')


def load_arm(name, directory):
    path = OUT / directory / 'per_event.csv'
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    keyed = {row['event_id']: row for row in rows}
    if len(keyed) != 46:
        raise ValueError('{} expected 46 unique events, found {}'.format(name, len(keyed)))
    return keyed


def val(row, key):
    return float(row[key])


def summarize(values):
    values = sorted(values)
    n = len(values)
    return {
        'n': n,
        'mean': sum(values) / n if n else None,
        'median': values[n // 2] if n % 2 else
            ((values[n // 2 - 1] + values[n // 2]) / 2 if n else None),
        'positive': sum(v > 0 for v in values),
        'negative': sum(v < 0 for v in values),
        'zero': sum(v == 0 for v in values),
    }


def main():
    arms = {name: load_arm(name, directory) for name, directory in ARMS.items()}
    ids = sorted(arms['parent'])
    interventions = [name for name in ARMS if name != 'parent']
    details = []
    for event_id in ids:
        base = arms['parent'][event_id]
        row = {'event_id': event_id, 'sequence': base['sequence'],
               'phase': base['phase'], 'frame': base['frame']}
        for arm in interventions:
            for metric in METRICS:
                delta = val(arms[arm][event_id], metric) - val(base, metric)
                if metric == 'center_gt_rank':
                    delta = -delta
                row[arm + '_' + metric + '_delta'] = delta
        # Oracle selects the intervention using replay IoU on this same state.
        best = max(interventions,
                   key=lambda arm: val(arms[arm][event_id], 'fixed_replay_iou'))
        row['oracle_iou_arm'] = best
        row['oracle_iou_delta'] = (val(arms[best][event_id], 'fixed_replay_iou') -
                                   val(base, 'fixed_replay_iou'))
        # Independent oracles expose whether one metric's winner helps another.
        best_margin = max(interventions,
                          key=lambda arm: val(arms[arm][event_id], 'center_gt_hard_bg_margin'))
        best_rank = min(interventions,
                        key=lambda arm: val(arms[arm][event_id], 'center_gt_rank'))
        row['oracle_margin_arm'] = best_margin
        row['oracle_margin_delta'] = (val(arms[best_margin][event_id], 'center_gt_hard_bg_margin') -
                                      val(base, 'center_gt_hard_bg_margin'))
        row['oracle_rank_arm'] = best_rank
        row['oracle_rank_delta'] = (val(base, 'center_gt_rank') -
                                    val(arms[best_rank][event_id], 'center_gt_rank'))
        details.append(row)

    summary = {}
    for phase in ('pre_onset', 'severe_onset', 'all'):
        subset = [r for r in details if phase == 'all' or r['phase'] == phase]
        summary[phase] = {'n': len(subset), 'arms': {},
                          'oracle': {}, 'sequence_oracle': {}}
        for arm in interventions:
            summary[phase]['arms'][arm] = {
                metric: summarize([r[arm + '_' + metric + '_delta'] for r in subset])
                for metric in METRICS
            }
        for metric in ('oracle_iou_delta', 'oracle_margin_delta', 'oracle_rank_delta'):
            summary[phase]['oracle'][metric] = summarize([r[metric] for r in subset])
        by_sequence = {}
        for r in subset:
            by_sequence.setdefault(r['sequence'], []).append(r)
        seq_deltas = [max(x['oracle_iou_delta'] for x in group)
                      for group in by_sequence.values()]
        seq_arm_means = {
            seq: {
                arm: sum(r[arm + '_fixed_replay_iou_delta'] for r in group) / len(group)
                for arm in interventions
            }
            for seq, group in by_sequence.items()
        }
        seq_best_arm = {seq: max(scores, key=scores.get)
                        for seq, scores in seq_arm_means.items()}
        seq_fixed_action_deltas = [
            seq_arm_means[seq][seq_best_arm[seq]] for seq in seq_arm_means]
        oracle_sorted = sorted((r['oracle_iou_delta'] for r in subset), reverse=True)
        oracle_sum = sum(oracle_sorted)
        positive_sequences = sum(v > 0 for v in seq_deltas)
        summary[phase]['sequence_oracle'] = {
            'n_sequences': len(seq_deltas),
            'mean_best_state_delta_per_sequence': sum(seq_deltas) / len(seq_deltas),
            'sequences_with_any_positive_state': positive_sequences,
            'fixed_action_per_sequence': summarize(seq_fixed_action_deltas),
            'fixed_action_arm_counts': {
                arm: sum(chosen == arm for chosen in seq_best_arm.values())
                for arm in interventions
            },
            'oracle_iou_arm_counts': {
                arm: sum(r['oracle_iou_arm'] == arm for r in subset)
                for arm in interventions
            },
            'top_action_stability_within_sequence': sum(
                len(set(x['oracle_iou_arm'] for x in group)) == 1
                for group in by_sequence.values()),
        }
        summary[phase]['oracle']['largest_state_contribution_fraction'] = (
            oracle_sorted[0] / oracle_sum if oracle_sum > 0 else None)
        summary[phase]['oracle']['mean_excluding_largest_state'] = (
            sum(oracle_sorted[1:]) / (len(oracle_sorted) - 1)
            if len(oracle_sorted) > 1 else None)

    analysis_dir = OUT / 'M1_NATIVE_MSHA_DIRECTIONAL_ORACLE_ANALYSIS_20261023'
    analysis_dir.mkdir(exist_ok=True)
    with (analysis_dir / 'per_event_paired_oracle.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    (analysis_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'analysis_dir': str(analysis_dir), 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
