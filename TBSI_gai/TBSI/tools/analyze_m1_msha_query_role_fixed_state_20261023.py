#!/usr/bin/env python3
"""Summarize the preregistered 28-event MSHA query-role replay."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output/experiments/M1_MSHA_QUERY_ROLE_FIXED_STATE_20261023_FULL28'
OUT = ROOT / 'output/experiments/M1_MSHA_QUERY_ROLE_FIXED_STATE_20261023_ANALYSIS'


def read_csv(name):
    with (RUN / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def as_float(value):
    return float(value) if value not in ('', 'NA', None) else None


def win_counts(deltas, eps=1e-10):
    return {'wins': int(sum(x > eps for x in deltas)),
            'ties': int(sum(abs(x) <= eps for x in deltas)),
            'losses': int(sum(x < -eps for x in deltas))}


def main():
    center_rows = read_csv('center_results.csv')
    boundary_rows = read_csv('query_boundary_deltas.csv')
    branch_rows = read_csv('branch_stats.csv')
    grouped = {}
    for row in center_rows:
        event = (row['sequence'], int(row['frame']), row['role'])
        grouped.setdefault(event, {})[row['condition']] = row
    if len(grouped) != 28 or any(set(arms) != {'M1', 'FG_QUERY', 'SHIFTED_QUERY'}
                                 for arms in grouped.values()):
        raise RuntimeError('expected 28 events with all three paired arms')
    for event, arms in grouped.items():
        states = {row['state_before'] for row in arms.values()}
        crops = {row['crop_xywh'] for row in arms.values()}
        if len(states) != 1 or len(crops) != 1:
            raise RuntimeError('state/crop mismatch: {}'.format(event))

    event_results = []
    summary = {'n_events': len(grouped), 'n_sequences': len({e[0] for e in grouped}),
               'events_by_role': {}, 'arms': {}, 'severe_target_alignment': {},
               'capture_contract': {}, 'query_spatial_response': {}}
    for role in ('matched_good', 'severe_onset'):
        events = [e for e in grouped if e[2] == role]
        summary['events_by_role'][role] = len(events)

    for arm in ('M1', 'FG_QUERY', 'SHIFTED_QUERY'):
        ious = np.asarray([float(grouped[e][arm]['fixed_state_iou']) for e in grouped])
        summary['arms'][arm] = {'mean_iou': float(ious.mean()),
                                'median_iou': float(np.median(ious))}
        if arm == 'M1':
            continue
        delta = np.asarray([float(grouped[e][arm]['fixed_state_iou']) -
                            float(grouped[e]['M1']['fixed_state_iou']) for e in grouped])
        summary['arms'][arm].update({'delta_mean_iou_vs_m1': float(delta.mean()),
                                     'delta_median_iou_vs_m1': float(np.median(delta)),
                                     'iou_win_tie_loss_vs_m1': win_counts(delta)})

    for arm in ('FG_QUERY', 'SHIFTED_QUERY'):
        severe = [e for e in grouped if e[2] == 'severe_onset']
        rank_deltas, margin_deltas, either = [], [], []
        for event in severe:
            parent = grouped[event]['M1']
            candidate = grouped[event][arm]
            rp, rc = as_float(parent['raw_gt_rank']), as_float(candidate['raw_gt_rank'])
            mp, mc = as_float(parent['raw_gt_hard_bg_margin']), as_float(candidate['raw_gt_hard_bg_margin'])
            if None in (rp, rc, mp, mc):
                continue
            rank_delta = rp - rc
            margin_delta = mc - mp
            rank_deltas.append(rank_delta)
            margin_deltas.append(margin_delta)
            either.append(rank_delta > 0 or margin_delta > 0)
            event_results.append({
                'sequence': event[0], 'frame': event[1], 'role': event[2], 'arm': arm,
                'rank_delta_positive_is_better': rank_delta,
                'margin_delta_positive_is_better': margin_delta,
                'iou_delta_vs_m1': float(candidate['fixed_state_iou']) - float(parent['fixed_state_iou']),
            })
        summary['severe_target_alignment'][arm] = {
            'n': len(rank_deltas),
            'gt_rank_improved_tied_worsened': win_counts(rank_deltas),
            'mean_gt_rank_improvement': float(np.mean(rank_deltas)) if rank_deltas else None,
            'margin_improved_tied_worsened': win_counts(margin_deltas),
            'mean_margin_delta': float(np.mean(margin_deltas)) if margin_deltas else None,
            'rank_or_margin_improved_count': int(sum(either)),
        }

    by_event = {}
    for row in boundary_rows:
        event = (row['sequence'], int(row['frame']), row['role'], row['condition'],
                 row['boundary'], row['layer'])
        by_event[event] = row
    capture = {}
    for arm in ('FG_QUERY', 'SHIFTED_QUERY'):
        for boundary in ('projection_input', 'projection_output', 'fusion_return', 'center_input'):
            rows = [row for key, row in by_event.items() if key[3] == arm and key[4] == boundary]
            event_counts = {}
            for row in rows:
                event = (row['sequence'], int(row['frame']), row['role'])
                event_counts.setdefault(event, []).append(float(row['max_abs_delta']))
            capture[arm + ':' + boundary] = {
                'rows': len(rows), 'events_with_nonzero_delta': int(sum(
                    max(values) > 1e-8 for values in event_counts.values())),
                'events': len(event_counts), 'nonfinite_rows': int(sum(
                    not np.isfinite(float(row['max_abs_delta'])) for row in rows)),
            }
    summary['capture_contract'] = capture

    spatial = {}
    for arm, expected in (('FG_QUERY', 'bg_l2_delta'), ('SHIFTED_QUERY', 'fg_l2_delta')):
        layer_rows = [row for row in boundary_rows
                      if row['condition'] == arm and row['boundary'] == 'fusion_return' and
                      row['layer'] == '2']
        event_scores = {}
        for row in layer_rows:
            fg, bg = as_float(row['fg_l2_delta']), as_float(row['bg_l2_delta'])
            if fg is None or bg is None:
                continue
            event = (row['sequence'], int(row['frame']), row['role'])
            event_scores[event] = (fg, bg)
        directional = sum((bg > fg) if expected == 'bg_l2_delta' else (fg > bg)
                          for fg, bg in event_scores.values())
        spatial[arm] = {
            'n': len(event_scores), 'directional_count': int(directional),
            'directional_fraction': float(directional / len(event_scores)) if event_scores else None,
            'mean_fg_delta': float(np.mean([v[0] for v in event_scores.values()])) if event_scores else None,
            'mean_bg_delta': float(np.mean([v[1] for v in event_scores.values()])) if event_scores else None,
        }
    summary['query_spatial_response'] = spatial

    modes = {}
    for row in branch_rows:
        stats = json.loads(row['stats'])
        modes.setdefault(row['condition'], []).append(stats.get('mode'))
    summary['branch_modes'] = {arm: {'count': len(values), 'unique_modes': sorted(set(values))}
                               for arm, values in modes.items()}

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (OUT / 'severe_event_deltas.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(event_results[0]))
        writer.writeheader(); writer.writerows(event_results)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
