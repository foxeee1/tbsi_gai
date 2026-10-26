#!/usr/bin/env python
"""Offline reanalysis for SJI-v2; never loads a checkpoint or uses CUDA."""
from __future__ import print_function

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def cluster_bootstrap(seq_values, seed=3407, draws=20000):
    names = sorted(seq_values)
    values = np.asarray([seq_values[n] for n in names], dtype=float)
    if not len(values):
        return {'mean': None, 'ci95': [None, None], 'n_sequences': 0}
    rng = np.random.RandomState(seed)
    sample = rng.randint(0, len(values), size=(draws, len(values)))
    means = values[sample].mean(axis=1)
    return {'mean': float(values.mean()),
            'median_sequence_value': float(np.median(values)),
            'ci95': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            'n_sequences': int(len(values)), 'draws': draws, 'seed': seed}


def paired_permutation(seq_values, seed=3407, draws=20000):
    values = np.asarray(list(seq_values.values()), dtype=float)
    if not len(values):
        return {'mean': None, 'p_two_sided': None, 'n_sequences': 0}
    rng = np.random.RandomState(seed)
    observed = float(values.mean())
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(values)))
    null = (signs * values).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(observed)) + 1.0) / (draws + 1.0)
    return {'mean': observed, 'p_two_sided': float(p), 'n_sequences': int(len(values)),
            'draws': draws, 'seed': seed}


def summarize_pair(rows, left, right, name):
    grouped = defaultdict(dict)
    seq_for_event = {}
    for row in rows:
        if row['condition'] in (left, right):
            grouped[row['event_id']][row['condition']] = float(row['iou'])
            seq_for_event[row['event_id']] = row['seq']
    event_rows = []
    seq_values = defaultdict(list)
    for event_id in sorted(grouped):
        x = grouped[event_id]
        if left not in x or right not in x:
            continue
        # Positive means the right condition has higher IoU than the left.
        delta_iou = x[right] - x[left]
        delta_error = (1.0 - x[left]) - (1.0 - x[right])
        event_rows.append({'event_id': event_id, 'seq': seq_for_event[event_id],
                           'iou_{}'.format(left): x[left], 'iou_{}'.format(right): x[right],
                           'delta_iou_{}_minus_{}'.format(right, left): delta_iou,
                           'delta_error_{}_minus_{}'.format(left, right): delta_error})
        seq_values[seq_for_event[event_id]].append(delta_iou)
    seq_means = {seq: float(np.mean(values)) for seq, values in seq_values.items()}
    return event_rows, {
        'name': name, 'left': left, 'right': right, 'n_events': len(event_rows),
        'n_sequences': len(seq_means), 'event_mean_delta_iou': float(np.mean([x['delta_iou_{}_minus_{}'.format(right, left)] for x in event_rows])) if event_rows else None,
        'event_median_delta_iou': float(np.median([x['delta_iou_{}_minus_{}'.format(right, left)] for x in event_rows])) if event_rows else None,
        'sequence_bootstrap': cluster_bootstrap(seq_means),
        'sequence_permutation': paired_permutation(seq_means),
        'sequence_values': seq_means,
    }


def closed_loop_post(rows, horizon):
    grouped = defaultdict(dict)
    seq_for_event = {}
    for row in rows:
        if int(row['horizon']) == horizon and int(row['frame_offset']) >= 1:
            if row['condition'] in ('00', '11_same'):
                grouped[row['event_id']][row['condition'], int(row['frame_offset'])] = float(row['iou'])
                seq_for_event[row['event_id']] = row['seq']
    event_rows = []
    seq_values = defaultdict(list)
    recovery = []
    for event_id in sorted(seq_for_event):
        offsets = sorted({k[1] for k in grouped[event_id]})
        clean = [grouped[event_id].get(('00', o)) for o in offsets]
        same = [grouped[event_id].get(('11_same', o)) for o in offsets]
        if not clean or any(x is None for x in clean + same):
            continue
        deltas = [a - b for a, b in zip(clean, same)]
        mean_delta = float(np.mean(deltas))
        seq = seq_for_event[event_id]
        seq_values[seq].append(mean_delta)
        event_rows.append({'event_id': event_id, 'seq': seq, 'horizon': horizon,
                           'n_post_frames': len(offsets), 'post_mean_clean_minus_same': mean_delta,
                           'post_min_clean_minus_same': float(np.min(deltas)),
                           'post_offsets': ','.join(str(x) for x in offsets)})
        # Recovery is defined on the same-condition trajectory, post intervention only.
        recover = -1
        for i, offset in enumerate(offsets):
            if offset >= 1 and same[i] >= 0.5 and i + 1 < len(same) and same[i + 1] >= 0.5:
                recover = offset
                break
        recovery.append(recover)
    seq_means = {seq: float(np.mean(values)) for seq, values in seq_values.items()}
    recovered = [x for x in recovery if x >= 0]
    return event_rows, {
        'horizon': horizon, 'n_events': len(event_rows), 'n_sequences': len(seq_means),
        'event_mean_post_clean_minus_same': float(np.mean([x['post_mean_clean_minus_same'] for x in event_rows])) if event_rows else None,
        'event_median_post_clean_minus_same': float(np.median([x['post_mean_clean_minus_same'] for x in event_rows])) if event_rows else None,
        'sequence_bootstrap': cluster_bootstrap(seq_means),
        'sequence_permutation': paired_permutation(seq_means),
        'sequence_values': seq_means,
        'recovery': {'n': len(recovery), 'recovered_n': len(recovered),
                     'unrecovered_n': len(recovery) - len(recovered),
                     'unrecovered_rate': float(np.mean(np.asarray(recovery) < 0)) if recovery else None,
                     'recovered_median_frame': float(np.median(recovered)) if recovered else None},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--controls', required=True)
    ap.add_argument('--closed-loop', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    controls = read_csv(args.controls)
    closed = read_csv(args.closed_loop)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    same_ind_rows, same_ind = summarize_pair(controls, '11_same', '11_independent', 'same_vs_independent')
    same_tex_rows, same_tex = summarize_pair(controls, '11_same', '11_texture', 'same_vs_texture')
    # The requested pairing is direct: no incomplete texture four-cell interaction is inferred.
    write_csv(os.path.join(out_dir, 'paired_same_independent_event.csv'), same_ind_rows, list(same_ind_rows[0]) if same_ind_rows else ['event_id', 'seq'])
    write_csv(os.path.join(out_dir, 'paired_same_texture_event.csv'), same_tex_rows, list(same_tex_rows[0]) if same_tex_rows else ['event_id', 'seq'])
    post = {str(h): closed_loop_post(closed, h) for h in (1, 5, 10)}
    for h, (_, summary) in post.items():
        summary['event_rows_path'] = os.path.join(out_dir, 'closed_loop_post_h{}_event.csv'.format(h))
    for h, (event_rows, _) in post.items():
        write_csv(os.path.join(out_dir, 'closed_loop_post_h{}_event.csv'.format(h)), event_rows,
                  list(event_rows[0]) if event_rows else ['event_id', 'seq', 'horizon'])
    result = {'analysis': 'sji_v2_paired_post_only_reanalysis_v1',
              'controls_source': os.path.abspath(args.controls),
              'closed_loop_source': os.path.abspath(args.closed_loop),
              'texture_definition': 'direct same-event 11_same vs 11_texture IoU pairing; no texture interaction term',
              'closed_loop_definition': 'offset 1..H only; offset 0 excluded from post-only means',
              'recovery_definition': 'first post offset with same IoU >= 0.5 and next post frame also >= 0.5; -1 is unrecovered and summarized separately',
              'same_vs_independent': same_ind, 'same_vs_texture': same_tex,
              'closed_loop_post_only': {h: summary for h, (_, summary) in post.items()}}
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
