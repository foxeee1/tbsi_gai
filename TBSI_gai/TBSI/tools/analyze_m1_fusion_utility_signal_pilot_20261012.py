#!/usr/bin/env python3
"""Sequence-aware analysis for the fixed-crop fusion utility pilot."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def stats(rows, signal, direction):
    x = np.asarray([float(row[signal]) * direction for row in rows])
    y = np.asarray([float(row['utility_gap_rgb_minus_tir']) for row in rows])
    valid = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    if valid.sum() < 3:
        return {'n': int(valid.sum()), 'directional_accuracy': None, 'spearman': None}
    xr, yr = rankdata(x[valid]), rankdata(y[valid])
    rho = float(np.corrcoef(xr, yr)[0, 1]) if np.std(xr) and np.std(yr) else None
    return {'n': int(valid.sum()),
            'directional_accuracy': float(np.mean(np.sign(x[valid]) == np.sign(y[valid]))),
            'spearman': rho}


def magnitude_association(rows, signal):
    x = np.asarray([float(row[signal]) for row in rows])
    y = np.abs(np.asarray([float(row['utility_gap_rgb_minus_tir']) for row in rows]))
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return {'n': int(valid.sum()), 'spearman_abs_utility': None}
    xr, yr = rankdata(x[valid]), rankdata(y[valid])
    rho = float(np.corrcoef(xr, yr)[0, 1]) if np.std(xr) and np.std(yr) else None
    return {'n': int(valid.sum()), 'spearman_abs_utility': rho}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, action='append')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--provenance-note', default='See sequence manifest and source cohort list.')
    args = parser.parse_args()
    rows = []
    for input_path in args.input:
        cohort = 'expansion72' if 'expansion72' in input_path else 'initial17'
        for row in csv.DictReader(open(input_path, newline='')):
            row['cohort'] = cohort
            rows.append(row)
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[row['sequence']].append(row)

    candidates = {
        'energy_gap_rgb_minus_tir': 1.0,
        'template_support_gap_rgb_minus_tir': 1.0,
        'template_psr_gap_rgb_minus_tir': 1.0,
    }
    subsets = {
        'all_states': rows,
        'severe_onset': [r for r in rows if r['label_severe'] == '1'],
        'nonsevere': [r for r in rows if r['label_severe'] != '1'],
    }
    signal_metrics = {}
    for subset_name, subset in subsets.items():
        utility = np.asarray([float(r['utility_gap_rgb_minus_tir']) for r in subset])
        majority = 1 if np.count_nonzero(utility > 0) >= np.count_nonzero(utility < 0) else -1
        signal_metrics[subset_name] = {
            'n': len(subset),
            'utility_rgb_positive': int(np.count_nonzero(utility > 0)),
            'utility_tir_positive': int(np.count_nonzero(utility < 0)),
            'constant_majority_accuracy': float(max(np.count_nonzero(utility > 0),
                                                    np.count_nonzero(utility < 0)) / len(subset)) if subset else None,
            'candidate_signals': {name: stats(subset, name, direction)
                                  for name, direction in candidates.items()},
            'conflict_magnitude': magnitude_association(subset, 'conflict') if subset and 'conflict' in subset[0] else None,
        }
    sequence_rows = []
    all_sequence_rows, severe_sequence_rows = [], []
    for sequence, items in sorted(by_sequence.items()):
        severe_items = [r for r in items if r['label_severe'] == '1']
        row = {'sequence': sequence, 'n_states_all': len(items),
               'n_severe': len(severe_items)}
        for suffix, pool in (('all', items), ('severe', severe_items)):
            if not pool:
                continue
            compact = {'sequence': sequence,
                       'cohort': pool[0]['cohort'],
                       'utility_gap_rgb_minus_tir': float(np.mean([float(r['utility_gap_rgb_minus_tir']) for r in pool]))}
            for key in candidates:
                compact[key] = float(np.mean([float(r[key]) for r in pool]))
            if 'conflict' in pool[0]:
                compact['conflict'] = float(np.mean([float(r['conflict']) for r in pool]))
            for key, value in compact.items():
                if key != 'sequence':
                    row[key + '_' + suffix] = value
            (all_sequence_rows if suffix == 'all' else severe_sequence_rows).append(compact)
        sequence_rows.append(row)

    sequence_stats = {}
    sequence_subsets = [('all_sequences', all_sequence_rows),
                        ('severe_sequences', severe_sequence_rows)]
    for cohort in sorted(set(row['cohort'] for row in rows)):
        sequence_subsets.append((cohort + '_all_sequences',
                                 [r for r in all_sequence_rows if r['cohort'] == cohort]))
        sequence_subsets.append((cohort + '_severe_sequences',
                                 [r for r in severe_sequence_rows if r['cohort'] == cohort]))
    for subset_name, seqs in sequence_subsets:
        y = np.asarray([r['utility_gap_rgb_minus_tir'] for r in seqs])
        sequence_stats[subset_name] = {
            'n': len(seqs),
            'utility_rgb_positive': int(np.count_nonzero(y > 0)),
            'utility_tir_positive': int(np.count_nonzero(y < 0)),
            'constant_majority_accuracy': float(max(np.count_nonzero(y > 0),
                                                    np.count_nonzero(y < 0)) / len(seqs)) if len(seqs) else None,
            'candidate_signals': {name: stats(seqs, name, direction)
                                  for name, direction in candidates.items()},
            'conflict_magnitude': magnitude_association(seqs, 'conflict') if seqs and 'conflict' in seqs[0] else None,
        }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per_sequence_utility.csv').open('w', newline='') as stream:
        fields = sorted({key for row in sequence_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(sequence_rows)
    summary = {
        'id': 'M1-FUSION-UTILITY-SIGNAL-PILOT-20261012',
        'analysis_version': 'sequence-aware-v2',
        'n_states': len(rows), 'n_sequences': len(by_sequence),
        'intervention_utility': 'target attenuation effect minus equal-area background-control effect on frozen CENTER Hann target-background margin',
        'state_level': signal_metrics,
        'sequence_level': sequence_stats,
        'decision': 'No signal admitted. Template support gap is a severe-onset follow-up candidate only; current cohort and small severe count do not justify selecting it.',
        'cohort_provenance': args.provenance_note,
        'limitations': ['sequence overlap status follows the supplied cohort provenance note',
                        'fixed-crop selection-margin utility is not final IoU or closed-loop tracker benefit',
                        'state-level observations are clustered within sequences',
                        'directional accuracy is compared with constant-majority baseline']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
