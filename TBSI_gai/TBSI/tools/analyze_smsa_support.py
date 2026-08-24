#!/usr/bin/env python3
"""Aggregate SMSA support diagnostics into analysis packages A/B/C."""
import argparse
import csv
import json
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_lasher_attributes import sequence_curves


METRICS = ['Keff', 'GTMass', 'GTPrecision', 'GTRecall', 'BGMass']


def mean_dict(rows):
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS}


def corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {'pearson': None, 'spearman': None, 'N': len(x)}
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {'pearson': pearson, 'spearman': spearman, 'N': len(x)}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--results-root', required=True)
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--attr-zip', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    raw = []
    with open(args.stats) as f:
        for line in f:
            if line.strip():
                raw.append(json.loads(line))
    if not raw:
        raise RuntimeError('No support statistics found: {}'.format(args.stats))

    # Per-sequence AUC/AO deltas, using the same sequence-level curves as the
    # attribute evaluator. This is the definition of positive/negative.
    result_root = args.results_root
    seq_delta = {}
    for seq in sorted(set(r['sequence'] for r in raw)):
        gt = os.path.join(args.dataset_root, seq, 'init.txt')
        bp = os.path.join(result_root, args.baseline, seq + '.txt')
        mp = os.path.join(result_root, args.method, seq + '.txt')
        if not (os.path.isfile(gt) and os.path.isfile(bp) and os.path.isfile(mp)):
            continue
        b = sequence_curves(bp, gt)
        m = sequence_curves(mp, gt)
        seq_delta[seq] = {
            'delta_AUC': float((m['overlap'].mean() - b['overlap'].mean()) * 100.0),
            'delta_AO': float((m['ao'] - b['ao']) * 100.0),
        }

    # Collapse frame rows to sequence x layer x direction so each sequence has
    # equal weight in all downstream analyses.
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row['sequence'], int(row['layer']), row['direction'])].append(row)
    seq_layer = []
    for (seq, layer, direction), rows in grouped.items():
        if seq not in seq_delta:
            continue
        item = {'sequence': seq, 'layer': layer, 'direction': direction}
        item.update(mean_dict(rows))
        item.update(seq_delta[seq])
        item['group'] = 'positive' if item['delta_AUC'] >= 0 else 'negative'
        seq_layer.append(item)

    with zipfile.ZipFile(args.attr_zip) as zf:
        order = zf.read('Attributes_order.txt').decode().strip().split(', ')
        attr_map = {}
        for seq in seq_delta:
            try:
                vals = [int(x) for x in zf.read('AttriSeqsTxt/{}.txt'.format(seq)).decode().strip().split(',')]
                attr_map[seq] = dict(zip(order, vals))
            except KeyError:
                attr_map[seq] = {}

    # Analysis A: attributes, aggregating directions and layers per sequence.
    seq_all = defaultdict(list)
    for row in seq_layer:
        seq_all[row['sequence']].append(row)
    seq_summary = []
    for seq, rows in seq_all.items():
        item = {'sequence': seq, 'delta_AUC': rows[0]['delta_AUC'], 'delta_AO': rows[0]['delta_AO']}
        item.update({m: float(np.mean([r[m] for r in rows])) for m in METRICS})
        seq_summary.append(item)
    attr_rows = []
    for attr in order:
        selected = [r for r in seq_summary if attr_map.get(r['sequence'], {}).get(attr, 0) == 1]
        if not selected:
            continue
        out = {'attribute': attr, 'N_seq': len(selected)}
        out.update({m: float(np.mean([r[m] for r in selected])) for m in METRICS})
        out['delta_AUC_mean'] = float(np.mean([r['delta_AUC'] for r in selected]))
        out['delta_AO_mean'] = float(np.mean([r['delta_AO'] for r in selected]))
        attr_rows.append(out)
    write_csv(os.path.join(args.out_dir, 'A_support_by_attribute.csv'), attr_rows)

    # Analysis B: layers, with all/positive/negative sequence groups.
    layer_rows = []
    for layer in sorted(set(r['layer'] for r in seq_layer)):
        for group in ('all', 'positive', 'negative'):
            for direction in ('RGB2F', 'TIR2F', 'both'):
                selected = [r for r in seq_layer if r['layer'] == layer and
                            (group == 'all' or r['group'] == group) and
                            (direction == 'both' or r['direction'] == direction)]
                # For both, average the two directions within each sequence.
                if direction == 'both':
                    per_seq = defaultdict(list)
                    for r in selected:
                        per_seq[r['sequence']].append(r)
                    selected = []
                    for seq, rs in per_seq.items():
                        x = {'sequence': seq, 'delta_AUC': rs[0]['delta_AUC'], 'delta_AO': rs[0]['delta_AO']}
                        x.update({m: float(np.mean([r[m] for r in rs])) for m in METRICS})
                        selected.append(x)
                if not selected:
                    continue
                out = {'layer': layer, 'group': group, 'direction': direction, 'N_seq': len(selected)}
                out.update({m: float(np.mean([r[m] for r in selected])) for m in METRICS})
                layer_rows.append(out)
    write_csv(os.path.join(args.out_dir, 'B_support_by_layer.csv'), layer_rows)

    # Analysis C: sequence-level support/performance correlations.
    perf_rows = []
    for item in seq_summary:
        out = dict(item)
        perf_rows.append(out)
    write_csv(os.path.join(args.out_dir, 'C_sequence_support_performance.csv'), perf_rows)
    correlations = {}
    for metric in METRICS:
        x = np.array([r[metric] for r in perf_rows])
        correlations[metric + '_vs_delta_AUC'] = corr(x, np.array([r['delta_AUC'] for r in perf_rows]))
        correlations[metric + '_vs_delta_AO'] = corr(x, np.array([r['delta_AO'] for r in perf_rows]))
    for layer in sorted(set(r['layer'] for r in seq_layer)):
        for direction in ('RGB2F', 'TIR2F'):
            subset = [r for r in seq_layer if r['layer'] == layer and r['direction'] == direction]
            key_prefix = 'L{}_{}'.format(layer, direction)
            for metric in METRICS:
                x = np.array([r[metric] for r in subset])
                correlations[key_prefix + '_' + metric + '_vs_delta_AUC'] = corr(
                    x, np.array([r['delta_AUC'] for r in subset]))
                correlations[key_prefix + '_' + metric + '_vs_delta_AO'] = corr(
                    x, np.array([r['delta_AO'] for r in subset]))
    with open(os.path.join(args.out_dir, 'C_correlations.json'), 'w') as f:
        json.dump(correlations, f, indent=2)

    print('Support rows:', len(raw), 'sequence summaries:', len(seq_summary))
    print('\nAnalysis A written:', os.path.join(args.out_dir, 'A_support_by_attribute.csv'))
    print('Analysis B written:', os.path.join(args.out_dir, 'B_support_by_layer.csv'))
    print('Analysis C correlations:')
    print(json.dumps(correlations, indent=2))


if __name__ == '__main__':
    main()
