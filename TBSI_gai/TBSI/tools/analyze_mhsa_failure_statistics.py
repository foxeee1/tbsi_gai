#!/usr/bin/env python3
"""Summarize paired IoU deltas against compact MHSA statistics."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = (
    'entropy_all', 'cross_mass_rgb_to_tir', 'cross_mass_tir_to_rgb',
    'self_mass_rgb', 'self_mass_tir', 'fg_mass_rgb', 'fg_mass_tir',
    'bg_mass_rgb', 'bg_mass_tir', 'cross_modal_cosine',
    'template_disagreement',
)


def mean(values):
    return sum(values) / len(values) if values else None


def rank(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        value = (index + end - 1) / 2.0 + 1.0
        for pos in range(index, end):
            ranks[ordered[pos][0]] = value
        index = end
    return ranks


def pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = mean(x), mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denom if denom else None


def read_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--attention', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = {(row['seq'], int(row['frame'])): row for row in read_csv(args.pairs)}
    attention = read_csv(args.attention)
    merged = []
    for row in attention:
        key = (row['seq'], int(row['frame']))
        pair = pairs.get(key)
        if pair is None:
            continue
        item = {'seq': row['seq'], 'frame': int(row['frame']), 'layer': int(row['layer']),
                'failure_label': pair['failure_label'], 'delta_iou': float(pair['delta_iou'])}
        for metric in METRICS:
            try:
                item[metric] = float(row[metric])
            except (TypeError, ValueError):
                item[metric] = None
        merged.append(item)
    if not merged:
        raise RuntimeError('no attention/pair rows matched')

    grouped = defaultdict(list)
    for row in merged:
        grouped[(row['layer'], row['failure_label'])].append(row)
    summary_rows = []
    for (layer, label), rows in sorted(grouped.items()):
        result = {'layer': layer, 'failure_label': label, 'rows': len(rows),
                  'mean_delta_iou': mean([row['delta_iou'] for row in rows])}
        for metric in METRICS:
            result[metric] = mean([row[metric] for row in rows if row[metric] is not None])
        summary_rows.append(result)

    seq_layer = defaultdict(list)
    for row in merged:
        seq_layer[(row['layer'], row['seq'])].append(row)
    correlations = []
    for layer in sorted({key[0] for key in seq_layer}):
        rows = []
        for (row_layer, seq), values in seq_layer.items():
            if row_layer == layer:
                rows.append({'seq': seq, 'delta_iou': mean([v['delta_iou'] for v in values]),
                             **{metric: mean([v[metric] for v in values if v[metric] is not None]) for metric in METRICS}})
        for metric in METRICS:
            valid = [row for row in rows if row[metric] is not None]
            x, y = [row[metric] for row in valid], [row['delta_iou'] for row in valid]
            correlations.append({'layer': layer, 'metric': metric, 'sequences': len(valid),
                                 'pearson': pearson(x, y), 'spearman': pearson(rank(x), rank(y))})

    labels = defaultdict(int)
    for row in merged:
        labels[row['failure_label']] += 1
    result = {'experiment': 'mhsa_failure_analysis_v1', 'stage': 'B',
              'pairs_rows': len(pairs), 'attention_rows': len(attention),
              'matched_rows': len(merged), 'labels': dict(sorted(labels.items())),
              'sampling': 'all 244 sequences; deterministic prefix at most 64 frames per sequence',
              'summary': summary_rows, 'sequence_correlations': correlations,
              'causal_claim': False}
    with open(output / 'failure_statistics_summary.json', 'w') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    with open(output / 'layer_failure_summary.csv', 'w', newline='') as handle:
        fields = list(summary_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary_rows)
    with open(output / 'sequence_correlations.csv', 'w', newline='') as handle:
        fields = list(correlations[0])
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(correlations)
    row_fields = list(merged[0])
    for label, filename in (('hard_regression', 'hard_regression.csv'),
                            ('regression', 'regression.csv'),
                            ('gain', 'gain_cases.csv'),
                            ('common_failure', 'common_failure.csv')):
        with open(output / filename, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=row_fields); writer.writeheader()
            writer.writerows(row for row in merged if row['failure_label'] == label)
    print(json.dumps({'matched_rows': len(merged), 'labels': dict(sorted(labels.items()))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
