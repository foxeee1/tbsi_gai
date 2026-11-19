#!/usr/bin/env python3
"""Paired 62-sequence evaluation for matched M1 and three single-signal arms."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'output/experiments/M1_FULL244_SCENE_MINING_20261009/per_sequence_scene.csv'
ARCHIVED_METRICS = ROOT / 'output/experiments/M1_ARCHIVED_PR_SR_SCENES_20261010/per_sequence_metrics.csv'
METRICS = ('auc', 'precision', 'norm_precision', 'op50', 'op75')
FLOW = ('gt_rank_hann', 'peak_distance_tokens', 'score_entropy',
        'rgb_norm', 'tir_norm', 'fused_norm', 'branch_cosine', 'iou')


def load_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def macro_by_sequence(rows, field):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['sequence'], []).append(float(row[field]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def bootstrap(values, seed=20261012):
    rng = np.random.RandomState(seed)
    array = np.asarray(values, dtype=np.float64)
    samples = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(args.root_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing nonempty output directory')
    paths = {'m1': root / 'matched_m1_full62',
             's1': root / 's1_full62', 's2': root / 's2_full62',
             's3': root / 's3_full62'}
    scene = {row['sequence']: row for row in load_csv(SCENE)}
    expected = sorted(name for name, row in scene.items() if float(row['m1_auc']) < 40)
    if len(expected) != 62:
        raise RuntimeError('archived low-AUC set changed')
    data = {}
    for arm, path in paths.items():
        summary = json.loads((path / 'summary.json').read_text())
        if summary['status'] != 'COMPLETE' or summary['n_sequences'] != 62:
            raise RuntimeError('incomplete arm: ' + arm)
        rows = {row['sequence']: row for row in load_csv(path / 'per_sequence.csv')}
        if sorted(rows) != expected:
            raise RuntimeError('sequence set mismatch: ' + arm)
        info = load_csv(path / 'information_flow.csv')
        if sorted(set(row['sequence'] for row in info)) != expected:
            raise RuntimeError('information-flow coverage mismatch: ' + arm)
        data[arm] = {'rows': rows, 'info': info, 'summary': summary}

    table = []
    for arm in paths:
        item = {'arm': arm, 'n_sequences': 62, 'n_info_samples': len(data[arm]['info'])}
        for metric in METRICS:
            values = [float(data[arm]['rows'][name][metric]) for name in expected]
            item[metric] = float(np.mean(values))
            if arm != 'm1':
                delta = [float(data[arm]['rows'][name][metric]) -
                         float(data['m1']['rows'][name][metric]) for name in expected]
                item[metric + '_delta'] = float(np.mean(delta))
                item[metric + '_delta_ci95'] = bootstrap(delta)
                if metric == 'auc':
                    item['auc_wins'] = sum(value > 0 for value in delta)
                    item['auc_losses'] = sum(value < 0 for value in delta)
        item['fps_sequence_macro'] = float(np.mean([
            float(data[arm]['rows'][name]['fps']) for name in expected]))
        for metric in FLOW:
            item[metric + '_sequence_macro'] = macro_by_sequence(data[arm]['info'], metric)
        item['gt_top1_rate_sequence_macro'] = macro_by_sequence(
            [{'sequence': row['sequence'], 'top1': int(float(row['gt_rank_hann']) == 1)}
             for row in data[arm]['info']], 'top1')
        table.append(item)

    strata = {}
    for group in ('TC', 'SA', 'LR', 'LI'):
        members = [name for name in expected if group in scene[name]['attributes'].split(';')]
        strata[group] = {'n_sequences': len(members)}
        for arm in paths:
            strata[group][arm + '_auc'] = float(np.mean([
                float(data[arm]['rows'][name]['auc']) for name in members]))
            if arm != 'm1':
                strata[group][arm + '_auc_delta'] = strata[group][arm + '_auc'] - strata[group]['m1_auc']
    archive_rows = {row['sequence']: row for row in load_csv(ARCHIVED_METRICS)}
    if not set(expected).issubset(archive_rows):
        raise RuntimeError('archived metric rows missing')
    archive_columns = {'auc': 'success_auc', 'precision': 'precision_20px',
                       'norm_precision': 'norm_precision_0p2',
                       'op50': 'op50', 'op75': 'op75'}
    archived = {metric: float(np.mean([
        float(archive_rows[name][column]) for name in expected]))
        for metric, column in archive_columns.items()}
    output.mkdir(parents=True)
    result = {'id': 'M1-SINGLE-SIGNAL-LOWAUC-PAIRED-20261012',
              'scope': '62 archived-M1 AUC<40 sequences; matched current-code reruns',
              'archived_m1_reference': archived, 'table': table, 'strata': strata,
              'limitations': ['archived M1 selection defines this subgroup',
                              'current-code reruns are not frame-parity reproductions of archived M1',
                              'sampled tensor norms and rank are descriptive, not proof of modality utility']}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (output / 'four_rows.csv').open('w', newline='') as stream:
        fields = [key for key in table[0] if not isinstance(table[0][key], list)]
        fields += [key for key in table[1] if key not in fields and not isinstance(table[1][key], list)]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows([{key: row.get(key, '') for key in fields} for row in table])
    print(json.dumps({'table': table, 'strata': strata}, indent=2), flush=True)


if __name__ == '__main__':
    main()
