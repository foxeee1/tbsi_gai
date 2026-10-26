#!/usr/bin/env python3
"""对 PVL 三组固定序列测试结果做逐序列和分层配对汇总。"""
import argparse
import csv
import json
import os
import pickle
from pathlib import Path


ARMS = {
    'control': ('M1_PVL_CONTROL_PRETRAIN_5EP_20261020', '52001'),
    'template_loc6': ('M1_PVL_TEMPLATE_LOC6_MATCHED_5EP_20261020', '52003'),
    'search_neck': ('M1_PVL_SEARCH_NECK_MATCHED_5EP_20261020', '52004'),
}


def nearest_index(values, target):
    return min(range(len(values)), key=lambda i: abs(float(values[i]) - target))


def read_eval(path):
    with open(path, 'rb') as stream:
        data = pickle.load(stream)
    overlap_x = data['threshold_set_overlap']
    center_x = data['threshold_set_center']
    norm_x = data['threshold_set_center_norm']
    op50_idx = nearest_index(overlap_x, 0.50)
    op75_idx = nearest_index(overlap_x, 0.75)
    pr_idx = nearest_index(center_x, 20.0)
    npr_idx = nearest_index(norm_x, 0.20)
    metrics = {}
    for i, name in enumerate(data['sequences']):
        overlap = data['ave_success_rate_plot_overlap'][i][0]
        precision = data['ave_success_rate_plot_center'][i][0]
        norm_precision = data['ave_success_rate_plot_center_norm'][i][0]
        metrics[name] = {
            'auc': sum(overlap) / len(overlap) * 100.0,
            'op50': overlap[op50_idx] * 100.0,
            'op75': overlap[op75_idx] * 100.0,
            'pr': precision[pr_idx] * 100.0,
            'npr': norm_precision[npr_idx] * 100.0,
            'mean_iou': data['avg_overlap_all'][i][0] * 100.0,
        }
    return metrics


def read_speed(result_dir):
    frame_count = 0
    elapsed = 0.0
    for name in os.listdir(result_dir):
        if not name.endswith('_time.txt'):
            continue
        with open(os.path.join(result_dir, name)) as stream:
            values = [float(line.strip()) for line in stream if line.strip()]
        frame_count += len(values)
        elapsed += sum(values)
    return {'frames': frame_count, 'tracking_seconds': elapsed,
            'fps': frame_count / elapsed if elapsed else None}


def cohort_summary(rows, field):
    groups = {}
    for row in rows:
        groups.setdefault(row['cohort'], []).append(row)
    result = {}
    for cohort, group in groups.items():
        result[cohort] = {'n': len(group)}
        for arm in ('control', 'template_loc6', 'search_neck'):
            result[cohort][arm] = sum(r[arm][field] for r in group) / len(group)
        for arm in ('template_loc6', 'search_neck'):
            result[cohort][arm + '_delta'] = result[cohort][arm] - result[cohort]['control']
            result[cohort][arm + '_wins'] = sum(r[arm][field] > r['control'][field] for r in group)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--output-dir', default='output/experiments/M1_PVL_POSITION_SCREEN_MATCHED_20261020/analysis')
    args = parser.parse_args()
    root = Path(args.project_root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_data = json.loads((root / 'experiments/manifests/m1_pvl_position_screen_20261020.json').read_text())
    cohort_by_sequence = {sequence: cohort for cohort, names in cohort_data['cohorts'].items()
                          for sequence in names}
    m1_auc = cohort_data['m1_auc']

    per_arm = {}
    speeds = {}
    for arm, (run, run_id) in ARMS.items():
        eval_path = root / 'output/experiments' / run / 'analysis/eval_data.pkl'
        per_arm[arm] = read_eval(eval_path)
        result_dir = root / 'output/test/tracking_results/tbsi_track' / (run + '_' + run_id)
        speeds[arm] = read_speed(result_dir)

    sequences = list(per_arm['control'])
    rows = []
    for sequence in sequences:
        row = {'sequence': sequence, 'cohort': cohort_by_sequence[sequence],
               'm1_auc': m1_auc[sequence]}
        for arm in ARMS:
            row[arm] = per_arm[arm][sequence]
        for arm in ('template_loc6', 'search_neck'):
            row[arm + '_auc_delta'] = row[arm]['auc'] - row['control']['auc']
            row[arm + '_mean_iou_delta'] = row[arm]['mean_iou'] - row['control']['mean_iou']
        rows.append(row)

    csv_path = out_dir / 'paired_per_sequence.csv'
    columns = ['sequence', 'cohort', 'm1_auc']
    for arm in ARMS:
        columns.extend(arm + '_' + metric for metric in ('auc', 'pr', 'npr', 'op50', 'op75', 'mean_iou'))
    columns.extend(['template_loc6_auc_delta', 'template_loc6_mean_iou_delta',
                    'search_neck_auc_delta', 'search_neck_mean_iou_delta'])
    with open(csv_path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {'sequence': row['sequence'], 'cohort': row['cohort'], 'm1_auc': row['m1_auc']}
            for arm in ARMS:
                flat.update({arm + '_' + metric: row[arm][metric]
                             for metric in ('auc', 'pr', 'npr', 'op50', 'op75', 'mean_iou')})
            flat.update({key: row[key] for key in columns if key.endswith('_delta')})
            writer.writerow(flat)

    summary = {
        'scope': '13 previously observed LasHeR test sequences; exploratory screening only',
        'sequence_count': len(rows),
        'arms': {arm: {'run': ARMS[arm][0], **speeds[arm]} for arm in ARMS},
        'overall': {},
        'cohort_auc': cohort_summary(rows, 'auc'),
        'cohort_mean_iou': cohort_summary(rows, 'mean_iou'),
        'sequence_win_loss': {},
        'limitations': ['single seed', '5-epoch 512-sample/epoch screening',
                        'test sequences were previously observed and selected retrospectively'],
    }
    for arm in ARMS:
        summary['overall'][arm] = {
            metric: sum(r[arm][metric] for r in rows) / len(rows)
            for metric in ('auc', 'pr', 'npr', 'op50', 'op75', 'mean_iou')}
    for arm in ('template_loc6', 'search_neck'):
        deltas = [r[arm]['auc'] - r['control']['auc'] for r in rows]
        summary['sequence_win_loss'][arm] = {
            'wins': sum(d > 0 for d in deltas),
            'ties': sum(d == 0 for d in deltas),
            'losses': sum(d < 0 for d in deltas),
            'best_delta': max(deltas), 'worst_delta': min(deltas),
        }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
