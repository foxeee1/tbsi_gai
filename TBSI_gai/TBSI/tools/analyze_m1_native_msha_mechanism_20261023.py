#!/usr/bin/env python3
"""Offline mechanism decomposition for the captured native MSHA controls."""
import csv
import json
import math
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
INTERVENTIONS = tuple(k for k in ARMS if k != 'parent')


def load(directory):
    with (OUT / directory / 'per_event.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    return {r['event_id']: r for r in rows}


def f(row, key):
    return float(row[key])


def b(row, key):
    return 1.0 if row[key] == 'True' else 0.0


def bbox(row):
    return [float(x) for x in row['predicted_bbox'].strip('[]').split(',')]


def bbox_delta(a, b):
    pa, pb = bbox(a), bbox(b)
    return {'x': pb[0] - pa[0], 'y': pb[1] - pa[1],
            'w': pb[2] - pa[2], 'h': pb[3] - pa[3],
            'center_shift': math.hypot(pb[0] + pb[2] / 2 - pa[0] - pa[2] / 2,
                                       pb[1] + pb[3] / 2 - pa[1] - pa[3] / 2),
            'size_shift': math.hypot(pb[2] - pa[2], pb[3] - pa[3])}


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarize(rows, arm):
    parent = rows['parent']
    cand = rows[arm]
    keys = {
        'iou_delta': ('fixed_replay_iou', 1),
        'raw_center_margin_delta': ('center_raw_score_gt_hard_bg_margin', 1),
        'hann_center_margin_delta': ('center_gt_hard_bg_margin', 1),
        'center_rank_delta': ('center_gt_rank', -1),
        'center_distance_delta': ('center_top1_to_gt_center_cells', 1),
        'query_gt_peak_delta': ('postfusion_query_gt_peak', 1),
        'query_gt_hard_bg_margin_delta': ('postfusion_query_gt_hard_bg_margin', 1),
        'query_gt_rank_delta': ('postfusion_query_gt_rank', -1),
        'query_strong_bg_margin_delta': ('postfusion_query_gt_strongest_bg_margin', 1),
        'query_peak_in_gt_delta': ('postfusion_query_peak_in_gt', 1),
        'pred_bbox_center_shift': (None, 1),
        'pred_bbox_size_shift': (None, 1),
    }
    out = {}
    for name, (key, sign) in keys.items():
        if key is not None:
            out[name] = mean([sign * (f(cand[e], key) - f(parent[e], key)) for e in rows['parent']])
        elif name == 'pred_bbox_center_shift':
            out[name] = mean([bbox_delta(parent[e], cand[e])['center_shift'] for e in rows['parent']])
        else:
            out[name] = mean([bbox_delta(parent[e], cand[e])['size_shift'] for e in rows['parent']])
    out['center_rank_changed'] = sum(f(cand[e], 'center_gt_rank') != f(parent[e], 'center_gt_rank') for e in rows['parent'])
    out['hann_top1_changed'] = sum((f(cand[e], 'center_top1_x') != f(parent[e], 'center_top1_x') or
                                    f(cand[e], 'center_top1_y') != f(parent[e], 'center_top1_y'))
                                   for e in rows['parent'])
    return out


def main():
    arms = {name: load(directory) for name, directory in ARMS.items()}
    if any(len(v) != 46 for v in arms.values()):
        raise ValueError('all arms must contain 46 rows')
    event_ids = sorted(arms['parent'])
    details = []
    for event_id in event_ids:
        row = {'event_id': event_id, 'sequence': arms['parent'][event_id]['sequence'],
               'phase': arms['parent'][event_id]['phase']}
        for arm in INTERVENTIONS:
            p, c = arms['parent'][event_id], arms[arm][event_id]
            bd = bbox_delta(p, c)
            row[arm] = {
                'iou_delta': f(c, 'fixed_replay_iou') - f(p, 'fixed_replay_iou'),
                'raw_center_margin_delta': f(c, 'center_raw_score_gt_hard_bg_margin') - f(p, 'center_raw_score_gt_hard_bg_margin'),
                'hann_center_margin_delta': f(c, 'center_gt_hard_bg_margin') - f(p, 'center_gt_hard_bg_margin'),
                'center_rank_delta': f(p, 'center_gt_rank') - f(c, 'center_gt_rank'),
                'center_distance_delta': f(c, 'center_top1_to_gt_center_cells') - f(p, 'center_top1_to_gt_center_cells'),
                'query_gt_peak_delta': f(c, 'postfusion_query_gt_peak') - f(p, 'postfusion_query_gt_peak'),
                'query_gt_hard_bg_margin_delta': f(c, 'postfusion_query_gt_hard_bg_margin') - f(p, 'postfusion_query_gt_hard_bg_margin'),
                'query_gt_rank_delta': f(p, 'postfusion_query_gt_rank') - f(c, 'postfusion_query_gt_rank'),
                'query_strong_bg_margin_delta': f(c, 'postfusion_query_gt_strongest_bg_margin') - f(p, 'postfusion_query_gt_strongest_bg_margin'),
                'query_peak_in_gt_delta': b(c, 'postfusion_query_peak_in_gt') - b(p, 'postfusion_query_peak_in_gt'),
                'pred_bbox': bd,
            }
        details.append(row)

    def select(predicate):
        return [r for r in details if predicate(r)]

    cohorts = {
        'all': details,
        'bike150': select(lambda r: r['sequence'] == 'bike150'),
        'theleftestrunningboy': select(lambda r: r['sequence'] == 'theleftestrunningboy'),
        'outlier_pair': select(lambda r: r['event_id'] in {
            'bike150:severe_onset:144', 'theleftestrunningboy:pre_onset:17'}),
        'adjacent_to_outliers': select(lambda r: r['sequence'] in {'bike150', 'theleftestrunningboy'}),
    }
    summary = {'n_events': len(details), 'cohorts': {}}
    metric_names = [
        'iou_delta', 'raw_center_margin_delta', 'hann_center_margin_delta',
        'center_rank_delta', 'center_distance_delta', 'query_gt_peak_delta',
        'query_gt_hard_bg_margin_delta', 'query_gt_rank_delta',
        'query_strong_bg_margin_delta', 'query_peak_in_gt_delta',
    ]
    for cohort, cohort_rows in cohorts.items():
        summary['cohorts'][cohort] = {'n': len(cohort_rows), 'arms': {}}
        for arm in INTERVENTIONS:
            values = {m: [r[arm][m] for r in cohort_rows] for m in metric_names}
            summary['cohorts'][cohort]['arms'][arm] = {m: mean(v) for m, v in values.items()}
            summary['cohorts'][cohort]['arms'][arm]['center_rank_changed'] = sum(
                r[arm]['center_rank_delta'] != 0 for r in cohort_rows)
            summary['cohorts'][cohort]['arms'][arm]['center_top1_changed'] = sum(
                r[arm]['center_distance_delta'] != 0 for r in cohort_rows)
            summary['cohorts'][cohort]['arms'][arm]['bbox_center_shift_mean'] = mean(
                [r[arm]['pred_bbox']['center_shift'] for r in cohort_rows])
            summary['cohorts'][cohort]['arms'][arm]['bbox_size_shift_mean'] = mean(
                [r[arm]['pred_bbox']['size_shift'] for r in cohort_rows])

    out = OUT / 'M1_NATIVE_MSHA_MECHANISM_ANALYSIS_20261023'
    out.mkdir(exist_ok=True)
    with (out / 'per_event_mechanism.jsonl').open('w') as handle:
        for row in details:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
