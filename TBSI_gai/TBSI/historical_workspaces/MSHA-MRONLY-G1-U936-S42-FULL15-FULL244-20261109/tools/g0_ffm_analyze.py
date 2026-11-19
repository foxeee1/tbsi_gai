#!/usr/bin/env python3
"""Analyze the fixed 24-sequence FFM short-run outputs."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.test.evaluation.lasherdataset import CplainLasHeRTrainDataset


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def metrics(result_dir, sequence):
    pred = np.loadtxt(Path(result_dir) / (sequence.name + '.txt'), delimiter='\t', ndmin=2).astype(float)
    gt = np.asarray(sequence.ground_truth_rect, dtype=float)
    n = min(len(pred), len(gt))
    overlaps = np.asarray([iou_xywh(pred[i], gt[i]) for i in range(n)])
    thresholds = np.linspace(0.0, 1.0, 101)
    auc = float(np.asarray([(overlaps >= t).mean() for t in thresholds]).mean())
    return {'sequence': sequence.name, 'frames': int(n), 'mean_iou': float(overlaps.mean()),
            'auc': auc, 'fail_iou_lt_05': int((overlaps < 0.5).sum()),
            'pred_len': int(len(pred)), 'gt_len': int(len(gt))}


def trim_mean(values, trim=0.1):
    values = np.sort(np.asarray(values, dtype=float))
    k = int(len(values) * trim)
    return float(values[k:len(values) - k].mean()) if len(values) > 2 * k else float(values.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-list', required=True)
    ap.add_argument('--c0', required=True)
    ap.add_argument('--cdiag', required=True)
    ap.add_argument('--cspatial', required=True)
    ap.add_argument('--gamma0', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    os.environ['TBSI_CPLAIN_EVAL_LIST'] = args.eval_list
    sequences = CplainLasHeRTrainDataset().get_sequence_list()
    dirs = {'C0': args.c0, 'Cdiag': args.cdiag, 'Cspatial': args.cspatial, 'Cspatial_gamma0': args.gamma0}
    rows = {name: [metrics(path, seq) for seq in sequences] for name, path in dirs.items()}
    by = {name: {x['sequence']: x for x in value} for name, value in rows.items()}
    deltas = {}
    for name in ('Cdiag', 'Cspatial'):
        deltas[name] = []
        for seq in sequences:
            a, b = by['C0'][seq.name], by[name][seq.name]
            deltas[name].append({'sequence': seq.name, 'delta_auc': b['auc'] - a['auc'],
                                 'delta_iou': b['mean_iou'] - a['mean_iou'],
                                 'c0_auc': a['auc'], 'candidate_auc': b['auc']})
    parity = []
    for seq in sequences:
        a = np.loadtxt(Path(args.c0) / (seq.name + '.txt'), delimiter='\t', ndmin=2)
        b = np.loadtxt(Path(args.gamma0) / (seq.name + '.txt'), delimiter='\t', ndmin=2)
        n = min(len(a), len(b))
        parity.append({'sequence': seq.name, 'length_equal': bool(len(a) == len(b)),
                       'frames': n, 'max_abs_box_diff': float(np.max(np.abs(a[:n] - b[:n])))})
    summary = {}
    for name, values in rows.items():
        aucs = [x['auc'] for x in values]
        ious = [x['mean_iou'] for x in values]
        summary[name] = {'sequences': len(values), 'mean_auc': float(np.mean(aucs)),
                         'mean_iou': float(np.mean(ious)), 'median_auc': float(np.median(aucs)),
                         'trimmed_mean_auc_10pct': trim_mean(aucs), 'trimmed_mean_iou_10pct': trim_mean(ious)}
    for name, values in deltas.items():
        da = [x['delta_auc'] for x in values]
        di = [x['delta_iou'] for x in values]
        summary[name + '_vs_C0'] = {
            'mean_delta_auc': float(np.mean(da)), 'mean_delta_iou': float(np.mean(di)),
            'median_delta_auc': float(np.median(da)), 'trimmed_mean_delta_auc_10pct': trim_mean(da),
            'wins_auc': int(sum(x > 0 for x in da)), 'losses_auc': int(sum(x < 0 for x in da)),
            'ties_auc': int(sum(x == 0 for x in da)),
            'severe_auc_regressions_le_-0.05': int(sum(x <= -0.05 for x in da)),
            'largest_gain_auc': float(max(da)), 'largest_loss_auc': float(min(da)),
        }
    result = {'eval_list': args.eval_list, 'sequences': [s.name for s in sequences],
              'summary': summary, 'per_sequence_deltas': deltas, 'metrics': rows,
              'gamma0_parity': {'passed': all(x['length_equal'] and x['max_abs_box_diff'] == 0.0 for x in parity),
                                'max_abs_box_diff': max(x['max_abs_box_diff'] for x in parity), 'per_sequence': parity},
              'decision': 'PENDING_WINDOW_ANALYSIS'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'gamma0_parity': result['gamma0_parity']}, indent=2))


if __name__ == '__main__':
    main()
