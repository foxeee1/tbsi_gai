#!/usr/bin/env python3
"""Analyze the three G0-Cplain closed-loop result directories."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lib.test.evaluation.lasherdataset import CplainLasHeRTrainDataset


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def sequence_metrics(result_dir, sequence):
    pred_path = Path(result_dir) / (sequence.name + '.txt')
    pred = np.loadtxt(pred_path, delimiter='\t', ndmin=2).astype(float)
    gt = np.asarray(sequence.ground_truth_rect, dtype=float)
    n = min(len(pred), len(gt))
    overlaps = np.asarray([iou_xywh(pred[i], gt[i]) for i in range(n)])
    thresholds = np.linspace(0.0, 1.0, 101)
    success = np.asarray([(overlaps >= threshold).mean() for threshold in thresholds])
    return {'sequence': sequence.name, 'frames': int(n), 'mean_iou': float(overlaps.mean()),
            'auc': float(success.mean()), 'failure_frames_iou_lt_05': int((overlaps < 0.5).sum()),
            'pred_len': int(len(pred)), 'gt_len': int(len(gt))}


def residual_ratio(feature_dir, checkpoint):
    state = torch.load(checkpoint, map_location='cpu')['net']
    rgb_weight = state['backbone.early_search_residual.rgb_proj.weight']
    tir_weight = state['backbone.early_search_residual.tir_proj.weight']
    gamma_rgb = state['backbone.early_search_residual.gamma_rgb']
    gamma_tir = state['backbone.early_search_residual.gamma_tir']
    ratios = []
    for path in sorted(Path(feature_dir).glob('GE-*.npz')):
        z = np.load(path)
        late_rgb = torch.tensor(z['late_input_rgb'], dtype=torch.float32)
        late_tir = torch.tensor(z['late_input_tir'], dtype=torch.float32)
        early_rgb = torch.tensor(z['early_rgb'], dtype=torch.float32)
        early_tir = torch.tensor(z['early_tir'], dtype=torch.float32)
        delta_rgb = float(gamma_rgb) * torch.nn.functional.linear(early_rgb, rgb_weight)
        delta_tir = float(gamma_tir) * torch.nn.functional.linear(early_tir, tir_weight)
        ratios.append({'event_id': path.stem,
                       'rgb_residual_over_late': float(delta_rgb.norm() / late_rgb.norm().clamp_min(1e-8)),
                       'tir_residual_over_late': float(delta_tir.norm() / late_tir.norm().clamp_min(1e-8))})
    return {'gamma_rgb': float(gamma_rgb), 'gamma_tir': float(gamma_tir),
            'mean_rgb_residual_over_late': float(np.mean([x['rgb_residual_over_late'] for x in ratios])),
            'mean_tir_residual_over_late': float(np.mean([x['tir_residual_over_late'] for x in ratios])),
            'events': len(ratios), 'note': 'Estimated on frozen G0-Expand exported crops; not a tracker-side hook.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-list', required=True)
    parser.add_argument('--c0', required=True)
    parser.add_argument('--cplain', required=True)
    parser.add_argument('--gamma0', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--feature-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import os
    os.environ['TBSI_CPLAIN_EVAL_LIST'] = args.eval_list
    dataset = CplainLasHeRTrainDataset()
    sequences = dataset.get_sequence_list()
    dirs = {'C0': args.c0, 'Cplain': args.cplain, 'gamma0': args.gamma0}
    metrics = {name: [sequence_metrics(path, sequence) for sequence in sequences] for name, path in dirs.items()}
    by_name = {name: {row['sequence']: row for row in rows} for name, rows in metrics.items()}
    parity = []
    c0_dir, g0_dir = Path(args.c0), Path(args.gamma0)
    for sequence in sequences:
        a = np.loadtxt(c0_dir / (sequence.name + '.txt'), delimiter='\t', ndmin=2)
        b = np.loadtxt(g0_dir / (sequence.name + '.txt'), delimiter='\t', ndmin=2)
        n = min(len(a), len(b))
        parity.append({'sequence': sequence.name, 'frames_compared': n,
                       'max_abs_box_diff': float(np.max(np.abs(a[:n] - b[:n]))) if n else None,
                       'length_equal': bool(len(a) == len(b))})
    deltas = []
    for sequence in sequences:
        base = by_name['C0'][sequence.name]
        current = by_name['Cplain'][sequence.name]
        deltas.append({'sequence': sequence.name, 'delta_auc': current['auc'] - base['auc'],
                       'delta_mean_iou': current['mean_iou'] - base['mean_iou'],
                       'c0_auc': base['auc'], 'cplain_auc': current['auc'],
                       'c0_mean_iou': base['mean_iou'], 'cplain_mean_iou': current['mean_iou']})
    result = {'sequences': len(sequences), 'metrics': metrics, 'paired_deltas': deltas,
              'summary': {}, 'gamma0_parity': {'passed': all(x['max_abs_box_diff'] == 0.0 and x['length_equal'] for x in parity),
                                               'max_abs_box_diff': max(x['max_abs_box_diff'] for x in parity), 'per_sequence': parity},
              'residual': residual_ratio(args.feature_dir, args.checkpoint)}
    for name, rows in metrics.items():
        result['summary'][name] = {'mean_auc': float(np.mean([x['auc'] for x in rows])),
                                   'mean_iou': float(np.mean([x['mean_iou'] for x in rows])),
                                   'median_iou': float(np.median([x['mean_iou'] for x in rows])),
                                   'sequences': len(rows)}
    result['summary']['Cplain_vs_C0'] = {
        'mean_delta_auc': float(np.mean([x['delta_auc'] for x in deltas])),
        'mean_delta_iou': float(np.mean([x['delta_mean_iou'] for x in deltas])),
        'wins_auc': int(sum(x['delta_auc'] > 0 for x in deltas)),
        'losses_auc': int(sum(x['delta_auc'] < 0 for x in deltas)),
        'ties_auc': int(sum(x['delta_auc'] == 0 for x in deltas)),
        'hard_c0_sequences_auc_lt_05': int(sum(x['c0_auc'] < 0.5 for x in deltas)),
        'hard_c0_improved_auc': int(sum(x['c0_auc'] < 0.5 and x['delta_auc'] > 0 for x in deltas)),
        'severe_regression_auc_drop_ge_005': int(sum(x['delta_auc'] <= -0.05 for x in deltas)),
        'severe_regression_iou_drop_ge_005': int(sum(x['delta_mean_iou'] <= -0.05 for x in deltas)),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))
    print(json.dumps(result['gamma0_parity'], indent=2))
    print(json.dumps(result['residual'], indent=2))


if __name__ == '__main__':
    main()
