#!/usr/bin/env python3
"""Analyze C0/Cplain/Cscsu/gamma0 on the fixed Cplain diagnostic split."""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from lib.test.evaluation.lasherdataset import CplainLasHeRTrainDataset

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by); ix2, iy2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0., ix2-ix1) * max(0., iy2-iy1); union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.

def metrics(root, seq):
    pred = np.loadtxt(Path(root) / (seq.name + '.txt'), delimiter='\t', ndmin=2).astype(float)
    gt = np.asarray(seq.ground_truth_rect, dtype=float); n = min(len(pred), len(gt))
    ov = np.asarray([iou(pred[i], gt[i]) for i in range(n)])
    auc = float(np.mean([(ov >= t).mean() for t in np.linspace(0., 1., 101)]))
    return {'sequence': seq.name, 'frames': int(n), 'auc': auc, 'mean_iou': float(ov.mean()),
            'failure_frames_iou_lt_05': int((ov < .5).sum()), 'pred_len': int(len(pred)), 'gt_len': int(len(gt))}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--eval-list', required=True)
    ap.add_argument('--c0', required=True); ap.add_argument('--cplain', required=True)
    ap.add_argument('--cscsu', required=True); ap.add_argument('--gamma0', required=True)
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--output', required=True)
    a = ap.parse_args(); os.environ['TBSI_CPLAIN_EVAL_LIST'] = a.eval_list
    seqs = CplainLasHeRTrainDataset().get_sequence_list()
    roots = {'C0': a.c0, 'Cplain': a.cplain, 'Cscsu': a.cscsu, 'gamma0': a.gamma0}
    rows = {k: [metrics(v, s) for s in seqs] for k, v in roots.items()}
    by = {k: {x['sequence']: x for x in v} for k, v in rows.items()}
    deltas = {k: [] for k in ('Cplain_vs_C0', 'Cscsu_vs_C0')}
    for s in seqs:
        base = by['C0'][s.name]
        for key, cond in (('Cplain_vs_C0', 'Cplain'), ('Cscsu_vs_C0', 'Cscsu')):
            cur = by[cond][s.name]
            deltas[key].append({'sequence': s.name, 'delta_auc': cur['auc']-base['auc'],
                                'delta_mean_iou': cur['mean_iou']-base['mean_iou'],
                                'c0_auc': base['auc'], 'condition_auc': cur['auc']})
    parity = []
    for s in seqs:
        a0 = np.loadtxt(Path(a.c0)/(s.name+'.txt'), delimiter='\t', ndmin=2)
        ag = np.loadtxt(Path(a.gamma0)/(s.name+'.txt'), delimiter='\t', ndmin=2); n=min(len(a0),len(ag))
        parity.append({'sequence': s.name, 'length_equal': len(a0)==len(ag),
                       'max_abs_box_diff': float(np.max(np.abs(a0[:n]-ag[:n])))})
    summary = {}
    for k, v in rows.items(): summary[k] = {'mean_auc': float(np.mean([x['auc'] for x in v])), 'mean_iou': float(np.mean([x['mean_iou'] for x in v]))}
    for key, ds in deltas.items():
        vals = [x['delta_auc'] for x in ds]; ivals = [x['delta_mean_iou'] for x in ds]
        summary[key] = {'mean_delta_auc': float(np.mean(vals)), 'mean_delta_iou': float(np.mean(ivals)),
                        'wins_auc': int(sum(x>0 for x in vals)), 'losses_auc': int(sum(x<0 for x in vals)),
                        'ties_auc': int(sum(x==0 for x in vals)), 'severe_auc_drop_ge_005': int(sum(x<=-.05 for x in vals)),
                        'severe_iou_drop_ge_005': int(sum(x<=-.05 for x in ivals)),
                        'mean_delta_auc_without_max_gain': float(np.mean(sorted(vals)[:-1]))}
    state = torch.load(a.checkpoint, map_location='cpu')['net']
    trainable = {k: v for k, v in state.items() if 'early_search_residual' in k}
    result = {'sequences': len(seqs), 'metrics': rows, 'summary': summary, 'paired_deltas': deltas,
              'gamma0_parity': {'passed': all(x['length_equal'] and x['max_abs_box_diff']==0.0 for x in parity), 'per_sequence': parity},
              'cscsu_trainable_state_items': len(trainable), 'decision': 'PENDING'}
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(summary, indent=2)); print(json.dumps(result['gamma0_parity'], indent=2))

if __name__ == '__main__': main()
