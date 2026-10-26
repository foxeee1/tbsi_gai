#!/usr/bin/env python3
"""Compute fixed rescue-event [t,t+30] windows from completed FFM runs."""
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


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    u = aw * ah + bw * bh - ix * iy
    return ix * iy / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-list', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--c0', required=True)
    ap.add_argument('--cdiag', required=True)
    ap.add_argument('--cspatial', required=True)
    ap.add_argument('--gamma0', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    os.environ['TBSI_CPLAIN_EVAL_LIST'] = args.eval_list
    names = {s.name for s in CplainLasHeRTrainDataset().get_sequence_list()}
    events = []
    for line in Path(args.events).read_text().splitlines():
        x = json.loads(line)
        if 'rescue_failure' in x.get('groups', []) and x['sequence'] in names:
            events.append(x)
    dirs = {'C0': args.c0, 'Cdiag': args.cdiag, 'Cspatial': args.cspatial, 'Cspatial_gamma0': args.gamma0}
    rows = []
    for e in events:
        out = {'event_id': e['event_id'], 'sequence': e['sequence'], 'frame': e['frame'],
               'm1_iou': e['m1_iou'], 'center_distance_tokens': e['center_distance_tokens'],
               'pred_area_over_gt_area': (e['pred_box'][2] * e['pred_box'][3]) / max(1e-9, e['gt_box'][2] * e['gt_box'][3])}
        for name, path in dirs.items():
            pred = np.loadtxt(Path(path) / (e['sequence'] + '.txt'), delimiter='\t', ndmin=2)
            start = max(0, int(e['frame']) - 1)
            vals = [iou(pred[i], e['gt_box']) for i in range(start, min(len(pred), start + 31))]
            out[name] = {'window_frames': len(vals), 'mean_iou': float(np.mean(vals)) if vals else None,
                         'reached_05': bool(any(v >= 0.5 for v in vals)),
                         'first_recovery_offset': next((i for i, v in enumerate(vals) if v >= 0.5), None)}
        rows.append(out)
    summary = {}
    for name in dirs:
        vals = [x[name]['mean_iou'] for x in rows if x[name]['mean_iou'] is not None]
        rec = [x[name]['reached_05'] for x in rows]
        summary[name] = {'events': len(vals), 'mean_window_iou': float(np.mean(vals)) if vals else None,
                         'recovery_rate': float(np.mean(rec)) if rec else None}
    result = {'contract': '[t,t+30], event frame converted from 1-based to zero-based; GT used only for post-run scoring',
              'events': rows, 'summary': summary, 'coverage_note': 'Only rescue events on the fixed 24-sequence FFM dev list are included.'}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
