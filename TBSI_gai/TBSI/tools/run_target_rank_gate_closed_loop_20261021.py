#!/usr/bin/env python3
"""Matched closed-loop comparison for the minimal TargetRankGate screen."""
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker

MANIFEST = ROOT / 'output/experiments/M1_PRECISION_SCREEN_MANIFEST_20261020/manifest.json'


def iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


def run_arm(config, checkpoint, sequences, events):
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(checkpoint)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_DISABLE_TRACKER_CACHE'] = '1'
    Tracker._tracker_cache = None
    Tracker._tracker_cache_params_key = None
    info = Tracker('tbsi_track', config, 'lasher_test')
    dataset = {s.name: s for s in get_dataset('lasher_test') if s.name in sequences}
    rows = []
    for idx, name in enumerate(sequences, 1):
        seq = dataset[name]
        event = events[name]
        frame = int(event['frame'])
        class Prefix: pass
        prefix = Prefix()
        prefix.name = seq.name
        prefix.frames = [seq.frames[0][:frame + 1], seq.frames[1][:frame + 1]]
        prefix.ground_truth_rect = seq.ground_truth_rect[:frame + 1]
        prefix.init_info = seq.init_info
        prefix.frame_info = seq.frame_info
        params = info.get_parameters()
        params.debug = 0
        tracker = info.create_tracker(params)
        output = info._track_sequence(tracker, prefix, prefix.init_info())
        preds = output.get('target_bbox', [])
        pred = preds[frame] if frame < len(preds) else [0, 0, 0, 0]
        gt = event['gt_box']
        event_iou = iou(pred, gt)
        vals = [iou(p, g) for p, g in zip(preds, prefix.ground_truth_rect)]
        rows.append({'sequence': name, 'role': event.get('role', 'unknown'), 'frame': frame,
                     'event_iou': event_iou,
                     'mean_prefix_iou': float(np.mean(vals)) if vals else 0.0,
                     'frames': len(preds)})
        print('[{}] {}/{} {} event_iou={:.5f}'.format(config, idx, len(sequences), name, event_iou), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate-config', required=True)
    ap.add_argument('--control-config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    payload = json.loads(MANIFEST.read_text())
    selected = payload['selection'] if 'selection' in payload else payload
    events = {e['sequence']: e for e in selected}
    sequences = sorted(events)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    candidate = run_arm(args.candidate_config, args.checkpoint, sequences, events)
    # Matched parent: remove only the newly introduced module parameters so the
    # strict loader can instantiate the disabled configuration.
    parent_ckpt = out / 'matched_parent_without_target_rank_gate.pth.tar'
    payload = torch.load(args.checkpoint, map_location='cpu')
    state = payload.get('net', payload) if isinstance(payload, dict) else payload
    state = {k: v for k, v in state.items() if not k.startswith('target_rank_gate.')}
    if isinstance(payload, dict) and 'net' in payload:
        payload = dict(payload); payload['net'] = state
    else:
        payload = state
    torch.save(payload, str(parent_ckpt))
    control = run_arm(args.control_config, str(parent_ckpt), sequences, events)
    by_c = {r['sequence']: r for r in candidate}
    by_p = {r['sequence']: r for r in control}
    paired = []
    for name in sequences:
        c, p = by_c[name], by_p[name]
        paired.append({'sequence': name, 'role': c['role'], 'frame': c['frame'],
                       'candidate_event_iou': c['event_iou'], 'control_event_iou': p['event_iou'],
                       'delta_event_iou': c['event_iou'] - p['event_iou'],
                       'candidate_mean_prefix_iou': c['mean_prefix_iou'],
                       'control_mean_prefix_iou': p['mean_prefix_iou'],
                       'delta_mean_prefix_iou': c['mean_prefix_iou'] - p['mean_prefix_iou']})
    with (out / 'paired_per_sequence.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
    summary = {'candidate_config': args.candidate_config, 'control_config': args.control_config,
               'checkpoint': args.checkpoint, 'events': len(paired),
               'event_iou_delta_mean': float(np.mean([r['delta_event_iou'] for r in paired])),
               'event_iou_delta_median': float(np.median([r['delta_event_iou'] for r in paired])),
               'event_wins': int(sum(r['delta_event_iou'] > 1e-6 for r in paired)),
               'event_losses': int(sum(r['delta_event_iou'] < -1e-6 for r in paired)),
               'prefix_iou_delta_mean': float(np.mean([r['delta_mean_prefix_iou'] for r in paired])),
               'failure_delta_mean': float(np.mean([r['delta_event_iou'] for r in paired if r['role'] == 'failure'])),
               'control_delta_mean': float(np.mean([r['delta_event_iou'] for r in paired if r['role'] == 'control']))}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
