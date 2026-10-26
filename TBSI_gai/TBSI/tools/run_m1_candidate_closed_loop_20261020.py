#!/usr/bin/env python3
"""Run one trained AMCM candidate on the fixed 25-event M1 cohort."""
import argparse, csv, json, os
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker

MANIFEST = ROOT / 'output/experiments/M1_PRECISION_SCREEN_MANIFEST_20261020/manifest.json'
BASELINE = ROOT / 'output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42'

def iou(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = max(0.0, a[2]) * max(0.0, a[3]) + max(0.0, b[2]) * max(0.0, b[3]) - inter
    return inter / union if union > 0 else 0.0

def read_baseline(seq, frame):
    p = BASELINE / (seq + '.txt')
    if not p.exists(): return None
    rows = []
    for line in p.read_text().splitlines():
        try: rows.append([float(x) for x in line.replace(',', '\t').split()[:4]])
        except ValueError: continue
    return rows[frame] if frame < len(rows) and len(rows[frame]) == 4 else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    os.environ['TBSI_INFER_FP16'] = '0'
    manifest_payload = json.loads(MANIFEST.read_text())
    # Fixed-event manifests exist in both the screening format
    # ({"selection": [...]}) and the earlier direct-list format.
    manifest = (manifest_payload['selection']
                if isinstance(manifest_payload, dict) and 'selection' in manifest_payload
                else manifest_payload)
    names = sorted({e['sequence'] for e in manifest})
    dataset = {s.name: s for s in get_dataset('lasher_test') if s.name in names}
    tracker_info = Tracker('tbsi_track', args.config, 'lasher_test')
    params = tracker_info.get_parameters()
    params.debug = 0
    tracker = tracker_info.create_tracker(params)
    by_key = {(e['sequence'], int(e['frame'])): e for e in manifest}
    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n, name in enumerate(names, 1):
        seq = dataset[name]
        frame = int(next(e for e in manifest if e['sequence'] == name)['frame'])
        # The diagnostic only needs the closed-loop path up to its fixed event.
        # Truncating here preserves initialization, per-frame state updates and
        # decoding while avoiding an unnecessary full-sequence replay.
        class PrefixSequence:
            pass
        prefix = PrefixSequence()
        prefix.name = seq.name
        prefix.frames = [seq.frames[0][:frame + 1], seq.frames[1][:frame + 1]]
        prefix.ground_truth_rect = seq.ground_truth_rect[:frame + 1]
        prefix.init_info = seq.init_info
        prefix.frame_info = seq.frame_info
        if hasattr(tracker, 'reset_temporal_tokens'):
            tracker.reset_temporal_tokens()
        output = tracker_info._track_sequence(tracker, prefix, prefix.init_info())
        preds = output.get('target_bbox', [])
        gt = seq.ground_truth_rect
        event = next(e for e in manifest if e['sequence'] == name)
        pred = preds[frame] if frame < len(preds) else None
        event_iou = iou(pred, event['gt_box']) if pred is not None else 0.0
        base = read_baseline(name, frame)
        base_iou = iou(base, event['gt_box']) if base is not None else float('nan')
        seq_ious = [iou(p, g) for p, g in zip(preds, gt)]
        rows.append({'sequence': name, 'role': event.get('role', 'fixed10'), 'frame': frame,
                     'm1_event_iou': base_iou, 'candidate_event_iou': event_iou,
                     'delta_event_iou': event_iou - base_iou if np.isfinite(base_iou) else float('nan'),
                     'm1_mean_iou': float('nan'), 'candidate_mean_iou': float(np.mean(seq_ious)) if seq_ious else float('nan'),
                     'frames': len(preds)})
        print('[{}/{}] {} event IoU {:.4f} delta {:.4f}'.format(n, len(names), name, event_iou, rows[-1]['delta_event_iou']), flush=True)
    with (args.out_dir / 'per_sequence.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for role in ('failure', 'control', 'all'):
        q = rows if role == 'all' else [r for r in rows if r['role'] == role]
        vals = [r['delta_event_iou'] for r in q if np.isfinite(r['delta_event_iou'])]
        print(json.dumps({'role': role, 'events': int(len(q)), 'mean_event_iou_delta': float(np.mean(vals)) if vals else None,
                          'wins': int(sum(v > 1e-6 for v in vals)), 'losses': int(sum(v < -1e-6 for v in vals))}, indent=2))
    (args.out_dir / 'summary.json').write_text(json.dumps({'config': args.config, 'events': len(rows), 'rows': rows}, indent=2))

if __name__ == '__main__': main()
