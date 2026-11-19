#!/usr/bin/env python3
"""Fixed-state residual scale screen for E5/E6 checkpoints."""
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack

EVENTS = ROOT / 'output/experiments/M1_CANDIDATE_RECALL_READOUT_20261010/analysis/per_event.csv'
TRACE = ROOT / 'output/experiments/M1_CANDIDATE_RECALL_READOUT_20261010/trace'
OUT = ROOT / 'output/experiments/M1_AMCM_RESIDUAL_SCALE_SCREEN_E5_E6_20261019/fixed_state'
ARMS = {
    'E5': ('M1_AMCM_DR_LOSS_E5_NOSUP_QUICK15_3EP_20261019',
           ROOT / 'output/experiments/M1_AMCM_DR_LOSS_E5_NOSUP_QUICK15_3EP_20261019/checkpoints/TBSITrack_ep0003.pth.tar'),
    'E6': ('M1_AMCM_DR_LOSS_E6_REPR_QUICK15_3EP_20261019',
           ROOT / 'output/experiments/M1_AMCM_DR_LOSS_E6_REPR_QUICK15_3EP_20261019/checkpoints/TBSITrack_ep0003.pth.tar'),
}
SCALES = (0.0, 0.5, 1.0, 2.0)


def read_trace(sequence, frame):
    path = TRACE / sequence / 'frames.jsonl'
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if int(row['frame_id']) == int(frame):
            return row
    raise RuntimeError('missing trace {} {}'.format(sequence, frame))


def box_iou(a, b):
    ix = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


def load_events():
    with EVENTS.open() as handle:
        return list(csv.DictReader(handle))


def attach_scale_hook(tracker, scale):
    module = getattr(tracker.network, 'amcm_reliability', None)
    if module is None:
        raise RuntimeError('checkpoint config did not construct AMCM reliability module')
    def rescale(_module, _inputs, output):
        rgb, tir, stats = output
        in_rgb, in_tir = _inputs[:2]
        return (in_rgb + scale * (rgb - in_rgb),
                in_tir + scale * (tir - in_tir), stats)
    return module.register_forward_hook(rescale)


def run_arm(arm, cfg_name, checkpoint, events, seqmap):
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(checkpoint)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_INTERNAL_DIAGNOSTIC'] = '1'
    rows = []
    for scale in SCALES:
        os.environ['TBSI_M1_TRACE_DIR'] = str(OUT / ('trace_{}_a{}'.format(arm, scale)))
        params = parameters(cfg_name)
        params.debug = 0
        params.save_all_boxes = False
        params.checkpoint = str(checkpoint)
        tracker = TBSITrack(params, 'm1_amcm_residual_scale_screen')
        handle = attach_scale_hook(tracker, scale)
        current_seq = None
        for idx, event in enumerate(events, 1):
            seq, frame = event['sequence'], int(event['frame'])
            trace = read_trace(seq, frame)
            state = np.asarray(trace['tracker_state_before'], dtype=np.float64)
            gt = np.asarray(trace['gt_box'], dtype=np.float64)
            ds = seqmap[seq]
            if current_seq != seq:
                v = cv2.cvtColor(cv2.imread(ds.frames[0][0]), cv2.COLOR_BGR2RGB)
                i = cv2.cvtColor(cv2.imread(ds.frames[1][0]), cv2.COLOR_BGR2RGB)
                tracker.initialize(np.concatenate((v, i), 2), {
                    'init_bbox': np.asarray(ds.ground_truth_rect[0]), 'sequence_name': seq})
                current_seq = seq
            tracker.state = state.tolist()
            v = cv2.cvtColor(cv2.imread(ds.frames[0][frame]), cv2.COLOR_BGR2RGB)
            i = cv2.cvtColor(cv2.imread(ds.frames[1][frame]), cv2.COLOR_BGR2RGB)
            pred = tracker.track(np.concatenate((v, i), 2), {
                'gt_bbox': gt, 'sequence_name': seq, 'frame_num': frame})
            diag = tracker._last_fixed_state_diag
            trace_out = json.loads((Path(os.environ['TBSI_M1_TRACE_DIR']) / seq / 'frames.jsonl').read_text().splitlines()[-1])
            pdiag = trace_out.get('peak_diagnostics', {})
            amap = diag['score_map'][0, 0].numpy()
            hann = tracker.output_window.cpu().numpy().reshape(amap.shape)
            response = amap * hann
            order = np.argsort(response.reshape(-1))[::-1]
            peak, second = int(order[0]), int(order[1])
            rows.append({
                'arm': arm, 'alpha': scale, 'event_index': idx, 'sequence': seq,
                'frame': frame, 'state_label': event.get('state_label', ''),
                'fixed_state_iou': box_iou(np.asarray(pred['target_bbox'], dtype=np.float64), gt),
                'gt_rank_hann': pdiag.get('gt_rank_hann', ''),
                'hann_peak_gt_margin': pdiag.get('hann_peak_gt_margin', ''),
                'peak_flat_index': peak, 'peak_score': float(response.reshape(-1)[peak]),
                'peak2_score': float(response.reshape(-1)[second]),
                'peak_margin': float(response.reshape(-1)[peak] - response.reshape(-1)[second]),
                'crop_contains_gt': pdiag.get('crop_contains_gt', ''),
                'rgb_residual_norm': trace_out.get('amcm_diagnostics', {}).get('residual_rgb_norm', ''),
                'tir_residual_norm': trace_out.get('amcm_diagnostics', {}).get('residual_tir_norm', ''),
            })
            if idx % 30 == 0:
                print('[{} alpha={}] {}/{}'.format(arm, scale, idx, len(events)), flush=True)
        handle.remove()
        del tracker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events()
    sequences = sorted({r['sequence'] for r in events})
    seqfile = OUT / 'sequence_list.txt'
    seqfile.write_text('\n'.join(sequences) + '\n')
    dataset = LasHeRDataset('trainingset', str(seqfile)).get_sequence_list()
    seqmap = {s.name: s for s in dataset}
    rows = []
    for arm, (cfg, ckpt) in ARMS.items():
        if not ckpt.is_file():
            raise FileNotFoundError(str(ckpt))
        rows.extend(run_arm(arm, cfg, ckpt, events, seqmap))
    fields = list(rows[0])
    with (OUT / 'per_event.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summaries = []
    for arm in ARMS:
        for scale in SCALES:
            subset = [r for r in rows if r['arm'] == arm and r['alpha'] == scale]
            summaries.append({'arm': arm, 'alpha': scale, 'events': len(subset),
                'sequences': len({r['sequence'] for r in subset}),
                'mean_fixed_state_iou': float(np.mean([r['fixed_state_iou'] for r in subset])),
                'positive_events': int(sum(r['fixed_state_iou'] > 0 for r in subset))})
    with (OUT / 'summary.json').open('w') as f:
        json.dump({'events': len(events), 'sequences': len(sequences), 'arms': summaries}, f, indent=2)
    print(json.dumps({'events': len(events), 'sequences': len(sequences), 'arms': summaries}, indent=2))


if __name__ == '__main__':
    main()
