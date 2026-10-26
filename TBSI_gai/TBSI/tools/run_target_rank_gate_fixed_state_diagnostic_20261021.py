#!/usr/bin/env python3
"""Fixed template/state/crop diagnostic for TargetRankGate."""
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker

MANIFEST = ROOT / 'output/experiments/M1_PRECISION_SCREEN_MANIFEST_20261020/manifest.json'


def run_arm(config, checkpoint, events, trace_root):
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(checkpoint)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_DISABLE_TRACKER_CACHE'] = '1'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_M1_TRACE_DIR'] = str(trace_root)
    Tracker._tracker_cache = None
    Tracker._tracker_cache_params_key = None
    info = Tracker('tbsi_track', config, 'lasher_test')
    dataset = {s.name: s for s in get_dataset('lasher_test')}
    rows = []
    for name in sorted(events):
        event = events[name]
        seq = dataset[name]
        params = info.get_parameters()
        params.debug = 0
        tracker = info.create_tracker(params)
        init_info = seq.init_info()
        init_info['sequence_name'] = name
        init_image = np.concatenate((info._read_image(seq.frames[0][0]),
                                     info._read_image(seq.frames[1][0])), axis=2)
        tracker.initialize(init_image, init_info)
        # Reuse the manifest state while keeping the same initialized template.
        tracker.state = [float(v) for v in event['tracker_state_before']]
        tracker.frame_id = int(event['frame']) - 1
        frame_idx = int(event['frame'])
        image = np.concatenate((info._read_image(seq.frames[0][frame_idx]),
                                info._read_image(seq.frames[1][frame_idx])), axis=2)
        frame_info = seq.frame_info(int(event['frame']))
        frame_info.update({'sequence_name': name, 'frame_num': int(event['frame']),
                           'gt_bbox': np.asarray(event['gt_box'], dtype=np.float32)})
        tracker.track(image, frame_info)
        trace_file = Path(trace_root) / name / 'frames.jsonl'
        trace_rows = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
        row = trace_rows[-1]
        peak = row.get('peak_diagnostics', {})
        rows.append({'sequence': name, 'role': event.get('role', 'unknown'),
                     'frame': int(event['frame']),
                     'state': event['tracker_state_before'],
                     'target_rank_score_gt': peak.get('target_rank_score_gt'),
                     'target_rank_score_background': peak.get('target_rank_score_background'),
                     'target_rank_score_margin': peak.get('target_rank_score_margin'),
                     'target_rank_score_gt_rank': peak.get('target_rank_score_gt_rank'),
                     'target_rank_score_top1_xy': peak.get('target_rank_score_top1_xy'),
                     'target_rank_score_std': peak.get('target_rank_score_std'),
                     'center_margin_raw': peak.get('target_background_margin_raw'),
                     'center_gt_rank_raw': peak.get('gt_rank_raw'),
                     'center_top1_xy': peak.get('raw_top1_xy'),
                     'center_peak_gt_margin': peak.get('raw_peak_gt_margin')})
        print('[{}] {}/{} {}'.format(config, len(rows), len(events), name), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate-config', required=True)
    ap.add_argument('--control-config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--control-checkpoint', default='')
    ap.add_argument('--output', required=True)
    ap.add_argument('--control-keep-gate', action='store_true')
    args = ap.parse_args()
    payload = json.loads(MANIFEST.read_text())
    events = {e['sequence']: e for e in payload['selection']}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    candidate_trace = out / 'candidate_trace'
    control_trace = out / 'control_trace'
    candidate = run_arm(args.candidate_config, args.checkpoint, events, candidate_trace)
    if args.control_checkpoint:
        parent = Path(args.control_checkpoint)
    else:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        state = ckpt.get('net', ckpt) if isinstance(ckpt, dict) else ckpt
        if not args.control_keep_gate:
            state = {k: v for k, v in state.items() if not k.startswith('target_rank_gate.')}
        if isinstance(ckpt, dict) and 'net' in ckpt:
            ckpt = dict(ckpt); ckpt['net'] = state
        else:
            ckpt = state
        parent = out / 'matched_parent_without_target_rank_gate.pth.tar'
        torch.save(ckpt, parent)
    control = run_arm(args.control_config, str(parent), events, control_trace)
    c_by = {r['sequence']: r for r in candidate}; p_by = {r['sequence']: r for r in control}
    paired = []
    for name in sorted(events):
        c, p = c_by[name], p_by[name]
        row = {'sequence': name, 'role': c['role'], 'frame': c['frame']}
        for key in ('target_rank_score_gt', 'target_rank_score_background', 'target_rank_score_margin',
                    'target_rank_score_gt_rank', 'target_rank_score_std', 'center_margin_raw',
                    'center_gt_rank_raw', 'center_peak_gt_margin'):
            row['candidate_' + key] = c[key]
            row['control_' + key] = p[key]
            if isinstance(c[key], (int, float)) and isinstance(p[key], (int, float)):
                row['delta_' + key] = c[key] - p[key]
        row['candidate_center_top1_xy'] = json.dumps(c['center_top1_xy'])
        row['control_center_top1_xy'] = json.dumps(p['center_top1_xy'])
        paired.append(row)
    with (out / 'fixed_state_paired.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    numeric = lambda key: [r['delta_' + key] for r in paired if r.get('delta_' + key) is not None]
    candidate_numeric = lambda key: [r['candidate_' + key] for r in paired
                                     if r.get('candidate_' + key) is not None]
    summary = {'events': len(paired),
               'candidate_target_rank_margin_mean': float(np.mean(candidate_numeric('target_rank_score_margin'))),
               'candidate_target_rank_margin_median': float(np.median(candidate_numeric('target_rank_score_margin'))),
               'candidate_target_rank_positive_events': int(sum(v > 0 for v in candidate_numeric('target_rank_score_margin'))),
               'candidate_target_rank_gt_rank_mean': float(np.mean(candidate_numeric('target_rank_score_gt_rank'))),
               'candidate_target_rank_gt_rank_one_events': int(sum(v <= 1 for v in candidate_numeric('target_rank_score_gt_rank'))),
               'candidate_target_rank_score_std_mean': float(np.mean(candidate_numeric('target_rank_score_std'))),
               'center_margin_delta_mean': float(np.mean(numeric('center_margin_raw'))),
               'center_margin_delta_median': float(np.median(numeric('center_margin_raw'))),
               'center_gt_rank_delta_mean': float(np.mean(numeric('center_gt_rank_raw'))),
               'center_peak_gt_margin_delta_mean': float(np.mean(numeric('center_peak_gt_margin')))}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
