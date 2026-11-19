#!/usr/bin/env python3
"""Fixed-state attribution of AMCM reliability signals.

The M1 and AMCM arms see identical frames, crop, template, and state_before.
Only the AMCM pre-fusion adapter differs. No candidate or state is fed back.
"""
import csv, json, os, sys
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
M1_CFG = 'TBSI_E_MHSA_FULL_4090D_seed42'
AMCM_CFG = 'M1_AMCM_DR_DIAG2_PROXY_20261018'
M1_CKPT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
AMCM_CKPT = ROOT / 'output/experiments/M1_AMCM_DR_DIAG2_PROXY_20261018/checkpoints/TBSITrack_ep0003.pth.tar'
OUT = ROOT / 'output/experiments/M1_AMCM_DR_DIAG2_PROXY_20261018/fixed_state_signal_audit'


def iou(a, b):
    ix = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


def trace_row(seq, frame):
    for line in (TRACE / seq / 'frames.jsonl').read_text().splitlines():
        row = json.loads(line)
        if int(row['frame_id']) == int(frame):
            return row
    raise RuntimeError('missing trace {} {}'.format(seq, frame))


def load_events():
    with EVENTS.open() as handle:
        return list(csv.DictReader(handle))


def run_arm(arm, cfg_name, checkpoint, events, seqmap):
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(checkpoint)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_INTERNAL_DIAGNOSTIC'] = '1'
    os.environ['TBSI_M1_TRACE_DIR'] = str(OUT / ('trace_' + arm))
    params = parameters(cfg_name)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(checkpoint)
    tracker = TBSITrack(params, 'm1_amcm_fixed_state_signal_audit')
    rows = []
    current_seq = None
    for idx, event in enumerate(events, 1):
        seq = event['sequence']
        frame = int(event['frame'])
        trace = trace_row(seq, frame)
        state = np.asarray(trace['tracker_state_before'], dtype=np.float64)
        gt = np.asarray(trace['gt_box'], dtype=np.float64)
        dataset_seq = seqmap[seq]
        if current_seq != seq:
            first_v = cv2.cvtColor(cv2.imread(dataset_seq.frames[0][0]), cv2.COLOR_BGR2RGB)
            first_i = cv2.cvtColor(cv2.imread(dataset_seq.frames[1][0]), cv2.COLOR_BGR2RGB)
            first = np.concatenate((first_v, first_i), 2)
            tracker.initialize(first, {'init_bbox': np.asarray(dataset_seq.ground_truth_rect[0]),
                                       'sequence_name': seq})
            current_seq = seq
        tracker.state = state.tolist()
        rgb = cv2.cvtColor(cv2.imread(dataset_seq.frames[0][frame]), cv2.COLOR_BGR2RGB)
        tir = cv2.cvtColor(cv2.imread(dataset_seq.frames[1][frame]), cv2.COLOR_BGR2RGB)
        image = np.concatenate((rgb, tir), 2)
        out = tracker.track(image, {'gt_bbox': gt, 'sequence_name': seq, 'frame_num': frame})
        diag = tracker._last_fixed_state_diag
        trace_row_latest = json.loads((Path(os.environ['TBSI_M1_TRACE_DIR']) / seq / 'frames.jsonl').read_text().splitlines()[-1])
        peak_diag = trace_row_latest.get('peak_diagnostics', {})
        amcm_diag = trace_row_latest.get('amcm_diagnostics', {})
        pred_iou = iou(np.asarray(out['target_bbox'], dtype=np.float64), gt)
        score = diag['score_map'][0, 0]
        flat = score.flatten()
        peak = int(flat.argmax())
        gate = getattr(tracker.network, '_last_amcm_stats', None)
        if gate is None:
            gate_rgb = gate_tir = gate_disagreement = gate_rgb_signed = gate_tir_signed = 0.0
        else:
            gate_rgb_signed = float(gate['rgb_gate_map'].mean())
            gate_tir_signed = float(gate['tir_gate_map'].mean())
            gate_rgb = float(gate['rgb_gate_map'].abs().mean())
            gate_tir = float(gate['tir_gate_map'].abs().mean())
            gate_disagreement = float((gate['rgb_gate_map'] - gate['tir_gate_map']).abs().mean())
        rows.append({
            'event_index': idx, 'arm': arm, 'sequence': seq, 'frame': frame,
            'state_label': event.get('state_label', ''), 'fixed_state_iou': pred_iou,
            'peak_x': peak % 16, 'peak_y': peak // 16,
            'peak_score': float(flat[peak]), 'gate_rgb_abs_mean': gate_rgb,
            'gate_tir_abs_mean': gate_tir, 'gate_disagreement_mean': gate_disagreement,
            'gate_rgb_signed_mean': gate_rgb_signed, 'gate_tir_signed_mean': gate_tir_signed,
            'gt_rank_raw': peak_diag.get('gt_rank_raw', ''),
            'gt_rank_hann': peak_diag.get('gt_rank_hann', ''),
            'raw_peak_gt_margin': peak_diag.get('raw_peak_gt_margin', ''),
            'hann_peak_gt_margin': peak_diag.get('hann_peak_gt_margin', ''),
            'raw_peak_gt_distance': peak_diag.get('raw_peak_gt_distance', ''),
            'crop_contains_gt': peak_diag.get('crop_contains_gt', ''),
            'rgb_prob_mean': amcm_diag.get('prob_rgb_mean', ''),
            'tir_prob_mean': amcm_diag.get('prob_tir_mean', ''),
            'rgb_residual_norm': amcm_diag.get('residual_rgb_norm', ''),
            'tir_residual_norm': amcm_diag.get('residual_tir_norm', ''),
        })
        if idx % 20 == 0:
            print('[{}] {}/{} events'.format(arm, idx, len(events)), flush=True)
    return rows


def main():
    events = load_events()
    sequences = sorted({row['sequence'] for row in events})
    OUT.mkdir(parents=True, exist_ok=True)
    list_path = OUT / 'sequence_list.txt'
    list_path.write_text('\n'.join(sequences) + '\n')
    dataset = LasHeRDataset('trainingset', str(list_path)).get_sequence_list()
    seqmap = {seq.name: seq for seq in dataset}
    rows = []
    rows.extend(run_arm('m1', M1_CFG, M1_CKPT, events, seqmap))
    rows.extend(run_arm('amcm', AMCM_CFG, AMCM_CKPT, events, seqmap))
    with (OUT / 'per_event.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    by_key = {(r['sequence'], r['frame']): r for r in rows}
    paired = []
    for key in sorted({(r['sequence'], r['frame']) for r in rows}):
        m1, amcm = by_key[(key[0], key[1])], None
        for arm in ('m1', 'amcm'):
            if by_key[(key[0], key[1])]['arm'] == arm:
                pass
        arms = [r for r in rows if (r['sequence'], r['frame']) == key]
        m1 = [r for r in arms if r['arm'] == 'm1'][0]
        amcm = [r for r in arms if r['arm'] == 'amcm'][0]
        paired.append({
            'sequence': key[0], 'frame': key[1],
            'm1_iou': m1['fixed_state_iou'], 'amcm_iou': amcm['fixed_state_iou'],
            'delta_iou': amcm['fixed_state_iou'] - m1['fixed_state_iou'],
            'gate_rgb_abs_mean': amcm['gate_rgb_abs_mean'],
            'gate_tir_abs_mean': amcm['gate_tir_abs_mean'],
            'gate_disagreement_mean': amcm['gate_disagreement_mean'],
            'state_label': amcm['state_label'],
            'gate_rgb_signed_mean': amcm['gate_rgb_signed_mean'],
            'gate_tir_signed_mean': amcm['gate_tir_signed_mean'],
            'm1_gt_rank_hann': m1['gt_rank_hann'],
            'amcm_gt_rank_hann': amcm['gt_rank_hann'],
            'm1_hann_peak_gt_margin': m1['hann_peak_gt_margin'],
            'amcm_hann_peak_gt_margin': amcm['hann_peak_gt_margin'],
            'm1_raw_peak_gt_distance': m1['raw_peak_gt_distance'],
            'amcm_raw_peak_gt_distance': amcm['raw_peak_gt_distance'],
        })
    with (OUT / 'paired_events.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    rank_pairs = [(int(r['m1_gt_rank_hann']), int(r['amcm_gt_rank_hann'])) for r in paired
                  if r['m1_gt_rank_hann'] != '' and r['amcm_gt_rank_hann'] != '']
    summary = {'events': len(paired), 'sequences': len(sequences), 'output': str(OUT),
               'mean_m1_iou': float(np.mean([r['m1_iou'] for r in paired])),
               'mean_amcm_iou': float(np.mean([r['amcm_iou'] for r in paired])),
               'mean_delta_iou': float(np.mean([r['delta_iou'] for r in paired])),
               'positive_delta_events': int(sum(r['delta_iou'] > 0 for r in paired)),
               'negative_delta_events': int(sum(r['delta_iou'] < 0 for r in paired)),
               'rank_events': len(rank_pairs),
               'rank_improved': int(sum(a > b for a, b in rank_pairs)),
               'rank_unchanged': int(sum(a == b for a, b in rank_pairs)),
               'rank_harmed': int(sum(a < b for a, b in rank_pairs)),
               'mean_m1_gt_rank_hann': float(np.mean([a for a, _ in rank_pairs])) if rank_pairs else None,
               'mean_amcm_gt_rank_hann': float(np.mean([b for _, b in rank_pairs])) if rank_pairs else None}
    with (OUT / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
