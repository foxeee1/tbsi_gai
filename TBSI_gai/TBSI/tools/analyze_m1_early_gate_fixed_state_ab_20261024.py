import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp/TBSI_gai/TBSI')
EXP = ROOT / 'output/experiments/M1_EARLY_GATE_CANDIDATE_CHECKPOINT_AB_20261024/fixed_state_maps'
payload = json.loads((EXP / 'paired_trace.json').read_text())
off = {(r['sequence_id'], int(r['frame_id'])): r for r in payload['off']}
on = {(r['sequence_id'], int(r['frame_id'])): r for r in payload['on']}
maps_off = np.load(EXP / 'head_maps_off.npz')
maps_on = np.load(EXP / 'head_maps_on.npz')
map_index_off = {str(key): i for i, key in enumerate(maps_off['keys'])}
map_index_on = {str(key): i for i, key in enumerate(maps_on['keys'])}
rows = []


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


for event in payload['events']:
    key = (event['sequence'], int(event['frame_id']))
    a, b = off[key], on[key]
    pa, pb = a.get('peak_diagnostics', {}), b.get('peak_diagnostics', {})
    ba, bb, gt = a['pred_box'], b['pred_box'], event['gt_box']
    map_key = '{}:{}'.format(key[0], key[1])
    ia, ib = map_index_off[map_key], map_index_on[map_key]
    score_a, score_b = maps_off['score_map'][ia], maps_on['score_map'][ib]
    size_a, size_b = maps_off['size_map'][ia], maps_on['size_map'][ib]
    offset_a, offset_b = maps_off['offset_map'][ia], maps_on['offset_map'][ib]
    hann = maps_off['hann'][ia]
    hann_a, hann_b = score_a * hann, score_b * hann
    center_delta = math.hypot((ba[0] + ba[2] / 2) - (bb[0] + bb[2] / 2),
                              (ba[1] + ba[3] / 2) - (bb[1] + bb[3] / 2))
    rows.append({
        'sequence': key[0], 'frame_id': key[1],
        'crop_equal': a.get('current_crop_xywh') == b.get('current_crop_xywh'),
        'state_before_equal': a.get('tracker_state_before') == b.get('tracker_state_before'),
        'hann_top1_off': json.dumps(pa.get('hann_top1_xy')),
        'hann_top1_on': json.dumps(pb.get('hann_top1_xy')),
        'hann_top1_changed': pa.get('hann_top1_xy') != pb.get('hann_top1_xy'),
        'gt_rank_off': pa.get('gt_rank_hann'), 'gt_rank_on': pb.get('gt_rank_hann'),
        'gt_rank_delta_on_minus_off': (pb.get('gt_rank_hann') - pa.get('gt_rank_hann')
                                       if pa.get('gt_rank_hann') is not None and pb.get('gt_rank_hann') is not None else None),
        'hann_margin_off': pa.get('target_background_margin_hann'),
        'hann_margin_on': pb.get('target_background_margin_hann'),
        'hann_margin_delta_on_minus_off': (pb.get('target_background_margin_hann') - pa.get('target_background_margin_hann')
                                            if pa.get('target_background_margin_hann') is not None and pb.get('target_background_margin_hann') is not None else None),
        'iou_off': iou(ba, gt), 'iou_on': iou(bb, gt),
        'iou_delta_on_minus_off': iou(bb, gt) - iou(ba, gt),
        'box_center_delta_px': center_delta,
        'raw_score_map_mean_abs_delta': float(np.mean(np.abs(score_b - score_a))),
        'raw_score_map_max_abs_delta': float(np.max(np.abs(score_b - score_a))),
        'hann_score_map_mean_abs_delta': float(np.mean(np.abs(hann_b - hann_a))),
        'size_map_mean_abs_delta': float(np.mean(np.abs(size_b - size_a))),
        'offset_map_mean_abs_delta': float(np.mean(np.abs(offset_b - offset_a))),
        'target_rank_score_margin_off': pa.get('target_rank_score_margin'),
        'target_rank_score_margin_on': pb.get('target_rank_score_margin'),
    })

with (EXP / 'fixed_state_paired.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
summary = {
    'events': len(rows),
    'all_crop_equal': all(r['crop_equal'] for r in rows),
    'all_state_before_equal': all(r['state_before_equal'] for r in rows),
    'valid_hann_top1_events': sum(r['hann_top1_off'] != 'null' and r['hann_top1_on'] != 'null' for r in rows),
    'hann_top1_changed': sum(r['hann_top1_changed'] and r['hann_top1_off'] != 'null' and r['hann_top1_on'] != 'null' for r in rows),
    'valid_gt_rank_events': sum(r['gt_rank_off'] is not None and r['gt_rank_off'] not in ('', 'None', 'null') and
                                r['gt_rank_on'] is not None and r['gt_rank_on'] not in ('', 'None', 'null') for r in rows),
    'gt_rank_improved': sum(r['gt_rank_delta_on_minus_off'] < 0 for r in rows if r['gt_rank_delta_on_minus_off'] is not None),
    'gt_rank_worsened': sum(r['gt_rank_delta_on_minus_off'] > 0 for r in rows if r['gt_rank_delta_on_minus_off'] is not None),
    'mean_iou_delta': sum(r['iou_delta_on_minus_off'] for r in rows) / len(rows),
    'median_iou_delta': statistics.median(r['iou_delta_on_minus_off'] for r in rows),
    'mean_hann_margin_delta': sum(r['hann_margin_delta_on_minus_off'] for r in rows if r['hann_margin_delta_on_minus_off'] is not None) / max(1, sum(r['hann_margin_delta_on_minus_off'] is not None for r in rows)),
    'positive_iou_events': sum(r['iou_delta_on_minus_off'] > 0 for r in rows),
    'negative_iou_events': sum(r['iou_delta_on_minus_off'] < 0 for r in rows),
    'sequence_count_positive_mean_iou': sum(sum(r['iou_delta_on_minus_off'] for r in rows if r['sequence'] == s) > 0 for s in sorted(set(r['sequence'] for r in rows))),
    'mean_raw_score_map_abs_delta': statistics.mean(r['raw_score_map_mean_abs_delta'] for r in rows),
    'mean_hann_score_map_abs_delta': statistics.mean(r['hann_score_map_mean_abs_delta'] for r in rows),
    'mean_size_map_abs_delta': statistics.mean(r['size_map_mean_abs_delta'] for r in rows),
    'mean_offset_map_abs_delta': statistics.mean(r['offset_map_mean_abs_delta'] for r in rows),
}
(EXP / 'fixed_state_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
