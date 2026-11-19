import csv
import json
import math
from pathlib import Path


ROOT = Path('/root/autodl-tmp/TBSI_gai/TBSI')
OUT = ROOT / 'output/experiments/M1_EARLY_GATE_CANDIDATE_CHECKPOINT_AB_20261024'
SEQUENCES = [
    'womanback2car', 'shinycarcoming', 'boydownplatform', '4thboywithwhite',
    'ab_bikeoccluded', '3rdfatboy', 'leftfarboycomingpicktheball',
]


def load(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iou(a, b):
    if not a or not b:
        return float('nan')
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def same_vector(a, b, tol=1e-6):
    return (a is not None and b is not None and len(a) == len(b) and
            all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b)))


rows, all_details = [], {}
for seq in SEQUENCES:
    off = load(OUT / 'trace/off' / seq / 'frames.jsonl')
    on = load(OUT / 'trace/on' / seq / 'frames.jsonl')
    off_by_id = {int(row['frame_id']): row for row in off}
    on_by_id = {int(row['frame_id']): row for row in on}
    aligned = []
    for fid in sorted(set(off_by_id) & set(on_by_id)):
        a, b = off_by_id[fid], on_by_id[fid]
        same_input = (same_vector(a.get('current_crop_xywh'), b.get('current_crop_xywh')) and
                      same_vector(a.get('tracker_state_before'), b.get('tracker_state_before')))
        pa, pb = a.get('peak_diagnostics', {}), b.get('peak_diagnostics', {})
        ba, bb = a.get('pred_box'), b.get('pred_box')
        center_delta = math.hypot((ba[0] + ba[2] / 2) - (bb[0] + bb[2] / 2),
                                  (ba[1] + ba[3] / 2) - (bb[1] + bb[3] / 2))
        aligned.append({
            'frame_id': fid,
            'same_crop_and_state': same_input,
            'iou_off': iou(ba, a.get('gt_box')),
            'iou_on': iou(bb, b.get('gt_box')),
            'iou_delta_on_minus_off': iou(bb, b.get('gt_box')) - iou(ba, a.get('gt_box')),
            'box_center_delta_px': center_delta,
            'hann_top1_off': pa.get('hann_top1_xy'),
            'hann_top1_on': pb.get('hann_top1_xy'),
            'hann_top1_changed': pa.get('hann_top1_xy') != pb.get('hann_top1_xy'),
            'gt_rank_hann_off': pa.get('gt_rank_hann'),
            'gt_rank_hann_on': pb.get('gt_rank_hann'),
            'gt_rank_hann_changed': pa.get('gt_rank_hann') != pb.get('gt_rank_hann'),
            'margin_hann_off': pa.get('target_background_margin_hann'),
            'margin_hann_on': pb.get('target_background_margin_hann'),
            'margin_hann_delta': (pb.get('target_background_margin_hann') - pa.get('target_background_margin_hann')
                                  if pa.get('target_background_margin_hann') is not None and
                                  pb.get('target_background_margin_hann') is not None else None),
            'crop_off': a.get('current_crop_xywh'), 'crop_on': b.get('current_crop_xywh'),
            'state_before_off': a.get('tracker_state_before'),
            'state_before_on': b.get('tracker_state_before'),
            'gt_valid': a.get('gt_valid'),
        })
    direct = [r for r in aligned if r['same_crop_and_state'] and r['gt_valid']]
    post = [r for r in aligned if not r['same_crop_and_state'] and r['gt_valid']]
    first_mismatch = next((r['frame_id'] for r in aligned if not r['same_crop_and_state']), None)
    rows.append({
        'sequence': seq, 'frames_off': len(off), 'frames_on': len(on),
        'aligned_frames': len(aligned), 'same_crop_state_frames': len(direct),
        'first_state_or_crop_mismatch_frame': first_mismatch,
        'direct_hann_top1_changed': sum(r['hann_top1_changed'] for r in direct),
        'direct_gt_rank_improved': sum(r['gt_rank_hann_on'] < r['gt_rank_hann_off']
                                       for r in direct if r['gt_rank_hann_on'] is not None and r['gt_rank_hann_off'] is not None),
        'direct_gt_rank_worsened': sum(r['gt_rank_hann_on'] > r['gt_rank_hann_off']
                                       for r in direct if r['gt_rank_hann_on'] is not None and r['gt_rank_hann_off'] is not None),
        'direct_mean_iou_delta': (sum(r['iou_delta_on_minus_off'] for r in direct) / len(direct) if direct else None),
        'direct_mean_hann_margin_delta': (sum(r['margin_hann_delta'] for r in direct if r['margin_hann_delta'] is not None) /
                                          max(1, sum(r['margin_hann_delta'] is not None for r in direct))),
        'closed_loop_mean_iou_delta': (sum(r['iou_delta_on_minus_off'] for r in post) / len(post) if post else None),
    })
    all_details[seq] = aligned

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / 'paired_summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with (OUT / 'paired_frames.json').open('w') as handle:
    json.dump(all_details, handle, indent=2)

print(json.dumps({'sequences': rows,
                  'direct_sequence_count_iou_positive': sum((r['direct_mean_iou_delta'] or 0) > 0 for r in rows),
                  'direct_sequence_count_rank_net_positive': sum(r['direct_gt_rank_improved'] > r['direct_gt_rank_worsened'] for r in rows)},
                 indent=2))
