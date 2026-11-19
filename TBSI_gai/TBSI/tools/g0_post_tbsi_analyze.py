#!/usr/bin/env python3
"""Sequence-level and fixed-stratum analysis for the post-TBSI G0 export."""
import argparse
import json
from pathlib import Path

import numpy as np


def iou(a, b):
    ax, ay, aw, ah = [float(x) for x in a]
    bx, by, bw, bh = [float(x) for x in b]
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def bootstrap(values, seed=42, draws=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    sampled = values[rng.randint(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()),
            'lo95': float(np.quantile(sampled, 0.025)),
            'hi95': float(np.quantile(sampled, 0.975)),
            'n_sequences': int(len(values))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    metrics = json.loads((root / 'event_metrics.json').read_text())
    by_id = {row['event_id']: row for row in metrics}
    usable = [row for row in rows if row['event_id'] in by_id]

    enriched = []
    for row in usable:
        m = by_id[row['event_id']]
        gt = row['gt_box']; pred = row['pred_box']
        cx = float(gt[0] + gt[2] / 2.0)
        cy = float(gt[1] + gt[3] / 2.0)
        state = row['state_before']
        crop_center_x = float(state[0] + state[2] / 2.0)
        crop_center_y = float(state[1] + state[3] / 2.0)
        # Fixed before-results strata: target center within 2 token units of crop center.
        center_distance_tokens = float(np.hypot(cx - crop_center_x, cy - crop_center_y) /
                                       (256.0 / 16.0))
        enriched.append({
            'event_id': row['event_id'], 'sequence': row['sequence'],
            'center_distance_tokens': center_distance_tokens,
            'center_stratum': 'centered_le_2_tokens' if center_distance_tokens <= 2.0 else 'offcenter_gt_2_tokens',
            'm1_iou': iou(pred, gt),
            'm1_iou_stratum': 'easy_iou_ge_0.5' if iou(pred, gt) >= 0.5 else 'hard_iou_lt_0.5',
            'correct_margin': m['post_tbsi']['diag_target_background_margin'],
            'event_margin': m['post_tbsi_event_shuffle']['diag_target_background_margin'],
            'spatial_margin': m['post_tbsi_spatial_shift']['diag_target_background_margin'],
            'correct_ap': m['post_tbsi']['token_ap'],
            'event_ap': m['post_tbsi_event_shuffle']['token_ap'],
            'spatial_ap': m['post_tbsi_spatial_shift']['token_ap'],
            'xy_ap': m['post_tbsi']['xy_ap'],
        })

    def group(rows_in, key, value):
        selected = [r for r in rows_in if r[key] == value]
        if not selected:
            return {'events': 0}
        return {
            'events': len(selected),
            'correct_margin': float(np.mean([r['correct_margin'] for r in selected])),
            'event_margin': float(np.mean([r['event_margin'] for r in selected])),
            'spatial_margin': float(np.mean([r['spatial_margin'] for r in selected])),
            'correct_ap': float(np.mean([r['correct_ap'] for r in selected])),
            'event_ap': float(np.mean([r['event_ap'] for r in selected])),
            'spatial_ap': float(np.mean([r['spatial_ap'] for r in selected])),
            'xy_ap': float(np.mean([r['xy_ap'] for r in selected])),
            'correct_minus_event_ap': float(np.mean([r['correct_ap'] - r['event_ap'] for r in selected])),
            'correct_minus_spatial_ap': float(np.mean([r['correct_ap'] - r['spatial_ap'] for r in selected])),
        }

    seq = {}
    for name in sorted(set(r['sequence'] for r in enriched)):
        selected = [r for r in enriched if r['sequence'] == name]
        seq[name] = {
            'events': len(selected),
            'correct_minus_event_margin': float(np.mean([r['correct_margin'] - r['event_margin'] for r in selected])),
            'correct_minus_spatial_margin': float(np.mean([r['correct_margin'] - r['spatial_margin'] for r in selected])),
            'correct_minus_event_ap': float(np.mean([r['correct_ap'] - r['event_ap'] for r in selected])),
            'correct_minus_spatial_ap': float(np.mean([r['correct_ap'] - r['spatial_ap'] for r in selected])),
        }
    seq_values = {key: [value[key] for value in seq.values()] for key in next(iter(seq.values())) if key != 'events'}
    result = {
        'events_usable': len(enriched), 'sequences_with_usable_events': len(seq),
        'fixed_strata': {
            'centered_le_2_tokens': group(enriched, 'center_stratum', 'centered_le_2_tokens'),
            'offcenter_gt_2_tokens': group(enriched, 'center_stratum', 'offcenter_gt_2_tokens'),
            'easy_iou_ge_0.5': group(enriched, 'm1_iou_stratum', 'easy_iou_ge_0.5'),
            'hard_iou_lt_0.5': group(enriched, 'm1_iou_stratum', 'hard_iou_lt_0.5'),
        },
        'sequence_bootstrap_95': {key: bootstrap(values) for key, values in seq_values.items()},
        'sequence_rows': seq,
        'event_rows': enriched,
        'interpretation': {
            'xy_only_is_position_control': True,
            'strata_fixed_before_result_inspection': True,
            'sequence_is_bootstrap_unit': True,
        },
    }
    (root / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['fixed_strata'], indent=2))
    print(json.dumps(result['sequence_bootstrap_95'], indent=2))


if __name__ == '__main__':
    main()
