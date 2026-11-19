import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path('/root/autodl-tmp/TBSI_gai/TBSI')
OUT = ROOT / 'output/experiments/M1_EARLY_CONDITIONED_GATE_TRACE_REPRESENTATIVE_20261023'
PARENT = OUT / 'parent_trace'
CANDIDATE = OUT / 'candidate_trace'
SEQUENCES = [
    'womanback2car', 'shinycarcoming', 'boydownplatform', '4thboywithwhite',
    'ab_bikeoccluded', '3rdfatboy', 'leftfarboycomingpicktheball'
]


def load(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def iou(a, b):
    if not a or not b:
        return float('nan')
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - inter
    return inter / union if union > 0 else 0.0


def mean_gate(row):
    vals = []
    for layer in row.get('early_gate_stats', []):
        for key in ('gate_rgb_mean', 'gate_tir_mean'):
            if layer.get(key) is not None:
                vals.append(float(layer[key]))
    return float(np.mean(vals)) if vals else float('nan')


def first_sustained(values, predicate, length=3):
    for i in range(len(values) - length + 1):
        if all(predicate(values[j]) for j in range(i, i + length)):
            return i
    return None


def xy_equal(a, b):
    return a == b or (a is None and b is None)


rows = []
details = {}
for seq in SEQUENCES:
    pp = PARENT / seq / 'frames.jsonl'
    cp = CANDIDATE / seq / 'frames.jsonl'
    parent, cand = load(pp), load(cp)
    pmap = {int(r['frame_id']): r for r in parent}
    cmap = {int(r['frame_id']): r for r in cand}
    frames = sorted(set(pmap) & set(cmap))
    aligned = []
    for fid in frames:
        p, c = pmap[fid], cmap[fid]
        ppbox, cpbox = p['pred_box'], c['pred_box']
        dx = math.hypot((cpbox[0] + cpbox[2] / 2) - (ppbox[0] + ppbox[2] / 2),
                        (cpbox[1] + cpbox[3] / 2) - (ppbox[1] + ppbox[3] / 2))
        peak_p, peak_c = p.get('peak_diagnostics', {}), c.get('peak_diagnostics', {})
        raw_peak_changed = not xy_equal(peak_p.get('raw_top1_xy'), peak_c.get('raw_top1_xy'))
        raw_rank_changed = peak_p.get('gt_rank_raw') != peak_c.get('gt_rank_raw')
        # The tracker decodes boxes from Hann-weighted response, not raw score.
        peak_changed = not xy_equal(peak_p.get('hann_top1_xy'), peak_c.get('hann_top1_xy'))
        rank_changed = peak_p.get('gt_rank_hann') != peak_c.get('gt_rank_hann')
        target_rank_p = peak_p.get('target_rank_score_gt_rank')
        target_rank_c = peak_c.get('target_rank_score_gt_rank')
        target_rank_changed = target_rank_p != target_rank_c
        aligned.append({
            'frame_id': fid,
            'parent_iou': iou(ppbox, p.get('gt_box')),
            'candidate_iou': iou(cpbox, c.get('gt_box')),
            'box_center_delta_px': dx,
            'peak_changed': peak_changed,
            'rank_changed': rank_changed,
            'raw_peak_changed': raw_peak_changed,
            'raw_rank_changed': raw_rank_changed,
            'target_rank_changed': target_rank_changed,
            'parent_gt_rank_raw': peak_p.get('gt_rank_raw'),
            'candidate_gt_rank_raw': peak_c.get('gt_rank_raw'),
            'parent_gt_rank_hann': peak_p.get('gt_rank_hann'),
            'candidate_gt_rank_hann': peak_c.get('gt_rank_hann'),
            'parent_raw_top1_xy': peak_p.get('raw_top1_xy'),
            'candidate_raw_top1_xy': peak_c.get('raw_top1_xy'),
            'parent_hann_top1_xy': peak_p.get('hann_top1_xy'),
            'candidate_hann_top1_xy': peak_c.get('hann_top1_xy'),
            'parent_target_rank_score_gt_rank': target_rank_p,
            'candidate_target_rank_score_gt_rank': target_rank_c,
            'parent_state_before': p.get('tracker_state_before'),
            'candidate_state_before': c.get('tracker_state_before'),
            'parent_state_after': p.get('tracker_state_after'),
            'candidate_state_after': c.get('tracker_state_after'),
            'gate_mean': mean_gate(c),
            'parent_margin_raw': peak_p.get('target_background_margin_raw'),
            'candidate_margin_raw': peak_c.get('target_background_margin_raw'),
        })
    div_idx = first_sustained(aligned, lambda x: x['box_center_delta_px'] > 2.0)
    strong_div_idx = first_sustained(aligned, lambda x: x['box_center_delta_px'] > 5.0)
    peak_idx = first_sustained(aligned, lambda x: x['peak_changed'])
    rank_idx = first_sustained(aligned, lambda x: x['rank_changed'])
    target_rank_idx = first_sustained(aligned, lambda x: x['target_rank_changed'])
    state_before_idx = first_sustained(aligned, lambda x: x['box_center_delta_px'] > 2.0)
    gate_vals = [x['gate_mean'] for x in aligned if not math.isnan(x['gate_mean'])]
    pious = np.array([x['parent_iou'] for x in aligned], dtype=float)
    cious = np.array([x['candidate_iou'] for x in aligned], dtype=float)
    # first divergence index is deliberately defined on output boxes; the
    # preceding rank/peak indices test whether CENTER changed first.
    first_frame = lambda idx: aligned[idx]['frame_id'] if idx is not None else None
    row = {
        'sequence': seq,
        'parent_frames': len(parent),
        'candidate_frames': len(cand),
        'aligned_frames': len(aligned),
        'mean_iou_parent': float(np.nanmean(pious)),
        'mean_iou_candidate': float(np.nanmean(cious)),
        'mean_iou_delta': float(np.nanmean(cious - pious)),
        'frames_peak_changed_hann': sum(x['peak_changed'] for x in aligned),
        'frames_gt_rank_changed_hann': sum(x['rank_changed'] for x in aligned),
        'frames_peak_changed_raw': sum(x['raw_peak_changed'] for x in aligned),
        'frames_gt_rank_changed_raw': sum(x['raw_rank_changed'] for x in aligned),
        'frames_target_rank_changed': sum(x['target_rank_changed'] for x in aligned),
        'first_peak_change_frame': first_frame(peak_idx),
        'first_gt_rank_change_frame': first_frame(rank_idx),
        'first_target_rank_change_frame': first_frame(target_rank_idx),
        'first_box_center_delta_gt2_frame': first_frame(div_idx),
        'first_box_center_delta_gt5_frame': first_frame(strong_div_idx),
        'gate_mean_over_frames': float(np.mean(gate_vals)) if gate_vals else None,
        'gate_min': float(np.min(gate_vals)) if gate_vals else None,
        'gate_max': float(np.max(gate_vals)) if gate_vals else None,
        'margin_raw_delta_mean': (float(np.nanmean([
            x['candidate_margin_raw'] - x['parent_margin_raw']
            for x in aligned
            if x['candidate_margin_raw'] is not None and x['parent_margin_raw'] is not None
        ])) if any(x['candidate_margin_raw'] is not None and x['parent_margin_raw'] is not None for x in aligned) else None),
    }
    rows.append(row)
    details[seq] = aligned

with (OUT / 'trace_summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with (OUT / 'trace_aligned_frames.json').open('w') as f:
    json.dump(details, f, indent=2)

def fmt(v):
    return 'NA' if v is None else str(v)

report = [
    '# Early-conditioned MSHA gate representative trace diagnosis', '',
    'Parent: `TBSI_E_MHSA_FULL_4090D_seed42`; candidate: `M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023`.',
    'All pairs use the same LasHeR test frames, `threads=0`, FP16 inference, template update off, and explicit checkpoints.', '',
    'The tracker selects CENTER from the Hann-weighted response. Peak and GT-rank changes below use that actual selection map; raw-score changes are retained in the CSV as auxiliary evidence.',
    'The seven paired replays contain {:,} aligned tracking frames ({:,} parent/candidate frame records).'.format(sum(r['aligned_frames'] for r in rows), 2 * sum(r['aligned_frames'] for r in rows)), '',
    '| Sequence | IoU delta | Hann peak changes | Hann GT-rank changes | first Hann peak | first Hann rank | first >2px box divergence | gate mean [min,max] |',
    '|---|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    report.append('| {sequence} | {mean_iou_delta:.4f} | {frames_peak_changed_hann} | {frames_gt_rank_changed_hann} | {first_peak_change_frame} | {first_gt_rank_change_frame} | {first_box_center_delta_gt2_frame} | {gate_mean_over_frames:.4f} [{gate_min:.4f},{gate_max:.4f}] |'.format(**r))
report += ['', '## Interpretation rules',
           '- `first Hann peak/rank` earlier than box divergence supports a CENTER selection change preceding closed-loop drift.',
           '- Box divergence with no Hann peak/rank change indicates the gate affects regression features, sub-cell geometry, or tracker state without changing the selected CENTER cell.',
           '- The trace is diagnostic evidence on seven fixed sequences, not a full-dataset performance claim.', '',
           '## Findings', '',
           '- Candidate gate means are approximately `0.9950` across the seven sequences, with observed ranges `0.9947--0.9954`; the learned test-time modulation is small and close to identity.',
           '- Using the actual Hann-weighted CENTER response, only `womanback2car` has a clear selected-peak change before sustained (>2 px) center-box divergence (frame 26 vs 39). `shinycarcoming`, `boydownplatform`, `ab_bikeoccluded`, `3rdfatboy`, and `leftfarboycomingpicktheball` diverge first; `4thboywithwhite` has a Hann-weighted GT-rank change at the same frame as the box divergence.',
           '- A candidate replay with `TBSI_INTERNAL_DIAGNOSTIC=0` matched the diagnostic replay exactly on `shinycarcoming` and `4thboywithwhite`, so the diagnostic hook did not cause the observed trajectories.',
           '- The earlier full-244 sequence CSV has opposite signs from clean same-checkpoint replay for several representative sequences. Treat those old sequence-level artifacts as replay-mismatched until full-244 is reproduced from a clean result directory; do not use them for causal attribution.',
           '- Decision: `MECHANISM-NOT-ESTABLISHED / STOP-GATE-EXPANSION`. Do not couple this gate with template update or a second fusion branch. First run a fixed-state candidate-checkpoint replay with `TBSI_TEST_EARLY_GATE_STRENGTH=0` versus `1`; this isolates the gate\'s current inference-time contribution while holding learned weights, frame, crop, and state fixed. Only if the gate changes Hann rank/box in the expected direction should a new spatially indexed gate be considered.', '']
(OUT / 'trace_report.md').write_text('\n'.join(report))
print((OUT / 'trace_report.md').read_text())
