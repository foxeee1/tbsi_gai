#!/usr/bin/env python3
"""Audit S1 closed-loop actions and calibrate an entropy admission threshold."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXISTING = ROOT / 'output/experiments/M1_READOUT_STRUCTURES_PROXY_20261014'
TRAIN = ROOT / 'output/experiments/M1_BOUNDED_READOUT_PREDICTOR_20261014_retry4'
FEATURES = ROOT / 'output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/selected_features.npz'
PREDICTOR = ROOT / 'experiments/artifacts/m1_bounded_readout_predictor_peakmargin_20261014.json'
CONFIG = 'TBSI_E_MHSA_FULL_4090D_seed42'


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def read_boxes(path):
    delimiter = '\t' if 'tracking_results' in str(path) else ','
    return np.loadtxt(path, delimiter=delimiter, ndmin=2).astype(float)


def audit_closed_loop(out):
    base_dir = ROOT / 'output/test/tracking_results/tbsi_track' / (CONFIG + '_1027')
    s1_dir = ROOT / 'output/test/tracking_results/tbsi_track' / (CONFIG + '_1028')
    action_root = ROOT / 'output/experiments/M1_S1_FIRST_TWO_STEPS_20261014_step2_retry1/action_audit_s1_same_env'
    seqs = [x.strip() for x in (EXISTING / 'sequence_list.txt').read_text().splitlines() if x.strip()]
    frame_rows, seq_rows = [], []
    for seq in seqs:
        gt = read_boxes(ROOT / 'data/lasher/testingset' / seq / 'init.txt')
        base = read_boxes(base_dir / (seq + '.txt'))
        s1 = read_boxes(s1_dir / (seq + '.txt'))
        actions = [json.loads(x) for x in (action_root / seq / 'actions.jsonl').read_text().splitlines() if x.strip()]
        n = min(len(gt), len(base), len(s1))
        base_iou = np.asarray([iou(base[i], gt[i]) for i in range(n)])
        s1_iou = np.asarray([iou(s1[i], gt[i]) for i in range(n)])
        delta = s1_iou - base_iou
        action_by_frame = {int(r['frame_id']): r for r in actions}
        for frame in range(n):
            a = action_by_frame.get(frame)
            frame_rows.append({
                'sequence': seq, 'frame': frame,
                'action': a['action'] if a else 'init',
                'offset_utility': a['offset_utility'] if a else '',
                'peak_margin': a['peak_margin'] if a else '',
                'normalized_entropy': a['normalized_entropy'] if a else '',
                'm1_iou': base_iou[frame], 's1_iou': s1_iou[frame],
                'delta_iou': delta[frame],
                'box_changed': bool(np.max(np.abs(base[frame] - s1[frame])) > 1e-5),
            })
        intervention = [int(r['frame_id']) for r in actions if r['action'] != 'noop']
        first_action = min(intervention) if intervention else -1
        first_box_change = next((i for i, x in enumerate(base[:n]) if np.max(np.abs(x - s1[i])) > 1e-5), -1)
        seq_rows.append({
            'sequence': seq, 'frames': n,
            'm1_auc': float(np.mean([(base_iou >= t).mean() for t in np.linspace(0, 1, 101)]) * 100),
            's1_auc': float(np.mean([(s1_iou >= t).mean() for t in np.linspace(0, 1, 101)]) * 100),
            'delta_auc': float(np.mean([(s1_iou >= t).mean() - (base_iou >= t).mean() for t in np.linspace(0, 1, 101)]) * 100),
            'delta_iou_first_action': float(delta[first_action]) if 0 <= first_action < n else '',
            'delta_iou_next1': float(delta[first_action + 1]) if 0 <= first_action + 1 < n else '',
            'delta_iou_next5_macro': float(np.mean(delta[first_action:min(first_action + 6, n)])) if 0 <= first_action < n else '',
            'first_action_frame': first_action, 'first_box_change_frame': first_box_change,
            'offset_actions': sum(r['action'] == 'offset' for r in actions),
            'noop_frames': sum(r['action'] == 'noop' for r in actions),
        })
    for name, rows in [('per_frame_action_iou.csv', frame_rows), ('per_sequence_first_divergence.csv', seq_rows)]:
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    action_delta = [r['delta_iou'] for r in frame_rows if r['action'] == 'offset']
    noop_delta = [r['delta_iou'] for r in frame_rows if r['action'] == 'noop']
    summary = {
        'sequences': len(seq_rows), 'frames': len(frame_rows),
        'offset_frame_mean_delta_iou': float(np.mean(action_delta)),
        'offset_frame_positive_fraction': float(np.mean(np.asarray(action_delta) > 0)),
        'noop_frame_mean_delta_iou': float(np.mean(noop_delta)),
        'noop_frame_positive_fraction': float(np.mean(np.asarray(noop_delta) > 0)),
        'sequence_auc_mean_delta': float(np.mean([r['delta_auc'] for r in seq_rows])),
        'limitation': 'S1 and M1 follow different closed-loop states after intervention; frame-aligned deltas are descriptive, not same-crop causal effects.'
    }
    (out / 'step1_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def calibrate_entropy(out):
    import os
    import sys
    import torch
    sys.path.insert(0, str(ROOT))
    from lib.test.parameter.tbsi_track import parameters
    from lib.test.tracker.tbsi_track import TBSITrack

    data = np.load(str(FEATURES), allow_pickle=False)
    sequence = data['sequence'].astype(str)
    frame = data['frame'].astype(int)
    utility_rows = list(csv.DictReader((TRAIN / 'capture_rows.csv').open()))
    cv_rows = [r for r in csv.DictReader((TRAIN / 'per_state_selection.csv').open()) if float(r['threshold']) == 0.08]
    u_map = {(r['sequence'], int(r['frame'])): r for r in utility_rows}
    cv_map = {(r['sequence'], int(r['frame'])): r for r in cv_rows}
    checkpoint = ROOT / 'output/experiments' / CONFIG / 'checkpoints/TBSITrack_ep0015.pth.tar'
    os.environ['TBSI_INFER_FP16'] = '0'
    torch.set_num_threads(1)
    params = parameters(CONFIG); params.debug = 0; params.checkpoint = str(checkpoint)
    tracker = TBSITrack(params, 'm1_s1_entropy_calibration')
    network = tracker.network.eval()
    records = []
    with torch.no_grad():
        for i in range(len(sequence)):
            key = (sequence[i], int(frame[i]))
            if key not in u_map or key not in cv_map:
                raise RuntimeError('train-side utility join failed: {}'.format(key))
            fused = torch.as_tensor(data['fused'][i].transpose(2, 0, 1)[None], dtype=torch.float32, device='cuda')
            score, _, _, _ = network.box_head(fused)
            response = tracker.output_window * score
            flat = response.flatten(1)[0].float()
            prob = torch.softmax(flat, dim=0)
            entropy = float((-(prob * torch.log(prob.clamp_min(1e-12))).sum() / math.log(max(int(flat.numel()), 2))).item())
            margin = float(torch.topk(flat, 2).values.diff().abs().item())
            cv = cv_map[key]
            records.append({
                'sequence': key[0], 'frame': key[1], 'normalized_entropy': entropy,
                'entropy_deficit': 1.0 - entropy, 'peak_margin_recomputed': margin,
                'peak_margin_saved': float(u_map[key]['peak_margin']),
                'action_at_utility_threshold_008': cv['action'],
                'selected_utility': float(cv['selected_utility']),
                'offset_utility': float(u_map[key]['offset_utility']),
            })
            if (i + 1) % 32 == 0:
                print('entropy states={}/{}'.format(i + 1, len(sequence)), flush=True)
    with (out / 'train_side_entropy_states.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    ent = np.asarray([r['normalized_entropy'] for r in records])
    sel = np.asarray([r['action_at_utility_threshold_008'] != 'noop' for r in records])
    utility = np.asarray([r['selected_utility'] for r in records])
    harms = utility < -1e-6
    thresholds = sorted(set([float(np.quantile(ent, q)) for q in np.linspace(0, 1, 101)] + [1.0]))
    risk = []
    for threshold in thresholds:
        keep = sel & (ent <= threshold)
        coverage = float(keep.mean())
        harmful_rate = float(np.mean(harms[keep])) if keep.any() else 0.0
        risk.append({'max_entropy': threshold, 'coverage': coverage,
                     'harmful_rate_covered': harmful_rate,
                     'mean_selected_utility_all': float(np.mean(np.where(keep, utility, 0.0)))})
    eligible = [r for r in risk if r['coverage'] >= 0.35 and r['harmful_rate_covered'] <= 0.01]
    selected_threshold = max(eligible, key=lambda r: r['coverage']) if eligible else None
    result = {
        'train_sequences': int(len(set(sequence))), 'states': len(records),
        'entropy_min': float(ent.min()), 'entropy_p05': float(np.quantile(ent, .05)),
        'entropy_median': float(np.median(ent)), 'entropy_p95': float(np.quantile(ent, .95)),
        'entropy_max': float(ent.max()), 'utility_gate_coverage': float(sel.mean()),
        'utility_gate_harmful_count': int(harms[sel].sum()),
        'selected_threshold': selected_threshold,
        'coverage_floor': 0.35, 'harmful_rate_ceiling': 0.01,
        'risk_coverage': risk,
        'limitations': ['train-side fixed-state replay; no closed-loop outcome',
                        'selection and threshold calibration reuse the existing 72-sequence cohort',
                        'test evaluation is permitted only if a threshold meets the predeclared coverage and harm criteria']
    }
    (out / 'step2_calibration.json').write_text(json.dumps(result, indent=2) + '\n')
    if selected_threshold:
        policy = {'mode': 'reliability', 'description': 'train-side calibrated entropy admission for S1 offset predictor',
                  'min_peak_margin': 0.0, 'max_entropy': selected_threshold['max_entropy']}
        policy_path = ROOT / 'experiments/artifacts/m1_readout_entropy_only_calibrated_20261014.json'
        policy_path.write_text(json.dumps(policy, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'risk_coverage'}, indent=2), flush=True)
    return result


def tracking_metrics(boxes, gt):
    n = min(len(boxes), len(gt))
    boxes, gt = boxes[:n], gt[:n]
    ov = np.asarray([iou(boxes[i], gt[i]) for i in range(n)])
    pred_c = np.column_stack((boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2))
    gt_c = np.column_stack((gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2))
    dist = np.linalg.norm(pred_c - gt_c, axis=1)
    diag = np.sqrt(gt[:, 2] * gt[:, 3])
    return {
        'AUC': float(np.mean([(ov >= t).mean() for t in np.linspace(0, 1, 101)]) * 100),
        'PR': float((dist <= 20).mean() * 100),
        'NPR': float((dist / np.maximum(diag, 1e-9) <= .2).mean() * 100),
        'OP50': float((ov >= .5).mean() * 100),
        'OP75': float((ov >= .75).mean() * 100),
        'frames': int(n),
    }


def summarize_paired(rows, cohort):
    subset = rows if cohort == 'union82' else [r for r in rows if r['cohort'] == cohort]
    result = {'n': len(subset)}
    for arm in ('S1', 'ENTROPY'):
        result[arm] = {}
        for metric in ('AUC', 'PR', 'NPR', 'OP50', 'OP75'):
            vals = np.asarray([r[arm + '_delta_' + metric] for r in subset], dtype=float)
            result[arm][metric] = {'delta': float(vals.mean()), 'wins': int((vals > 0).sum()),
                                   'losses': int((vals < 0).sum()), 'ties': int((vals == 0).sum())}
    return result


def analyze_closed_loop(out):
    result_root = ROOT / 'output/test/tracking_results/tbsi_track'
    arms = {'M1': result_root / (CONFIG + '_1027'),
            'S1': result_root / (CONFIG + '_1028'),
            'ENTROPY': result_root / (CONFIG + '_1026')}
    seqs = [x.strip() for x in (EXISTING / 'sequence_list.txt').read_text().splitlines() if x.strip()]
    low = set((ROOT / 'output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/matched_m1_full62/sequence_list.txt').read_text().split())
    normal = set((ROOT / 'experiments/manifests/m1_normal20_control_20261012.txt').read_text().split())
    rows, action_counts = [], {'noop': 0, 'offset': 0, 'size': 0, 'joint': 0, 'frames': 0}
    for seq in seqs:
        gt = read_boxes(ROOT / 'data/lasher/testingset' / seq / 'init.txt')
        ms = {name: tracking_metrics(read_boxes(path / (seq + '.txt')), gt) for name, path in arms.items()}
        cohort = 'lowauc62' if seq in low else 'normal20' if seq in normal else 'other'
        row = {'sequence': seq, 'cohort': cohort, 'frames': ms['M1']['frames']}
        for arm in ('M1', 'S1', 'ENTROPY'):
            for metric in ('AUC', 'PR', 'NPR', 'OP50', 'OP75'):
                row[arm + '_' + metric] = ms[arm][metric]
                if arm != 'M1':
                    row[arm + '_delta_' + metric] = ms[arm][metric] - ms['M1'][metric]
        actions = [json.loads(x) for x in (out / 'action_audit' / seq / 'actions.jsonl').read_text().splitlines() if x.strip()]
        for r in actions:
            action_counts[r['action']] = action_counts.get(r['action'], 0) + 1
            action_counts['frames'] += 1
        row['entropy_noop_frames'] = sum(r['action'] == 'noop' for r in actions)
        row['entropy_offset_frames'] = sum(r['action'] == 'offset' for r in actions)
        row['entropy_size_frames'] = sum(r['action'] == 'size' for r in actions)
        rows.append(row)
    with (out / 'per_sequence_three_arm_metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    entropy_vs_s1 = {}
    rng = np.random.RandomState(20261014)
    for cohort in ('lowauc62', 'normal20', 'union82'):
        subset = rows if cohort == 'union82' else [r for r in rows if r['cohort'] == cohort]
        delta = np.asarray([r['ENTROPY_delta_AUC'] - r['S1_delta_AUC'] for r in subset], dtype=float)
        boot = np.asarray([np.mean(rng.choice(delta, size=len(delta), replace=True)) for _ in range(10000)])
        entropy_vs_s1[cohort] = {
            'AUC_delta': float(np.mean(delta)), 'wins': int((delta > 0).sum()),
            'losses': int((delta < 0).sum()), 'ties': int((delta == 0).sum()),
            'paired_sequence_bootstrap95': [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        }
        for metric in ('PR', 'NPR', 'OP50', 'OP75'):
            vals = np.asarray([r['ENTROPY_delta_' + metric] - r['S1_delta_' + metric] for r in subset], dtype=float)
            entropy_vs_s1[cohort][metric + '_delta'] = float(np.mean(vals))
    summary = {
        'n_sequences': len(rows), 'cohort_sizes': {'lowauc62': len(low), 'normal20': len(normal)},
        'arms_vs_matched_current_m1_1027': {c: summarize_paired(rows, c) for c in ('lowauc62', 'normal20', 'union82')},
        'entropy_vs_s1': entropy_vs_s1,
        'entropy_action_counts': action_counts,
        's1_action_counts': {'noop': 0, 'offset': 0, 'size': 0, 'joint': 0, 'frames': 0},
        'fps': 'NA; tracker logs show per-sequence FPS but full aggregate was not captured as a structured field',
        'comparability': 'same 82-sequence manifest, protected M1 epoch15 checkpoint, current tracker code, Python 3.8.20/PyTorch 1.9.0+cu111/CUDA 11.1, FP32 serial evaluation; entropy arm differs from S1 only by fixed train-calibrated entropy admission threshold',
    }
    s1_counts = summary['s1_action_counts']
    for seq in seqs:
        for line in (ROOT / 'output/experiments/M1_S1_FIRST_TWO_STEPS_20261014_step2_retry1/action_audit_s1_same_env' / seq / 'actions.jsonl').read_text().splitlines():
            r = json.loads(line); s1_counts[r['action']] += 1; s1_counts['frames'] += 1
    (out / 'step2_closed_loop_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--phase', choices=['audit', 'calibrate', 'analyze', 'both'], default='both')
    args = ap.parse_args()
    out = Path(args.output_dir)
    if args.phase != 'analyze' and out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing non-empty output {}'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    if args.phase in ('audit', 'both'):
        print('STEP1', json.dumps(audit_closed_loop(out), sort_keys=True), flush=True)
    if args.phase in ('calibrate', 'both'):
        print('STEP2', json.dumps(calibrate_entropy(out), sort_keys=True), flush=True)
    if args.phase == 'analyze':
        analyze_closed_loop(out)


if __name__ == '__main__':
    main()
