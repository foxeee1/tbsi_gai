#!/usr/bin/env python3
"""Fit two train-side, sequence-held-out S1 admission policies."""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack

NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
PREDICTOR = ROOT / 'experiments/artifacts/m1_bounded_readout_predictor_peakmargin_20261014.json'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def trace_index(dirs):
    out = {}
    for root in dirs:
        for path in Path(root).glob('*/frames.jsonl'):
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    out[(path.parent.name, int(row['frame_id']))] = row
    return out


def logistic_fit(x, y, l2=1.0, steps=80):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = np.clip((x - mean) / scale, -8.0, 8.0)
    design = np.column_stack((np.ones(len(z)), z))
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(len(beta), dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    for _ in range(steps):
        logits = np.clip(design @ beta, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        weight = np.maximum(prob * (1.0 - prob), 1e-6)
        hessian = (design.T * weight) @ design / len(y) + penalty
        grad = design.T @ (prob - y) / len(y) + penalty @ beta
        delta = np.linalg.solve(hessian, grad)
        beta -= delta
        if np.max(np.abs(delta)) < 1e-7:
            break
    return {'mean': mean.tolist(), 'scale': scale.tolist(),
            'bias': float(beta[0]), 'weight': beta[1:].tolist()}


def logistic_predict(model, x):
    z = np.clip((x - np.asarray(model['mean'])) / np.asarray(model['scale']), -8.0, 8.0)
    logits = np.clip(model['bias'] + z @ np.asarray(model['weight']), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def motion_innovation(current, previous, previous_previous):
    if current is None or previous is None or previous_previous is None:
        return 0.0
    def center(box):
        return np.asarray([box[0] + 0.5 * box[2], box[1] + 0.5 * box[3]], dtype=np.float64)
    c0, c1, c2 = center(previous_previous), center(previous), center(current)
    expected = 2.0 * c1 - c0
    scale = max(math.sqrt(max(float(previous[2]) * float(previous[3]), 1.0)), 1.0)
    return float(np.linalg.norm(c2 - expected) / scale)


def select_threshold(values, utility, eligible, risk_budget=0.05, min_coverage=0.20):
    finite = [float(v) for v in values if np.isfinite(v)]
    candidates = sorted(set([0.0] + finite + [max(finite) + 1e-9]))
    feasible = []
    for threshold in candidates:
        selected = eligible & (values <= threshold)
        count = int(selected.sum())
        coverage = count / max(len(utility), 1)
        if count == 0:
            continue
        harm = float(np.mean(utility[selected] <= 0.0))
        mean_gain = float(np.sum(utility[selected]) / max(len(utility), 1))
        if harm <= risk_budget and coverage >= min_coverage:
            feasible.append((mean_gain, coverage, -harm, threshold, count, harm))
    if not feasible:
        return {'threshold': 0.0, 'coverage': 0.0, 'harmful_rate': 0.0,
                'selected': 0, 'mean_utility_per_state': 0.0, 'feasible': False}
    best = max(feasible)
    return {'threshold': best[3], 'coverage': best[1], 'harmful_rate': best[5],
            'selected': best[4], 'mean_utility_per_state': best[0], 'feasible': True}


def risk_frontier(values, utility, eligible, budgets):
    result = []
    for budget in budgets:
        selection = select_threshold(values, utility, eligible,
                                     risk_budget=float(budget), min_coverage=0.0)
        selection['risk_budget'] = float(budget)
        result.append(selection)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--capture-rows', required=True)
    ap.add_argument('--trace-dir', action='append', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--risk-budget', type=float, default=0.15)
    ap.add_argument('--min-coverage', type=float, default=0.10)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing nonempty output directory: {}'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ.pop('TBSI_BOUNDED_READOUT_PREDICTOR', None)
    os.environ.pop('TBSI_READOUT_POLICY', None)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    features = np.load(args.features, allow_pickle=False)
    captures = {(r['sequence'], int(r['frame'])): r
                for r in csv.DictReader(open(args.capture_rows, newline=''))}
    traces = trace_index(args.trace_dir)
    sequences = [str(s) for s in features['sequence']]
    rows = []
    params = parameters(NAME)
    params.debug = 0
    params.save_all_boxes = False
    tracker = TBSITrack(params, 's1_admission_train')
    network = tracker.network.eval()
    with torch.no_grad():
        for idx, (seq, frame) in enumerate(zip(sequences, features['frame'])):
            frame = int(frame)
            key = (seq, frame)
            if key not in captures or key not in traces:
                raise RuntimeError('missing training label/trace: {}'.format(key))
            fused = torch.as_tensor(features['fused'][idx].transpose(2, 0, 1)[None],
                                    dtype=torch.float32, device='cuda')
            score, _, _, _ = network.box_head(fused)
            response = tracker.output_window * score
            flat = response.flatten(1)[0].float()
            top = torch.topk(flat, k=min(5, flat.numel()), largest=True, sorted=True).values
            peak = int(torch.argmax(flat).item())
            px, py = peak % tracker.feat_sz, peak // tracker.feat_sz
            probs = torch.softmax(flat, dim=0)
            entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum() /
                             math.log(max(int(flat.numel()), 2))).item())
            top5_margin = float((top[0] - top[-1]).item())
            peak_margin = float((top[0] - top[1]).item())
            edge = min(px, py, tracker.feat_sz - 1 - px,
                       tracker.feat_sz - 1 - py) / max(tracker.feat_sz / 2.0, 1.0)
            tr = traces[key]
            prev = traces.get((seq, frame - 1))
            prevprev = traces.get((seq, frame - 2))
            innovation = motion_innovation(
                tr.get('pred_box'), prev.get('pred_box') if prev else None,
                prevprev.get('pred_box') if prevprev else None)
            utility = float(captures[key]['offset_utility'])
            rows.append({'sequence': seq, 'frame': frame, 'peak_margin': peak_margin,
                         'normalized_entropy': entropy, 'top5_margin': top5_margin,
                         'peak_edge_distance': float(edge), 'motion_innovation': innovation,
                         'offset_utility': utility})
            if (idx + 1) % 24 == 0:
                print('feature_states={}/{}'.format(idx + 1, len(sequences)), flush=True)

    feature_names = ['peak_margin', 'normalized_entropy', 'top5_margin', 'peak_edge_distance']
    x = np.asarray([[r[name] for name in feature_names] for r in rows], dtype=np.float64)
    y = np.asarray([r['offset_utility'] <= 0.0 for r in rows], dtype=np.float64)
    utility = np.asarray([r['offset_utility'] for r in rows], dtype=np.float64)
    margin = np.asarray([r['peak_margin'] for r in rows], dtype=np.float64)
    base_model = json.load(open(PREDICTOR, encoding='utf-8'))['models']['offset_utility']
    z = np.clip((margin - base_model['mean'][0]) / base_model['scale'][0], -8.0, 8.0)
    base_utility = base_model['bias'] + z * base_model['weight'][0]
    eligible = base_utility > 0.08
    unique_sequences = sorted(set(r['sequence'] for r in rows))
    oof_risk = np.full(len(rows), np.nan, dtype=np.float64)
    for held in unique_sequences:
        test = np.asarray([r['sequence'] == held for r in rows])
        train = ~test
        model = logistic_fit(x[train], y[train])
        oof_risk[test] = logistic_predict(model, x[test])
    if not np.isfinite(oof_risk).all():
        raise RuntimeError('sequence-held-out risk predictions incomplete')
    risk_selection = select_threshold(oof_risk, utility, eligible,
                                      risk_budget=args.risk_budget,
                                      min_coverage=args.min_coverage)
    final_risk_model = logistic_fit(x, y)

    innovation = np.asarray([r['motion_innovation'] for r in rows], dtype=np.float64)
    innovation_selection = select_threshold(innovation, utility, eligible,
                                            risk_budget=args.risk_budget,
                                            min_coverage=args.min_coverage)
    risk_policy = {
        'schema_version': 1, 'method': 'cost-sensitive selective risk gate',
        'mode': 'independent', 'admission': 'selective_risk', 'risk_model': final_risk_model,
        'risk_feature_names': feature_names,
        'max_harm_probability': float(risk_selection['threshold']),
        'risk_budget': args.risk_budget, 'minimum_train_coverage': args.min_coverage,
        'train_oof': risk_selection, 'train_sequences': len(unique_sequences),
        'train_states': len(rows), 'target': 'same-state offset utility <= 0',
        'fallback': 'no-op'
    }
    motion_policy = {
        'schema_version': 1, 'method': 'constant-velocity normalized innovation gate',
        'mode': 'independent', 'admission': 'motion_innovation',
        'max_normalized_innovation': float(innovation_selection['threshold']),
        'risk_budget': args.risk_budget, 'minimum_train_coverage': args.min_coverage,
        'train_oof': innovation_selection, 'train_sequences': len(unique_sequences),
        'train_states': len(rows), 'target': 'same-state offset utility <= 0',
        'fallback': 'no-op'
    }
    (out / 'policy_selective_risk.json').write_text(json.dumps(risk_policy, indent=2) + '\n')
    (out / 'policy_motion_innovation.json').write_text(json.dumps(motion_policy, indent=2) + '\n')
    with (out / 'train_side_states.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cv = {'states': len(rows), 'sequences': len(unique_sequences),
          'harmful_labels': int(y.sum()), 'baseline_s1_coverage': float(eligible.mean()),
          'baseline_s1_harmful_rate': float(np.mean(utility[eligible] <= 0.0)) if eligible.any() else 0.0,
          'selective_risk_oof': risk_selection,
          'selective_risk_frontier': risk_frontier(oof_risk, utility, eligible,
                                                    [0.05, 0.10, 0.15, 0.188, 0.20, 0.25]),
          'motion_innovation_oof': innovation_selection,
          'motion_innovation_frontier': risk_frontier(innovation, utility, eligible,
                                                       [0.05, 0.10, 0.15, 0.188, 0.20, 0.25]),
          'features': feature_names,
          'sequence_holdout': True,
          'predictor_sha256': sha256(PREDICTOR),
          'feature_archive_sha256': sha256(args.features),
          'capture_rows_sha256': sha256(args.capture_rows)}
    cv['risk_budget'] = args.risk_budget
    cv['minimum_train_coverage'] = args.min_coverage
    cv['oof_predictions'] = [{'sequence': rows[i]['sequence'], 'frame': rows[i]['frame'],
                              'risk_probability': float(oof_risk[i]),
                              'normalized_motion_innovation': float(innovation[i]),
                              's1_eligible': bool(eligible[i]),
                              'offset_utility': float(utility[i])}
                             for i in range(len(rows))]
    (out / 'training_summary.json').write_text(json.dumps(cv, indent=2) + '\n')
    print(json.dumps(cv, indent=2), flush=True)


if __name__ == '__main__':
    main()
