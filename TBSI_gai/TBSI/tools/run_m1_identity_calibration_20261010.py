#!/usr/bin/env python3
"""Export fixed-state candidate features, fit a tiny ranker, and evaluate once."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ['TBSI_INFER_FP16'] = '0'
os.environ['TBSI_PERTURBATION'] = ''

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_m1_search_stage_20261009 import (
    CONFIG, CHECKPOINT, EXPECTED_CONFIG, EXPECTED_CHECKPOINT, iou,
    predicted_box, read_pair, select_frames, sha256, target_masks)
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target, transform_image_to_crop

COHORT = ROOT / 'experiments/diagnostics/m1_identity_calibration_v2'
OUT = ROOT / 'output/experiments/M1_IDENTITY_CALIBRATION_20261010'
TOPK = 8
FOLDS = ('calibration', 'validation', 'final_holdout')


def pool_template(tokens, mask):
    weight = mask.reshape(1, -1, 1).to(device=tokens.device, dtype=tokens.dtype)
    return (tokens * weight).sum(1) / weight.sum(1).clamp_min(1e-8)


class CandidateMatcher(nn.Module):
    def __init__(self, fused_dim, modality_dim, projection=32):
        super().__init__()
        self.fused = nn.Sequential(nn.Linear(fused_dim, projection), nn.LayerNorm(projection))
        self.rgb = nn.Sequential(nn.Linear(modality_dim, projection), nn.LayerNorm(projection))
        self.tir = nn.Sequential(nn.Linear(modality_dim, projection), nn.LayerNorm(projection))
        self.mlp = nn.Sequential(nn.Linear(projection * 5, 64), nn.GELU(),
                                 nn.Linear(64, 1))

    def forward(self, fused, rgb, tir):
        f, r, t = self.fused(fused), self.rgb(rgb), self.tir(tir)
        return self.mlp(torch.cat((f, r, t, (f - r).abs(), (f - t).abs()), dim=-1)).squeeze(-1)


def load_rows(split, trace_root):
    seq_file = COHORT / (split + '_sequences.txt')
    names = seq_file.read_text().splitlines()
    trace_dir = Path(trace_root) / 'trace'
    dataset = LasHeRDataset('trainingset', str(seq_file)).get_sequence_list()
    seq_map = {seq.name: seq for seq in dataset}
    missing = [name for name in names if name not in seq_map or
               not (trace_dir / name / 'frames.jsonl').is_file()]
    if missing:
        raise RuntimeError('{} missing sequences/traces: {}'.format(split, missing[:5]))
    return names, seq_map, trace_dir


def extract_split(split, trace_root, tracker, network, template, fg_mask, ce_mask, hook_capture):
    names, seq_map, trace_dir = load_rows(split, trace_root)
    records = []
    for name in names:
        seq = seq_map[name]
        trace = [json.loads(line) for line in (trace_dir / name / 'frames.jsonl').read_text().splitlines()
                 if line.strip()]
        selected = [(i, 'matched_good' if label == 'good_only' else label)
                    for i, label in select_frames(trace, 0, 'onset-3-iou-0.5')
                    if label in ('matched_good', 'pre_onset', 'severe_onset', 'good_only')]
        if not selected:
            continue
        tracker.initialize(read_pair(seq, 0), {'init_bbox': seq.ground_truth_rect[0], 'sequence_name': name})
        seq_template = tracker.z_dict1
        seq_fg = tracker.template_fg_mask
        seq_ce = tracker.box_mask_z
        for index, label in selected:
            item = trace[index]
            frame = int(item['frame_id'])
            state = [float(v) for v in item['tracker_state_before']]
            gt = np.asarray(seq.ground_truth_rect[frame], dtype=np.float64)
            image = read_pair(seq, frame)
            crop, resize, amask = sample_target(image, state, tracker.params.search_factor,
                                                output_sz=tracker.params.search_size)
            search = tracker.preprocessor.process(crop, amask, half=False)
            search_pair = [search.tensors[:, :3], search.tensors[:, 3:]]
            gt_crop = transform_image_to_crop(
                torch.tensor(gt, dtype=torch.float32), torch.tensor(state, dtype=torch.float32), resize,
                torch.tensor([tracker.params.search_size, tracker.params.search_size]), normalize=True).numpy()
            masks = target_masks(gt_crop)
            gx, gy = masks[2], masks[3]
            hook_capture.clear()
            with torch.inference_mode():
                out = network(template=[seq_template.tensors[:, :3], seq_template.tensors[:, 3:]],
                              search=search_pair, ce_template_mask=seq_ce,
                              template_fg_masks=(seq_fg, seq_fg))
            fused = hook_capture.get('fused')
            if fused is None or tuple(fused.shape[-2:]) != (16, 16):
                raise RuntimeError('tbsi_fuse_search hook missing or unexpected shape')
            feat = out['backbone_feat']
            p_rgb = pool_template(feat[:, :64], seq_fg).float()[0]
            p_tir = pool_template(feat[:, 320:384], seq_fg).float()[0]
            fused_tokens = fused.flatten(2).transpose(1, 2).float()[0]
            online = (out['score_map'].float() * tracker.output_window.float())[0, 0].reshape(-1)
            cand = torch.topk(online, TOPK, sorted=True).indices
            target = torch.as_tensor(masks[0].reshape(-1), dtype=torch.bool, device=cand.device)
            background = torch.as_tensor(masks[1].reshape(-1), dtype=torch.bool, device=cand.device)
            top_f = fused_tokens[cand]
            top_rgb = p_rgb.expand(TOPK, -1)
            top_tir = p_tir.expand(TOPK, -1)
            cosine = .5 * (F.cosine_similarity(top_f, top_rgb) + F.cosine_similarity(top_f, top_tir))
            row = {
                'sequence': name, 'frame': frame, 'label': label,
                'candidate_idx': cand.cpu().numpy().astype(np.int16),
                'candidate_score': online[cand].cpu().numpy().astype(np.float32),
                'candidate_cosine': cosine.cpu().numpy().astype(np.float32),
                'target': target[cand].cpu().numpy().astype(np.uint8),
                'candidate_is_hard_bg': background[cand].cpu().numpy().astype(np.uint8),
                'fused': top_f.cpu().numpy().astype(np.float32),
                'rgb': top_rgb.cpu().numpy().astype(np.float32),
                'tir': top_tir.cpu().numpy().astype(np.float32),
                'gt': gt.tolist(), 'state': state, 'resize': float(resize),
                'image_shape': list(image.shape), 'gt_center': [masks[4], masks[5]],
                'target_candidate_count': int(target[cand].sum().item()),
                'raw_target_rank': int((~target[cand]).sum().item() + 1)
                    if bool(target[cand[0]].item()) else next(
                        (j + 1 for j, idx in enumerate(cand) if bool(target[idx].item())), TOPK + 1),
                'm1_cell': int(cand[0].item()), 'gt_cell': int(gy * 16 + gx),
                'size_map': out['size_map'].float().cpu().numpy()[0],
                'offset_map': out['offset_map'].float().cpu().numpy()[0],
                'score_map': online.reshape(16, 16).cpu().numpy(),
                'online_window': tracker.output_window.float().cpu().numpy()[0, 0],
            }
            # Identity check: the replayed M1 cell must equal the closed-loop trace box.
            parity_box = predicted_box(network, tracker, out, state, image.shape, resize)
            if np.max(np.abs(np.asarray(parity_box) - np.asarray(item['pred_box']))) > .01:
                raise RuntimeError('fixed-state M1 parity failure: {} {}'.format(name, frame))
            row['trace_iou'] = float(iou(item['pred_box'], gt))
            records.append(row)
        print('{} {} events={} total={}'.format(split, name, len(selected), len(records)), flush=True)
    return records


def tensor_data(records, device):
    f = torch.tensor(np.stack([r['fused'] for r in records]), device=device)
    rgb = torch.tensor(np.stack([r['rgb'] for r in records]), device=device)
    tir = torch.tensor(np.stack([r['tir'] for r in records]), device=device)
    target = torch.tensor(np.stack([r['target'] for r in records]).astype(bool), device=device)
    return f, rgb, tir, target


def save_features(path, records):
    np.savez_compressed(
        str(path), fused=np.stack([r['fused'] for r in records]),
        rgb=np.stack([r['rgb'] for r in records]), tir=np.stack([r['tir'] for r in records]),
        candidate_score=np.stack([r['candidate_score'] for r in records]),
        candidate_cosine=np.stack([r['candidate_cosine'] for r in records]),
        target=np.stack([r['target'] for r in records]),
        candidate_idx=np.stack([r['candidate_idx'] for r in records]),
        sequence=np.asarray([r['sequence'] for r in records]),
        frame=np.asarray([r['frame'] for r in records]),
        label=np.asarray([r['label'] for r in records]))


def train_matcher(records, seed):
    torch.manual_seed(seed)
    device = torch.device('cuda')
    f, rgb, tir, target = tensor_data(records, device)
    model = CandidateMatcher(f.shape[-1], rgb.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    usable = target.any(1) & (~target).any(1)
    if int(usable.sum()) < 12:
        raise RuntimeError('too few calibration events with both positive and hard negative candidates: {}'.format(int(usable.sum())))
    for epoch in range(200):
        optimizer.zero_grad(set_to_none=True)
        scores = model(f, rgb, tir)
        losses = []
        for i in range(len(records)):
            if not bool(usable[i]):
                continue
            pos = scores[i][target[i]]
            neg = scores[i][~target[i]]
            losses.append(F.softplus(1.0 - pos[:, None] + neg[None, :]).mean())
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    return model, float(loss.detach().item()), int(usable.sum().item())


def fit_linear(records, q_values):
    device = torch.device('cuda')
    params = torch.zeros(3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=.03)
    usable_count = 0
    tensors = []
    for row, q in zip(records, q_values):
        s = torch.tensor(row['candidate_score'], device=device)
        s = (s - s.mean()) / s.std(unbiased=False).clamp_min(1e-6)
        q = torch.tensor(q, device=device)
        q = (q - q.mean()) / q.std(unbiased=False).clamp_min(1e-6)
        y = torch.tensor(row['target'].astype(bool), device=device)
        tensors.append((s, q, y))
        usable_count += int(bool(y.any()) and bool((~y).any()))
    if usable_count < 12:
        raise RuntimeError('too few calibration events for C3 fit')
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        a, b, c = params
        for s, q, y in tensors:
            if not bool(y.any()) or not bool((~y).any()):
                continue
            score = a * s + b * q + c * s * q
            losses.append(F.softplus(1.0 - score[y, None] + score[None, ~y]).mean())
        loss = torch.stack(losses).mean() + 1e-3 * params.square().sum()
        loss.backward()
        optimizer.step()
    return params.detach().cpu().numpy().tolist(), float(loss.detach().item()), usable_count


def candidate_decision(row, q, coeff):
    s = row['candidate_score'].astype(np.float64)
    sz = (s - s.mean()) / max(float(s.std()), 1e-6)
    q = np.asarray(q, dtype=np.float64)
    qz = (q - q.mean()) / max(float(q.std()), 1e-6)
    c1 = sz + .5 * ((row['candidate_cosine'] - row['candidate_cosine'].mean()) /
                    max(float(row['candidate_cosine'].std()), 1e-6))
    c3 = coeff[0] * sz + coeff[1] * qz + coeff[2] * sz * qz
    return {'C0_M1': 0, 'C1_RAW_COSINE': int(np.argmax(c1)),
            'C2_LEARNED_MATCH': int(np.argmax(q)), 'C3_LEARNED_COMBO': int(np.argmax(c3))}, q, c3


def decode_and_metrics(row, cand_pos, tracker, network):
    cell = int(row['candidate_idx'][cand_pos])
    x, y = cell % 16, cell // 16
    # Stored maps are sufficient to compute cell-conditioned localization error.
    off = row['offset_map'][:, y, x]
    cx, cy = row['gt_center']
    center_error = math.hypot(x + float(off[0]) - cx, y + float(off[1]) - cy)
    score = torch.zeros((1, 1, 16, 16), dtype=torch.float32, device=tracker.output_window.device)
    score[0, 0, y, x] = 1.
    out = {
        'score_map': score,
        'size_map': torch.as_tensor(row['size_map'][None], device=score.device),
        'offset_map': torch.as_tensor(row['offset_map'][None], device=score.device),
    }
    state = row['state']
    box = predicted_box(network, tracker, out, state, row['image_shape'], row['resize'],
                        window=torch.ones_like(tracker.output_window))
    return {'iou': float(iou(box, row['gt'])), 'center_error_tokens': center_error,
            'selected_target': int(bool(row['target'][cand_pos])),
            'selected_idx': cell, 'box': box}


def choose_safe_gates(val_rows, val_q, coeff, val_scores, tracker, network):
    chosen = {}
    for arm in ('C2_LEARNED_MATCH', 'C3_LEARNED_COMBO'):
        candidates = []
        for delta_min in (0., .05, .10, .20, .40):
            for radius in (1., 2., 3.):
                rows_by_label = {key: [] for key in ('matched_good', 'pre_onset', 'severe_onset')}
                switches = 0
                safe = True
                for row, q in zip(val_rows, val_q):
                    decisions, _, c3 = candidate_decision(row, q, coeff)
                    pos = decisions[arm]
                    old = 0
                    utility = ((q - q.mean()) / max(float(q.std()), 1e-6)
                               if arm == 'C2_LEARNED_MATCH' else c3)
                    if float(utility[pos] - utility[old]) < delta_min:
                        pos = old
                    xy0 = np.array([row['candidate_idx'][0] % 16, row['candidate_idx'][0] // 16])
                    xy1 = np.array([row['candidate_idx'][pos] % 16, row['candidate_idx'][pos] // 16])
                    if np.linalg.norm(xy1 - xy0) > radius:
                        pos = old
                    switches += int(pos != old)
                    result = decode_and_metrics(row, pos, tracker, network)
                    rows_by_label[row['label']].append(result)
                good = rows_by_label['matched_good']
                pre = rows_by_label['pre_onset']
                severe = rows_by_label['severe_onset']
                base = {label: [decode_and_metrics(row, 0, tracker, network)
                                for row in val_rows if row['label'] == label]
                        for label in rows_by_label}
                good_loss = sum(x['selected_target'] for x in base['matched_good']) - sum(x['selected_target'] for x in good)
                pre_delta = (np.mean([x['iou'] for x in pre]) - np.mean([x['iou'] for x in base['pre_onset']])) if pre else 0.
                if good_loss > 1 or pre_delta < -1e-9:
                    safe = False
                if safe:
                    gain = sum(x['selected_target'] for x in severe) - sum(x['selected_target'] for x in base['severe_onset'])
                    candidates.append((gain, switches, delta_min, radius, good_loss, pre_delta))
        if candidates:
            best = sorted(candidates, key=lambda x: (-x[0], x[1], x[2], x[3]))[0]
            chosen[arm] = {'enabled': True, 'delta_min': best[2], 'max_distance_tokens': best[3],
                           'validation_severe_top1_gain': int(best[0]), 'switches': int(best[1]),
                           'matched_good_lost_events': int(best[4]), 'pre_onset_iou_delta': float(best[5])}
        else:
            chosen[arm] = {'enabled': False, 'reason': 'no gate met predeclared validation safety constraints'}
    return chosen


def evaluate(rows, model, coeff, gates, tracker, network):
    device = torch.device('cuda')
    output_rows = []
    for row in rows:
        with torch.inference_mode():
            q = model(torch.as_tensor(row['fused'], device=device),
                      torch.as_tensor(row['rgb'], device=device),
                      torch.as_tensor(row['tir'], device=device)).cpu().numpy()
        decisions, q, c3 = candidate_decision(row, q, coeff)
        for arm, pos in decisions.items():
            result = decode_and_metrics(row, pos, tracker, network)
            output_rows.append({'sequence': row['sequence'], 'frame': row['frame'],
                                'label': row['label'], 'arm': arm, **result,
                                'raw_target_rank': row['raw_target_rank'],
                                'top8_target_recall': int(row['target'].any())})
        for arm in ('C2_LEARNED_MATCH', 'C3_LEARNED_COMBO'):
            gate = gates.get(arm, {})
            pos = decisions[arm]
            utility = ((q - q.mean()) / max(float(q.std()), 1e-6)
                       if arm == 'C2_LEARNED_MATCH' else c3)
            old = 0
            d = math.hypot(int(row['candidate_idx'][pos]) % 16 - int(row['candidate_idx'][old]) % 16,
                           int(row['candidate_idx'][pos]) // 16 - int(row['candidate_idx'][old]) // 16)
            if (not gate.get('enabled') or float(utility[pos] - utility[old]) < gate['delta_min']
                    or d > gate['max_distance_tokens']):
                pos = old
            result = decode_and_metrics(row, pos, tracker, network)
            output_rows.append({'sequence': row['sequence'], 'frame': row['frame'],
                                'label': row['label'], 'arm': arm + '_SAFE', **result,
                                'raw_target_rank': row['raw_target_rank'],
                                'top8_target_recall': int(row['target'].any())})
    return output_rows


def summarize(rows):
    result = {}
    for label in ('matched_good', 'pre_onset', 'severe_onset'):
        result[label] = {}
        for arm in sorted({r['arm'] for r in rows}):
            subset = [r for r in rows if r['label'] == label and r['arm'] == arm]
            if not subset:
                continue
            result[label][arm] = {'events': len(subset), 'sequences': len({r['sequence'] for r in subset}),
                                  'target_top1': float(np.mean([r['selected_target'] for r in subset])),
                                  'center_error_tokens_median': float(np.median([r['center_error_tokens'] for r in subset])),
                                  'iou_median': float(np.median([r['iou'] for r in subset])),
                                  'top8_target_recall': float(np.mean([r['top8_target_recall'] for r in subset]))}
    return result


def main():
    global COHORT
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-calibration', required=True)
    parser.add_argument('--trace-validation', required=True)
    parser.add_argument('--trace-final', required=True)
    parser.add_argument('--output-dir', default=str(OUT))
    parser.add_argument('--cohort-dir', default=str(COHORT))
    args = parser.parse_args()
    COHORT = Path(args.cohort_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing nonempty output directory: ' + str(output))
    if sha256(CONFIG) != EXPECTED_CONFIG or sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.manual_seed(20261010)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    params = parameters('TBSI_E_MHSA_FULL_4090D_seed42')
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'lasher_train')
    network = tracker.network.eval()
    captured = {}

    def capture_fused(_module, _inputs, result):
        captured['fused'] = result.detach()

    hook = network.tbsi_fuse_search.register_forward_hook(capture_fused)
    start = time.time()
    roots = {'calibration': args.trace_calibration, 'validation': args.trace_validation,
             'final_holdout': args.trace_final}
    data = {}
    for split in ('calibration', 'validation'):
        data[split] = extract_split(split, roots[split], tracker, network, None, None, None, captured)
    output.mkdir(parents=True, exist_ok=False)
    for split in ('calibration', 'validation'):
        save_features(output / ('features_' + split + '.npz'), data[split])
    # Fit only on calibration; the validation split selects fixed safe-gate settings.
    model, train_loss, train_events = train_matcher(data['calibration'], 20261010)
    with torch.inference_mode():
        cal_q = [model(torch.as_tensor(r['fused'], device='cuda'),
                       torch.as_tensor(r['rgb'], device='cuda'),
                       torch.as_tensor(r['tir'], device='cuda')).cpu().numpy()
                 for r in data['calibration']]
    coeff, c3_loss, c3_events = fit_linear(data['calibration'], cal_q)
    with torch.inference_mode():
        val_q = [model(torch.as_tensor(r['fused'], device='cuda'),
                       torch.as_tensor(r['rgb'], device='cuda'),
                       torch.as_tensor(r['tir'], device='cuda')).cpu().numpy()
                 for r in data['validation']]
    gates = choose_safe_gates(data['validation'], val_q, coeff, None, tracker, network)
    # Lock all learned parameters and gates before opening final-holdout labels.
    torch.save({'state_dict': model.state_dict(), 'c3_coefficients': coeff,
                'seed': 20261010, 'calibration_manifest_sha256': sha256(COHORT / 'manifest.json')},
               output / 'calibrated_ranker.pth')
    (output / 'validation_gates.json').write_text(json.dumps(gates, indent=2) + '\n')
    data['final_holdout'] = extract_split('final_holdout', roots['final_holdout'], tracker,
                                          network, None, None, None, captured)
    save_features(output / 'features_final_holdout.npz', data['final_holdout'])
    final_rows = evaluate(data['final_holdout'], model, coeff, gates, tracker, network)
    hook.remove()
    with (output / 'final_per_event.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(final_rows[0]))
        writer.writeheader(); writer.writerows(final_rows)
    summary = {
        'id': 'M1-IDENTITY-CALIBRATION-20261010', 'status': 'DIAGNOSTIC-COMPLETE',
        'parent': 'Protected M1 epoch15', 'cohort_manifest_sha256': sha256(COHORT / 'manifest.json'),
        'config_sha256': EXPECTED_CONFIG, 'checkpoint_sha256': EXPECTED_CHECKPOINT,
        'candidate_k': TOPK, 'train_loss_pairwise': train_loss, 'calibration_events_usable': train_events,
        'c3_pairwise_loss': c3_loss, 'c3_events_usable': c3_events, 'c3_coefficients_a_b_c': coeff,
        'validation_gates_locked_before_final': gates, 'final_event_count': len(data['final_holdout']),
        'final_sequence_count': len({r['sequence'] for r in data['final_holdout']}),
        'final_summary': summarize(final_rows),
        'environment': {'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
                        'torch': torch.__version__, 'cuda': torch.version.cuda},
        'elapsed_seconds': time.time() - start,
        'protocol': 'Fixed-state top-8 selection; one event per sequence using M1 onset rule; C2 learned pairwise identity score; C3 learned a*S+b*q+c*S*q; validation gate constrained to <=1 matched-good lost event, no pre-onset IoU decrease, and max 3-token switch distance; final split evaluated once after model/gates saved.',
        'limitations': ['Train-side sequences remain within M1 training distribution.',
                        'Final split has at most 18 sequences; screening evidence only.',
                        'Current working-tree replay is not frame-equivalent to archived M1 outputs.',
                        'This is fixed-state candidate reranking, not closed-loop gain or full test performance.'],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
