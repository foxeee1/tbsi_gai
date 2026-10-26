#!/usr/bin/env python3
"""Fixed-state B0-B3 decision-path test for MHSA template identity reranking."""
import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ['TBSI_INFER_FP16'] = '0'
os.environ['TBSI_PERTURBATION'] = ''

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_m1_search_stage_20261009 import (
    CONFIG, CHECKPOINT, EXPECTED_CONFIG, EXPECTED_CHECKPOINT, iou,
    predicted_box, read_pair, select_frames, sha256)
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target, transform_image_to_crop

COHORT = ROOT / 'experiments/diagnostics/m1_two_direction_cohort_v1'
TRACE = ROOT / 'output/experiments/M1_TWO_DIRECTION_DIAGNOSTIC_20261010/trace_run/trace'
OUT = ROOT / 'output/experiments/M1_MHSA_CANDIDATE_RERANK_V3_20261010'
TOPK = 8
PRIMARY_LAMBDA = 0.5
LAMBDA_SWEEP = (0.25, 0.5, 1.0)


def pool_template(tokens, mask):
    weight = mask.reshape(1, -1, 1).to(device=tokens.device, dtype=tokens.dtype)
    return (tokens * weight).sum(1) / weight.sum(1).clamp_min(1e-8)


def map_stats(score, target, background):
    flat = score.reshape(-1).float()
    t = torch.as_tensor(target.reshape(-1), device=flat.device, dtype=torch.bool)
    b = torch.as_tensor(background.reshape(-1), device=flat.device, dtype=torch.bool)
    order = torch.argsort(flat, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(flat.numel(), device=flat.device)
    p = int(flat.argmax().item())
    ranks = rank[torch.nonzero(t, as_tuple=False).flatten()]
    return {
        'target_top1': int(bool(t[p].item())),
        'target_rank': int(ranks.min().item()) + 1,
        'target_top5': int(bool((ranks < 5).any().item())),
        'target_bg_margin': float((flat[t].max() - flat[b].max()).item()),
        'peak_index': p,
    }


def decode_cell(network, tracker, out, state, image_shape, resize, cell):
    score = torch.zeros_like(out['score_map'])
    score[0, 0, int(cell[1]), int(cell[0])] = 1.0
    box = predicted_box(network, tracker, dict(out, score_map=score), state,
                        image_shape, resize, window=torch.ones_like(tracker.output_window))
    return box


def center_error_tokens(out, cell, gt_center):
    x, y = int(cell[0]), int(cell[1])
    offset = out['offset_map'].float()[0, :, y, x]
    return math.hypot(x + float(offset[0]) - gt_center[0],
                      y + float(offset[1]) - gt_center[1])


def size_l1_tokens(out, cell, gt_crop):
    x, y = int(cell[0]), int(cell[1])
    size = out['size_map'].float()[0, :, y, x]
    return (abs(float(size[0]) * 16.0 - float(gt_crop[2]) * 16.0) +
            abs(float(size[1]) * 16.0 - float(gt_crop[3]) * 16.0)) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=str(OUT))
    parser.add_argument('--limit-sequences', type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing nonempty output: ' + str(output))
    if sha256(CONFIG) != EXPECTED_CONFIG or sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
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
    names = (COHORT / 'sequence_list.txt').read_text().splitlines()
    if args.limit_sequences:
        names = names[:args.limit_sequences]
    seqs = LasHeRDataset('trainingset', str(COHORT / 'sequence_list.txt')).get_sequence_list()
    seq_map = {seq.name: seq for seq in seqs}
    rows = []
    started = time.time()
    selected_total = 0

    for name in names:
        seq = seq_map[name]
        tracker.initialize(read_pair(seq, 0), {
            'init_bbox': seq.ground_truth_rect[0], 'sequence_name': name})
        template = tracker.z_dict1
        fg_mask = tracker.template_fg_mask
        ce_mask = tracker.box_mask_z
        trace = [json.loads(line) for line in
                 (TRACE / name / 'frames.jsonl').read_text().splitlines() if line.strip()]
        selected = [(i, label) for i, label in select_frames(
            trace, 0, 'onset-3-iou-0.5')
                    if label in ('matched_good', 'pre_onset', 'severe_onset')]
        for index, label in selected:
            item = trace[index]
            frame = int(item['frame_id'])
            state = [float(v) for v in item['tracker_state_before']]
            gt = np.asarray(seq.ground_truth_rect[frame], dtype=np.float64)
            image = read_pair(seq, frame)
            crop, resize, amask = sample_target(
                image, state, params.search_factor, output_sz=params.search_size)
            search = tracker.preprocessor.process(crop, amask, half=False)
            search_pair = [search.tensors[:, :3], search.tensors[:, 3:]]
            gt_crop = transform_image_to_crop(
                torch.tensor(gt, dtype=torch.float32), torch.tensor(state, dtype=torch.float32), resize,
                torch.tensor([params.search_size, params.search_size]), normalize=True).numpy()
            gt_center = (float((gt_crop[0] + .5 * gt_crop[2]) * 16.0),
                         float((gt_crop[1] + .5 * gt_crop[3]) * 16.0))
            gx, gy = min(15, max(0, int(gt_center[0]))), min(15, max(0, int(gt_center[1])))
            yy, xx = np.mgrid[:16, :16]
            radius = max(1.0, min(3.0, 4.0 * math.sqrt(max(0., float(gt_crop[2] * gt_crop[3])))))
            distance = (xx + .5 - gt_center[0]) ** 2 + (yy + .5 - gt_center[1]) ** 2
            target = distance <= radius ** 2
            target[gy, gx] = True
            background = distance > (radius + 1.) ** 2

            captured.clear()
            with torch.inference_mode():
                out = network(template=[template.tensors[:, :3], template.tensors[:, 3:]],
                              search=search_pair, ce_template_mask=ce_mask,
                              template_fg_masks=(fg_mask, fg_mask))
            fused = captured.get('fused')
            if fused is None or fused.shape[-2:] != (16, 16):
                raise RuntimeError('fused search hook missing or wrong shape')
            feat = out['backbone_feat']
            p_rgb = pool_template(feat[:, :64], fg_mask).float()
            p_tir = pool_template(feat[:, 320:384], fg_mask).float()
            fused_tokens = fused.flatten(2).transpose(1, 2).float()
            fused_norm = F.normalize(fused_tokens, dim=-1)
            q_rgb = torch.bmm(fused_norm, F.normalize(p_rgb, dim=-1).unsqueeze(-1)).squeeze(-1).reshape(16, 16)
            q_tir = torch.bmm(fused_norm, F.normalize(p_tir, dim=-1).unsqueeze(-1)).squeeze(-1).reshape(16, 16)
            q_map = .5 * (q_rgb + q_tir)
            online = (out['score_map'].float() * tracker.output_window.float())[0, 0]
            flat = online.reshape(-1)
            cand_idx = torch.topk(flat, k=TOPK, largest=True, sorted=True).indices
            cand_xy = [(int(i.item()) % 16, int(i.item()) // 16) for i in cand_idx]
            cand_score = flat[cand_idx]
            q_flat = q_map.reshape(-1)
            cand_q = q_flat[cand_idx]
            score_std = cand_score.std(unbiased=False).clamp_min(1e-8)
            q_std = cand_q.std(unbiased=False).clamp_min(1e-8)
            score_z = (cand_score - cand_score.mean()) / score_std
            q_z = (cand_q - cand_q.mean()) / q_std
            q_sorted = torch.sort(cand_q, descending=True).values
            q_gap_z = float(((q_sorted[0] - q_sorted[1]) / q_std).item())
            confidence = float(torch.sigmoid(torch.tensor((q_gap_z - 1.0) / .5)).item())

            b0 = (cand_xy[0][0], cand_xy[0][1])
            b1 = (gx, gy)
            b0_box = predicted_box(network, tracker, out, state, image.shape, resize)
            b1_box = decode_cell(network, tracker, out, state, image.shape, resize, b1)
            parity = float(np.max(np.abs(np.asarray(b0_box) - np.asarray(item['pred_box']))))
            if parity > .01:
                raise RuntimeError('M1 fixed-state replay parity failure {} {} delta={}'.format(name, frame, parity))
            cand_iou = [iou(decode_cell(network, tracker, out, state, image.shape, resize, xy), gt)
                        for xy in cand_xy]
            oracle_topk_iou = max(cand_iou)

            arms = [('B0_M1', b0, cand_idx)]
            arms.append(('B1_GT_CELL', b1, torch.tensor([gy * 16 + gx], device=cand_idx.device)))
            for lam in LAMBDA_SWEEP:
                rerank_score = score_z + float(lam) * q_z
                order = torch.argsort(rerank_score, descending=True)
                pick = int(order[0].item())
                arms.append(('B2_RERANK_L{}'.format(lam), cand_xy[pick], cand_idx[order]))
            if confidence > .5:
                rerank_score = score_z + PRIMARY_LAMBDA * confidence * q_z
                order = torch.argsort(rerank_score, descending=True)
                pick = int(order[0].item())
                b3 = cand_xy[pick]
            else:
                order = torch.arange(len(cand_xy), device=cand_idx.device)
                b3 = b0
            arms.append(('B3_CONF_GATED', b3, cand_idx[order]))

            for arm, cell, candidate_order in arms:
                box = b0_box if arm == 'B0_M1' else (b1_box if arm == 'B1_GT_CELL' else
                       decode_cell(network, tracker, out, state, image.shape, resize, cell))
                chosen_score = online.clone()
                if arm.startswith('B2_'):
                    lam = float(arm.rsplit('L', 1)[1])
                    adjusted = cand_score + lam * score_std * q_z
                    chosen_score.reshape(-1)[cand_idx] = adjusted
                elif arm == 'B1_GT_CELL':
                    chosen_score.zero_()
                    chosen_score[gy, gx] = 1.0
                elif arm == 'B3_CONF_GATED' and confidence > .5:
                    adjusted = cand_score + PRIMARY_LAMBDA * confidence * score_std * q_z
                    chosen_score.reshape(-1)[cand_idx] = adjusted
                stats = map_stats(chosen_score, target, background)
                candidate_xy = [(int(i.item()) % 16, int(i.item()) // 16) for i in candidate_order]
                candidate_target_ranks = [rank + 1 for rank, (x, y) in enumerate(candidate_xy)
                                          if target[y, x]]
                candidate_gt_rank = min(candidate_target_ranks) if candidate_target_ranks else TOPK + 1
                selected_is_target = int(bool(target[cell[1], cell[0]]))
                candidate_values = cand_score
                if arm.startswith('B2_'):
                    lam = float(arm.rsplit('L', 1)[1])
                    utility = score_z + lam * q_z
                    local_order = torch.argsort(utility, descending=True)
                    candidate_values = (cand_score + lam * score_std * q_z)[local_order]
                elif arm == 'B1_GT_CELL':
                    candidate_values = torch.ones(1, device=cand_score.device)
                elif arm == 'B3_CONF_GATED' and confidence > .5:
                    utility = score_z + PRIMARY_LAMBDA * confidence * q_z
                    local_order = torch.argsort(utility, descending=True)
                    candidate_values = (cand_score + PRIMARY_LAMBDA * confidence * score_std * q_z)[local_order]
                target_values = [candidate_values[j] for j, (x, y) in enumerate(candidate_xy)
                                 if target[y, x]]
                background_values = [candidate_values[j] for j, (x, y) in enumerate(candidate_xy)
                                     if background[y, x]]
                candidate_margin = (float((torch.stack(target_values).max() -
                                            torch.stack(background_values).max()).item())
                                    if target_values and background_values else None)
                rows.append({
                    'sequence': name, 'frame': frame, 'label': label, 'arm': arm,
                    'lambda_primary': PRIMARY_LAMBDA, 'candidate_k': TOPK,
                    'identity_gap_z': q_gap_z, 'identity_confidence': confidence,
                    'gate_fired': int(confidence > .5), 'selected_x': cell[0], 'selected_y': cell[1],
                    'gt_x': gx, 'gt_y': gy, 'selected_in_gt_support': selected_is_target,
                    'parity_max_px': parity if arm == 'B0_M1' else float('nan'),
                    'candidate_gt_rank': candidate_gt_rank,
                    'candidate_gt_top5': int(candidate_gt_rank <= 5),
                    'candidate_target_top1': selected_is_target,
                    'gt_in_raw_top5': int(any(bool(target[y, x]) for x, y in cand_xy[:5])),
                    'gt_in_raw_top8': int(any(bool(target[y, x]) for x, y in cand_xy)),
                    'raw_topk_oracle_iou': oracle_topk_iou,
                    'iou': iou(box, gt),
                    'center_error_tokens': center_error_tokens(out, cell, gt_center),
                    'size_l1_tokens': size_l1_tokens(out, cell, gt_crop),
                    'target_rank': candidate_gt_rank, 'target_top1': selected_is_target,
                    'target_top5': int(candidate_gt_rank <= 5), 'target_bg_margin': candidate_margin,
                    'trace_iou': iou(item['pred_box'], gt),
                })
            selected_total += 1
        print('sequence {} selected={} elapsed_s={:.1f}'.format(name, len(selected), time.time() - started), flush=True)

    hook.remove()
    if not rows:
        raise RuntimeError('no selected fixed-state frames')
    output.mkdir(parents=True)
    with (output / 'per_arm.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    arm_names = ['B0_M1', 'B1_GT_CELL', 'B2_RERANK_L0.25', 'B2_RERANK_L0.5',
                 'B2_RERANK_L1.0', 'B3_CONF_GATED']
    metrics = ['target_top1', 'target_top5', 'center_error_tokens', 'iou', 'size_l1_tokens',
               'target_bg_margin', 'selected_in_gt_support']
    for label in ('matched_good', 'pre_onset', 'severe_onset'):
        groups[label] = {}
        for arm in arm_names:
            subset = [r for r in rows if r['label'] == label and r['arm'] == arm]
            groups[label][arm] = {'n': len(subset), 'sequences': len({r['sequence'] for r in subset})}
            for metric in metrics:
                vals = [float(r[metric]) for r in subset if r[metric] not in (None, '')]
                groups[label][arm][metric] = {
                    'mean': float(np.mean(vals)) if vals else None,
                    'median': float(np.median(vals)) if vals else None,
                }
            groups[label][arm]['gt_in_raw_top5_rate'] = (
                float(np.mean([r['gt_in_raw_top5'] for r in subset])) if subset else None)
            groups[label][arm]['gt_in_raw_top8_rate'] = (
                float(np.mean([r['gt_in_raw_top8'] for r in subset])) if subset else None)
            groups[label][arm]['gate_fire_rate'] = (
                float(np.mean([r['gate_fired'] for r in subset])) if subset else None)
            groups[label][arm]['raw_topk_oracle_iou_median'] = (
                float(np.median([r['raw_topk_oracle_iou'] for r in subset])) if subset else None)

    # Frame-paired deltas against B0, reported by sequence to avoid frame pooling as independent evidence.
    by_key = {(r['sequence'], r['frame'], r['arm']): r for r in rows}
    paired_rows = []
    for row in rows:
        if row['arm'] == 'B0_M1':
            continue
        base = by_key[(row['sequence'], row['frame'], 'B0_M1')]
        delta_fields = {}
        for metric in ('target_top1', 'center_error_tokens', 'iou', 'size_l1_tokens', 'target_bg_margin'):
            delta_fields['delta_' + metric] = (
                None if row[metric] is None or base[metric] is None else
                float(row[metric]) - float(base[metric]))
        paired_rows.append({
            'sequence': row['sequence'], 'frame': row['frame'], 'label': row['label'], 'arm': row['arm'],
            **delta_fields,
        })
    with (output / 'paired_deltas.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    summary = {
        'id': 'M1-MHSA-CANDIDATE-RERANK-20261010',
        'status': 'DIAGNOSTIC-COMPLETE', 'parent': 'Protected M1 epoch15',
        'analysis_version': 3,
        'config_sha256': EXPECTED_CONFIG, 'checkpoint_sha256': EXPECTED_CHECKPOINT,
        'cohort_manifest_sha256': sha256(COHORT / 'manifest.json'),
        'sequence_list_sha256': sha256(COHORT / 'sequence_list.txt'),
        'fixed_state_frames': selected_total, 'candidate_k': TOPK,
        'identity_score': 'Equal-weight mean cosine of fused M1 search feature to RGB/TIR foreground-pooled post-TBSI template tokens; no learned projection.',
        'normalization': 'Candidate CENTER and identity scores independently z-scored within raw Hann top-K.',
        'primary_lambda': PRIMARY_LAMBDA,
        'sensitivity_lambdas': list(LAMBDA_SWEEP),
        'B3_confidence': 'sigmoid((top1_top2_identity_gap_in_candidate_std - 1.0)/0.5); rerank only if confidence > 0.5.',
        'gate_fire_rate_all_frames': float(np.mean([r['gate_fired'] for r in rows if r['arm'] == 'B0_M1'])),
        'environment': {'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
                        'torch': torch.__version__, 'cuda': torch.version.cuda},
        'elapsed_seconds': time.time() - started,
        'max_B0_trace_parity_px': max((float(r['parity_max_px']) for r in rows if r['arm'] == 'B0_M1'), default=None),
        'groups': groups,
        'metric_contract': 'target_top1 is the actually selected cell inside GT support; target_top5/rank and target-hard-negative margin are computed within each arm\'s candidate list. V1 whole-map ranks and V2 whole-map margins are superseded.',
        'limitations': [
            'B2/B3 are hand-defined zero-training diagnostic rules with fixed raw cosine, not learned calibrated identity scores.',
            'B1 is an offline GT-cell oracle, never an online method.',
            'Top-K and normalized lambda are fixed in advance; lambda sweep is sensitivity analysis, not candidate selection.',
            'Current working-tree replay is not frame-equivalent to archived M1 predictions.',
            'Train-side cohort is held out from prior diagnostics but was part of M1 training; proxy attributes are descriptive.',
            'A successful fixed-state rerank would not yet establish closed-loop gain or full-test performance.',
        ],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
