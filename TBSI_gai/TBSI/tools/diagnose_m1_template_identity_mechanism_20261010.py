#!/usr/bin/env python3
"""Fixed-state counterfactual test of template identity use in M1."""
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_m1_search_stage_20261009 import (
    CONFIG, CHECKPOINT, EXPECTED_CONFIG, EXPECTED_CHECKPOINT, coverage, iou,
    predicted_box, read_pair, select_frames, sha256)
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target, transform_image_to_crop

COHORT = ROOT / 'experiments/diagnostics/m1_two_direction_cohort_v1'
TRACE = ROOT / 'output/experiments/M1_TWO_DIRECTION_DIAGNOSTIC_20261010/trace_run/trace'
OUT_DEFAULT = ROOT / 'output/experiments/M1_MHSA_TEMPLATE_IDENTITY_MECHANISM_20261010'


def mask_metrics(score, target, background):
    flat = score.reshape(-1).float()
    target = torch.as_tensor(target.reshape(-1), device=flat.device, dtype=torch.bool)
    background = torch.as_tensor(background.reshape(-1), device=flat.device, dtype=torch.bool)
    order = torch.argsort(flat, descending=True)
    rank_map = torch.empty_like(order)
    rank_map[order] = torch.arange(flat.numel(), device=flat.device)
    target_idx = torch.nonzero(target, as_tuple=False).flatten()
    bg_idx = torch.nonzero(background, as_tuple=False).flatten()
    peak = int(flat.argmax().item())
    target_rank = int(rank_map[target_idx].min().item()) + 1
    return {
        'target_top1': int(bool(target[peak].item())),
        'target_rank': target_rank,
        'target_max': float(flat[target_idx].max().item()),
        'hard_bg_max': float(flat[bg_idx].max().item()),
        'target_bg_margin': float((flat[target_idx].max() - flat[bg_idx].max()).item()),
    }


def foreground_pool(tokens, weights):
    weights = weights.reshape(1, -1, 1).to(device=tokens.device, dtype=tokens.dtype)
    return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=str(OUT_DEFAULT))
    parser.add_argument('--wrong-templates-per-query', type=int, default=3)
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
    net = tracker.network.eval()
    sequence_names = (COHORT / 'sequence_list.txt').read_text().splitlines()
    if args.limit_sequences:
        sequence_names = sequence_names[:args.limit_sequences]
    sequences = LasHeRDataset('trainingset', str(COHORT / 'sequence_list.txt')).get_sequence_list()
    seq_map = {seq.name: seq for seq in sequences}
    cohort = json.loads((COHORT / 'manifest.json').read_text())
    strata = {row['name']: row['quartile_bins'] for row in cohort['sequences']}
    templates, chosen_by_seq = {}, {}
    started = time.time()

    # Cache each sequence's actual initialization crop and its target-token mask.
    for name in sequence_names:
        seq = seq_map[name]
        tracker.initialize(read_pair(seq, 0), {
            'init_bbox': seq.ground_truth_rect[0], 'sequence_name': name})
        templates[name] = {
            'tensor': tracker.z_dict1.tensors.detach().clone(),
            'mask': tracker.template_fg_mask.detach().clone(),
            'ce_mask': None if tracker.box_mask_z is None else tracker.box_mask_z.detach().clone(),
        }
        trace = [json.loads(line) for line in (TRACE / name / 'frames.jsonl').read_text().splitlines() if line.strip()]
        chosen_by_seq[name] = select_frames(trace, 0, 'onset-3-iou-0.5')

    rows = []
    n_wrong = max(1, min(8, int(args.wrong_templates_per_query)))
    for seq_index, name in enumerate(sequence_names):
        seq = seq_map[name]
        trace = [json.loads(line) for line in (TRACE / name / 'frames.jsonl').read_text().splitlines() if line.strip()]
        selected = [(idx, label) for idx, label in chosen_by_seq[name]
                    if label in ('matched_good', 'pre_onset', 'severe_onset')]
        if not selected:
            continue
        donors = [candidate for candidate in sequence_names if candidate != name and
                  strata[candidate]['relative_area_median'] == strata[name]['relative_area_median'] and
                  strata[candidate]['relative_motion_p90'] == strata[name]['relative_motion_p90']]
        if len(donors) < n_wrong:
            donors = [candidate for candidate in sequence_names if candidate != name]
        offset = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % len(donors)
        donors = [donors[(offset + j) % len(donors)] for j in range(min(n_wrong, len(donors)))]

        for index, label in selected:
            item = trace[index]
            frame = int(item['frame_id'])
            state = [float(v) for v in item['tracker_state_before']]
            gt = np.asarray(seq.ground_truth_rect[frame], dtype=np.float64)
            image = read_pair(seq, frame)
            crop, resize, amask = sample_target(image, state, params.search_factor,
                                                output_sz=params.search_size)
            search = tracker.preprocessor.process(crop, amask, half=False)
            search_pair = [search.tensors[:, :3], search.tensors[:, 3:]]
            gt_crop = transform_image_to_crop(
                torch.tensor(gt, dtype=torch.float32), torch.tensor(state, dtype=torch.float32), resize,
                torch.tensor([params.search_size, params.search_size]), normalize=True).numpy()
            cx = float((gt_crop[0] + .5 * gt_crop[2]) * 16)
            cy = float((gt_crop[1] + .5 * gt_crop[3]) * 16)
            yy, xx = np.mgrid[:16, :16]
            radius = max(1., min(3., 4 * math.sqrt(max(0., float(gt_crop[2] * gt_crop[3])))))
            dist = (xx + .5 - cx) ** 2 + (yy + .5 - cy) ** 2
            target, background = dist <= radius ** 2, dist > (radius + 1.) ** 2
            if not target.any() or not background.any():
                continue
            arms = [('self', name)] + [('wrong', donor) for donor in donors]
            for arm, template_name in arms:
                tpl = templates[template_name]
                z = [tpl['tensor'][:, :3], tpl['tensor'][:, 3:]]
                with torch.inference_mode():
                    out = net(template=z, search=search_pair,
                              ce_template_mask=tpl['ce_mask'],
                              template_fg_masks=(tpl['mask'], tpl['mask']))
                feat = out['backbone_feat']
                if feat.ndim != 3 or feat.shape[1] != 640:
                    raise RuntimeError('unexpected final token layout {}'.format(tuple(feat.shape)))
                rgb_t = foreground_pool(feat[:, :64], tpl['mask'])
                tir_t = foreground_pool(feat[:, 320:384], tpl['mask'])
                rgb_s = feat[:, 64:320]
                tir_s = feat[:, 384:640]
                rgb_map = torch.nn.functional.normalize(rgb_s.float(), dim=-1) @ torch.nn.functional.normalize(rgb_t.float(), dim=-1).transpose(0, 1)
                tir_map = torch.nn.functional.normalize(tir_s.float(), dim=-1) @ torch.nn.functional.normalize(tir_t.float(), dim=-1).transpose(0, 1)
                rgb_map = rgb_map.reshape(16, 16).detach()
                tir_map = tir_map.reshape(16, 16).detach()
                fused_scores = (tracker.output_window.float() * out['score_map'].float())[0, 0]
                ctr = mask_metrics(fused_scores, target, background)
                rm = mask_metrics(rgb_map, target, background)
                tm = mask_metrics(tir_map, target, background)
                pred = predicted_box(net, tracker, out, state, image.shape, resize)
                trace_iou = iou(item['pred_box'], gt)
                parity = float(np.max(np.abs(np.asarray(pred) - np.asarray(item['pred_box'])))) if arm == 'self' else float('nan')
                rows.append({
                    'sequence': name, 'frame': frame, 'label': label,
                    'arm': arm, 'template_sequence': template_name,
                    'donor_same_area_motion_bin': int(strata[template_name]['relative_area_median'] == strata[name]['relative_area_median'] and strata[template_name]['relative_motion_p90'] == strata[name]['relative_motion_p90']),
                    'trace_iou': trace_iou, 'counterfactual_iou': iou(pred, gt),
                    'parity_max_px': parity,
                    'center_target_top1': ctr['target_top1'], 'center_target_rank': ctr['target_rank'],
                    'center_target_bg_margin': ctr['target_bg_margin'],
                    'rgb_template_target_top1': rm['target_top1'], 'rgb_template_target_rank': rm['target_rank'],
                    'rgb_template_target_bg_margin': rm['target_bg_margin'],
                    'tir_template_target_top1': tm['target_top1'], 'tir_template_target_rank': tm['target_rank'],
                    'tir_template_target_bg_margin': tm['target_bg_margin'],
                })
        print('sequence {} selected={} elapsed_s={:.1f}'.format(name, len(selected), time.time() - started), flush=True)

    output.mkdir(parents=True)
    fields = list(rows[0]) if rows else []
    with (output / 'per_arm.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    self_rows = [row for row in rows if row['arm'] == 'self']
    wrong_rows = [row for row in rows if row['arm'] == 'wrong']
    by_key_wrong = {}
    for row in wrong_rows:
        by_key_wrong.setdefault((row['sequence'], row['frame'], row['label']), []).append(row)
    paired = []
    for row in self_rows:
        controls = by_key_wrong[(row['sequence'], row['frame'], row['label'])]
        for key in ('counterfactual_iou', 'center_target_top1', 'center_target_rank',
                    'center_target_bg_margin', 'rgb_template_target_top1',
                    'rgb_template_target_rank', 'rgb_template_target_bg_margin',
                    'tir_template_target_top1', 'tir_template_target_rank',
                    'tir_template_target_bg_margin'):
            vals = [float(control[key]) for control in controls]
            row['wrong_median_' + key] = float(np.median(vals))
            row['self_minus_wrong_' + key] = float(row[key]) - row['wrong_median_' + key]
        paired.append(row)
    with (output / 'paired_self_vs_wrong.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]) if paired else [])
        writer.writeheader()
        writer.writerows(paired)

    def summary_metric(rows_, key):
        vals = [float(r[key]) for r in rows_]
        return {'n': len(vals), 'mean': float(np.mean(vals)) if vals else None,
                'median': float(np.median(vals)) if vals else None,
                'positive': int(sum(v > 0 for v in vals)) if vals else 0}

    groups = {}
    for label in ('matched_good', 'pre_onset', 'severe_onset'):
        group = [r for r in paired if r['label'] == label]
        groups[label] = {
            'sequences': len({r['sequence'] for r in group}),
            'n_frames': len(group),
            'self_center_top1_rate': float(np.mean([r['center_target_top1'] for r in group])) if group else None,
            'wrong_center_top1_rate': float(np.mean([r['wrong_median_center_target_top1'] for r in group])) if group else None,
            'self_iou_median': float(np.median([r['counterfactual_iou'] for r in group])) if group else None,
            'wrong_iou_median': float(np.median([r['wrong_median_counterfactual_iou'] for r in group])) if group else None,
            'center_margin_self_minus_wrong': summary_metric(group, 'self_minus_wrong_center_target_bg_margin'),
            'rgb_identity_margin_self_minus_wrong': summary_metric(group, 'self_minus_wrong_rgb_template_target_bg_margin'),
            'tir_identity_margin_self_minus_wrong': summary_metric(group, 'self_minus_wrong_tir_template_target_bg_margin'),
            'center_rank_self_minus_wrong': summary_metric(group, 'self_minus_wrong_center_target_rank'),
        }
    summary = {
        'id': 'M1-MHSA-TEMPLATE-IDENTITY-MECHANISM-20261010',
        'status': 'DIAGNOSTIC-COMPLETE',
        'parent': 'Protected M1 epoch15',
        'config_sha256': EXPECTED_CONFIG, 'checkpoint_sha256': EXPECTED_CHECKPOINT,
        'cohort_manifest_sha256': sha256(COHORT / 'manifest.json'),
        'sequence_list_sha256': sha256(COHORT / 'sequence_list.txt'),
        'trace_dir': str(TRACE), 'selected_sequences': len(sequence_names),
        'fixed_state_frames': len(self_rows), 'wrong_templates_per_frame': n_wrong,
        'matching': 'Wrong templates preferentially share target-area and motion proxy bins; deterministic cyclic donors.',
        'environment': {'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
                        'torch': torch.__version__, 'cuda': torch.version.cuda},
        'elapsed_seconds': time.time() - started,
        'max_self_trace_parity_px': max((r['parity_max_px'] for r in self_rows), default=None),
        'groups': groups,
        'limitations': [
            'Cosine similarity is a diagnostic readout, not the trained CENTER score or causal module gain.',
            'Wrong-template counterfactual changes initialization appearance and is an identity-use stress test, not a deployable alternative.',
            'Trace uses current working-tree replay and differs from archived M1 frame predictions.',
            'M1 checkpoint has seen LasHeR training sequences; sequence cohort is held out from prior diagnostics, not backbone training.',
            'Proxy strata are dataset-derived and not official attribute labels.',
        ],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
