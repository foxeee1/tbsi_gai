#!/usr/bin/env python3
"""Paired fixed-state feature attribution on ten M1 full-244 test failures."""
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack

AUDIT = ROOT / 'output/experiments/M1_FULL244_SEQUENCE_AUDIT_20261009/per_sequence.csv'
TRACE_ROOT = ROOT / 'output/experiments/M1_FUSION_FLOW_FULL244_DIAG_20261009/closed_loop_trace/trace'
M1_CFG = 'TBSI_E_MHSA_FULL_4090D_seed42'
E6_CFG = 'M1_AMCM_DR_LOSS_E6_REPR_QUICK15_3EP_20261019'
M1_CKPT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
E6_CKPT = ROOT / 'output/experiments/M1_AMCM_DR_LOSS_E6_REPR_QUICK15_3EP_20261019/checkpoints/TBSITrack_ep0003.pth.tar'
OUT = ROOT / 'output/experiments/M1_AMCM_TEST_FAILURE_STRUCTURE_DIAG_V2_20261019'

# Chosen deterministically from valid M1 closed-loop frames: crop contains GT,
# GT is decodable, and IoU is near 0.30. The exact trace state is frozen below.
FRAME_IDS = {
    'girldownstairfromlight': 708,
    'leftopenexersicebook': 120,
    'blkboydown': 49,
    'foamatgirl`srighthand': 67,
    '7rightorangegirl': 343,
    'rightbluewhite': 236,
    'umbrella': 197,
    'ab_bolstershaking': 2952,
    'boyss': 218,
    'rightdarksingleman': 883,
}


def box_iou(a, b):
    ix = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


def load_trace(seq, frame):
    path = TRACE_ROOT / seq / 'frames.jsonl'
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if int(row['frame_id']) == frame:
                return row
    raise RuntimeError('trace frame missing: {} {}'.format(seq, frame))


def feature_summary(tensor, mask):
    # [B,C,H,W] -> vectors at target and selected hard-background cells.
    x = tensor.detach().float()[0].permute(1, 2, 0).cpu().numpy()
    tgt = x[mask]
    bg = x[~mask]
    if len(tgt) == 0 or len(bg) == 0:
        raise RuntimeError('empty target or background region')
    target_mean = tgt.mean(axis=0)
    bg_norm = np.linalg.norm(bg, axis=1)
    bg_peak = bg[int(np.argmax(bg_norm))]
    denom = np.linalg.norm(target_mean) * np.linalg.norm(bg_peak) + 1e-12
    cosine = float(np.dot(target_mean, bg_peak) / denom)
    return {'target_bg_cosine': cosine,
            'target_bg_l2': float(np.linalg.norm(target_mean - bg_peak)),
            'target_feature_norm': float(np.linalg.norm(target_mean)),
            'background_peak_feature_norm': float(np.linalg.norm(bg_peak))}


def feature_summary_at_background(array, mask, y, x):
    feature = array[0].transpose(1, 2, 0)
    target = feature[mask].mean(axis=0)
    background = feature[y, x]
    cosine = np.dot(target, background) / (np.linalg.norm(target) * np.linalg.norm(background) + 1e-12)
    return float(cosine), float(np.linalg.norm(target - background))


def make_mask(gt, crop, size=16):
    x, y, side, _ = crop
    gx, gy, gw, gh = gt
    x1, x2 = (gx - x) * size / side, (gx + gw - x) * size / side
    y1, y2 = (gy - y) * size / side, (gy + gh - y) * size / side
    yy, xx = np.mgrid[:size, :size]
    mask = ((xx + .5 >= x1) & (xx + .5 <= x2) &
            (yy + .5 >= y1) & (yy + .5 <= y2))
    if not mask.any():
        mask[int(np.clip((y1 + y2) / 2, 0, size - 1)),
             int(np.clip((x1 + x2) / 2, 0, size - 1))] = True
    return mask


def read_pair(seq, frame, seqmap):
    item = seqmap[seq]
    rgb = cv2.cvtColor(cv2.imread(item.frames[0][frame]), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(item.frames[1][frame]), cv2.COLOR_BGR2RGB)
    if rgb is None or tir is None:
        raise RuntimeError('could not read frame {} {}'.format(seq, frame))
    return np.concatenate((rgb, tir), axis=2)


def run_arm(arm, cfg_name, checkpoint, seqmap, events):
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(checkpoint)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_INTERNAL_DIAGNOSTIC'] = '0'
    params = parameters(cfg_name)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(checkpoint)
    tracker = TBSITrack(params, 'm1_amcm_test_failure_structure_diag')
    captured = {}
    net = tracker.network

    def fuse_pre(module, inputs):
        captured['fuse_input'] = inputs[0].detach().float().cpu()

    def fuse_post(module, inputs, output):
        captured['fuse_output'] = output.detach().float().cpu()

    handles = [net.tbsi_fuse_search.register_forward_pre_hook(fuse_pre),
               net.tbsi_fuse_search.register_forward_hook(fuse_post)]
    if getattr(net, 'amcm_reliability', None) is not None:
        def amcm_pre(module, inputs):
            captured['amcm_input_rgb'] = inputs[0].detach().float().cpu()
            captured['amcm_input_tir'] = inputs[1].detach().float().cpu()

        def amcm_post(module, inputs, output):
            captured['amcm_output_rgb'] = output[0].detach().float().cpu()
            captured['amcm_output_tir'] = output[1].detach().float().cpu()
            if len(output) > 2 and isinstance(output[2], dict):
                for key in ('nlformer_local_live', 'nlformer_channel_live',
                            'nlformer_complementary_live'):
                    if key in output[2]:
                        captured[key] = output[2][key].detach().float().cpu()

        handles.extend([net.amcm_reliability.register_forward_pre_hook(amcm_pre),
                        net.amcm_reliability.register_forward_hook(amcm_post)])

    rows = []
    for i, event in enumerate(events, 1):
        seq, frame, trace = event['sequence'], event['frame'], event['trace']
        # Reinitialize template identically for each isolated event, then restore
        # the exact M1 state immediately before the selected frame.
        first = read_pair(seq, 0, seqmap)
        tracker.initialize(first, {'init_bbox': np.asarray(seqmap[seq].ground_truth_rect[0]),
                                   'sequence_name': seq})
        tracker.state = list(map(float, trace['tracker_state_before']))
        tracker.prev_search = None
        tracker.prev_temporal_tokens = None
        tracker.prev_head_feature = None
        image = read_pair(seq, frame, seqmap)
        captured.clear()
        pred = tracker.track(image, {'gt_bbox': trace['gt_box'], 'sequence_name': seq,
                                     'frame_num': frame})
        diag = tracker._last_fixed_state_diag
        score = diag['score_map'][0, 0].numpy()
        score = score * tracker.output_window[0, 0].detach().float().cpu().numpy()
        mask = make_mask(trace['gt_box'], trace['current_crop_xywh'])
        target_score = float(score[mask].max())
        bg_score = float(score[~mask].max())
        target_idx = np.argwhere(mask & (score == target_score))[0]
        bg_rank_idx = np.unravel_index(np.argmax(np.where(~mask, score, -np.inf)), score.shape)
        gt_rank = 1 + int(np.sum(score[~mask] > target_score))
        fused = captured['fuse_output']
        in_fused = captured['fuse_input']
        # Fusion input channels are [RGB, TIR]. Its output has C channels.
        pre_rgb, pre_tir = in_fused[:, :fused.shape[1]], in_fused[:, fused.shape[1]:]
        if 'amcm_input_rgb' in captured:
            pre_rgb, pre_tir = captured['amcm_input_rgb'], captured['amcm_input_tir']
            post_rgb, post_tir = captured['amcm_output_rgb'], captured['amcm_output_tir']
        else:
            post_rgb, post_tir = pre_rgb, pre_tir
        delta_rgb, delta_tir = post_rgb - pre_rgb, post_tir - pre_tir
        target_coord = (int(target_idx[1]), int(target_idx[0]))
        bg_coord = (int(bg_rank_idx[1]), int(bg_rank_idx[0]))
        out_path = OUT / 'tensors' / '{}_{}_{}'.format(arm, seq, frame)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path.with_suffix('.npz'),
                            pre_rgb=pre_rgb.numpy(), pre_tir=pre_tir.numpy(),
                            post_rgb=post_rgb.numpy(), post_tir=post_tir.numpy(),
                            residual_rgb=delta_rgb.numpy(), residual_tir=delta_tir.numpy(),
                            fused_input=in_fused.numpy(), fused_output=fused.numpy(),
                            hann_score=score, gt_region=mask.astype(np.uint8))
        row = {'arm': arm, 'sequence': seq, 'frame': frame,
               'm1_closed_loop_iou': box_iou(trace['pred_box'], trace['gt_box']),
               'fixed_state_iou': box_iou(pred['target_bbox'], trace['gt_box']),
               'gt_rank': gt_rank, 'target_peak_score': target_score,
               'strongest_background_score': bg_score,
               'target_background_margin': target_score - bg_score,
               'top1_is_gt_region': bool(mask[np.unravel_index(np.argmax(score), score.shape)]),
               'target_peak_xy': target_coord, 'background_peak_xy': bg_coord,
               'snapshot': str(out_path.with_suffix('.npz'))}
        for name, feat in [('rgb_pre', pre_rgb), ('tir_pre', pre_tir),
                           ('rgb_post', post_rgb), ('tir_post', post_tir),
                           ('fused', fused)]:
            row.update({name + '_' + k: v for k, v in feature_summary(feat, mask).items()})
        for name in ('nlformer_local_live', 'nlformer_channel_live',
                     'nlformer_complementary_live'):
            if name in captured:
                branch = captured[name]
                values = torch.linalg.vector_norm(branch, dim=1)[0].numpy()
                row[name + '_target_norm'] = float(values[mask].mean())
                row[name + '_background_norm'] = float(values[~mask].mean())
                row[name + '_target_bg_ratio'] = float(values[mask].mean() /
                                                        (values[~mask].mean() + 1e-8))
        for name, delta, source in [('rgb', delta_rgb, pre_rgb), ('tir', delta_tir, pre_tir)]:
            norm = torch.linalg.vector_norm(delta[0].float(), dim=0).numpy()
            source_norm = torch.linalg.vector_norm(source[0].float(), dim=0).numpy()
            row[name + '_residual_mean'] = float(norm.mean())
            row[name + '_residual_rel_mean'] = float((norm / (source_norm + 1e-8)).mean())
            row[name + '_residual_target_mean'] = float(norm[mask].mean())
            row[name + '_residual_background_mean'] = float(norm[~mask].mean())
            row[name + '_residual_target_bg_ratio'] = float(norm[mask].mean() / (norm[~mask].mean() + 1e-8))
            row[name + '_residual_spatial_cv'] = float(norm.std() / (norm.mean() + 1e-8))
        rows.append(row)
        print('[{}] {}/{} {} frame={} IoU={:.3f} rank={} margin={:.5f}'.format(
            arm, i, len(events), seq, frame, row['fixed_state_iou'], gt_rank,
            row['target_background_margin']), flush=True)
    for h in handles:
        h.remove()
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this diagnostic')
    OUT.mkdir(parents=True, exist_ok=True)
    with AUDIT.open() as f:
        aucs = {r['sequence']: float(r['m1_auc']) for r in csv.DictReader(f)}
    events, names = [], sorted(FRAME_IDS)
    list_path = OUT / 'sequence_list.txt'
    list_path.write_text('\n'.join(names) + '\n')
    dataset = LasHeRDataset('testingset', str(list_path)).get_sequence_list()
    seqmap = {s.name: s for s in dataset}
    for seq in names:
        frame = FRAME_IDS[seq]
        trace = load_trace(seq, frame)
        if not trace['gt_valid']:
            raise RuntimeError('invalid GT in {} frame {}'.format(seq, frame))
        events.append({'sequence': seq, 'frame': frame, 'trace': trace})
    manifest = [{'sequence': e['sequence'], 'm1_auc': aucs[e['sequence']],
                 'frame': e['frame'], 'm1_state_before': e['trace']['tracker_state_before'],
                 'gt_box': e['trace']['gt_box'], 'crop_xywh': e['trace']['current_crop_xywh'],
                 'm1_trace_pred_box': e['trace']['pred_box'],
                 'm1_trace_iou': box_iou(e['trace']['pred_box'], e['trace']['gt_box'])}
                for e in events]
    (OUT / 'fixed_event_manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = run_arm('m1', M1_CFG, M1_CKPT, seqmap, events)
    rows += run_arm('e6_amcm', E6_CFG, E6_CKPT, seqmap, events)
    with (OUT / 'per_event.csv').open('w', newline='') as f:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    pairs = []
    for seq in names:
        a = next(r for r in rows if r['sequence'] == seq and r['arm'] == 'm1')
        b = next(r for r in rows if r['sequence'] == seq and r['arm'] == 'e6_amcm')
        with np.load(a['snapshot']) as z:
            base = {k: z[k].copy() for k in z.files}
        with np.load(b['snapshot']) as z:
            adapted = {k: z[k].copy() for k in z.files}
        mask = base['gt_region'].astype(bool)
        base_map, adapted_map = base['hann_score'], adapted['hann_score']
        bg_y, bg_x = np.unravel_index(np.argmax(np.where(~mask, base_map, -np.inf)), base_map.shape)
        def peak_metrics(score_map):
            gt_peak = float(score_map[mask].max())
            bg_peak = float(score_map[~mask].max())
            rank = 1 + int(np.sum(score_map[~mask] > gt_peak))
            top1 = bool(mask[np.unravel_index(np.argmax(score_map), score_map.shape)])
            return gt_peak, bg_peak, rank, top1
        m1_gt, m1_bg, m1_rank, m1_top = peak_metrics(base_map)
        e6_gt, e6_bg, e6_rank, e6_top = peak_metrics(adapted_map)
        residuals = {}
        feature_metrics = {}
        for modality in ('rgb', 'tir'):
            delta = adapted['residual_' + modality][0]
            delta_norm = np.linalg.norm(delta, axis=0)
            residuals[modality + '_residual_target_mean'] = float(delta_norm[mask].mean())
            residuals[modality + '_residual_background_peak'] = float(delta_norm[bg_y, bg_x])
            residuals[modality + '_residual_target_bg_ratio'] = float(
                delta_norm[mask].mean() / (delta_norm[bg_y, bg_x] + 1e-8))
            residuals[modality + '_residual_spatial_cv'] = float(delta_norm.std() / (delta_norm.mean() + 1e-8))
            for stage, arrays in (('pre', base), ('post', adapted)):
                cosine, distance = feature_summary_at_background(
                    arrays[stage + '_' + modality], mask, bg_y, bg_x)
                feature_metrics[modality + '_' + stage + '_cosine_to_m1_hard_bg'] = cosine
                feature_metrics[modality + '_' + stage + '_distance_to_m1_hard_bg'] = distance
        for stage, arrays in (('m1', base), ('e6', adapted)):
            cosine, distance = feature_summary_at_background(arrays['fused_output'], mask, bg_y, bg_x)
            feature_metrics[stage + '_fused_cosine_to_m1_hard_bg'] = cosine
            feature_metrics[stage + '_fused_distance_to_m1_hard_bg'] = distance
        pairs.append({'sequence': seq, 'frame': a['frame'], 'm1_auc': aucs[seq],
                      'm1_iou': a['fixed_state_iou'], 'e6_amcm_iou': b['fixed_state_iou'],
                      'delta_iou': b['fixed_state_iou'] - a['fixed_state_iou'],
                      'm1_gt_rank': m1_rank, 'e6_amcm_gt_rank': e6_rank,
                      'm1_margin': m1_gt - m1_bg, 'e6_amcm_margin': e6_gt - e6_bg,
                      'm1_top1_gt': m1_top, 'e6_amcm_top1_gt': e6_top,
                      'm1_hard_bg_xy': '{},{}'.format(bg_x, bg_y),
                      'm1_gt_peak_score': m1_gt, 'm1_bg_peak_score': m1_bg,
                      'e6_gt_peak_score': e6_gt, 'e6_bg_peak_score': e6_bg,
                      **residuals, **feature_metrics})
    with (OUT / 'paired_per_sequence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    summary = {'events': len(pairs), 'output': str(OUT),
               'mean_m1_fixed_iou': float(np.mean([p['m1_iou'] for p in pairs])),
               'mean_e6_amcm_fixed_iou': float(np.mean([p['e6_amcm_iou'] for p in pairs])),
               'mean_delta_iou': float(np.mean([p['delta_iou'] for p in pairs])),
               'iou_positive_events': sum(p['delta_iou'] > 0 for p in pairs),
               'iou_negative_events': sum(p['delta_iou'] < 0 for p in pairs),
               'gt_rank_improved': sum(p['e6_amcm_gt_rank'] < p['m1_gt_rank'] for p in pairs),
               'top1_changed_to_gt': sum(not p['m1_top1_gt'] and p['e6_amcm_top1_gt'] for p in pairs),
               'top1_changed_away_from_gt': sum(p['m1_top1_gt'] and not p['e6_amcm_top1_gt'] for p in pairs),
               'claim_scope': 'fixed-state, ten selected M1 full-244 test events; not sequence-level closed-loop performance',
               'score_protocol': 'CENTER maps multiplied by the tracker Hann window; hard-background feature cell fixed from M1 strongest non-GT response'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
