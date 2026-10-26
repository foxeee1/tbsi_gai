#!/usr/bin/env python3
"""Fixed-state AMCM residual interventions on a frozen train-side cohort."""
import csv
import copy
import argparse
import hashlib
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

M1_NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
ADAPTIVE_NAME = 'M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021'
M1_CKPT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
ADAPTIVE_CKPT = ROOT / 'output/experiments/M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021/checkpoints/TBSITrack_ep0003.pth.tar'
M1_CONFIG = ROOT / 'experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml'
ADAPTIVE_CONFIG = ROOT / 'experiments/tbsi_track/M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021.yaml'
TRACE_ROOT = ROOT / 'output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/trace_run/trace'
TRACE_MANIFEST = TRACE_ROOT / 'trace_sequences.txt'
OUT = ROOT / 'output/experiments/M1_AMCM_ADAPTIVE_FIRST_UPDATE_CAUSAL_DIAG_20261021_RETRY5'
ARMS = ('m1', 'adaptive', 'zero', 'reverse', 'shift')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_pair(sequence, index):
    rgb = cv2.imread(sequence.frames[0][index], cv2.IMREAD_COLOR)
    tir = cv2.imread(sequence.frames[1][index], cv2.IMREAD_COLOR)
    if rgb is None or tir is None or rgb.shape[:2] != tir.shape[:2]:
        raise RuntimeError('missing or misaligned frame {} {}'.format(sequence.name, index))
    return np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)), axis=2)


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def crop_contains(box, crop):
    x, y, w, h = box
    cx, cy, cw, ch = crop
    return x >= cx and y >= cy and x + w <= cx + cw and y + h <= cy + ch


def trace_rows(sequence):
    path = TRACE_ROOT / sequence / 'frames.jsonl'
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_events(names):
    """Select first crop-contained M1 low-IoU and stable states, outcome stratified."""
    failures, controls = [], []
    for name in names:
        rows = trace_rows(name)
        for row in rows:
            if int(row['frame_id']) < 2 or not row.get('gt_valid', False):
                continue
            if row['gt_box'][2] <= 0 or row['gt_box'][3] <= 0:
                continue
            value = iou(row['pred_box'], row['gt_box'])
            if not crop_contains(row['gt_box'], row['current_crop_xywh']):
                continue
            event = {
                'sequence': name, 'frame': int(row['frame_id']),
                'state_before': row['tracker_state_before'], 'gt_box': row['gt_box'],
                'crop_xywh': row['current_crop_xywh'], 'm1_trace_iou': value,
                'selection_source': 'pre-frozen train-side M1 trace',
            }
            if value < 0.30:
                failures.append(event)
                break
        for row in rows:
            if int(row['frame_id']) < 2 or not row.get('gt_valid', False):
                continue
            if row['gt_box'][2] <= 0 or row['gt_box'][3] <= 0:
                continue
            value = iou(row['pred_box'], row['gt_box'])
            if value > 0.80 and crop_contains(row['gt_box'], row['current_crop_xywh']):
                controls.append({
                    'sequence': name, 'frame': int(row['frame_id']),
                    'state_before': row['tracker_state_before'], 'gt_box': row['gt_box'],
                    'crop_xywh': row['current_crop_xywh'], 'm1_trace_iou': value,
                    'selection_source': 'pre-frozen train-side M1 trace',
                })
                break
    # Sequence order is fixed by the pre-existing manifest; avoid selecting the
    # same sequence twice so failure and stable-control cohorts remain distinct.
    failure_names = {event['sequence'] for event in failures[:6]}
    controls = [event for event in controls if event['sequence'] not in failure_names]
    selected = failures[:6] + controls[:6]
    if len(failures[:6]) < 6 or len(controls[:6]) < 6:
        raise RuntimeError('manifest yielded insufficient events: failures={}, controls={}'.format(
            len(failures[:6]), len(controls[:6])))
    return selected


def select_first_update_events(names):
    selected = []
    for name in names:
        rows = trace_rows(name)
        event = None
        for row in rows:
            if int(row['frame_id']) < 1 or not row.get('gt_valid', False):
                continue
            if row['gt_box'][2] <= 0 or row['gt_box'][3] <= 0:
                continue
            if not crop_contains(row['gt_box'], row['current_crop_xywh']):
                continue
            event = {
                'sequence': name, 'frame': int(row['frame_id']),
                'state_before': row['tracker_state_before'], 'gt_box': row['gt_box'],
                'crop_xywh': row['current_crop_xywh'],
                'm1_trace_iou': iou(row['pred_box'], row['gt_box']),
                'selection_source': 'first crop-contained tracker update from pre-frozen train-side M1 trace',
            }
            break
        if event is not None:
            selected.append(event)
    if len(selected) < 24:
        raise RuntimeError('only {} eligible first-update events from the frozen manifest'.format(len(selected)))
    return selected


def select_sampled_first_update_events(all_sequences, excluded_names, count):
    candidates = []
    excluded = set(excluded_names)
    for sequence in all_sequences:
        if sequence.name in excluded or len(sequence.frames[0]) < 2:
            continue
        init_box = [float(v) for v in sequence.ground_truth_rect[0]]
        gt_box = [float(v) for v in sequence.ground_truth_rect[1]]
        if min(init_box[2], init_box[3], gt_box[2], gt_box[3]) <= 0:
            continue
        cx, cy = init_box[0] + init_box[2] / 2.0, init_box[1] + init_box[3] / 2.0
        side = int(np.ceil(np.sqrt(init_box[2] * init_box[3]) * 4.0))
        crop = [float(int(round(cx - side / 2.0))),
                float(int(round(cy - side / 2.0))), float(side), float(side)]
        if not crop_contains(gt_box, crop):
            continue
        key = hashlib.sha256(('M1-AMCM-ADAPTIVE-FIRST50-20261021:' + sequence.name).encode('utf-8')).hexdigest()
        candidates.append((key, sequence.name, {
            'sequence': sequence.name, 'frame': 1, 'state_before': init_box,
            'gt_box': gt_box, 'crop_xywh': crop, 'm1_trace_iou': 1.0,
            'selection_source': 'SHA256 outcome-blind sample from LasHeR trainingset; first crop-contained update',
            'cohort': 'sampled_first_update',
        }))
    candidates.sort(key=lambda row: (row[0], row[1]))
    selected = candidates[:count]
    if len(selected) < count:
        raise RuntimeError('only {} eligible outcome-blind first-update sequences; requested {}'.format(
            len(selected), count))
    return [row[2] for row in selected], [row[1] for row in selected]


def feature_stats(feature, mask, bg_yx):
    z = feature.detach().float()[0].permute(1, 2, 0).cpu().numpy()
    target = z[mask].mean(0)
    background = z[bg_yx[0], bg_yx[1]]
    return {
        'target_bg_l2': float(np.linalg.norm(target - background)),
        'target_norm': float(np.linalg.norm(target)),
        'background_norm': float(np.linalg.norm(background)),
    }


def grid_mask(box, crop, grid=16):
    x, y, w, h = box
    cx, cy, cw, ch = crop
    x1, x2 = (x - cx) * grid / cw, (x + w - cx) * grid / cw
    y1, y2 = (y - cy) * grid / ch, (y + h - cy) * grid / ch
    yy, xx = np.mgrid[:grid, :grid]
    mask = ((xx + 0.5 >= x1) & (xx + 0.5 <= x2) &
            (yy + 0.5 >= y1) & (yy + 0.5 <= y2))
    if not mask.any():
        mask[int(np.clip((y1 + y2) / 2, 0, grid - 1)),
             int(np.clip((x1 + x2) / 2, 0, grid - 1))] = True
    return mask


def spatial_map(tensor):
    value = tensor.detach().float()
    while value.ndim > 2:
        value = value[0]
    return value.cpu().numpy()


def intervention_hook(arm):
    if arm == 'adaptive':
        return None

    def hook(module, inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) < 3:
            raise RuntimeError('unexpected AMCM output contract')
        out_rgb, out_tir, stats = output
        in_rgb, in_tir = inputs[:2]
        delta_rgb, delta_tir = out_rgb - in_rgb, out_tir - in_tir
        if arm == 'zero':
            delta_rgb, delta_tir = torch.zeros_like(delta_rgb), torch.zeros_like(delta_tir)
        elif arm == 'reverse':
            delta_rgb, delta_tir = -delta_rgb, -delta_tir
        elif arm == 'shift':
            delta_rgb = torch.roll(delta_rgb, shifts=(4, 4), dims=(-2, -1))
            delta_tir = torch.roll(delta_tir, shifts=(4, 4), dims=(-2, -1))
        else:
            raise ValueError('unsupported intervention {}'.format(arm))
        stats = dict(stats)
        stats['rgb_residual_live'] = delta_rgb
        stats['tir_residual_live'] = delta_tir
        stats['rgb_output_live'] = in_rgb + delta_rgb
        stats['tir_output_live'] = in_tir + delta_tir
        return in_rgb + delta_rgb, in_tir + delta_tir, stats
    return hook


def build_tracker(arm):
    if arm == 'm1':
        cfg_name, checkpoint = M1_NAME, M1_CKPT
    else:
        cfg_name, checkpoint = ADAPTIVE_NAME, ADAPTIVE_CKPT
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(checkpoint)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_INTERNAL_DIAGNOSTIC'] = '0'
    os.environ['TBSI_M1_TRACE_DIR'] = ''
    params = parameters(cfg_name)
    params.cfg = copy.deepcopy(params.cfg)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(checkpoint)
    if Path(params.checkpoint).resolve() != Path(checkpoint).resolve():
        raise RuntimeError('checkpoint override mismatch: {}'.format(params.checkpoint))
    tracker = TBSITrack(params, 'm1_adaptive_first_update_causal_diag')
    if arm != 'm1' and getattr(tracker.network, 'amcm_reliability', None) is None:
        raise RuntimeError('adaptive checkpoint has no AMCM module')
    return tracker


def rank_and_margin(score, mask):
    response = score.detach().float().cpu().numpy()
    if response.ndim == 3:
        response = response[0]
    if response.ndim == 3:
        response = response[0]
    response = response.reshape(16, 16)
    target_peak = float(response[mask].max())
    bg = np.where(~mask, response, -np.inf)
    bg_flat = int(np.argmax(bg))
    top = int(np.argmax(response))
    rank = 1 + int(np.sum(bg > target_peak))
    order = np.sort(response.reshape(-1))
    return {
        'gt_peak': target_peak,
        'hard_bg_peak': float(response.reshape(-1)[bg_flat]),
        'gt_hard_bg_margin': target_peak - float(response.reshape(-1)[bg_flat]),
        'gt_rank': rank,
        'top1_y': top // 16,
        'top1_x': top % 16,
        'gt_cell_is_top1': bool(mask.reshape(-1)[top]),
        'top1_top2_margin': float(order[-1] - order[-2]),
    }, (bg_flat // 16, bg_flat % 16)


def run_event(arm, event, sequence, tracker):
    handle = None
    if arm in ('zero', 'reverse', 'shift'):
        handle = tracker.network.amcm_reliability.register_forward_hook(
            intervention_hook(arm))
    captured = {}
    hook_handles = []
    def fusion_pre(module, inputs):
        captured['fusion_input'] = inputs[0].detach().float()
    def fusion_post(module, inputs, output):
        captured['fusion_output'] = output.detach().float()
    def head_pre(module, inputs):
        captured['head_input'] = inputs[0].detach().float()
    hook_handles.extend([
        tracker.network.tbsi_fuse_search.register_forward_pre_hook(fusion_pre),
        tracker.network.tbsi_fuse_search.register_forward_hook(fusion_post),
        tracker.network.box_head.register_forward_pre_hook(head_pre),
    ])
    init = read_pair(sequence, 0)
    tracker.initialize(init, {'init_bbox': np.asarray(sequence.ground_truth_rect[0]),
                              'sequence_name': sequence.name})
    tracker.state = [float(value) for value in event['state_before']]
    tracker.prev_search = None
    tracker.prev_temporal_tokens = None
    tracker.prev_head_feature = None
    image = read_pair(sequence, int(event['frame']))
    pred = tracker.track(image, {'gt_bbox': event['gt_box'], 'sequence_name': sequence.name,
                                 'frame_num': int(event['frame'])})
    diag = tracker._last_fixed_state_diag
    mask = grid_mask(event['gt_box'], event['crop_xywh'])
    response = diag['score_map'][0, 0] * tracker.output_window[0, 0].detach().float().cpu()
    center, bg_yx = rank_and_margin(response, mask)
    stats = getattr(tracker.network, '_last_amcm_stats', None) or {}
    row = {
        'cohort': event.get('cohort',
                            'failure' if event['m1_trace_iou'] < 0.30 else 'stable_control'),
        'sequence': sequence.name, 'frame': int(event['frame']), 'arm': arm,
        'm1_trace_iou': event['m1_trace_iou'],
        'fixed_state_iou': iou(pred['target_bbox'], event['gt_box']),
        'predicted_bbox': json.dumps([float(v) for v in pred['target_bbox']]),
        'gt_box': json.dumps(event['gt_box']), 'state_before': json.dumps(event['state_before']),
        'crop_xywh': json.dumps(event['crop_xywh']), **center,
        'bg_y': bg_yx[0], 'bg_x': bg_yx[1],
        'amcm_residual_rgb_norm': float(stats['rgb_residual_live'].detach().float().norm().cpu())
            if stats.get('rgb_residual_live') is not None else 0.0,
        'amcm_residual_tir_norm': float(stats['tir_residual_live'].detach().float().norm().cpu())
            if stats.get('tir_residual_live') is not None else 0.0,
        'amcm_input_rgb_norm': float(stats['rgb_input_live'].detach().float().norm().cpu())
            if stats.get('rgb_input_live') is not None else 0.0,
        'amcm_input_tir_norm': float(stats['tir_input_live'].detach().float().norm().cpu())
            if stats.get('tir_input_live') is not None else 0.0,
        'amcm_rgb_sim_gt_bg_delta': float(spatial_map(stats['rgb_sim_map_live'])[mask].mean() -
            spatial_map(stats['rgb_sim_map_live'])[~mask].mean())
            if stats.get('rgb_sim_map_live') is not None else float('nan'),
        'amcm_tir_sim_gt_bg_delta': float(spatial_map(stats['tir_sim_map_live'])[mask].mean() -
            spatial_map(stats['tir_sim_map_live'])[~mask].mean())
            if stats.get('tir_sim_map_live') is not None else float('nan'),
    }
    for key, tensor in (('fusion_input', captured.get('fusion_input')),
                        ('fusion_output', captured.get('fusion_output')),
                        ('head_input', captured.get('head_input'))):
        if tensor is None:
            continue
        if key == 'fusion_input':
            channels = tensor.shape[1] // 2
            for modality, part in (('rgb', tensor[:, :channels]),
                                   ('tir', tensor[:, channels:])):
                for stat, value in feature_stats(part, mask, bg_yx).items():
                    row['{}_{}_{}'.format(key, modality, stat)] = value
        else:
            for stat, value in feature_stats(tensor, mask, bg_yx).items():
                row['{}_{}'.format(key, stat)] = value
    size = diag['size_map'][0].numpy()
    offset = diag['offset_map'][0].numpy()
    px, py = center['top1_x'], center['top1_y']
    row.update({
        'top1_size_w': float(size[0, py, px]),
        'top1_size_h': float(size[1, py, px]),
        'top1_offset_x': float(offset[0, py, px]),
        'top1_offset_y': float(offset[1, py, px]),
        'decoded_width_px': float(json.loads(row['predicted_bbox'])[2]),
        'decoded_height_px': float(json.loads(row['predicted_bbox'])[3]),
    })
    if handle is not None:
        handle.remove()
    for hook in hook_handles:
        hook.remove()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('stratified', 'first_update_all', 'sampled_first_update'), default='stratified')
    parser.add_argument('--output', default=str(OUT))
    parser.add_argument('--sample-count', type=int, default=50)
    args = parser.parse_args()
    output_root = Path(args.output).resolve()
    if sha256(M1_CONFIG) != 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e':
        raise RuntimeError('protected M1 config hash mismatch')
    if sha256(M1_CKPT) != '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef':
        raise RuntimeError('protected M1 checkpoint hash mismatch')
    if not TRACE_MANIFEST.is_file():
        raise FileNotFoundError(str(TRACE_MANIFEST))
    names = [line.strip() for line in TRACE_MANIFEST.read_text().splitlines() if line.strip()]
    if len(names) != 32 or len(set(names)) != 32:
        raise RuntimeError('expected the pre-frozen 32-sequence train-side manifest')
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError('refusing to reuse non-empty output directory: {}'.format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    if args.mode == 'stratified':
        selected = select_events(names)
        selected_names = sorted({event['sequence'] for event in selected})
    elif args.mode == 'first_update_all':
        selected = select_first_update_events(names)
        selected_names = sorted({event['sequence'] for event in selected})
    else:
        all_sequences = LasHeRDataset('trainingset').get_sequence_list()
        selected, selected_names = select_sampled_first_update_events(
            all_sequences, names, args.sample_count)
        manifest = {
            'id': 'M1-AMCM-ADAPTIVE-INITIAL-UPDATE-SAMPLED-20261021',
            'split': 'LasHeR trainingset', 'selection': 'SHA256 deterministic order, outcome-blind; excludes prior frozen 32-sequence manifest; positive GT and crop-contained first update required',
            'selection_seed_string': 'M1-AMCM-ADAPTIVE-FIRST50-20261021',
            'excluded_manifest_sha256': sha256(TRACE_MANIFEST),
            'sequence_count': len(selected_names), 'sequences': selected_names,
        }
        (output_root / 'sample_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output_root / 'selected_events.json').write_text(json.dumps(selected, indent=2) + '\n')
    seq_file = output_root / 'sequence_list.txt'
    seq_file.write_text('\n'.join(selected_names) + '\n')
    seqs = LasHeRDataset('trainingset', str(seq_file)).get_sequence_list()
    seqmap = {sequence.name: sequence for sequence in seqs}
    trackers = {arm: build_tracker(arm) for arm in ('m1', 'adaptive')}
    rows = []
    for event_idx, event in enumerate(selected, 1):
        event_rows = []
        for arm in ARMS:
            tracker = trackers['m1'] if arm == 'm1' else trackers['adaptive']
            result = run_event(arm, event, seqmap[event['sequence']], tracker)
            result['event_index'] = event_idx
            event_rows.append(result)
            print('[event {}/{}] {} frame={} arm={} IoU={:.3f} rank={} margin={:+.4f}'.format(
                event_idx, len(selected), event['sequence'], event['frame'], arm,
                result['fixed_state_iou'], result['gt_rank'], result['gt_hard_bg_margin']), flush=True)
        if args.mode == 'sampled_first_update':
            m1_iou = next(row['fixed_state_iou'] for row in event_rows if row['arm'] == 'm1')
            for row in event_rows:
                row['m1_trace_iou'] = m1_iou
                row['cohort'] = 'm1_first_update_iou_lt_0p3' if m1_iou < 0.30 else 'm1_first_update_iou_ge_0p3'
        rows.extend(event_rows)
    with (output_root / 'per_arm.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for cohort in sorted({row['cohort'] for row in rows}):
        for arm in ARMS:
            subset = [row for row in rows if row['cohort'] == cohort and row['arm'] == arm]
            if not subset:
                continue
            summaries.append({
                'cohort': cohort, 'arm': arm, 'n': len(subset),
                'mean_iou': float(np.mean([row['fixed_state_iou'] for row in subset])),
                'gt_rank_one': int(sum(row['gt_rank'] == 1 for row in subset)),
                'mean_gt_rank': float(np.mean([row['gt_rank'] for row in subset])),
                'mean_gt_bg_margin': float(np.mean([row['gt_hard_bg_margin'] for row in subset])),
                'mean_residual_rgb_norm': float(np.mean([row['amcm_residual_rgb_norm'] for row in subset])),
                'mean_residual_tir_norm': float(np.mean([row['amcm_residual_tir_norm'] for row in subset])),
            })
    summary = {
        'id': ('M1-AMCM-ADAPTIVE-FIRST-UPDATE-CAUSAL-DIAG-20261021' if args.mode == 'stratified'
               else ('M1-AMCM-ADAPTIVE-INITIAL-UPDATE-32SEQ-20261021' if args.mode == 'first_update_all'
                     else 'M1-AMCM-ADAPTIVE-INITIAL-UPDATE-SAMPLED-20261021')),
        'status': 'COMPLETE', 'parent': 'M1_AMCM_ADAPTIVE_NOSUP_QUICK15_3EP_20261021 vs protected M1',
        'question': 'Does the trained adaptive AMCM residual cause the first fixed-state CENTER/readout failure?',
        'arms': list(ARMS), 'sequence_count': len(selected_names), 'event_count': len(selected),
        'mode': args.mode,
        'summary_by_cohort_arm': summaries,
        'trace_manifest_sha256': sha256(TRACE_MANIFEST),
        'adaptive_config_sha256': sha256(ADAPTIVE_CONFIG),
        'adaptive_checkpoint_sha256': sha256(ADAPTIVE_CKPT),
        'm1_config_sha256': sha256(M1_CONFIG), 'm1_checkpoint_sha256': sha256(M1_CKPT),
        'git_head': os.popen('git -C "{}" rev-parse HEAD'.format(ROOT)).read().strip(),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'limitations': [
            'Train-side diagnostic states come from an existing fixed 32-sequence manifest; M1 checkpoint training exposure is unknown.',
            'Outcome-stratified states are mechanism cases, not a population estimate.',
            'M1 and candidate training progress differ; this is an intervention on the candidate residual, not a matched-progress method comparison.',
            'Full internal stage traces are limited to output response, decoded box and AMCM maps/norms; exact fused-feature margin hook is not captured in this pass.',
        ],
    }
    (output_root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
