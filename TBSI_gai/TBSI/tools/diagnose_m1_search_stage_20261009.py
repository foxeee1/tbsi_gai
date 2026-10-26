#!/usr/bin/env python3
"""Analyze frozen M1 train traces at search fusion, head, and decoder interfaces."""
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import clip_box

CONFIG_NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
CONFIG = ROOT / 'experiments/tbsi_track' / (CONFIG_NAME + '.yaml')
CHECKPOINT = ROOT / 'output/experiments' / CONFIG_NAME / 'checkpoints/TBSITrack_ep0015.pth.tar'
EXPECTED_CONFIG = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
EXPECTED_CHECKPOINT = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_pair(sequence, index):
    rgb = cv2.imread(sequence.frames[0][index], cv2.IMREAD_COLOR)
    tir = cv2.imread(sequence.frames[1][index], cv2.IMREAD_COLOR)
    if rgb is None or tir is None or rgb.shape[:2] != tir.shape[:2]:
        raise RuntimeError('missing/misaligned pair {} {}'.format(sequence.name, index))
    return np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)), axis=2)


def iou(a, b):
    iw = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


def coverage(gt, crop):
    if gt[2] <= 0 or gt[3] <= 0:
        return 0.
    iw = max(0., min(gt[0] + gt[2], crop[0] + crop[2]) - max(gt[0], crop[0]))
    ih = max(0., min(gt[1] + gt[3], crop[1] + crop[3]) - max(gt[1], crop[1]))
    return iw * ih / (gt[2] * gt[3])


def select_frames(trace, reference_count, selection_rule='sustained-30-iou-0.2'):
    gt_valid, inside = [], []
    for row in trace:
        gt = np.asarray(row['gt_box'], dtype=np.float64)
        valid = bool(row.get('gt_valid', True)) and np.isfinite(gt).all() and gt[2] > 0 and gt[3] > 0
        gt_valid.append(valid)
        inside.append(valid and coverage(gt, row['current_crop_xywh']) >= .999)
    severe = [gt_valid[index] and iou(row['pred_box'], row['gt_box']) < .2
              for index, row in enumerate(trace)]
    if selection_rule == 'onset-3-iou-0.5':
        onset = [gt_valid[index] and inside[index] and
                 all(inside[index:index + 3]) and
                 all(iou(trace[j]['pred_box'], trace[j]['gt_box']) < .5
                     for j in range(index, index + 3))
                 for index in range(max(0, len(trace) - 2))]
        start = next((i for i, found in enumerate(onset) if found), None)
    elif selection_rule == 'sustained-30-iou-0.2':
        start = next((i for i in range(len(trace) - 29)
                      if inside[i] and severe[i] and all(severe[i:i + 30])), None)
    else:
        raise ValueError('unknown event selection rule: ' + selection_rule)
    if start is None:
        good = [i for i, row in enumerate(trace) if inside[i] and
                iou(row['pred_box'], row['gt_box']) >= .5]
        result = [(good[len(good) // 2], 'good_only')] if good else []
    else:
        good = [i for i in range(max(0, start - 150), max(0, start - 4))
                if inside[i] and iou(trace[i]['pred_box'], trace[i]['gt_box']) >= .5]
        result = [(good[-1], 'matched_good')] if good else []
        if start > 0 and inside[start - 1]:
            result.append((start - 1, 'pre_onset'))
        result.append((start, 'severe_onset'))
    available = [i for i, flag in enumerate(inside) if flag and
                 i not in {selected for selected, _label in result}]
    for part in range(reference_count):
        if available:
            index = min(len(available) - 1,
                        int((part + 1) * len(available) / (reference_count + 1)))
            result.append((available[index], 'probe_reference'))
    return result


def target_masks(gt_crop):
    cx = float((gt_crop[0] + .5 * gt_crop[2]) * 16)
    cy = float((gt_crop[1] + .5 * gt_crop[3]) * 16)
    gx = min(15, max(0, int(cx)))
    gy = min(15, max(0, int(cy)))
    yy, xx = np.mgrid[:16, :16]
    radius = max(1., min(3., 4 * math.sqrt(max(0., float(gt_crop[2] * gt_crop[3])))))
    distance = (xx + .5 - cx) ** 2 + (yy + .5 - cy) ** 2
    target = distance <= radius ** 2
    target[gy, gx] = True
    background = distance > (radius + 1.) ** 2
    return target, background, gx, gy, cx, cy


def response_metrics(score_map, window, size_map, offset_map, masks, gt_crop):
    target, background, gx, gy, cx, cy = masks
    result = {}
    for name, score in (('raw', score_map[0, 0]),
                        ('window', (score_map * window)[0, 0])):
        score = score.detach().float()
        flat = score.reshape(-1)
        idx = int(flat.argmax().item())
        px, py = idx % 16, idx // 16
        prob = (flat / flat.sum().clamp_min(1e-12)).clamp_min(1e-12)
        result[name + '_gt_score'] = float(score[gy, gx].item())
        result[name + '_target_bg_margin'] = float(
            score[torch.as_tensor(target, device=score.device)].max().item() -
            score[torch.as_tensor(background, device=score.device)].max().item())
        result[name + '_gt_rank'] = int((flat > score[gy, gx]).sum().item() + 1)
        result[name + '_peak_x'] = px
        result[name + '_peak_y'] = py
        result[name + '_peak_distance_tokens'] = math.hypot(px + .5 - cx, py + .5 - cy)
        result[name + '_entropy_norm'] = float(
            (-(prob * prob.log()).sum() / math.log(flat.numel())).item())
        offset = offset_map.detach().float()[0, :, py, px]
        size = size_map.detach().float()[0, :, py, px]
        result[name + '_decoded_center_distance_tokens'] = math.hypot(
            px + float(offset[0]) - cx, py + float(offset[1]) - cy)
        result[name + '_size_l1_tokens'] = (
            abs(float(size[0]) * 16 - float(gt_crop[2]) * 16) +
            abs(float(size[1]) * 16 - float(gt_crop[3]) * 16)) / 2
    gt_offset = offset_map.detach().float()[0, :, gy, gx]
    gt_size = size_map.detach().float()[0, :, gy, gx]
    result['gt_cell_decoded_center_distance_tokens'] = math.hypot(
        gx + float(gt_offset[0]) - cx, gy + float(gt_offset[1]) - cy)
    result['gt_cell_size_l1_tokens'] = (
        abs(float(gt_size[0]) * 16 - float(gt_crop[2]) * 16) +
        abs(float(gt_size[1]) * 16 - float(gt_crop[3]) * 16)) / 2
    result['hann_changed_peak'] = int(
        (result['raw_peak_x'], result['raw_peak_y']) !=
        (result['window_peak_x'], result['window_peak_y']))
    return result


def predicted_box(network, tracker, output, state, image_shape, resize, window=None):
    response = (tracker.output_window if window is None else window) * output['score_map']
    box = network.box_head.cal_bbox(response, output['size_map'],
                                    output['offset_map']).view(-1, 4).mean(dim=0)
    box = box * (tracker.params.search_size / resize)
    cx = state[0] + .5 * state[2]
    cy = state[1] + .5 * state[3]
    half = .5 * tracker.params.search_size / resize
    xywh = [float(box[0].item() + cx - half - .5 * box[2].item()),
            float(box[1].item() + cy - half - .5 * box[3].item()),
            float(box[2].item()), float(box[3].item())]
    return clip_box(xywh, image_shape[0], image_shape[1], margin=10)


def blur_local(search, modality, gt_crop, control, strength):
    result = [search[0].clone(), search[1].clone()]
    value = result[modality]
    size = value.shape[-1]
    cx = float((gt_crop[0] + .5 * gt_crop[2]) * size)
    cy = float((gt_crop[1] + .5 * gt_crop[3]) * size)
    width = min(64, max(24, int(gt_crop[2] * size + 12)))
    height = min(64, max(24, int(gt_crop[3] * size + 12)))
    if control:
        cx = size - 24 if cx < size / 2 else 24
        cy = size - 24 if cy < size / 2 else 24
    x1 = max(0, min(size - width, int(cx - width / 2)))
    y1 = max(0, min(size - height, int(cy - height / 2)))
    blurred = F.avg_pool2d(value, kernel_size=9, stride=1, padding=4)
    value[:, :, y1:y1 + height, x1:x1 + width] = (
        (1. - strength) * value[:, :, y1:y1 + height, x1:x1 + width] +
        strength * blurred[:, :, y1:y1 + height, x1:x1 + width])
    return result


def attenuate_feature(fusion_input, modality, masks, control, strength):
    """Replace selected S1 cells toward the same-modality background mean."""
    value = fusion_input.clone()
    channels = value.shape[1] // 2
    branch = value[:, modality * channels:(modality + 1) * channels]
    target = masks[0].reshape(-1)
    background = masks[1].reshape(-1)
    if control:
        indices = np.flatnonzero(background)[:int(target.sum())]
    else:
        indices = np.flatnonzero(target)
    selected = torch.as_tensor(indices, device=value.device, dtype=torch.long)
    background_index = torch.as_tensor(np.flatnonzero(background),
                                       device=value.device, dtype=torch.long)
    flat = branch.flatten(2)
    background_mean = flat.index_select(2, background_index).mean(2, keepdim=True)
    previous = flat.index_select(2, selected)
    flat[:, :, selected] = (1. - strength) * previous + strength * background_mean
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--limit-sequences', type=int, default=0)
    parser.add_argument('--no-perturb', action='store_true')
    parser.add_argument('--probe-reference-frames', type=int, default=2)
    parser.add_argument('--selection-rule', choices=('sustained-30-iou-0.2', 'onset-3-iou-0.5'),
                        default='sustained-30-iou-0.2')
    parser.add_argument('--diagnostic-id', default='M1-SEARCH-STAGE-TRAIN-DIAG-20261009')
    parser.add_argument('--capture-template-tokens', action='store_true',
                        help='also save final template tokens for target-conditioned signal diagnostics')
    args = parser.parse_args()
    trace_dir, output_dir = Path(args.trace_dir).resolve(), Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError('refusing nonempty output: ' + str(output_dir))
    if sha256(CONFIG) != EXPECTED_CONFIG or sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    names = (trace_dir / 'trace_sequences.txt').read_text().splitlines()
    if args.limit_sequences:
        names = names[:args.limit_sequences]
    if not names or len(names) != len(set(names)) or not torch.cuda.is_available():
        raise RuntimeError('nonempty unique sequence list and CUDA required')
    output_dir.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    params = parameters(CONFIG_NAME)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'lasher_train')
    network = tracker.network.eval()
    sequences = LasHeRDataset('trainingset', str(trace_dir / 'trace_sequences.txt')).get_sequence_list()
    seq_map = {seq.name: seq for seq in sequences}
    fusion = {}

    def capture(_module, inputs, output):
        fusion['input'] = inputs[0].detach()
        fusion['output'] = output.detach()

    handle = network.tbsi_fuse_search.register_forward_hook(capture)
    rows, feature_rows = [], []
    started = time.time()
    for name in names:
        sequence = seq_map[name]
        trace = [json.loads(line) for line in
                 (trace_dir / name / 'frames.jsonl').read_text().splitlines() if line.strip()]
        chosen = select_frames(trace, args.probe_reference_frames, args.selection_rule)
        tracker.initialize(read_pair(sequence, 0), {
            'init_bbox': sequence.ground_truth_rect[0], 'sequence_name': name})
        for index, label in chosen:
            item = trace[index]
            frame = int(item['frame_id'])
            state = [float(v) for v in item['tracker_state_before']]
            gt = np.asarray(sequence.ground_truth_rect[frame], dtype=np.float64)
            image = read_pair(sequence, frame)
            crop, resize, mask = sample_target(image, state, params.search_factor,
                                                output_sz=params.search_size)
            search_nested = tracker.preprocessor.process(crop, mask, half=False)
            search = [search_nested.tensors[:, :3], search_nested.tensors[:, 3:]]
            gt_crop = transform_image_to_crop(
                torch.tensor(gt, dtype=torch.float32),
                torch.tensor(state, dtype=torch.float32), resize,
                torch.tensor([params.search_size, params.search_size]),
                normalize=True).numpy()
            masks = target_masks(gt_crop)

            def run(input_pair):
                with torch.no_grad():
                    out = network(template=[tracker.z_dict1.tensors[:, :3],
                                            tracker.z_dict1.tensors[:, 3:]],
                                  search=input_pair, ce_template_mask=tracker.box_mask_z,
                                  template_fg_masks=(tracker.template_fg_mask,
                                                     tracker.template_fg_mask))
                features = {key: value for key, value in fusion.items()}
                if args.capture_template_tokens:
                    backbone_tokens = out['backbone_feat']
                    if isinstance(backbone_tokens, (list, tuple)):
                        backbone_tokens = backbone_tokens[-1]
                    if backbone_tokens.ndim != 3 or backbone_tokens.shape[1] != 640:
                        raise RuntimeError('unexpected final backbone token shape: {}'.format(
                            tuple(backbone_tokens.shape)))
                    features['template_rgb'] = backbone_tokens[:, :64].detach()
                    features['template_tir'] = backbone_tokens[:, 320:384].detach()
                metrics = response_metrics(out['score_map'], tracker.output_window,
                                           out['size_map'], out['offset_map'], masks, gt_crop)
                pred = predicted_box(network, tracker, out, state, image.shape, resize)
                metrics['iou'] = iou(pred, gt)
                oracle_score = torch.zeros_like(out['score_map'])
                oracle_score[0, 0, masks[3], masks[2]] = 1.
                oracle_out = dict(out, score_map=oracle_score)
                oracle_box = predicted_box(network, tracker, oracle_out, state,
                                           image.shape, resize,
                                           window=torch.ones_like(tracker.output_window))
                metrics['oracle_gt_cell_iou'] = iou(oracle_box, gt)
                return features, metrics, pred

            base_features, base, pred = run(search)
            parity = float(np.max(np.abs(np.asarray(pred) - np.asarray(item['pred_box']))))
            if parity > .01:
                raise RuntimeError('trace parity failed {} {} delta={}'.format(name, frame, parity))
            common = {'sequence': name, 'frame': frame, 'label': label,
                      'trace_iou': iou(item['pred_box'], gt),
                      'crop_coverage': coverage(gt, item['current_crop_xywh']),
                      'parity_max_px': parity}
            rows.append(dict(common, arm='baseline', **base))
            inp, fused = base_features['input'][0], base_features['output'][0]
            channels = inp.shape[0] // 2
            feature_rows.append({
                'sequence': name, 'frame': frame, 'label': label,
                'target': masks[0], 'background': masks[1],
                'gt_x': masks[4], 'gt_y': masks[5],
                'rgb': inp[:channels].permute(1, 2, 0).cpu().half().numpy(),
                'tir': inp[channels:].permute(1, 2, 0).cpu().half().numpy(),
                'concat': inp.permute(1, 2, 0).cpu().half().numpy(),
                'fused': fused.permute(1, 2, 0).cpu().half().numpy(),
                'template_rgb': (base_features['template_rgb'][0].cpu().half().numpy()
                                 if args.capture_template_tokens else None),
                'template_tir': (base_features['template_tir'][0].cpu().half().numpy()
                                 if args.capture_template_tokens else None),
                'template_fg': (tracker.template_fg_mask[0].detach().float().cpu().numpy()
                                if args.capture_template_tokens else None)})
            if not args.no_perturb and label != 'probe_reference':
                for modality, mod_name in enumerate(('rgb', 'tir')):
                    for strength in (.5, .9):
                        for control in (False, True):
                            variant_features, metrics, _ = run(
                                blur_local(search, modality, gt_crop, control, strength))
                            ratio = float((variant_features['output'] -
                                           base_features['output']).float().norm().item() /
                                          base_features['output'].float().norm().clamp_min(1e-8).item())
                            arm = '{}_{}_{:02d}'.format(
                                mod_name, 'background_control' if control else 'target_blur',
                                int(strength * 100))
                            rows.append(dict(common, arm=arm, fused_delta_ratio=ratio, **metrics))
                    for strength in (.25, .5):
                        for control in (False, True):
                            modified = attenuate_feature(base_features['input'], modality,
                                                         masks, control, strength)
                            with torch.no_grad():
                                fused_variant = network.tbsi_fuse_search(modified)
                                score, _box, size, offset = network.box_head(fused_variant)
                            out = {'score_map': score, 'size_map': size, 'offset_map': offset}
                            metrics = response_metrics(score, tracker.output_window,
                                                       size, offset, masks, gt_crop)
                            metrics['iou'] = iou(predicted_box(
                                network, tracker, out, state, image.shape, resize), gt)
                            oracle_score = torch.zeros_like(score)
                            oracle_score[0, 0, masks[3], masks[2]] = 1.
                            oracle_out = dict(out, score_map=oracle_score)
                            metrics['oracle_gt_cell_iou'] = iou(predicted_box(
                                network, tracker, oracle_out, state, image.shape, resize,
                                window=torch.ones_like(tracker.output_window)), gt)
                            ratio = float((fused_variant - base_features['output']).float().norm().item() /
                                          base_features['output'].float().norm().clamp_min(1e-8).item())
                            arm = '{}_feature_{}_{:02d}'.format(
                                mod_name, 'background_control' if control else 'target_atten',
                                int(strength * 100))
                            rows.append(dict(common, arm=arm, fused_delta_ratio=ratio, **metrics))
        print('diagnosed {} selected={} elapsed_s={:.1f}'.format(
            name, len(chosen), time.time() - started), flush=True)
    handle.remove()
    if not rows:
        raise RuntimeError('no eligible frames')
    fields = sorted({key for row in rows for key in row})
    with (output_dir / 'per_arm.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # Keep only selected-frame feature grids for sequence-separated probe fitting.
    feature_payload = {
        'rgb': np.asarray([r['rgb'] for r in feature_rows]),
        'tir': np.asarray([r['tir'] for r in feature_rows]),
        'concat': np.asarray([r['concat'] for r in feature_rows]),
        'fused': np.asarray([r['fused'] for r in feature_rows]),
        'target': np.asarray([r['target'] for r in feature_rows]),
        'background': np.asarray([r['background'] for r in feature_rows]),
        'gt_xy': np.asarray([[r['gt_x'], r['gt_y']] for r in feature_rows]),
        'sequence': np.asarray([r['sequence'] for r in feature_rows]),
        'frame': np.asarray([r['frame'] for r in feature_rows]),
        'label': np.asarray([r['label'] for r in feature_rows])}
    if args.capture_template_tokens:
        feature_payload.update(
            template_rgb=np.asarray([r['template_rgb'] for r in feature_rows]),
            template_tir=np.asarray([r['template_tir'] for r in feature_rows]),
            template_fg=np.asarray([r['template_fg'] for r in feature_rows]))
    np.savez_compressed(output_dir / 'selected_features.npz', **feature_payload)
    summary = {
        'id': args.diagnostic_id, 'status': 'STAGE-CAPTURE-COMPLETE',
        'event_selection_rule': args.selection_rule,
        'trace_dir': str(trace_dir),
        'trace_sequence_list_sha256': sha256(trace_dir / 'trace_sequences.txt'),
        'config_sha256': EXPECTED_CONFIG, 'checkpoint_sha256': EXPECTED_CHECKPOINT,
        'script_sha256': sha256(Path(__file__)), 'selected_sequences': len(names),
        'captured_template_tokens': bool(args.capture_template_tokens),
        'selected_frames': len(feature_rows),
        'probe_reference_frames': sum(r['label'] == 'probe_reference' for r in feature_rows),
        'severe_onset_frames': sum(r['label'] == 'severe_onset' for r in feature_rows),
        'max_parity_px': max(row['parity_max_px'] for row in rows),
        'elapsed_seconds': time.time() - started, 'gpu': torch.cuda.get_device_name(0),
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'limitations': ['Current working-tree replay, not archived M1 frame parity.',
                       'GT-guided blur is an offline intervention, not an inference method.',
                       'No human visibility audit; crop-inside may include occlusion.']}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
