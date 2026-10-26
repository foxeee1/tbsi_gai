#!/usr/bin/env python3
"""Extract runtime-visible signals and fixed-state action utility for M1."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box

NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
CONFIG = ROOT / 'experiments/tbsi_track' / (NAME + '.yaml')
CHECKPOINT = ROOT / 'output/experiments' / NAME / 'checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG_SHA = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
CHECKPOINT_SHA = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def read_pair(sequence, index):
    rgb = cv2.imread(sequence.frames[0][index], cv2.IMREAD_COLOR)
    tir = cv2.imread(sequence.frames[1][index], cv2.IMREAD_COLOR)
    if rgb is None or tir is None or rgb.shape[:2] != tir.shape[:2]:
        raise RuntimeError('missing/misaligned modalities {} frame {}'.format(sequence.name, index))
    return np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)), axis=2)


def iou(a, b):
    iw = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


def decode(network, tracker, out, state, image_shape, resize):
    response = tracker.output_window * out['score_map']
    box = network.box_head.cal_bbox(response, out['size_map'], out['offset_map']).view(-1, 4).mean(0)
    box = box * (tracker.params.search_size / resize)
    cx, cy = state[0] + .5 * state[2], state[1] + .5 * state[3]
    half = .5 * tracker.params.search_size / resize
    xywh = [float(box[0] + cx - half - .5 * box[2]),
            float(box[1] + cy - half - .5 * box[3]), float(box[2]), float(box[3])]
    return clip_box(xywh, image_shape[0], image_shape[1], margin=10)


def soft_entropy(score, temperature):
    values = np.asarray(score, dtype=np.float64).reshape(-1) / temperature
    values -= values.max()
    prob = np.exp(values)
    prob /= max(float(prob.sum()), 1e-12)
    return float(-(prob * np.log(np.maximum(prob, 1e-12))).sum() / np.log(len(prob)))


def template_response(search, template, weights):
    s = np.asarray(search, dtype=np.float32).reshape(256, -1)
    t = np.asarray(template, dtype=np.float32)
    w = np.maximum(np.asarray(weights, dtype=np.float32).reshape(-1), 0.)
    if w.size != t.shape[0] or float(w.sum()) < 1e-8:
        w = np.ones(t.shape[0], dtype=np.float32)
    proto = (t * w[:, None]).sum(0) / max(float(w.sum()), 1e-8)
    proto /= max(float(np.linalg.norm(proto)), 1e-8)
    s /= np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-8)
    return (s @ proto).reshape(16, 16)


def response_summary(score, prefix):
    flat = np.asarray(score, dtype=np.float32).reshape(-1)
    order = np.argsort(flat)
    peak, second = int(order[-1]), int(order[-2])
    y, x = divmod(peak, 16)
    return {
        prefix + '_peak': float(flat[peak]),
        prefix + '_top_margin': float(flat[peak] - flat[second]),
        prefix + '_entropy': soft_entropy(flat, .08),
        prefix + '_peak_x': x,
        prefix + '_peak_y': y,
    }


def corr(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.
    return float(np.corrcoef(a, b)[0, 1])


def local_energy(feature, x, y):
    f = np.asarray(feature, dtype=np.float32)
    norm = np.linalg.norm(f, axis=-1)
    patch = norm[max(0, y-1):min(16, y+2), max(0, x-1):min(16, x+2)]
    return float(patch.mean() / max(float(norm.mean()), 1e-8))


def action_input(x, direction, rho):
    if direction == 0:
        return x
    y = x.clone()
    channels = y.shape[1] // 2
    if direction > 0:
        y[:, :channels] *= 1. + rho
        y[:, channels:] *= 1. - rho
    else:
        y[:, :channels] *= 1. - rho
        y[:, channels:] *= 1. + rho
    return y


def predict(network, tracker, x, state, image_shape, resize):
    with torch.no_grad():
        fused = network.tbsi_fuse_search(x)
        score, _bbox, size, offset = network.box_head(fused)
    out = {'score_map': score, 'size_map': size, 'offset_map': offset}
    box = decode(network, tracker, out, state, image_shape, resize)
    return box, (score * tracker.output_window)[0, 0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--center-csv', required=True)
    ap.add_argument('--trace-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=3407)
    args = ap.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing nonempty output directory: ' + str(output))
    if sha256(CONFIG) != CONFIG_SHA or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_PERTURBATION'] = ''
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    archive_path = Path(args.features).resolve()
    with np.load(str(archive_path), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    needed = {'rgb', 'tir', 'concat', 'target', 'background', 'gt_xy', 'sequence', 'frame',
              'label', 'template_rgb', 'template_tir', 'template_fg'}
    if not needed.issubset(data):
        raise RuntimeError('missing feature keys: {}'.format(sorted(needed - set(data))))
    center_rows = [r for r in read_csv(args.center_csv) if r['arm'] == 'baseline']
    center_map = {(r['sequence'], int(r['frame'])): r for r in center_rows}

    trace = {}
    trace_root = Path(args.trace_dir).resolve()
    for path in trace_root.glob('*/frames.jsonl'):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (path.parent.name, int(row['frame_id']))
            trace[key] = row
    names = sorted(set(str(v) for v in data['sequence']))
    list_path = output.parent / 'sequence_list.txt'
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text('\n'.join(names) + '\n')
    sequences = LasHeRDataset('trainingset', str(list_path)).get_sequence_list()
    seqmap = {sequence.name: sequence for sequence in sequences}

    params = parameters(NAME)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'm1_need_aware_action_sweep')
    network = tracker.network.eval()
    random_gen = random.Random(args.seed)
    rho_values = (.05, .10, .15)
    action_defs = [('rgb', rho, 1) for rho in rho_values] + [('tir', rho, -1) for rho in rho_values]
    signal_rows, action_rows = [], []
    parity_max = 0.
    started = time.time()

    for i, (seq_value, frame_value) in enumerate(zip(data['sequence'], data['frame'])):
        seq, frame = str(seq_value), int(frame_value)
        key = (seq, frame)
        if key not in trace or key not in center_map or seq not in seqmap:
            raise RuntimeError('unmatched saved feature, trace or CENTER row: {}'.format(key))
        tr = trace[key]
        state = np.asarray(tr['tracker_state_before'], dtype=np.float64)
        gt = np.asarray(tr['gt_box'], dtype=np.float64)
        image = read_pair(seqmap[seq], frame)
        crop, resize, _mask = sample_target(image, state, params.search_factor,
                                             output_sz=params.search_size)
        x = torch.as_tensor(data['concat'][i].transpose(2, 0, 1)[None],
                            dtype=torch.float32, device='cuda')
        rgb_map = template_response(data['rgb'][i], data['template_rgb'][i], data['template_fg'][i])
        tir_map = template_response(data['tir'][i], data['template_tir'][i], data['template_fg'][i])
        row = {'sequence': seq, 'frame': frame, 'label': str(data['label'][i])}
        row.update(response_summary(rgb_map, 'rgb_support'))
        row.update(response_summary(tir_map, 'tir_support'))
        row['support_top_margin_gap_rgb_minus_tir'] = row['rgb_support_top_margin'] - row['tir_support_top_margin']
        row['support_peak_gap_rgb_minus_tir'] = row['rgb_support_peak'] - row['tir_support_peak']
        row['support_entropy_gap_rgb_minus_tir'] = row['rgb_support_entropy'] - row['tir_support_entropy']
        row['support_peak_distance'] = float(math.hypot(
            row['rgb_support_peak_x'] - row['tir_support_peak_x'],
            row['rgb_support_peak_y'] - row['tir_support_peak_y']))
        row['support_map_correlation'] = corr(rgb_map, tir_map)
        center_row = center_map[key]
        cx, cy = int(float(center_row['window_peak_x'])), int(float(center_row['window_peak_y']))
        row['center_gt_rank_diagnostic_only'] = int(float(center_row['window_gt_rank']))
        row['rgb_local_energy_at_center_peak'] = local_energy(data['rgb'][i], cx, cy)
        row['tir_local_energy_at_center_peak'] = local_energy(data['tir'][i], cx, cy)
        row['local_energy_gap_rgb_minus_tir'] = row['rgb_local_energy_at_center_peak'] - row['tir_local_energy_at_center_peak']
        rgb_flat = crop[:, :, :3].astype(np.float32).mean(2)
        tir_flat = crop[:, :, 3:6].astype(np.float32).mean(2)
        row['rgb_crop_luma_mean'] = float(rgb_flat.mean())
        row['rgb_crop_luma_std'] = float(rgb_flat.std())
        row['tir_crop_luma_mean'] = float(tir_flat.mean())
        row['tir_crop_luma_std'] = float(tir_flat.std())
        row['crop_luma_mean_gap_rgb_minus_tir'] = row['rgb_crop_luma_mean'] - row['tir_crop_luma_mean']
        row['state_area_fraction'] = float(state[2] * state[3] / (image.shape[0] * image.shape[1]))
        row['state_aspect_ratio'] = float(state[2] / max(state[3], 1e-8))
        prior = trace.get((seq, frame - 1))
        if prior is not None:
            prev = np.asarray(prior['tracker_state_before'], dtype=np.float64)
            scale = max(math.sqrt(max(state[2] * state[3], 1.)), 1.)
            row['state_motion_norm'] = float(np.linalg.norm(
                (state[:2] + .5 * state[2:]) - (prev[:2] + .5 * prev[2:])) / scale)
            row['state_log_scale_change'] = float(math.log(max(state[2] * state[3], 1.) /
                                                           max(prev[2] * prev[3], 1.)))
        else:
            row['state_motion_norm'] = 0.
            row['state_log_scale_change'] = 0.

        base_box, base_score = predict(network, tracker, x, state, image.shape, resize)
        base_iou = iou(base_box, gt)
        with torch.no_grad():
            fused = network.tbsi_fuse_search(x)[0].permute(1, 2, 0).float().cpu().numpy()
        saved = data['fused'][i].astype(np.float32)
        parity_max = max(parity_max, float(np.max(np.abs(fused - saved))))
        row['center_top_margin'] = float(np.sort(base_score.reshape(-1))[-1] - np.sort(base_score.reshape(-1))[-2])
        row['center_entropy'] = soft_entropy(base_score, .08)
        signal_rows.append(row)
        action_rows.append({'sequence': seq, 'frame': frame, 'label': str(data['label'][i]),
                            'arm': 'no_op', 'rho': 0., 'direction': 0,
                            'iou': base_iou, 'delta_iou_vs_no_op': 0.,
                            'gt_rank': int(float(center_row['window_gt_rank']))})
        for modality, rho, direction in action_defs:
            box, score = predict(network, tracker, action_input(x, direction, rho),
                                 state, image.shape, resize)
            action_rows.append({'sequence': seq, 'frame': frame, 'label': str(data['label'][i]),
                                'arm': '{}_rho_{:.2f}'.format(modality, rho), 'rho': rho,
                                'direction': direction, 'iou': iou(box, gt),
                                'delta_iou_vs_no_op': iou(box, gt) - base_iou,
            'gt_rank': int((score.reshape(-1) > score[int(data['gt_xy'][i][1]),
                                                       int(data['gt_xy'][i][0])]).sum() + 1)})
        random_direction = 1 if random_gen.random() >= .5 else -1
        box, score = predict(network, tracker, action_input(x, random_direction, .10),
                             state, image.shape, resize)
        action_rows.append({'sequence': seq, 'frame': frame, 'label': str(data['label'][i]),
                            'arm': 'random_rho_0.10', 'rho': .10, 'direction': random_direction,
                            'iou': iou(box, gt), 'delta_iou_vs_no_op': iou(box, gt) - base_iou,
                            'gt_rank': int((score.reshape(-1) > score[int(data['gt_xy'][i][1]),
                                                             int(data['gt_xy'][i][0])]).sum() + 1)})
        if (i + 1) % 10 == 0:
            print('processed {}/{} states'.format(i + 1, len(data['sequence'])), flush=True)

    if len(set(row['sequence'] for row in signal_rows)) < 24 or len(signal_rows) < 30:
        raise RuntimeError('cohort did not meet preregistered minimum: {} states, {} sequences'.format(
            len(signal_rows), len(set(row['sequence'] for row in signal_rows))))
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('per_state_signals.csv', signal_rows), ('per_state_actions.csv', action_rows)):
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    meta = {
        'id': 'M1-NEED-AWARE-ACTION-SWEEP-20261012', 'parent': NAME,
        'config_sha256': sha256(CONFIG), 'checkpoint_sha256': sha256(CHECKPOINT),
        'features_sha256': sha256(archive_path),
        'center_csv_sha256': sha256(args.center_csv),
        'trace_dir': str(trace_root), 'trace_manifest_sha256': sha256(trace_root / 'manifest.json'),
        'seed': args.seed, 'n_states': len(signal_rows),
        'n_sequences': len(set(row['sequence'] for row in signal_rows)),
        'actions': [r['arm'] for r in action_rows[:len(action_defs) + 2]],
        'utility': 'decoded bbox IoU after original Hann peak and CENTER offset/size decode minus no-op on identical crop/state',
        'feature_replay_max_abs_delta_vs_saved_fp16': parity_max,
        'elapsed_seconds': time.time() - started,
        'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'limitations': ['fixed-state offline utility is not closed-loop AUC',
                        'M1 training exposure is unknown; cohort is fresh only relative to recorded prior signal diagnostics',
                        'signals are screening candidates, not calibrated probabilities or a validated controller']}
    (output / 'summary.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
