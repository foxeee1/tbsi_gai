#!/usr/bin/env python3
"""Sequence-held-out, fixed-state screen of three pre-fusion signals."""
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

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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
FEATURES = ROOT / 'output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/selected_features.npz'
SEQUENCES = ROOT / 'experiments/diagnostics/m1_fusion_utility_signal_pilot_20261012/expansion72/sequence_list.txt'
TRACE_ROOT = ROOT / 'output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009'
CONFIG_SHA = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
CHECKPOINT_SHA = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def iou(a, b):
    if min(a[2], a[3], b[2], b[3]) <= 0:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return overlap / max(1e-12, a[2] * a[3] + b[2] * b[3] - overlap)


def load_trace(keys):
    wanted = {}
    for sequence, frame in keys:
        wanted.setdefault(sequence, set()).add(frame)
    roots = [TRACE_ROOT / name / 'trace' for name in
             ('trace_run', 'trace_run_rank24', 'trace_run_rank48')]
    result = {}
    for sequence, frames in wanted.items():
        paths = [root / sequence / 'frames.jsonl' for root in roots]
        paths = [path for path in paths if path.is_file()]
        if len(paths) != 1:
            raise RuntimeError('trace ownership ambiguous: {} {}'.format(sequence, paths))
        with paths[0].open() as stream:
            for line in stream:
                row = json.loads(line)
                if int(row['frame_id']) in frames:
                    result[(sequence, int(row['frame_id']))] = row
    if set(result) != set(keys):
        raise RuntimeError('missing trace rows: ' + repr(sorted(set(keys) - set(result))[:10]))
    return result


def image_features(sequence, frame, state, dataset_sequence):
    rgb = cv2.imread(dataset_sequence.frames[0][frame], cv2.IMREAD_COLOR)
    tir = cv2.imread(dataset_sequence.frames[1][frame], cv2.IMREAD_COLOR)
    if rgb is None or tir is None:
        raise RuntimeError('missing image {} {}'.format(sequence, frame))
    pair = np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)), axis=2)
    patch, _resize, _mask = sample_target(pair, state, 4.0, output_sz=256)
    values = []
    for image in (patch[:, :, :3], patch[:, :, 3:]):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        center = gray[80:176, 80:176]
        ring = np.concatenate((gray[:64].reshape(-1), gray[-64:].reshape(-1),
                               gray[:, :64].reshape(-1), gray[:, -64:].reshape(-1)))
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        values.extend((float(gray.mean()), float(gray.std()),
                       float(((gray < 0.03) | (gray > 0.97)).mean()),
                       float(np.log1p(lap.var())),
                       float(abs(center.mean() - ring.mean()))))
    w, h = float(state[2]), float(state[3])
    values.extend((math.log(max(1.0, w * h)), math.log(max(1.0, min(w, h))),
                   math.log(max(w, h) / max(1.0, min(w, h)))))
    return values


def feature_summary(rgb, tir, trgb, ttir):
    result = []
    for search, template in ((rgb, trgb), (tir, ttir)):
        pooled = template.mean(axis=0)
        a = search.reshape(-1, search.shape[-1]).astype(np.float32)
        q = pooled.astype(np.float32)
        similarity = a @ q / (np.linalg.norm(a, axis=1) * np.linalg.norm(q) + 1e-8)
        top = np.sort(similarity)[-8:]
        result.extend((float(np.linalg.norm(a, axis=1).mean()),
                       float(np.linalg.norm(a, axis=1).std()),
                       float(top[-1]), float(top[-1] - top[:-1].mean()),
                       float(similarity.std())))
    return result


class Response(nn.Module):
    def __init__(self):
        super().__init__()
        self.search = nn.Linear(768, 64, bias=False)
        self.template = nn.Linear(768, 64, bias=False)
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, search, template):
        query = F.normalize(self.search(F.layer_norm(search, (768,))), dim=-1)
        key = F.normalize(self.template(F.layer_norm(template, (768,))), dim=-1)
        return self.scale * (query * key[:, None, None, :]).sum(-1) + self.bias


def fit_response(search, template, target, train_ids, seed):
    torch.manual_seed(seed)
    model = Response().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    x = torch.as_tensor(search[train_ids], device='cuda', dtype=torch.float32)
    q = torch.as_tensor(template[train_ids], device='cuda', dtype=torch.float32)
    y = torch.as_tensor(target[train_ids], device='cuda', dtype=torch.float32)
    weight = torch.clamp((1 - y.mean()) / y.mean().clamp_min(1e-3), 1, 30)
    for _ in range(100):
        optimizer.zero_grad()
        logits = model(x, q)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        all_x = torch.as_tensor(search, device='cuda', dtype=torch.float32)
        all_q = torch.as_tensor(template, device='cuda', dtype=torch.float32)
        scores = model(all_x, all_q).float().cpu().numpy()
    return model.cpu().state_dict(), scores, float(loss.item())


def response_summary(scores):
    values = []
    for score in scores:
        flat = score.reshape(-1)
        peak = int(flat.argmax())
        y, x = divmod(peak, 16)
        yy, xx = np.mgrid[:16, :16]
        other = score[((xx - x) ** 2 + (yy - y) ** 2) > 4]
        values.append((float(flat[peak]), float(flat[peak] - other.max()),
                       float(flat[peak] - flat.mean()),
                       float(math.hypot(x - 7.5, y - 7.5))))
    return np.asarray(values, dtype=np.float64)


def ridge_fit_predict(x, y, train_ids, l2=10.0):
    train = x[train_ids]
    mean = train.mean(axis=0)
    scale = train.std(axis=0).clip(min=1e-6)
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0
    beta = np.linalg.solve(design[train_ids].T @ design[train_ids] + penalty,
                           design[train_ids].T @ y[train_ids])
    return design @ beta, {'mean': mean.tolist(), 'scale': scale.tolist(),
                           'beta': beta.tolist(), 'l2': l2}


def decode_iou(tracker, network, x, trace, image_shape):
    with torch.no_grad():
        fused = network.tbsi_fuse_search(x)
        score, _box, size, offset = network.box_head(fused)
        selected = score * tracker.output_window
        bbox = network.box_head.cal_bbox(selected, size, offset).view(-1, 4).mean(0)
        side = float(trace['crop_size'][0])
        resize = 256.0 / side
        tracker.state = [float(v) for v in trace['tracker_state_before']]
        box = tracker.map_box_back(bbox * 256.0 / resize, resize)
        box = clip_box(box, image_shape[0], image_shape[1], margin=10)
    return iou([float(v) for v in box], [float(v) for v in trace['gt_box']])


def summarize(rows, selected):
    chosen = [rows[i] for i in selected]
    if not chosen:
        return {'n_states': 0}
    sequences = sorted(set(row['sequence'] for row in chosen))
    by_sequence = []
    for sequence in sequences:
        members = [row for row in chosen if row['sequence'] == sequence]
        by_sequence.append(float(np.mean([row['gain'] for row in members])))
    rng = np.random.RandomState(20261012)
    boot = np.asarray([np.mean(rng.choice(by_sequence, size=len(by_sequence), replace=True))
                       for _ in range(5000)])
    return {'n_states': len(chosen), 'n_sequences': len(sequences),
            'mean_gain_state': float(np.mean([row['gain'] for row in chosen])),
            'mean_gain_sequence': float(np.mean(by_sequence)),
            'sequence_bootstrap_ci95': [float(v) for v in np.percentile(boot, [2.5, 97.5])],
            'wins': int(sum(row['gain'] > 1e-6 for row in chosen)),
            'harms': int(sum(row['gain'] < -1e-6 for row in chosen))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-states', type=int, default=144,
                        help='preflight only below 144')
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing nonempty output directory')
    if sha256(CONFIG) != CONFIG_SHA or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('protected M1 hash mismatch')
    if not torch.cuda.is_available() or not 1 <= args.max_states <= 144:
        raise RuntimeError('invalid GPU or state count')
    data = np.load(FEATURES)
    keys = [(str(sequence), int(frame)) for sequence, frame in
            zip(data['sequence'][:args.max_states], data['frame'][:args.max_states])]
    if len(keys) != len(set(keys)):
        raise RuntimeError('duplicate fixed state')
    traces = load_trace(keys)
    names = [line.strip() for line in SEQUENCES.read_text().splitlines() if line.strip()]
    dataset = LasHeRDataset(split='trainingset',
                            sequence_list_file=str(SEQUENCES)).get_sequence_list()
    image_index = {sequence.name: sequence for sequence in dataset}
    if set(name for name, _ in keys) - set(image_index):
        raise RuntimeError('feature sequence not in dataset manifest')

    order = sorted(set(name for name, _ in keys), key=lambda name: hashlib.sha256(
        ('m1-single-signal-20261012:' + name).encode()).hexdigest())
    train_names = set(order[:int(len(order) * 2 / 3)])
    train_ids = np.asarray([i for i, (name, _) in enumerate(keys) if name in train_names])
    held_ids = np.asarray([i for i, (name, _) in enumerate(keys) if name not in train_names])
    if not train_ids.size or not held_ids.size:
        raise RuntimeError('empty sequence partition')

    os.environ['TBSI_INFER_FP16'] = '0'
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    params = parameters(NAME)
    params.checkpoint = str(CHECKPOINT)
    params.debug = 0
    params.save_all_boxes = False
    tracker = TBSITrack(params, 'm1_three_signal_screen')
    network = tracker.network.eval()
    labels = data['target'][:args.max_states].astype(np.float32)
    template_rgb = np.asarray([(t * w[:, None]).sum(0) / max(1e-6, w.sum())
                               for t, w in zip(data['template_rgb'][:args.max_states].astype(np.float32),
                                               data['template_fg'][:args.max_states])], dtype=np.float32)
    template_tir = np.asarray([(t * w[:, None]).sum(0) / max(1e-6, w.sum())
                               for t, w in zip(data['template_tir'][:args.max_states].astype(np.float32),
                                               data['template_fg'][:args.max_states])], dtype=np.float32)
    started = time.time()
    rgb_state, rgb_response, rgb_loss = fit_response(
        data['rgb'][:args.max_states], template_rgb, labels, train_ids, 42)
    tir_state, tir_response, tir_loss = fit_response(
        data['tir'][:args.max_states], template_tir, labels, train_ids, 43)
    responses = np.column_stack((response_summary(rgb_response),
                                  response_summary(tir_response)))

    rows, s2_features, s3_features = [], [], []
    for i, key in enumerate(keys):
        trace = traces[key]
        state = trace['tracker_state_before']
        image = image_index[key[0]]
        image_shape = cv2.imread(image.frames[0][key[1]], cv2.IMREAD_COLOR).shape
        x = torch.as_tensor(data['concat'][i].transpose(2, 0, 1)[None],
                            dtype=torch.float32, device='cuda')
        base = decode_iou(tracker, network, x, trace, image_shape)
        plus_x = x.clone(); minus_x = x.clone()
        plus_x[:, :768] *= 1.15; plus_x[:, 768:] *= 0.85
        minus_x[:, :768] *= 0.85; minus_x[:, 768:] *= 1.15
        plus = decode_iou(tracker, network, plus_x, trace, image_shape)
        minus = decode_iou(tracker, network, minus_x, trace, image_shape)
        rows.append({'sequence': key[0], 'frame': key[1], 'label': str(data['label'][i]),
                     'base_iou': base, 'plus_iou': plus, 'minus_iou': minus,
                     'oracle_gain': max(base, plus, minus) - base,
                     'trace_iou': iou(trace['pred_box'], trace['gt_box'])})
        s2_features.append(feature_summary(data['rgb'][i], data['tir'][i],
                                           template_rgb[i][None], template_tir[i][None]))
        s3_features.append(image_features(key[0], key[1], state, image))
        if (i + 1) % 12 == 0:
            print('replayed {}/{}'.format(i + 1, len(keys)), flush=True)

    parity = [abs(row['base_iou'] - row['trace_iou']) for row in rows]
    if max(parity) > 0.02:
        raise RuntimeError('fixed-state IoU replay differs from trace: max {}'.format(max(parity)))
    target = np.asarray([row['plus_iou'] - row['minus_iou'] for row in rows])
    signals = {}
    for signal, features in (('s1', responses), ('s2', np.asarray(s2_features)),
                             ('s3', np.asarray(s3_features))):
        prediction, calibration = ridge_fit_predict(
            np.asarray(features, dtype=np.float64), target, train_ids)
        signals[signal] = {'prediction': prediction, 'calibration': calibration}
    for i, row in enumerate(rows):
        row['split'] = 'train' if i in train_ids else 'heldout'
        row['rgb_response_top1_target'] = int(labels[i].reshape(-1)[rgb_response[i].argmax()] > 0)
        row['tir_response_top1_target'] = int(labels[i].reshape(-1)[tir_response[i].argmax()] > 0)
        for name, values in signals.items():
            row[name + '_predicted_gap'] = float(values['prediction'][i])
    constant = 1 if target[train_ids].mean() >= 0 else -1
    output.mkdir(parents=True)
    result = {'id': 'M1-THREE-SIGNAL-SCREEN-20261012',
              'status': 'FIXED-STATE-DIAGNOSTIC-COMPLETE',
              'parent': NAME, 'n_states': len(keys), 'n_sequences': len(order),
              'train_sequences': sorted(train_names),
              'heldout_sequences': sorted(set(order) - train_names),
              'config_sha256': CONFIG_SHA, 'checkpoint_sha256': CHECKPOINT_SHA,
              'features_sha256': sha256(FEATURES), 'sequence_list_sha256': sha256(SEQUENCES),
              'trace_parity_max_iou': max(parity),
              'response_train_loss': {'rgb': rgb_loss, 'tir': tir_loss},
              'response_target_top1_heldout': {
                  'rgb': float(np.mean([labels[i].reshape(-1)[rgb_response[i].argmax()]
                                        for i in held_ids])),
                  'tir': float(np.mean([labels[i].reshape(-1)[tir_response[i].argmax()]
                                        for i in held_ids]))},
              'arms': {}, 'elapsed_seconds': time.time() - started,
              'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'scope': 'train-side fixed crop; current-code M1 checkpoint; not closed-loop AUC'}
    for arm in ('constant', 's1', 's2', 's3', 'oracle'):
        arm_rows = []
        for i, row in enumerate(rows):
            if arm == 'oracle':
                gain = row['oracle_gain']
            else:
                direction = constant if arm == 'constant' else (1 if signals[arm]['prediction'][i] >= 0 else -1)
                gain = (row['plus_iou'] if direction > 0 else row['minus_iou']) - row['base_iou']
            arm_rows.append({'sequence': row['sequence'], 'gain': gain})
            row[arm + '_gain'] = gain
        result['arms'][arm] = {'heldout': summarize(arm_rows, held_ids),
                               'severe_heldout': summarize(arm_rows,
                                   [i for i in held_ids if rows[i]['label'] == 'severe_onset']),
                               'matched_good_heldout': summarize(arm_rows,
                                   [i for i in held_ids if rows[i]['label'] == 'matched_good'])}
    with (output / 'per_state.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    torch.save({'rgb': rgb_state, 'tir': tir_state}, str(output / 's1_response_heads.pth'))
    (output / 'signal_calibration.json').write_text(json.dumps({
        name: value['calibration'] for name, value in signals.items()}, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['arms'], indent=2), flush=True)


if __name__ == '__main__':
    main()
