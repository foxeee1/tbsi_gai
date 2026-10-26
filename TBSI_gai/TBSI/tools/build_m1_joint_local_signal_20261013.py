#!/usr/bin/env python3
"""Build local-action labels and sequence-held-out S1+S2(+S3) utility models."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.models.layers.local_signal_post_fusion import LocalSignalPostFusion
from lib.models.layers.single_signal_prefusion import image_features_from_patch
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box

NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
CHECKPOINT = ROOT / 'output/experiments' / NAME / 'checkpoints/TBSITrack_ep0015.pth.tar'
CHECKPOINT_SHA = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'
SOURCE_FEATURES = ROOT / 'output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/selected_features.npz'
SOURCE_TRACES = [
    ROOT / 'output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run/trace',
    ROOT / 'output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run_rank24/trace',
    ROOT / 'output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run_rank48/trace',
]
S1_ARTIFACT = ROOT / 'output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/train72_screen_v1'
S1_SUMMARY = ROOT / 'output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/train72_screen_v1/summary.json'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_pair(sequence, index):
    rgb = cv2.imread(sequence.frames[0][index], cv2.IMREAD_COLOR)
    tir = cv2.imread(sequence.frames[1][index], cv2.IMREAD_COLOR)
    if rgb is None or tir is None or rgb.shape[:2] != tir.shape[:2]:
        raise RuntimeError('missing/misaligned modalities {} frame {}'.format(sequence.name, index))
    return np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)), axis=2)


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


def decode(network, tracker, score, size, offset, state, image_shape, resize):
    response = tracker.output_window * score
    box = network.box_head.cal_bbox(response, size, offset).view(-1, 4).mean(0)
    box = box * (tracker.params.search_size / resize)
    cx, cy = state[0] + .5 * state[2], state[1] + .5 * state[3]
    half = .5 * tracker.params.search_size / resize
    xywh = [float(box[0] + cx - half - .5 * box[2]),
            float(box[1] + cy - half - .5 * box[3]), float(box[2]), float(box[3])]
    return clip_box(xywh, image_shape[0], image_shape[1], margin=10)


def ridge_fit(x, y, alpha=10.):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.
    xn = np.clip((x - mean) / scale, -8., 8.)
    ymean = y.mean(0)
    weight = np.linalg.solve(xn.T @ xn + alpha * np.eye(x.shape[1]),
                             xn.T @ (y - ymean))
    residual = y - (xn @ weight + ymean)
    return {'mean': mean.tolist(), 'scale': scale.tolist(),
            'weights': np.concatenate((ymean[:, None], weight.T), axis=1).tolist(),
            'rmse': np.sqrt(np.mean(residual ** 2, axis=0)).tolist()}


def predict(model, x):
    z = np.clip((x - np.asarray(model['mean'])) / np.asarray(model['scale']), -8., 8.)
    w = np.asarray(model['weights'])
    return w[:, 0] + z @ w[:, 1:].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--rho', type=float, default=.05)
    ap.add_argument('--ridge-alpha', type=float, default=10.)
    args = ap.parse_args()
    outdir = Path(args.output_dir).resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError('refusing nonempty output dir: ' + str(outdir))
    if not torch.cuda.is_available() or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('CUDA unavailable or protected M1 checkpoint hash mismatch')
    os.environ['TBSI_INFER_FP16'] = '0'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    archive = np.load(str(SOURCE_FEATURES), allow_pickle=False)
    data = {key: archive[key] for key in archive.files}
    summary = json.loads(S1_SUMMARY.read_text())
    train_names = set(summary['train_sequences'])
    held_names = set(summary['heldout_sequences'])
    if train_names & held_names or len(train_names) != 48 or len(held_names) != 24:
        raise RuntimeError('unexpected S1 sequence split')

    outdir.mkdir(parents=True, exist_ok=True)
    seq_list = outdir / 'sequence_list.txt'
    seq_list.write_text('\n'.join(sorted(set(str(v) for v in data['sequence']))) + '\n')
    sequences = LasHeRDataset('trainingset', str(seq_list)).get_sequence_list()
    seqmap = {s.name: s for s in sequences}
    trace = {}
    for trace_root in SOURCE_TRACES:
        for path in trace_root.glob('*/frames.jsonl'):
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    trace[(path.parent.name, int(row['frame_id']))] = row

    params = parameters(NAME)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'm1_joint_local_signal_label_build')
    network = tracker.network.eval()
    signal = LocalSignalPostFusion('fixed_rgb', str(S1_ARTIFACT), rho=args.rho).cuda().eval()
    rows, x1_rows, x3_rows, utilities = [], [], [], []
    started = time.time()

    for i in range(len(data['sequence'])):
        seq_name, frame = str(data['sequence'][i]), int(data['frame'][i])
        key = (seq_name, frame)
        if key not in trace or seq_name not in seqmap:
            raise RuntimeError('missing trace or dataset state: {}'.format(key))
        tr = trace[key]
        state = np.asarray(tr['tracker_state_before'], dtype=np.float64)
        gt = np.asarray(tr['gt_box'], dtype=np.float64)
        image = read_pair(seqmap[seq_name], frame)
        crop, resize, _ = sample_target(image, state, params.search_factor,
                                         output_sz=params.search_size)

        rgb = torch.as_tensor(data['rgb'][i].reshape(1, 256, 768), dtype=torch.float32, device='cuda')
        tir = torch.as_tensor(data['tir'][i].reshape(1, 256, 768), dtype=torch.float32, device='cuda')
        template_rgb = torch.as_tensor(data['template_rgb'][i][None], dtype=torch.float32, device='cuda')
        template_tir = torch.as_tensor(data['template_tir'][i][None], dtype=torch.float32, device='cuda')
        fg = torch.as_tensor(data['template_fg'][i][None], dtype=torch.float32, device='cuda')
        cat = torch.cat((template_rgb, rgb, template_tir, tir), dim=1)
        signal.template_fg = fg
        rgb_feat = rgb.transpose(1, 2).reshape(1, 768, 16, 16)
        tir_feat = tir.transpose(1, 2).reshape(1, 768, 16, 16)
        concat = torch.cat((rgb_feat, tir_feat), dim=1)
        with torch.no_grad():
            fused = network.tbsi_fuse_search(concat)
            zeros = torch.zeros_like(rgb_feat)
            rgb_only = network.tbsi_fuse_search(torch.cat((rgb_feat, zeros), 1))
            tir_only = network.tbsi_fuse_search(torch.cat((zeros, tir_feat), 1))
            signal_maps = signal._signals(cat, rgb, tir, fused)
            rgb_map, tir_map, x1 = signal_maps
            x1_np = x1.detach().cpu().numpy().astype(np.float64)
            s3 = np.asarray(image_features_from_patch(crop, state), dtype=np.float64)
            x3_np = np.concatenate((x1_np, s3))
            base_out = network.box_head(fused)
            base_box = decode(network, tracker, base_out[0], base_out[2], base_out[3],
                              state, image.shape, resize)
            base_iou = box_iou(base_box, gt)

            def local_variant(source, score):
                mask = signal._local_mask(score).to(device=fused.device, dtype=fused.dtype)
                return fused + args.rho * mask * (source - fused)

            rgb_out = network.box_head(local_variant(rgb_only, rgb_map))
            tir_out = network.box_head(local_variant(tir_only, tir_map))
            rgb_iou = box_iou(decode(network, tracker, rgb_out[0], rgb_out[2], rgb_out[3],
                                     state, image.shape, resize), gt)
            tir_iou = box_iou(decode(network, tracker, tir_out[0], tir_out[2], tir_out[3],
                                     state, image.shape, resize), gt)

        if not np.isfinite(np.concatenate((x1_np, x3_np, [base_iou, rgb_iou, tir_iou])).all()):
            raise RuntimeError('nonfinite feature or utility at {}'.format(key))
        x1_rows.append(x1_np)
        x3_rows.append(x3_np)
        utilities.append([rgb_iou - base_iou, tir_iou - base_iou])
        rows.append({'sequence': seq_name, 'frame': frame, 'label': str(data['label'][i]),
                     'base_iou': base_iou, 'rgb_local_iou': rgb_iou,
                     'tir_local_iou': tir_iou, 'u_rgb': rgb_iou - base_iou,
                     'u_tir': tir_iou - base_iou,
                     'rgb_peak': int(rgb_map.argmax().item()), 'tir_peak': int(tir_map.argmax().item())})
        if (i + 1) % 16 == 0:
            print('states={}/{} elapsed_s={:.1f}'.format(i + 1, len(data['sequence']),
                                                          time.time() - started), flush=True)

    x1 = np.asarray(x1_rows)
    x3 = np.asarray(x3_rows)
    y = np.asarray(utilities)
    names = np.asarray([r['sequence'] for r in rows])
    train = np.asarray([name in train_names for name in names])
    held = np.asarray([name in held_names for name in names])
    if train.sum() != 100 or held.sum() != 44:
        raise RuntimeError('expected 100 train / 44 heldout states; got {} / {}'.format(train.sum(), held.sum()))

    report = {'fit_split': {}, 'heldout_split': {}, 'rho': args.rho,
              'ridge_alpha': args.ridge_alpha, 'features_s1s2': int(x1.shape[1]),
              'features_s1s2s3': int(x3.shape[1])}
    held_choices = {}
    final_models = {}
    for name, x in (('s1s2', x1), ('s1s2s3', x3)):
        model = ridge_fit(x[train], y[train], args.ridge_alpha)
        pred = predict(model, x[held])
        choices = np.argmax(np.column_stack((np.zeros(held.sum()), pred)), axis=1) - 1
        actual = np.where(choices == 0, y[held, 0], np.where(choices == 1, y[held, 1], 0.))
        held_choices[name] = choices
        seqs = sorted(set(names[held]))
        per_seq = []
        held_indices = np.flatnonzero(held)
        for seq in seqs:
            mask = names[held_indices] == seq
            per_seq.append(float(actual[mask].mean()))
        report['heldout_split'][name] = {
            'sequences': len(seqs), 'states': int(held.sum()),
            'mean_utility': float(actual.mean()), 'sequence_macro_utility': float(np.mean(per_seq)),
            'action_counts': {'no_op': int(np.sum(choices < 0)), 'rgb': int(np.sum(choices == 0)),
                              'tir': int(np.sum(choices == 1))},
            'fixed_rgb_mean': float(y[held, 0].mean()), 'fixed_tir_mean': float(y[held, 1].mean()),
            'oracle_mean': float(np.maximum(0., y[held].max(1)).mean())}
        report['fit_split'][name] = {'states': int(train.sum()),
                                     'sequences': len(set(names[train]))}
        final_models[name] = ridge_fit(x, y, args.ridge_alpha)

    (outdir / 'local_utility_models.json').write_text(json.dumps(final_models, indent=2) + '\n')
    torch.save(torch.load(str(S1_ARTIFACT / 's1_response_heads.pth'), map_location='cpu'),
               str(outdir / 's1_response_heads.pth'))
    with (outdir / 'local_action_utility.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report.update({'id': 'M1-JOINT-LOCAL-SIGNAL-DECISION-20261013',
                   'status': 'UTILITY-MODELS-BUILT', 'n_sequences': len(set(names)),
                   'n_states': len(rows), 'train_sequences': sorted(train_names),
                   'heldout_sequences': sorted(held_names),
                   'checkpoint_sha256': sha256(CHECKPOINT),
                   'features_sha256': sha256(SOURCE_FEATURES),
                   'models_sha256': sha256(outdir / 'local_utility_models.json'),
                   'elapsed_seconds': time.time() - started,
                   'limitations': ['Train-side 72-state cohort was previously used in signal diagnostics.',
                                   'M1 training sequence exposure is unknown.',
                                   'Union82 is an observed stress cohort, not independent confirmation.']})
    (outdir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['heldout_split'], indent=2), flush=True)


if __name__ == '__main__':
    main()
