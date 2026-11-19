#!/usr/bin/env python3
"""Offline oracle headroom comparison for feature and readout action spaces."""
import argparse, csv, hashlib, json, os, sys
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
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def iou(a, b):
    ix = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.

def decode(network, tracker, out, state, shape, resize):
    response = tracker.output_window * out[0]
    box = network.box_head.cal_bbox(response, out[2], out[3]).view(-1, 4).mean(0)
    box = box * (tracker.params.search_size / resize)
    cx, cy = state[0] + .5 * state[2], state[1] + .5 * state[3]
    half = .5 * tracker.params.search_size / resize
    xywh = [float(box[0] + cx - half - .5 * box[2]),
            float(box[1] + cy - half - .5 * box[3]),
            float(box[2]), float(box[3])]
    return clip_box(xywh, shape[0], shape[1], margin=10)

def local_mask(h, w, gx, gy, radius=1):
    m = torch.zeros((1, 1, h, w), dtype=torch.float32)
    m[:, :, max(0, gy-radius):min(h, gy+radius+1),
            max(0, gx-radius):min(w, gx+radius+1)] = 1.
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--trace-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--rho', type=float, default=.05)
    args = ap.parse_args()
    outdir = Path(args.output_dir).resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError('refusing nonempty output dir: ' + str(outdir))
    if sha256(CONFIG) != CONFIG_SHA or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('protected M1 hash mismatch')
    if not (0.0 < args.rho <= .2):
        raise ValueError('rho must be in (0, .2]')
    data = np.load(args.features, allow_pickle=False)
    required = {'rgb', 'tir', 'fused', 'sequence', 'frame', 'gt_xy'}
    if not required.issubset(set(data.files)):
        raise RuntimeError('feature asset lacks required fields')
    trace = {}
    for p in Path(args.trace_dir).glob('*/frames.jsonl'):
        for line in p.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                trace[(p.parent.name, int(row['frame_id']))] = row
    seq_list = sorted(set(str(v) for v in data['sequence']))
    list_path = outdir.parent / 'replay_sequence_list.txt'
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text('\n'.join(seq_list) + '\n')
    seqmap = {s.name: s for s in LasHeRDataset('trainingset', str(list_path)).get_sequence_list()}
    os.environ['TBSI_INFER_FP16'] = '0'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    params = parameters(NAME)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'm1_action_space_oracle')
    network = tracker.network.eval()
    rows = []
    with torch.no_grad():
        for i in range(len(data['sequence'])):
            seq = str(data['sequence'][i]); frame = int(data['frame'][i])
            tr = trace[(seq, frame)]
            state = np.asarray(tr['tracker_state_before'], dtype=np.float64)
            gt = np.asarray(tr['gt_box'], dtype=np.float64)
            rgb_img = cv2.imread(seqmap[seq].frames[0][frame], cv2.IMREAD_COLOR)
            tir_img = cv2.imread(seqmap[seq].frames[1][frame], cv2.IMREAD_COLOR)
            image_shape = rgb_img.shape[:2]
            image = np.concatenate((cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB),
                                    cv2.cvtColor(tir_img, cv2.COLOR_BGR2RGB)), axis=2)
            _, resize, _ = sample_target(image, state, params.search_factor, output_sz=params.search_size)
            rgb = torch.as_tensor(data['rgb'][i].transpose(2, 0, 1)[None], dtype=torch.float32, device='cuda')
            tir = torch.as_tensor(data['tir'][i].transpose(2, 0, 1)[None], dtype=torch.float32, device='cuda')
            fused = torch.as_tensor(data['fused'][i].transpose(2, 0, 1)[None], dtype=torch.float32, device='cuda')
            rgb_only = network.tbsi_fuse_search(torch.cat((rgb, torch.zeros_like(tir)), 1))
            tir_only = network.tbsi_fuse_search(torch.cat((torch.zeros_like(rgb), tir), 1))
            base = network.box_head(fused)
            base_box = decode(network, tracker, base, state, image_shape, resize)
            base_iou = iou(base_box, gt)
            gx, gy = int(data['gt_xy'][i, 0]), int(data['gt_xy'][i, 1])
            gx = max(0, min(15, gx)); gy = max(0, min(15, gy))
            mask = local_mask(16, 16, gx, gy).to(device='cuda', dtype=fused.dtype)
            actions = {}
            for name, source in [('feature_rgb_gt3x3', rgb_only), ('feature_tir_gt3x3', tir_only)]:
                f = fused + args.rho * mask * (source - fused)
                actions[name] = network.box_head(f)

            score, bbox, size, offset = base
            score_gt = score.clone()
            score_gt[:, :, gy, gx] = torch.clamp(score.flatten(1).max(1).values + .05, max=.9999)
            # Desired readout values in the same normalized 16x16 contract.
            gt_cx = gt[0] + .5 * gt[2]; gt_cy = gt[1] + .5 * gt[3]
            state_cx = state[0] + .5 * state[2]; state_cy = state[1] + .5 * state[3]
            cx_norm = (gt_cx - state_cx) * resize / params.search_size + .5
            cy_norm = (gt_cy - state_cy) * resize / params.search_size + .5
            wh = torch.tensor([gt[2] * resize / params.search_size,
                               gt[3] * resize / params.search_size], device='cuda', dtype=size.dtype)
            gt_offset = torch.tensor([cx_norm * 16. - gx, cy_norm * 16. - gy], device='cuda', dtype=offset.dtype)
            score_only = (score_gt, bbox, size, offset)
            offset_gt = offset.clone(); offset_gt[:, :, gy, gx] = gt_offset
            size_gt = size.clone(); size_gt[:, :, gy, gx] = wh
            joint = (score_gt, bbox, size_gt, offset_gt)
            actions['readout_score_oracle'] = score_only
            actions['readout_offset_oracle'] = (score, bbox, size, offset_gt)
            actions['readout_size_oracle'] = (score, bbox, size_gt, offset)
            actions['readout_joint_oracle'] = joint
            for name, out in [('no_op', base)] + list(actions.items()):
                pred = decode(network, tracker, out, state, image_shape, resize)
                rows.append({'sequence': seq, 'frame': frame, 'label': str(data['label'][i]),
                             'action': name, 'base_iou': base_iou, 'iou': iou(pred, gt),
                             'utility': iou(pred, gt) - base_iou,
                             'gt_cell_x': gx, 'gt_cell_y': gy})
            if (i + 1) % 16 == 0:
                print('states={}/{}'.format(i + 1, len(data['sequence'])), flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / 'per_state.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    seqs = sorted(set(r['sequence'] for r in rows))
    actions = sorted(set(r['action'] for r in rows))
    summary = []
    for action in actions:
        vals = {(s): float(np.mean([float(r['utility']) for r in rows if r['action'] == action and r['sequence'] == s])) for s in seqs}
        diffs = np.asarray([vals[s] for s in seqs])
        summary.append({'action': action, 'states': sum(r['action'] == action for r in rows),
                        'sequences': len(seqs), 'mean_utility': float(diffs.mean()),
                        'median_utility': float(np.median(diffs)),
                        'sequence_positive': int((diffs > 0).sum()),
                        'sequence_negative': int((diffs < 0).sum())})
    with (outdir / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    meta = {'id': 'M1-ACTION-SPACE-ORACLE-20261013', 'parent': NAME,
            'rho': args.rho, 'states': len(data['sequence']), 'sequences': len(seqs),
            'actions': actions, 'config_sha256': sha256(CONFIG), 'checkpoint_sha256': sha256(CHECKPOINT),
            'features_sha256': sha256(Path(args.features)),
            'estimand': 'fixed-state decoded IoU utility relative to no-op',
            'interpretation': 'oracle headroom only; GT cell and GT readout values are not deployable',
            'status': 'ORACLE-REPLAY-COMPLETE'}
    (outdir / 'summary.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'meta': meta, 'summary': summary}, indent=2))

if __name__ == '__main__':
    main()
