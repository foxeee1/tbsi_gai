#!/usr/bin/env python3
"""Small code-audit experiment for the Innovation-1 path intervention."""
import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
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


def clone(v):
    if torch.is_tensor(v):
        return v.detach().clone()
    if isinstance(v, list):
        return [clone(x) for x in v]
    if isinstance(v, tuple):
        return tuple(clone(x) for x in v)
    if isinstance(v, dict):
        return {k: clone(x) for k, x in v.items()}
    return copy.deepcopy(v)


def snap(t):
    return {'state': clone(t.state), 'prev_search': clone(t.prev_search),
            'prev_temporal_tokens': clone(t.prev_temporal_tokens), 'frame_id': t.frame_id,
            'torch_rng': torch.get_rng_state().tolist(),
            'cuda_rng': [x.tolist() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            'numpy_rng': np.random.get_state()[1].tolist()}


def restore(t, s):
    t.state = clone(s['state'])
    t.prev_search = clone(s['prev_search'])
    t.prev_temporal_tokens = clone(s['prev_temporal_tokens'])
    t.frame_id = s['frame_id']
    torch.set_rng_state(torch.tensor(s['torch_rng'], dtype=torch.uint8))
    if torch.cuda.is_available() and s['cuda_rng']:
        torch.cuda.set_rng_state_all([torch.tensor(x, dtype=torch.uint8, device='cpu') for x in s['cuda_rng']])


def frame(seq, n):
    v = cv2.cvtColor(cv2.imread(seq.frames[0][n]), cv2.COLOR_BGR2RGB)
    i = cv2.cvtColor(cv2.imread(seq.frames[1][n]), cv2.COLOR_BGR2RGB)
    return np.concatenate([v, i], axis=2)


def iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    ix = max(0., min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0., min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.


def digest(v):
    return hashlib.sha256(repr(v).encode()).hexdigest()[:16]


def set_gate(action=None, layer=2, scale=0.):
    if action is None:
        os.environ.pop('TBSI_PATH_EFFECT_ACTION', None)
    else:
        os.environ['TBSI_PATH_EFFECT_ACTION'] = action
    os.environ['TBSI_PATH_EFFECT_LAYER'] = str(layer)
    os.environ['TBSI_PATH_EFFECT_SCALE'] = str(scale)


def install_hooks(tracker, layer_index, capture):
    layer = tracker.network.backbone.tbsi_layers[layer_index]
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            x = inputs[0]
            entry = capture.setdefault(name, {'calls': 0, 'input_shapes': [], 'message_norms': [],
                                               'output_norms': []})
            entry['calls'] += 1
            entry['input_shapes'].append(list(x.shape))
            out = output[0] if isinstance(output, tuple) else output
            lens_z = 64
            entry['message_norms'].append(float(out[:, lens_z:, :].detach().float().norm().item()))
            entry['output_norms'].append(float(out.detach().float().norm().item()))
        return hook

    for name, block in (('f2v', layer.ca_t2s_f2v), ('f2i', layer.ca_t2s_f2i)):
        handles.append(block.attn_reshape.register_forward_hook(make_hook(name)))
    return handles


def run_one(tracker, seq, frame_num, base, action, scale, layer, horizon, capture):
    restore(tracker, base)
    set_gate(action, layer, scale)
    boxes = []
    for k in range(horizon):
        out = tracker.track(frame(seq, frame_num + k), {'sequence_name': seq.name, 'frame_num': frame_num + k})
        boxes.append([float(x) for x in out['target_bbox']])
        set_gate(None, layer, 1.)
    return boxes, digest(tracker.state), digest(tracker.prev_temporal_tokens)


def run_current(tracker, seq, frame_num, action, scale, layer, capture):
    set_gate(action, layer, scale)
    out = tracker.track(frame(seq, frame_num), {'sequence_name': seq.name, 'frame_num': frame_num})
    set_gate(None, layer, 1.)
    return [float(x) for x in out['target_bbox']], digest(tracker.state), digest(tracker.prev_temporal_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='v1.0.3-单卡4090D-24G-等效全局128-seed3407')
    ap.add_argument('--output', default='output/experiments/innovation1_path_effect_audit')
    ap.add_argument('--layer', type=int, default=2)
    ap.add_argument('--sequences', type=int, default=8)
    ap.add_argument('--points', type=int, default=3)
    ap.add_argument('--horizon', type=int, default=3)
    args = ap.parse_args()
    os.environ['TBSI_INFER_FP16'] = '0'
    torch.manual_seed(3407)
    np.random.seed(3407)
    random.seed(3407)
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    params = parameters(args.config)
    params.debug = 0
    params.save_all_boxes = False
    tracker = TBSITrack(params, 'lasher_train')
    ckpt = torch.load(params.checkpoint, map_location='cpu')['net']
    model_keys = set(tracker.network.state_dict().keys())
    ckpt_keys = set(ckpt.keys())
    audit = {
        'git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'config': args.config, 'checkpoint': params.checkpoint,
        'checkpoint_sha256': hashlib.sha256(Path(params.checkpoint).read_bytes()).hexdigest(),
        'checkpoint_missing_keys': sorted(model_keys - ckpt_keys),
        'checkpoint_unexpected_keys': sorted(ckpt_keys - model_keys),
        'tbsi_layers': len(tracker.network.backbone.tbsi_layers),
        'tbsi_locations': list(tracker.cfg.MODEL.BACKBONE.TBSI_LOC),
        'selected_layer_index': args.layer,
        'manual_iou_checks': [iou([0, 0, 10, 10], [0, 0, 10, 10]), iou([0, 0, 10, 10], [5, 0, 10, 10]), iou([0, 0, 10, 10], [20, 0, 10, 10])],
        'rows': [], 'scale_rows': [], 'order_rows': [], 'module_calls': {},
    }
    list_file = ROOT / 'experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt'
    dataset = LasHeRDataset(split='trainingset', sequence_list_file=str(list_file)).get_sequence_list()
    sequences = list(dataset[:args.sequences])
    audit['sequences'] = [s.name for s in sequences]
    set_gate(None, args.layer, 1.)
    handles = install_hooks(tracker, args.layer, audit['module_calls'])
    try:
        for seq in sequences:
            tracker.initialize(frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0].tolist()})
            max_start = len(seq.frames[0]) - args.horizon
            starts = np.linspace(1, max_start, args.points).round().astype(int).tolist()
            for t in sorted(set(starts)):
                set_gate(None, args.layer, 1.)
                while tracker.frame_id < t:
                    tracker.track(frame(seq, tracker.frame_id + 1), {'sequence_name': seq.name, 'frame_num': tracker.frame_id + 1})
                base = snap(tracker)
                # Gate-off parity: unset environment versus explicit F/scale=1.
                set_gate(None, args.layer, 1.)
                f0, fstate0, _ = run_one(tracker, seq, t, base, None, 1., args.layer, args.horizon, audit['module_calls'])
                set_gate('F', args.layer, 1.)
                f1, fstate1, _ = run_one(tracker, seq, t, base, 'F', 1., args.layer, args.horizon, audit['module_calls'])
                audit['rows'].append({'sequence': seq.name, 'frame': int(t), 'parity_box_max_abs': max(abs(a-b) for x,y in zip(f0,f1) for a,b in zip(x,y)), 'parity_state_equal': fstate0 == fstate1})
                for action in ('GR', 'GT'):
                    for scale in (0.75, 0.5, 0.0):
                        boxes, state_hash, _ = run_one(tracker, seq, t, base, action, scale, args.layer, args.horizon, audit['module_calls'])
                        ds = [iou(boxes[k], seq.ground_truth_rect[t+k]) - iou(f0[k], seq.ground_truth_rect[t+k]) for k in range(args.horizon)]
                        audit['scale_rows'].append({'sequence': seq.name, 'frame': int(t), 'action': action, 'scale': scale,
                                                    'd0': ds[0], 'd1': ds[1], 'd2': ds[2], 'mean_delta': float(np.mean(ds)), 'state_hash': state_hash})
                # Order contamination: F-GR-GT-F and F-GT-GR-F from the same base.
                for order in (('F', 'GR', 'GT', 'F'), ('F', 'GT', 'GR', 'F')):
                    values = []
                    for action in order:
                        # Reuse the same action-before snapshot for each call;
                        # only invocation order is changed, so state evolution
                        # cannot be mistaken for cache/tensor contamination.
                        restore(tracker, base)
                        boxes, state_hash, _ = run_one(
                            tracker, seq, t, base, action,
                            0. if action in ('GR', 'GT') else 1., args.layer, 1,
                            audit['module_calls'])
                        box = boxes[0]
                        values.append({'action': action, 'box': box, 'state_hash': state_hash})
                    audit['order_rows'].append({'sequence': seq.name, 'frame': int(t), 'order': '-'.join(order), 'values': values})
    finally:
        for handle in handles:
            handle.remove()
        set_gate(None, args.layer, 1.)
    audit['finished'] = True
    audit['summary'] = {
        'parity_max_abs': max(x['parity_box_max_abs'] for x in audit['rows']),
        'parity_state_all_equal': all(x['parity_state_equal'] for x in audit['rows']),
        'scale_rows': len(audit['scale_rows']),
        'order_rows': len(audit['order_rows']),
        'manual_iou_ok': audit['manual_iou_checks'] == [1.0, 1.0 / 3.0, 0.0],
    }
    (out_dir / 'audit_results.json').write_text(json.dumps(audit, indent=2, ensure_ascii=True))
    print(json.dumps(audit['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
