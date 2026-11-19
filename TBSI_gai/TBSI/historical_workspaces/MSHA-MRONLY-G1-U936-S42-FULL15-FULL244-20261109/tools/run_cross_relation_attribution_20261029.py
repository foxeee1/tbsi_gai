#!/usr/bin/env python3
"""Fixed-state attribution for the cross-mass-preserving relation candidate."""
import copy
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['TBSI_INFER_FP16'] = '0'

import cv2
import numpy as np
import torch

CHECKPOINT = Path('/autodl-fs/data/early_gate_closed_loop/cross_relation_20261028/runs/CROSS_RELATION/checkpoint_u0468.pth.tar')
CHECKPOINT_SHA256 = 'b941047d4d8f8d38a7ef8d422c413b540a0acac40e63482c404faa18a58d8f48'
CONFIG_NAME = 'EG_EARLY_GATE_CROSS_RELATION_QUICK_SOT_S42_20261028'
CONFIG = ROOT / ('experiments/tbsi_track/' + CONFIG_NAME + '.yaml')
PREDICTIONS = Path('/autodl-fs/data/early_gate_closed_loop/cross_relation_20261028/eval96_max2000')
OUTPUT = Path('/autodl-fs/data/early_gate_closed_loop/cross_relation_20261029/attribution_beta_sweep')
CASES = [
    {'sequence': 'whitegirlinlight', 'role': 'worst-normal', 'frame': 69},
    {'sequence': 'boy`sheadingreycol', 'role': 'worst-low', 'frame': 69},
    {'sequence': 'basketballathand', 'role': 'worst-low', 'frame': 40},
    {'sequence': 'whitecarturn683', 'role': 'worst-middle', 'frame': 69},
    {'sequence': 'leftboy2jointhe4', 'role': 'largest-gain-low', 'frame': 69},
    {'sequence': 'rightbottlecomes', 'role': 'gain-low', 'frame': 69},
    {'sequence': 'mandownstair', 'role': 'normal-gain', 'frame': 69},
    {'sequence': 'rightgirltakingcup', 'role': 'normal-check', 'frame': 69},
]
HORIZON = 20
MODES = ['relation', 'identity', 'beta_pos_0p5', 'beta_neg_0p5',
         'beta_pos_1', 'beta_neg_1']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clone(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, list):
        return [clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone(v) for v in value)
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    return copy.deepcopy(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_rgbt(sequence, frame_index):
    visible = cv2.imread(sequence.frames[0][frame_index], cv2.IMREAD_COLOR)
    infrared = cv2.imread(sequence.frames[1][frame_index], cv2.IMREAD_COLOR)
    if visible is None or infrared is None:
        raise RuntimeError('Could not decode frame %s:%d' % (sequence.name, frame_index))
    return np.concatenate((cv2.cvtColor(visible, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(infrared, cv2.COLOR_BGR2RGB)), axis=2)


def iou(a, b):
    ax, ay, aw, ah = [float(x) for x in a]
    bx, by, bw, bh = [float(x) for x in b]
    ix = max(0., min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0., min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1.e-12)


def center_error(a, b):
    return float(np.hypot((a[0] + a[2] / 2.) - (b[0] + b[2] / 2.),
                          (a[1] + a[3] / 2.) - (b[1] + b[3] / 2.)))


class RelationTrace:
    def __init__(self, modules):
        self.modules = modules
        self.handles = [module.register_forward_hook(self._hook) for module in modules]
        self.enabled = False
        self.rows = []

    @staticmethod
    def _entropy(prob):
        prob = prob.float().clamp_min(1.e-8)
        return float((-(prob * prob.log()).sum(dim=-1).mean()).detach().cpu())

    def _hook(self, module, args, output):
        if not self.enabled:
            return
        query, source, native = args[:3]
        adjusted, diagnostics = output
        native_mass = native.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(native.dtype).tiny)
        adjusted_mass = adjusted.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(adjusted.dtype).tiny)
        p = native / native_mass
        q = adjusted / adjusted_mass
        self.rows.append({
            'module': next((name for name, item in self.module_names if item is module), 'unknown'),
            'beta': float(module.beta.detach().cpu()),
            'cross_mass_before': float(native_mass.mean().detach().cpu()),
            'cross_mass_after': float(adjusted_mass.mean().detach().cpu()),
            'entropy_before': self._entropy(p),
            'entropy_after': self._entropy(q),
            'top1_changed_fraction': float((p.argmax(-1) != q.argmax(-1)).float().mean().detach().cpu()),
            'conditional_kl_native_to_adjusted': float((p * (p.clamp_min(1.e-8).log() - q.clamp_min(1.e-8).log())).sum(-1).mean().detach().cpu()),
            'relation_bias_abs_mean': float(diagnostics['relation_bias'].abs().mean().cpu()),
            'relation_bias_abs_max': float(diagnostics['relation_bias'].abs().max().cpu()),
        })

    def begin(self, mode):
        self.enabled = True
        self.rows = []
        for module in self.modules:
            module._attribution_mode = mode
            module._attribution_saved_beta = module.beta.detach().clone()
            if mode == 'identity':
                module.beta.data.zero_()
            elif mode.startswith('beta_pos_') or mode.startswith('beta_neg_'):
                sign = -1. if mode.startswith('beta_neg_') else 1.
                value = float(mode.rsplit('_', 1)[-1].replace('p', '.'))
                module.beta.data.fill_(sign * value)

    def end(self, mode):
        if mode != 'relation':
            for module in self.modules:
                module.beta.data.copy_(module._attribution_saved_beta)
        self.enabled = False
        rows, self.rows = self.rows, []
        return rows

    def close(self):
        for handle in self.handles:
            handle.remove()


def mean_or_none(rows, key):
    return None if not rows else float(np.mean([row[key] for row in rows]))


def summarize_trace(rows):
    keys = ['cross_mass_before', 'cross_mass_after', 'entropy_before', 'entropy_after',
            'top1_changed_fraction', 'conditional_kl_native_to_adjusted',
            'relation_bias_abs_mean', 'relation_bias_abs_max']
    return {key: mean_or_none(rows, key) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize-existing', action='store_true')
    args = parser.parse_args()
    if args.finalize_existing:
        cases = [json.loads(path.read_text()) for path in sorted(OUTPUT.glob('case_*.json'))]
        if len(cases) != len(CASES) or any(len(case.get('branches', [])) != len(MODES) for case in cases):
            raise RuntimeError('Incomplete case outputs; cannot finalize')
        started = json.loads((OUTPUT / 'started.json').read_text())
        completed = {'status': 'ATTRIBUTION-COMPLETE', 'cases': cases,
                     'checkpoint_sha256': CHECKPOINT_SHA256,
                     'elapsed_seconds': time.time() - started['start_time'],
                     'finalization_note': 'All case branch outputs existed; finalized after reporting-only constant fix.'}
        write_json(OUTPUT / 'completed.json', completed)
        print(json.dumps({'status': completed['status'], 'cases': len(cases), 'output': str(OUTPUT)}))
        return
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError('Refusing nonempty output: ' + str(OUTPUT))
    if sha(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError('Checkpoint SHA256 mismatch')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')

    from lib.models.layers.cross_mass_preserving_relation import CrossMassPreservingReceiverSourceRelation
    from lib.test.evaluation import get_dataset, Tracker

    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(CHECKPOINT)
    dataset = {sequence.name: sequence for sequence in get_dataset('lasher_test')}
    tracker_wrapper = Tracker('tbsi_track', CONFIG_NAME, 'lasher_test')
    params = tracker_wrapper.get_parameters()
    if Path(params.checkpoint).resolve() != CHECKPOINT.resolve():
        raise RuntimeError('Tracker resolved a different checkpoint: ' + str(params.checkpoint))
    params.debug = 0
    manifest = {'id': 'EG-CROSS-RELATION-ATTRIBUTION-S42-20261029',
                'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': CHECKPOINT_SHA256,
                'config': str(CONFIG), 'config_sha256': sha(CONFIG), 'cases': CASES,
                'modes': ['relation', 'identity'], 'horizon': HORIZON,
                'precision': 'FP32', 'tracker_threads': 0, 'template_update': False,
                'gpu': torch.cuda.get_device_name(0), 'start_time': time.time()}
    write_json(OUTPUT / 'started.json', manifest)
    results = []
    for case in CASES:
        # A fresh tracker per sequence prevents hidden tracker/network state
        # from contaminating the fixed-state prefix contract.
        tracker = tracker_wrapper.create_tracker(params)
        modules = [module for module in tracker.network.modules()
                   if isinstance(module, CrossMassPreservingReceiverSourceRelation)]
        if len(modules) != 6:
            raise RuntimeError('Expected 6 relation modules, got %d' % len(modules))
        trace = RelationTrace(modules)
        trace.module_names = [(name, module) for name, module in tracker.network.named_modules()
                              if isinstance(module, CrossMassPreservingReceiverSourceRelation)]
        try:
            sequence = dataset[case['sequence']]
            target = min(int(case['frame']), len(sequence.frames[0]) - 2)
            tracker.initialize(read_rgbt(sequence, 0), sequence.init_info())
            saved = np.loadtxt(str(PREDICTIONS / (case['sequence'] + '.txt')), ndmin=2)
            for frame_num in range(1, target):
                image = read_rgbt(sequence, frame_num)
                info = sequence.frame_info(frame_num)
                info.update(sequence_name=sequence.name, frame_num=frame_num,
                            gt_bbox=sequence.ground_truth_rect[frame_num])
                prediction = np.asarray(tracker.track(image, info)['target_bbox'], dtype=np.float64).reshape(4)
                if not np.allclose(prediction, saved[frame_num], atol=1.e-5, rtol=1.e-6):
                    raise RuntimeError('Prefix replay diverged at %s frame %d' % (sequence.name, frame_num))
            anchor = {'state': clone(tracker.state), 'frame_id': tracker.frame_id,
                      'prev_search': clone(tracker.prev_search),
                      'prev_temporal_tokens': clone(tracker.prev_temporal_tokens)}
            case_result = {'sequence': sequence.name, 'role': case['role'],
                           'target_frame': target, 'branches': []}
            for mode in MODES:
                tracker.state = clone(anchor['state']); tracker.frame_id = anchor['frame_id']
                tracker.prev_search = clone(anchor['prev_search'])
                tracker.prev_temporal_tokens = clone(anchor['prev_temporal_tokens'])
                trace.begin(mode)
                frames = []
                for frame_num in range(target, min(target + HORIZON, len(sequence.frames[0]))):
                    image = read_rgbt(sequence, frame_num)
                    info = sequence.frame_info(frame_num)
                    info.update(sequence_name=sequence.name, frame_num=frame_num,
                                gt_bbox=sequence.ground_truth_rect[frame_num])
                    prediction = np.asarray(tracker.track(image, info)['target_bbox'], dtype=np.float64).reshape(4)
                    gt = np.asarray(sequence.ground_truth_rect[frame_num], dtype=np.float64).reshape(4)
                    frames.append({'frame': frame_num, 'iou': iou(prediction, gt),
                                   'center_error': center_error(prediction, gt),
                                   'prediction': prediction.tolist(),
                                   'trace': summarize_trace(trace.rows)})
                    trace.rows = []
                if mode == 'relation':
                    beta_values = [float(module.beta.detach().cpu()) for module in modules]
                elif mode == 'identity':
                    beta_values = [0.0 for module in modules]
                else:
                    sign = -1. if mode.startswith('beta_neg_') else 1.
                    value = float(mode.rsplit('_', 1)[-1].replace('p', '.'))
                    beta_values = [sign * value for module in modules]
                branch = {'mode': mode, 'beta_values': beta_values,
                          'frames': frames}
                branch['mean_iou'] = float(np.mean([frame['iou'] for frame in frames]))
                branch['mean_center_error'] = float(np.mean([frame['center_error'] for frame in frames]))
                case_result['branches'].append(branch)
            results.append(case_result)
            write_json(OUTPUT / ('case_' + sequence.name + '.json'), case_result)
        finally:
            trace.close()
    completed = {'status': 'ATTRIBUTION-COMPLETE', 'cases': results,
                 'checkpoint_sha256': CHECKPOINT_SHA256, 'elapsed_seconds': time.time() - manifest['start_time']}
    write_json(OUTPUT / 'completed.json', completed)
    print(json.dumps({'status': completed['status'], 'cases': len(results), 'output': str(OUTPUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
