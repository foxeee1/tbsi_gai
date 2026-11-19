#!/usr/bin/env python
"""Inference-only RGB/TIR search-branch scaling attribution probe.

The intervention is applied at the TBSILayer boundary, where the model calls
``forward(x_v, x_i, lens_z, ...)``.  The implementation records the actual
tensor shapes and norms instead of assuming token positions from an attention
module.  It never changes model weights or the canonical YAML.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.tracker.tbsi_track import TBSITrack
from lib.test.utils.params import TrackerParams
from synthetic_joint_interference_v1 import (box_iou, image_loader, load_train_sequence,
                                              restore_rng, rng_snapshot)
from single_modality_blur_v2 import degrade_rgb


EXP = 'tbsi_branch_scale_probe_v1'
CONDITIONS = ('F', 'B', 'B-R1.00', 'B-R0.75', 'B-R0.50', 'B-R0.25',
              'B-R0.00', 'B-T0.50', 'F-R1.00', 'F-R0.50', 'F-T0.50')


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def pred_hash(pred):
    return sha256_bytes(json.dumps([float(x) for x in pred]).encode())


def build_params(config_name, checkpoint):
    config_path = os.path.join(ROOT, 'experiments/tbsi_track', config_name + '.yaml')
    update_config_from_file(config_path)
    params = TrackerParams()
    params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = checkpoint
    params.save_all_boxes = False
    params.debug = 0
    return params, config_path


def clone(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return type(value)(clone(item) for item in value)
    if torch.is_tensor(value):
        return value.detach().clone()
    return value


def snapshot_tracker(tracker):
    return {'state': list(tracker.state), 'frame_id': tracker.frame_id,
            'prev_search': clone(tracker.prev_search),
            'prev_temporal_tokens': clone(tracker.prev_temporal_tokens)}


def restore_tracker(tracker, snapshot):
    tracker.state = list(snapshot['state'])
    tracker.frame_id = snapshot['frame_id']
    tracker.prev_search = clone(snapshot['prev_search'])
    tracker.prev_temporal_tokens = clone(snapshot['prev_temporal_tokens'])


class BranchScaleProbe(object):
    def __init__(self, backbone, rgb_scale, tir_scale):
        self.rgb_scale = float(rgb_scale)
        self.tir_scale = float(tir_scale)
        self.records = []
        self.handles = []
        self.layers = list(backbone.tbsi_layers)

    @staticmethod
    def _norm(value):
        return float(value.detach().float().norm().cpu())

    def _pre_hook(self, index, module, inputs):
        if len(inputs) < 3 or not torch.is_tensor(inputs[0]) or not torch.is_tensor(inputs[1]):
            raise RuntimeError('unexpected TBSILayer inputs at index {}'.format(index))
        x_rgb, x_tir, lens_z = inputs[:3]
        if not isinstance(lens_z, int):
            lens_z = int(lens_z)
        if x_rgb.ndim != 3 or x_tir.ndim != 3 or x_rgb.shape != x_tir.shape:
            raise RuntimeError('unexpected branch shapes {} / {}'.format(tuple(x_rgb.shape), tuple(x_tir.shape)))
        record = {'module_name': 'tbsi_layers[{}]'.format(index),
                  'input_shape_rgb': list(x_rgb.shape), 'input_shape_tir': list(x_tir.shape),
                  'lens_z': lens_z, 'rgb_branch_norm': self._norm(x_rgb[:, lens_z:, :]),
                  'tir_branch_norm': self._norm(x_tir[:, lens_z:, :]),
                  'rgb_scaled_norm': self._norm(x_rgb[:, lens_z:, :] * self.rgb_scale),
                  'tir_scaled_norm': self._norm(x_tir[:, lens_z:, :] * self.tir_scale)}
        self.records.append(record)
        if self.rgb_scale == 1.0 and self.tir_scale == 1.0:
            return inputs
        rgb_new = x_rgb.clone()
        tir_new = x_tir.clone()
        rgb_new[:, lens_z:, :] *= self.rgb_scale
        tir_new[:, lens_z:, :] *= self.tir_scale
        return (rgb_new, tir_new) + tuple(inputs[2:])

    def _post_hook(self, index, module, inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError('unexpected TBSILayer output at index {}'.format(index))
        record = self.records[-1]
        x_rgb, x_tir = output[:2]
        record.update({'output_shape_rgb': list(x_rgb.shape), 'output_shape_tir': list(x_tir.shape),
                       'fusion_output_norm': self._norm(torch.cat([x_rgb, x_tir], dim=1))})
        return output

    def install(self):
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(
                lambda module, inputs, index=index: self._pre_hook(index, module, inputs)))
            self.handles.append(layer.register_forward_hook(
                lambda module, inputs, output, index=index: self._post_hook(index, module, inputs, output)))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def summary(self):
        by_module = {}
        for record in self.records:
            by_module.setdefault(record['module_name'], []).append(record)
        return {'call_count': {name: len(items) for name, items in by_module.items()},
                'records': self.records}


def run_suffix(tracker, rgb_frames, tir_frames, gt_boxes, event, rgb_bad, rng,
               horizon, condition, probe):
    restore_rng(rng)
    restore_tracker(tracker, event['_snapshot'])
    if probe is not None:
        probe.install()
    try:
        outputs = []
        for frame_num in range(event['frame'], event['frame'] + horizon + 1):
            rgb = rgb_bad if frame_num == event['frame'] and condition.startswith('B') else rgb_frames[frame_num]
            out = tracker.track(np.concatenate([rgb, tir_frames[frame_num]], axis=2),
                                {'gt_bbox': gt_boxes[frame_num], 'sequence_name': event['seq'],
                                 'frame_num': frame_num})
            pred = [float(x) for x in out['target_bbox']]
            outputs.append({'offset': frame_num - event['frame'], 'iou': box_iou(pred, gt_boxes[frame_num]),
                            'pred': pred, 'pred_sha256': pred_hash(pred)})
        return outputs
    finally:
        if probe is not None:
            probe.close()


def scales(condition):
    return {'F': (1.0, 1.0), 'B': (1.0, 1.0), 'B-R1.00': (1.0, 1.0),
            'B-R0.75': (0.75, 1.0), 'B-R0.50': (0.50, 1.0),
            'B-R0.25': (0.25, 1.0), 'B-R0.00': (0.0, 1.0),
            'B-T0.50': (1.0, 0.50), 'F-R1.00': (1.0, 1.0),
            'F-R0.50': (0.50, 1.0), 'F-T0.50': (1.0, 0.50)}[condition]


def run(args):
    manifest = json.load(open(args.manifest))
    events = manifest['events'][:8] if args.mode == 'smoke' else manifest['events']
    settings = env_settings()
    grouped = {}
    for event in events:
        grouped.setdefault(event['seq'], []).append(event)
    params, config_path = build_params(args.config, args.checkpoint)
    tracker = TBSITrack(params, EXP)
    rows, audits = [], []
    for seq_name in sorted(grouped):
        sequence = load_train_sequence(settings.lasher_path, seq_name)
        rgb_frames = [image_loader(path) for path in sequence.frames[0]]
        tir_frames = [image_loader(path) for path in sequence.frames[1]]
        n = min(len(rgb_frames), len(tir_frames), len(sequence.ground_truth_rect))
        gt_boxes = [[float(x) for x in box] for box in sequence.ground_truth_rect[:n]]
        seq_events = sorted(grouped[seq_name], key=lambda item: item['frame'])
        if seq_events[-1]['frame'] + args.horizon >= n:
            raise RuntimeError('insufficient future frames for {}'.format(seq_name))
        tracker.initialize(np.concatenate([rgb_frames[0], tir_frames[0]], axis=2), {'init_bbox': gt_boxes[0]})
        snapshots, rng_cache = {}, {}
        for frame_num in range(1, seq_events[-1]['frame'] + 1):
            if frame_num in set(event['frame'] for event in seq_events):
                snapshots[frame_num] = snapshot_tracker(tracker)
                rng_cache[frame_num] = rng_snapshot()
            tracker.track(np.concatenate([rgb_frames[frame_num], tir_frames[frame_num]], axis=2),
                          {'gt_bbox': gt_boxes[frame_num], 'sequence_name': seq_name, 'frame_num': frame_num})
        for event0 in seq_events:
            event = dict(event0)
            event['_snapshot'] = snapshots[event['frame']]
            rgb_bad = degrade_rgb(rgb_frames[event['frame']])
            outputs, hook_meta = {}, {}
            for condition in CONDITIONS:
                rgb_scale, tir_scale = scales(condition)
                probe = BranchScaleProbe(tracker.network.backbone, rgb_scale, tir_scale)
                outputs[condition] = run_suffix(tracker, rgb_frames, tir_frames, gt_boxes, event,
                                                rgb_bad, rng_cache[event['frame']], args.horizon,
                                                condition, probe if condition != 'B' else None)
                hook_meta[condition] = probe.summary() if condition != 'B' else {'call_count': {}, 'records': []}
                for point in outputs[condition]:
                    rows.append({'event_id': event['event_id'], 'seq': seq_name, 'frame': event['frame'],
                                 'condition': condition, 'offset': point['offset'], 'iou': point['iou'],
                                 'error': 1.0 - point['iou'], 'pred_sha256': point['pred_sha256'], 'status': 'OK'})
            b, br1 = outputs['B'], outputs['B-R1.00']
            f, fr1 = outputs['F'], outputs['F-R1.00']
            audits.append({'event_id': event['event_id'], 'seq': seq_name, 'frame': event['frame'],
                           'raw_hook_parity_pred': [a['pred_sha256'] == z['pred_sha256'] for a, z in zip(b, br1)],
                           'raw_hook_parity_iou': [abs(a['iou'] - z['iou']) <= 1e-12 for a, z in zip(b, br1)],
                           'clean_r1_reference_pred': [a['pred_sha256'] == z['pred_sha256'] for a, z in zip(f, fr1)],
                           'clean_r1_reference_iou': [abs(a['iou'] - z['iou']) <= 1e-12 for a, z in zip(f, fr1)],
                           'hook_meta': hook_meta})
        print('processed sequence {} ({}/{})'.format(seq_name, len(audits), len(events)), flush=True)
    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'branch_scale_event_results.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['event_id', 'seq', 'frame', 'condition', 'offset', 'iou', 'error', 'pred_sha256', 'status'])
        writer.writeheader(); writer.writerows(rows)
    parity = all(all(item['raw_hook_parity_pred']) and all(item['raw_hook_parity_iou']) for item in audits)
    with open(os.path.join(out, 'hook_audit.json'), 'w') as handle:
        json.dump({'experiment': EXP, 'mode': args.mode, 'gate_raw_hook_parity': parity,
                   'branch_contract': 'TBSILayer.forward x_v[:,lens_z:] = RGB search; x_i[:,lens_z:] = TIR search',
                   'events': audits}, handle, indent=2)
    with open(os.path.join(out, 'metadata.json'), 'w') as handle:
        json.dump({'experiment': EXP, 'mode': args.mode, 'manifest': os.path.abspath(args.manifest),
                   'n_events': len(events), 'n_rows': len(rows), 'horizon': args.horizon,
                   'conditions': CONDITIONS, 'config': args.config,
                   'config_sha256': hashlib.sha256(open(config_path, 'rb').read()).hexdigest(),
                   'checkpoint': args.checkpoint,
                   'checkpoint_sha256': hashlib.sha256(open(args.checkpoint, 'rb').read()).hexdigest(),
                   'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                   'status': 'COMPLETED' if parity else 'FAILED_HOOK_AUDIT'}, handle, indent=2)
    if not parity:
        raise RuntimeError('raw branch-scale hook parity failed; batch must not proceed')
    print('COMPLETED {} gate={}'.format(args.mode, parity), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('smoke', 'batch'), default='smoke')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--config', default='v1.0.0-单卡3090-48G-等效全局128基线')
    ap.add_argument('--checkpoint', default=os.path.join(ROOT, 'output/experiments/v1.0.0-单卡3090-48G-等效全局128基线/checkpoints/TBSITrack_ep0015.pth.tar'))
    ap.add_argument('--horizon', type=int, default=5)
    ap.add_argument('--min-free-gb', type=float, default=8.0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if shutil.disk_usage(ROOT).free / 2 ** 30 < args.min_free_gb:
        raise RuntimeError('insufficient free disk')
    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    lock = os.path.join(out, '.lock')
    if os.path.exists(lock):
        raise RuntimeError('lock exists: {}'.format(lock))
    with open(lock, 'w') as handle:
        handle.write(str(os.getpid()))
    try:
        run(args)
    finally:
        if os.path.exists(lock):
            os.unlink(lock)


if __name__ == '__main__':
    main()
