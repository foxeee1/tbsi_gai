#!/usr/bin/env python
"""Inference-only message-recovery attribution probe for RGB Gaussian blur.

The probe uses forward hooks and never changes weights or the canonical config.
For the first TBSI layer, ca_s2t_v2f.attn_reshape returns
[template_message, RGB_search_tokens]. Only the first 64 tokens are eligible
for replacement; the search-token suffix is always retained from the blurred
run. A clean-message replacement is diagnostic-only and is not a deployable
method.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.tracker.tbsi_track import TBSITrack
from lib.test.utils.params import TrackerParams
from synthetic_joint_interference_v1 import image_loader, load_train_sequence, rng_snapshot, restore_rng
from single_modality_blur_v2 import box_iou, degrade_rgb


EXP = 'tbsi_message_recovery_v1'
CONDITIONS = ('F', 'B', 'B_RGB_MESSAGE_RECOVERY', 'B_RAW_MESSAGE_HOOK',
              'B_TIR_PATH_CONTROL', 'B_RANDOM_MESSAGE_CONTROL')


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


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


class HookProbe(object):
    def __init__(self, target_module, tir_module, event_frame, clean_target, clean_tir, action, permutation):
        self.target_module = target_module
        self.tir_module = tir_module
        self.event_frame = event_frame
        self.clean_target = clean_target
        self.clean_tir = clean_tir
        self.action = action
        self.permutation = permutation
        self.calls_target = 0
        self.calls_tir = 0
        self.target_shape = None
        self.tir_shape = None
        self.replacement_hash = None
        self.handles = []

    def _replace(self, output, clean):
        if not torch.is_tensor(output) or not torch.is_tensor(clean):
            raise TypeError('message hook expects tensor output')
        if output.ndim != 3 or clean.shape != output.shape or output.shape[1] < 64:
            raise ValueError('unexpected attention output shape {} / {}'.format(tuple(output.shape), tuple(clean.shape)))
        replacement = clean.to(device=output.device, dtype=output.dtype)
        if self.action == 'recovery':
            message = replacement[:, :64, :]
        elif self.action == 'random':
            message = replacement[:, self.permutation, :]
        elif self.action == 'tir_control':
            message = replacement[:, :64, :]
        else:
            return output
        result = output.clone()
        result[:, :64, :] = message
        self.replacement_hash = tensor_hash(message)
        return result

    def target_hook(self, module, inputs, output):
        self.calls_target += 1
        self.target_shape = tuple(output.shape)
        if self.calls_target == self.event_frame and self.action in ('recovery', 'random'):
            return self._replace(output, self.clean_target)
        return output

    def tir_hook(self, module, inputs, output):
        self.calls_tir += 1
        self.tir_shape = tuple(output.shape)
        if self.calls_tir == self.event_frame and self.action == 'tir_control':
            return self._replace(output, self.clean_tir)
        return output

    def install(self):
        self.handles.append(self.target_module.register_forward_hook(self.target_hook))
        self.handles.append(self.tir_module.register_forward_hook(self.tir_hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def run_track(tracker, rgb_frames, tir_frames, gt_boxes, event, rgb_event, rng, horizon, probe=None):
    if probe is not None:
        probe.install()
    try:
        restore_rng(rng)
        tracker.initialize(np.concatenate([rgb_frames[0], tir_frames[0]], axis=2), {'init_bbox': gt_boxes[0]})
        outputs = []
        for frame_num in range(1, event['frame'] + horizon + 1):
            rgb = rgb_frames[frame_num]
            if frame_num == event['frame'] and rgb_event is not None:
                rgb = rgb_event
            out = tracker.track(np.concatenate([rgb, tir_frames[frame_num]], axis=2),
                                {'gt_bbox': gt_boxes[frame_num], 'sequence_name': event['seq'],
                                 'frame_num': frame_num})
            if frame_num >= event['frame']:
                pred = [float(x) for x in out['target_bbox']]
                outputs.append({'offset': frame_num - event['frame'], 'pred': pred,
                                'iou': box_iou(pred, gt_boxes[frame_num]),
                                'pred_hash': hashlib.sha256(json.dumps(pred).encode()).hexdigest()})
        return outputs
    finally:
        if probe is not None:
            probe.close()


def get_message_cache(tracker, rgb_frames, tir_frames, gt_boxes, event, rng, horizon):
    records = {'target': [], 'tir': []}
    layer = tracker.network.backbone.tbsi_layers[0]

    def capture_target(module, inputs, output):
        records['target'].append(output.detach().clone())
        return output

    def capture_tir(module, inputs, output):
        records['tir'].append(output.detach().clone())
        return output

    h1 = layer.ca_s2t_v2f.attn_reshape.register_forward_hook(capture_target)
    h2 = layer.ca_s2t_i2f.attn_reshape.register_forward_hook(capture_tir)
    try:
        run_track(tracker, rgb_frames, tir_frames, gt_boxes, event, None, rng, horizon)
    finally:
        h1.remove(); h2.remove()
    expected = event['frame']
    if len(records['target']) < expected or len(records['tir']) < expected:
        raise RuntimeError('hook call count {} / {} below event frame {}'.format(
            len(records['target']), len(records['tir']), expected))
    return records['target'][expected - 1], records['tir'][expected - 1], len(records['target']), len(records['tir'])


def replay_event(tracker, event, sequence, rng, horizon):
    rgb_frames = [image_loader(path) for path in sequence.frames[0]]
    tir_frames = [image_loader(path) for path in sequence.frames[1]]
    n = min(len(rgb_frames), len(tir_frames), len(sequence.ground_truth_rect))
    rgb_frames, tir_frames = rgb_frames[:n], tir_frames[:n]
    gt_boxes = [[float(x) for x in box] for box in sequence.ground_truth_rect[:n]]
    if event['frame'] + horizon >= n:
        raise RuntimeError('insufficient future frames for {}'.format(event['event_id']))
    rgb_bad = degrade_rgb(rgb_frames[event['frame']])
    return rgb_frames, tir_frames, gt_boxes, rgb_bad


def clone_cache(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return type(value)(clone_cache(item) for item in value)
    if torch.is_tensor(value):
        return value.detach().clone()
    return value


def snapshot_tracker(tracker):
    return {'state': list(tracker.state), 'frame_id': tracker.frame_id,
            'prev_search': clone_cache(tracker.prev_search),
            'prev_temporal_tokens': clone_cache(tracker.prev_temporal_tokens)}


def restore_tracker(tracker, snapshot):
    tracker.state = list(snapshot['state'])
    tracker.frame_id = snapshot['frame_id']
    tracker.prev_search = clone_cache(snapshot['prev_search'])
    tracker.prev_temporal_tokens = clone_cache(snapshot['prev_temporal_tokens'])


def run_from_snapshot(tracker, rgb_frames, tir_frames, gt_boxes, event, rgb_bad, rng,
                      horizon, probe, condition):
    restore_rng(rng)
    restore_tracker(tracker, event['_snapshot'])
    probe.install()
    try:
        outputs = []
        for frame_num in range(event['frame'], event['frame'] + horizon + 1):
            rgb = rgb_bad if frame_num == event['frame'] and condition != 'F' else rgb_frames[frame_num]
            out = tracker.track(np.concatenate([rgb, tir_frames[frame_num]], axis=2),
                                {'gt_bbox': gt_boxes[frame_num], 'sequence_name': event['seq'],
                                 'frame_num': frame_num})
            pred = [float(x) for x in out['target_bbox']]
            outputs.append({'offset': frame_num - event['frame'], 'pred': pred,
                            'iou': box_iou(pred, gt_boxes[frame_num]),
                            'pred_hash': hashlib.sha256(json.dumps(pred).encode()).hexdigest()})
        return outputs
    finally:
        probe.close()


def run_fast(args):
    """Group by sequence: one clean prefix, then isolated event suffixes."""
    manifest = json.load(open(args.manifest))
    events = manifest['events'][:8] if args.mode == 'smoke' else manifest['events']
    settings = env_settings()
    grouped = {}
    for event in events:
        grouped.setdefault(event['seq'], []).append(event)
    params, config_path = build_params(args.config, args.checkpoint)
    tracker = TBSITrack(params, EXP)
    rows = []
    audit = []
    for seq_name in sorted(grouped):
        sequence = load_train_sequence(settings.lasher_path, seq_name)
        rgb_frames = [image_loader(path) for path in sequence.frames[0]]
        tir_frames = [image_loader(path) for path in sequence.frames[1]]
        n = min(len(rgb_frames), len(tir_frames), len(sequence.ground_truth_rect))
        rgb_frames, tir_frames = rgb_frames[:n], tir_frames[:n]
        gt_boxes = [[float(x) for x in box] for box in sequence.ground_truth_rect[:n]]
        seq_events = sorted(grouped[seq_name], key=lambda item: item['frame'])
        if seq_events[-1]['frame'] + args.horizon >= n:
            raise RuntimeError('insufficient future frames for {}'.format(seq_name))
        tracker.initialize(np.concatenate([rgb_frames[0], tir_frames[0]], axis=2), {'init_bbox': gt_boxes[0]})
        target_cache, tir_cache, snapshots, rng_cache = {}, {}, {}, {}
        target_calls = 0
        tir_calls = 0
        event_frames = set(event['frame'] for event in seq_events)
        layer = tracker.network.backbone.tbsi_layers[0]

        def capture_target(module, inputs, output):
            nonlocal target_calls
            target_calls += 1
            if target_calls in event_frames:
                target_cache[target_calls] = output.detach().clone()
            return output

        def capture_tir(module, inputs, output):
            nonlocal tir_calls
            tir_calls += 1
            if tir_calls in event_frames:
                tir_cache[tir_calls] = output.detach().clone()
            return output

        h1 = layer.ca_s2t_v2f.attn_reshape.register_forward_hook(capture_target)
        h2 = layer.ca_s2t_i2f.attn_reshape.register_forward_hook(capture_tir)
        try:
            for frame_num in range(1, seq_events[-1]['frame'] + 1):
                if frame_num in event_frames:
                    snapshots[frame_num] = snapshot_tracker(tracker)
                    rng_cache[frame_num] = rng_snapshot()
                tracker.track(np.concatenate([rgb_frames[frame_num], tir_frames[frame_num]], axis=2),
                              {'gt_bbox': gt_boxes[frame_num], 'sequence_name': seq_name,
                               'frame_num': frame_num})
        finally:
            h1.remove(); h2.remove()
        if set(target_cache) != event_frames or set(tir_cache) != event_frames:
            raise RuntimeError('clean hook cache incomplete for {}'.format(seq_name))
        for event in seq_events:
            event = dict(event)
            event['_snapshot'] = snapshots[event['frame']]
            clean_target, clean_tir = target_cache[event['frame']], tir_cache[event['frame']]
            generator = torch.Generator(); generator.manual_seed(1000003 + int(event['event_id'].split('-')[-1]))
            permutation = torch.randperm(64, generator=generator).to(clean_target.device)
            target_module = layer.ca_s2t_v2f.attn_reshape
            tir_module = layer.ca_s2t_i2f.attn_reshape
            rgb_bad = degrade_rgb(rgb_frames[event['frame']])
            outputs, hook_meta = {}, {}
            for condition in CONDITIONS:
                action = {'F': 'none', 'B': 'none', 'B_RGB_MESSAGE_RECOVERY': 'recovery',
                          'B_RAW_MESSAGE_HOOK': 'raw', 'B_TIR_PATH_CONTROL': 'tir_control',
                          'B_RANDOM_MESSAGE_CONTROL': 'random'}[condition]
                probe = HookProbe(target_module, tir_module, 1, clean_target, clean_tir, action, permutation)
                outputs[condition] = run_from_snapshot(tracker, rgb_frames, tir_frames, gt_boxes, event,
                                                       rgb_bad, rng_cache[event['frame']], args.horizon,
                                                       probe, condition)
                hook_meta[condition] = {'target_calls': probe.calls_target, 'tir_calls': probe.calls_tir,
                                        'target_shape': probe.target_shape, 'tir_shape': probe.tir_shape,
                                        'replacement_hash': probe.replacement_hash}
            b, raw = outputs['B'], outputs['B_RAW_MESSAGE_HOOK']
            audit_item = {'event_id': event['event_id'], 'seq': seq_name, 'frame': event['frame'],
                          'clean_target_shape': list(clean_target.shape), 'clean_tir_shape': list(clean_tir.shape),
                          'clean_target_calls': target_calls, 'clean_tir_calls': tir_calls,
                          'raw_hook_pred_hash_equal_B': [a['pred_hash'] == z['pred_hash'] for a, z in zip(raw, b)],
                          'raw_hook_iou_equal_B': [abs(a['iou'] - z['iou']) <= 1e-12 for a, z in zip(raw, b)],
                          'all_target_calls_equal': len(set(hook_meta[c]['target_calls'] for c in CONDITIONS)) == 1,
                          'all_tir_calls_equal': len(set(hook_meta[c]['tir_calls'] for c in CONDITIONS)) == 1,
                          'hook_meta': hook_meta}
            audit.append(audit_item)
            if not all(audit_item['raw_hook_pred_hash_equal_B']) or not all(audit_item['raw_hook_iou_equal_B']):
                raise RuntimeError('raw hook parity failed for {}'.format(event['event_id']))
            for condition, points in outputs.items():
                for point in points:
                    rows.append({'event_id': event['event_id'], 'seq': event['seq'], 'frame': event['frame'],
                                 'condition': condition, 'offset': point['offset'], 'iou': point['iou'],
                                 'error': 1.0 - point['iou'], 'pred_hash': point['pred_hash'], 'status': 'OK'})
        print('processed sequence {} ({}/{})'.format(seq_name, len(audit), len(events)), flush=True)
    out = os.path.abspath(args.output_dir); os.makedirs(out, exist_ok=True)
    fields = ['event_id', 'seq', 'frame', 'condition', 'offset', 'iou', 'error', 'pred_hash', 'status']
    with open(os.path.join(out, 'message_recovery_event_results.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    gate = all(all(item['raw_hook_pred_hash_equal_B']) and all(item['raw_hook_iou_equal_B']) and
               item['all_target_calls_equal'] and item['all_tir_calls_equal'] for item in audit)
    with open(os.path.join(out, 'hook_audit.json'), 'w') as handle:
        json.dump({'experiment': EXP, 'mode': args.mode, 'gate_raw_hook_parity': gate,
                   'events': audit, 'target_path': 'tbsi_layers[0].ca_s2t_v2f.attn_reshape[:,:64,:]',
                   'tir_control_path': 'tbsi_layers[0].ca_s2t_i2f.attn_reshape[:,:64,:]'}, handle, indent=2)
    with open(os.path.join(out, 'metadata.json'), 'w') as handle:
        json.dump({'experiment': EXP, 'mode': args.mode, 'manifest': os.path.abspath(args.manifest),
                   'n_events': len(events), 'n_rows': len(rows), 'horizon': args.horizon,
                   'conditions': CONDITIONS, 'config': args.config,
                   'config_sha256': hashlib.sha256(open(config_path, 'rb').read()).hexdigest(),
                   'checkpoint': args.checkpoint,
                   'checkpoint_sha256': hashlib.sha256(open(args.checkpoint, 'rb').read()).hexdigest(),
                   'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                   'status': 'COMPLETED' if gate else 'FAILED_HOOK_AUDIT'}, handle, indent=2)
    if not gate:
        raise RuntimeError('hook audit gate failed; batch must not proceed')
    print('COMPLETED {} gate={}'.format(args.mode, gate), flush=True)


def run(args):
    manifest = json.load(open(args.manifest))
    events = manifest['events'][:8] if args.mode == 'smoke' else manifest['events']
    settings = env_settings()
    sequences = {name: load_train_sequence(settings.lasher_path, name)
                 for name in sorted(set(event['seq'] for event in events))}
    params, config_path = build_params(args.config, args.checkpoint)
    tracker = TBSITrack(params, EXP)
    rng = rng_snapshot()
    rows = []
    audit = []
    for index, event in enumerate(events):
        rgb_frames, tir_frames, gt_boxes, rgb_bad = replay_event(tracker, event, sequences[event['seq']], rng, args.horizon)
        clean_target, clean_tir, clean_target_calls, clean_tir_calls = get_message_cache(
            tracker, rgb_frames, tir_frames, gt_boxes, event, rng, args.horizon)
        target_module = tracker.network.backbone.tbsi_layers[0].ca_s2t_v2f.attn_reshape
        tir_module = tracker.network.backbone.tbsi_layers[0].ca_s2t_i2f.attn_reshape
        generator = torch.Generator()
        generator.manual_seed(1000003 + index)
        permutation = torch.randperm(64, generator=generator).to(clean_target.device)
        outputs = {}
        hook_meta = {}
        for condition in CONDITIONS:
            action = {'F': 'none', 'B': 'none', 'B_RGB_MESSAGE_RECOVERY': 'recovery',
                      'B_RAW_MESSAGE_HOOK': 'raw', 'B_TIR_PATH_CONTROL': 'tir_control',
                      'B_RANDOM_MESSAGE_CONTROL': 'random'}[condition]
            probe = HookProbe(target_module, tir_module, event['frame'], clean_target, clean_tir, action, permutation)
            rgb_input = None if condition == 'F' else rgb_bad
            outputs[condition] = run_track(tracker, rgb_frames, tir_frames, gt_boxes, event, rgb_input, rng, args.horizon, probe)
            hook_meta[condition] = {'target_calls': probe.calls_target, 'tir_calls': probe.calls_tir,
                                    'target_shape': probe.target_shape, 'tir_shape': probe.tir_shape,
                                    'replacement_hash': probe.replacement_hash}
        b = outputs['B']
        raw = outputs['B_RAW_MESSAGE_HOOK']
        audit_item = {'event_id': event['event_id'], 'seq': event['seq'], 'frame': event['frame'],
                      'clean_target_shape': list(clean_target.shape), 'clean_tir_shape': list(clean_tir.shape),
                      'clean_target_calls': clean_target_calls, 'clean_tir_calls': clean_tir_calls,
                      'raw_hook_pred_hash_equal_B': [a['pred_hash'] == b['pred_hash'] for a, b in zip(raw, b)],
                      'raw_hook_iou_equal_B': [abs(a['iou'] - z['iou']) <= 1e-12 for a, z in zip(raw, b)],
                      'all_target_calls_equal': len(set(hook_meta[c]['target_calls'] for c in CONDITIONS)) == 1,
                      'all_tir_calls_equal': len(set(hook_meta[c]['tir_calls'] for c in CONDITIONS)) == 1,
                      'hook_meta': hook_meta}
        audit.append(audit_item)
        if not all(audit_item['raw_hook_pred_hash_equal_B']) or not all(audit_item['raw_hook_iou_equal_B']):
            raise RuntimeError('raw hook parity failed for {}'.format(event['event_id']))
        for condition, points in outputs.items():
            for point in points:
                rows.append({'event_id': event['event_id'], 'seq': event['seq'], 'frame': event['frame'],
                             'condition': condition, 'offset': point['offset'], 'iou': point['iou'],
                             'error': 1.0 - point['iou'], 'pred_hash': point['pred_hash'], 'status': 'OK'})
        print('processed {}/{} events'.format(index + 1, len(events)), flush=True)
    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    fields = ['event_id', 'seq', 'frame', 'condition', 'offset', 'iou', 'error', 'pred_hash', 'status']
    with open(os.path.join(out, 'message_recovery_event_results.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    gate = all(all(item['raw_hook_pred_hash_equal_B']) and all(item['raw_hook_iou_equal_B']) and
               item['all_target_calls_equal'] and item['all_tir_calls_equal'] for item in audit)
    with open(os.path.join(out, 'hook_audit.json'), 'w') as handle:
        json.dump({'experiment': EXP, 'mode': args.mode, 'gate_raw_hook_parity': gate,
                   'events': audit, 'target_path': 'tbsi_layers[0].ca_s2t_v2f.attn_reshape[:,:64,:]',
                   'tir_control_path': 'tbsi_layers[0].ca_s2t_i2f.attn_reshape[:,:64,:]'}, handle, indent=2)
    with open(os.path.join(out, 'metadata.json'), 'w') as handle:
        json.dump({'experiment': EXP, 'mode': args.mode, 'manifest': os.path.abspath(args.manifest),
                   'n_events': len(events), 'n_rows': len(rows), 'horizon': args.horizon,
                   'conditions': CONDITIONS, 'config': args.config,
                   'config_sha256': hashlib.sha256(open(config_path, 'rb').read()).hexdigest(),
                   'checkpoint': args.checkpoint,
                   'checkpoint_sha256': hashlib.sha256(open(args.checkpoint, 'rb').read()).hexdigest(),
                   'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                   'status': 'COMPLETED' if gate else 'FAILED_HOOK_AUDIT'}, handle, indent=2)
    if not gate:
        raise RuntimeError('hook audit gate failed; batch must not proceed')
    print('COMPLETED {} gate={}'.format(args.mode, gate), flush=True)


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
    lock = os.path.join(os.path.abspath(args.output_dir), '.lock')
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    if os.path.exists(lock):
        raise RuntimeError('lock exists: {}'.format(lock))
    open(lock, 'w').write(str(os.getpid()))
    try:
        run_fast(args)
    finally:
        if os.path.exists(lock):
            os.unlink(lock)


if __name__ == '__main__':
    main()
