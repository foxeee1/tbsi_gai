#!/usr/bin/env python
"""Inference-only local-modality stability probe.

This diagnostic replays fixed blur-v2 events from the same tracker/RNG
snapshot under four input conditions and records first-layer RGB->template
message changes together with IoU changes.  It never trains or edits a
configuration.
"""
from __future__ import print_function

import argparse
import csv
import copy
import hashlib
import json
import os
import random
import shutil
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.tracker.tbsi_track import TBSITrack
from lib.test.utils.params import TrackerParams
from synthetic_joint_interference_v1 import image_loader, load_train_sequence, rng_snapshot, restore_rng
from single_modality_blur_v2 import box_iou

EXP = 'isf_stability_probe_v1'
CONDITIONS = ('00', 'R_target_blur', 'T_target_blur', 'R_background_blur')
CANONICAL = 'v1.0.0-单卡3090-48G-等效全局128基线'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def clamp_box(box, width, height):
    x, y, w, h = [int(round(float(v))) for v in box]
    x0 = max(0, min(x, width - 1))
    y0 = max(0, min(y, height - 1))
    x1 = max(x0 + 1, min(x + max(1, w), width))
    y1 = max(y0 + 1, min(y + max(1, h), height))
    return x0, y0, x1, y1


def box_blur(image, box, sigma=3.0):
    out = image.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = clamp_box(box, w, h)
    out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (0, 0), sigmaX=sigma, sigmaY=sigma)
    return out


def choose_background_box(gt_box, image_shape, seed):
    h, w = image_shape[:2]
    gx, gy, gw, gh = [int(round(float(v))) for v in gt_box]
    rng = np.random.RandomState(seed)
    for _ in range(100):
        x = int(rng.randint(0, max(1, w - max(2, gw))))
        y = int(rng.randint(0, max(1, h - max(2, gh))))
        candidate = [x, y, max(2, gw), max(2, gh)]
        if box_iou(candidate, gt_box) <= 0.01:
            return candidate
    return [max(0, w - max(2, gw) - 1), max(0, h - max(2, gh) - 1), max(2, gw), max(2, gh)]


def condition_images(rgb, tir, gt_box, condition, seed):
    rgb_out, tir_out = rgb.copy(), tir.copy()
    if condition == '00':
        return rgb_out, tir_out, 'none', None
    if condition == 'R_target_blur':
        return box_blur(rgb_out, gt_box), tir_out, 'target', None
    if condition == 'T_target_blur':
        return rgb_out, box_blur(tir_out, gt_box), 'target', None
    if condition == 'R_background_blur':
        bg = choose_background_box(gt_box, rgb_out.shape, seed)
        return box_blur(rgb_out, bg), tir_out, 'background', bg
    raise ValueError('unknown condition {}'.format(condition))


def build_params(config_name, checkpoint):
    config_path = os.path.join(ROOT, 'experiments', 'tbsi_track', config_name + '.yaml')
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


def clone_cache(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return type(value)(clone_cache(item) for item in value)
    if torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)


def snapshot_tracker(tracker):
    return {
        'state': list(tracker.state),
        'frame_id': tracker.frame_id,
        'prev_search': clone_cache(tracker.prev_search),
        'prev_temporal_tokens': clone_cache(tracker.prev_temporal_tokens),
    }


def restore_tracker(tracker, snapshot):
    tracker.state = list(snapshot['state'])
    tracker.frame_id = snapshot['frame_id']
    tracker.prev_search = clone_cache(snapshot['prev_search'])
    tracker.prev_temporal_tokens = clone_cache(snapshot['prev_temporal_tokens'])


class MessageCapture(object):
    def __init__(self, module, call_index):
        self.calls = []
        self.call_index = int(call_index)
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not torch.is_tensor(output) or output.ndim != 3 or output.shape[1] < 64:
            raise RuntimeError('unexpected bridge output: {}'.format(type(output)))
        self.calls.append(output.detach().clone())

    def get(self, call_index=None):
        index = self.call_index if call_index is None else int(call_index)
        if len(self.calls) <= index:
            raise RuntimeError('message hook call {} unavailable; calls={}'.format(index, len(self.calls)))
        value = self.calls[index]
        return value[:, :64, :].float().cpu(), list(value.shape), len(self.calls)

    def close(self):
        self.handle.remove()


def message_metrics(clean, perturbed):
    a, b = clean.float(), perturbed.float()
    cosine = F.cosine_similarity(a, b, dim=-1)
    return {
        'msg_cos': float(cosine.mean().item()),
        'msg_cos_drop': float((1.0 - cosine).mean().item()),
        'msg_rel_l2': float((torch.norm(a - b) / torch.norm(a).clamp_min(1e-8)).item()),
        'msg_abs_l2': float(torch.norm(a - b).item()),
    }


def load_event_frames(settings, event):
    sequence = load_train_sequence(settings.lasher_path, event['seq'])
    rgb = [image_loader(path) for path in sequence.frames[0]]
    tir = [image_loader(path) for path in sequence.frames[1]]
    n = min(len(rgb), len(tir), len(sequence.ground_truth_rect))
    return rgb[:n], tir[:n], [[float(x) for x in box] for box in sequence.ground_truth_rect[:n]]


def run_event_condition(tracker, module, event, rgb, tir, gt_boxes, snapshot, rng, condition):
    restore_rng(rng)
    restore_tracker(tracker, snapshot)
    frame = event['frame']
    rgb_in, tir_in, region_type, region_box = condition_images(
        rgb[frame], tir[frame], gt_boxes[frame], condition, 3407 + int(event['event_id'].split('-')[-1]))
    capture = MessageCapture(module, 0)
    try:
        with torch.no_grad():
            output = tracker.track(np.concatenate([rgb_in, tir_in], axis=2), {
                'gt_bbox': gt_boxes[frame], 'sequence_name': event['seq'], 'frame_num': frame})
        message, shape, calls = capture.get()
    finally:
        capture.close()
    pred = [float(x) for x in output['target_bbox']]
    return {'pred_box': pred, 'gt_box': gt_boxes[frame], 'iou': box_iou(pred, gt_boxes[frame]),
            'message': message, 'output_shape': shape, 'hook_calls': calls,
            'region_type': region_type, 'region_box': region_box, 'pred_hash': tensor_hash(torch.tensor(pred))}


def preflight(args, config_path, output_dir):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if shutil.disk_usage(ROOT).free / 2 ** 30 < args.min_free_gb:
        raise RuntimeError('insufficient free disk')
    if os.path.abspath(output_dir).startswith(os.path.abspath(os.path.join(ROOT, 'output', 'experiments', CANONICAL))):
        raise RuntimeError('output path overlaps canonical baseline')
    if not os.path.isfile(args.manifest) or not os.path.isfile(args.checkpoint):
        raise RuntimeError('manifest or checkpoint missing')
    if os.path.exists(os.path.join(output_dir, '.lock')):
        raise RuntimeError('output lock exists')
    return {'canonical_yaml_sha256': sha256_file(config_path),
            'checkpoint_sha256': sha256_file(args.checkpoint),
            'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}


def run(args):
    os.environ['TBSI_INFER_FP16'] = '0'
    manifest = json.load(open(args.manifest, 'r'))
    events = manifest['events'][:8] if args.mode == 'smoke' else manifest['events']
    if args.limit is not None:
        events = events[:args.limit]
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    params, config_path = build_params(args.config, args.checkpoint)
    preflight_meta = preflight(args, config_path, output_dir)
    lock_path = os.path.join(output_dir, '.lock')
    open(lock_path, 'w').write(str(os.getpid()))
    rows, audits = [], []
    try:
        settings = env_settings()
        event_by_seq = {event['seq']: event for event in events}
        sequences = {name: load_event_frames(settings, event_by_seq[name])
                     for name in sorted(event_by_seq)}
        tracker = TBSITrack(params, EXP)
        module = tracker.network.backbone.tbsi_layers[0].ca_s2t_v2f.attn_reshape
        grouped = OrderedDict()
        for event in events:
            grouped.setdefault(event['seq'], []).append(event)
        for seq_name in sorted(grouped):
            rgb, tir, gt_boxes = sequences[seq_name]
            seq_events = sorted(grouped[seq_name], key=lambda item: item['frame'])
            if seq_events[-1]['frame'] >= len(gt_boxes):
                raise RuntimeError('event frame out of range for {}'.format(seq_name))
            tracker.initialize(np.concatenate([rgb[0], tir[0]], axis=2), {'init_bbox': gt_boxes[0]})
            snapshots, rngs, clean_messages = {}, {}, {}
            capture = MessageCapture(module, 0)
            try:
                event_frames = set(x['frame'] for x in seq_events)
                for frame in range(1, seq_events[-1]['frame'] + 1):
                    if frame in event_frames:
                        snapshots[frame] = snapshot_tracker(tracker)
                        rngs[frame] = rng_snapshot()
                    tracker.track(np.concatenate([rgb[frame], tir[frame]], axis=2),
                                  {'gt_bbox': gt_boxes[frame], 'sequence_name': seq_name, 'frame_num': frame})
                    if frame in event_frames:
                        clean_messages[frame], shape, calls = capture.get(frame - 1)
                        if shape != [1, 320, 768] or calls != frame:
                            raise RuntimeError('hook contract failed at {}: shape={} calls={}'.format(frame, shape, calls))
            finally:
                capture.close()
            for event in seq_events:
                frame = event['frame']
                outputs = {condition: run_event_condition(tracker, module, event, rgb, tir, gt_boxes,
                                                           snapshots[frame], rngs[frame], condition)
                           for condition in CONDITIONS}
                clean_iou = outputs['00']['iou']
                clean_message = clean_messages[frame]
                audit = {'event_id': event['event_id'], 'seq': seq_name, 'frame': frame,
                         'shape_set': [outputs[c]['output_shape'] for c in CONDITIONS],
                         'hook_calls': [outputs[c]['hook_calls'] for c in CONDITIONS],
                         'clean_pred_hash': outputs['00']['pred_hash'],
                         'clean_message_hash': tensor_hash(clean_message)}
                audits.append(audit)
                for condition in CONDITIONS:
                    result = outputs[condition]
                    metric = {'msg_cos': 1.0, 'msg_cos_drop': 0.0, 'msg_rel_l2': 0.0, 'msg_abs_l2': 0.0}
                    if condition != '00':
                        metric = message_metrics(clean_message, result['message'])
                    rows.append({'event_id': event['event_id'], 'seq': seq_name, 'frame': frame,
                                 'condition': condition, 'gt_box': json.dumps(result['gt_box']),
                                 'pred_box': json.dumps(result['pred_box']), 'iou': result['iou'],
                                 'clean_iou': clean_iou, 'iou_drop': clean_iou - result['iou'],
                                 'region_type': result['region_type'], 'region_box': json.dumps(result['region_box']),
                                 'output_shape': json.dumps(result['output_shape']), 'hook_calls': result['hook_calls'],
                                 'pred_hash': result['pred_hash'], **metric})
            print('processed sequence {} ({}/{})'.format(seq_name, len(audits), len(events)), flush=True)
        fields = list(rows[0].keys())
        with open(os.path.join(output_dir, 'event_results.csv'), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        gate = all(item['shape_set'] == [[1, 320, 768]] * 4 and item['hook_calls'][0] == item['hook_calls'][1] == item['hook_calls'][2] == item['hook_calls'][3] for item in audits)
        with open(os.path.join(output_dir, 'hook_audit.json'), 'w') as handle:
            json.dump({'experiment': EXP, 'gate': gate, 'events': audits,
                       'target_path': 'tbsi_layers[0].ca_s2t_v2f.attn_reshape[:,:64,:]'}, handle, indent=2)
        metadata = {'experiment': EXP, 'mode': args.mode, 'manifest': os.path.abspath(args.manifest),
                    'manifest_sha256': sha256_file(args.manifest), 'n_events': len(events), 'n_rows': len(rows),
                    'conditions': CONDITIONS, 'lens_z': 64, 'sigma': 3.0, 'config': args.config,
                    'config_sha256': preflight_meta['canonical_yaml_sha256'], 'checkpoint': os.path.abspath(args.checkpoint),
                    'checkpoint_sha256': preflight_meta['checkpoint_sha256'], 'gpu': preflight_meta['gpu'],
                    'torch': preflight_meta['torch'], 'fp16_env': os.environ['TBSI_INFER_FP16'],
                    'status': 'COMPLETED' if gate else 'FAILED_HOOK_AUDIT'}
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        if not gate:
            raise RuntimeError('hook audit gate failed')
    finally:
        if os.path.exists(lock_path):
            os.unlink(lock_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'batch'), default='smoke')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', default=CANONICAL)
    parser.add_argument('--checkpoint', default=os.path.join(ROOT, 'output/experiments', CANONICAL, 'checkpoints', 'TBSITrack_ep0015.pth.tar'))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--min-free-gb', type=float, default=8.0)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
