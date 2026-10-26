#!/usr/bin/env python3
"""Matched fixed-state sensitivity of T and S messages in protected M1."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from diagnose_m1_search_stage_20261009 import (
    CONFIG, CHECKPOINT, EXPECTED_CONFIG, EXPECTED_CHECKPOINT, sha256,
    read_pair, select_frames, target_masks, response_metrics, predicted_box, iou,
)

CONFIG_NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
T_SCALES = (0.0, 0.95, 1.05)


def make_message_hooks(network, state):
    handles = []
    backbone = network.backbone
    t_drop = backbone.tbsi_layers[2].v2_template_fusion.attn_drop
    s_drop = backbone.blocks[10].attn.attn_drop
    t_proj = backbone.tbsi_layers[2].v2_template_fusion.proj
    s_proj = backbone.blocks[10].attn.proj

    def scale_probs(site):
        def hook(_module, _inputs, output):
            active, scale = state['active'], state['scale']
            if active != site or scale == 1.0:
                return output
            value = output.clone()
            if site == 'T':
                n = 64
                value[:, :, :n, n:] *= scale
                value[:, :, n:, :n] *= scale
            else:
                n = 64
                value[:, :, n:, :n] *= scale
            return value
        return hook

    def capture(site):
        def hook(_module, _inputs, output):
            if state['active'] in (None, site):
                state['messages'].setdefault(site, []).append(output.detach().float())
            return None
        return hook

    handles.append(t_drop.register_forward_hook(scale_probs('T')))
    handles.append(s_drop.register_forward_hook(scale_probs('S')))
    handles.append(t_proj.register_forward_hook(capture('T')))
    handles.append(s_proj.register_forward_hook(capture('S')))
    return handles


def message_delta(base, variant, site):
    base_parts, variant_parts = base[site], variant[site]
    if len(base_parts) != len(variant_parts):
        raise RuntimeError('{} message call count differs'.format(site))
    deltas, references = [], []
    for left, right in zip(base_parts, variant_parts):
        if site == 'S':
            left, right = left[:, 64:], right[:, 64:]
        deltas.append((right - left).reshape(-1))
        references.append(left.reshape(-1))
    delta = torch.cat(deltas).norm().item()
    ref = torch.cat(references).norm().item()
    return delta, delta / max(ref, 1e-12)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--s-epsilon', type=float, default=0.05,
                        help='S scale perturbation; T remains fixed at +/-5 percent')
    args = parser.parse_args()
    if not 0.0 < args.s_epsilon < 1.0:
        raise RuntimeError('--s-epsilon must be in (0,1)')
    scales = {'T': T_SCALES,
              'S': (0.0, 1.0 - args.s_epsilon, 1.0 + args.s_epsilon)}
    trace_dir = Path(args.trace_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError('refusing nonempty output: {}'.format(output_dir))
    cohort_manifest = trace_dir.parent / 'cohort_manifest.json'
    if sha256(CONFIG) != EXPECTED_CONFIG or sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    names = (trace_dir / 'trace_sequences.txt').read_text().splitlines()
    if not names or len(names) != len(set(names)):
        raise RuntimeError('trace sequence list must be nonempty and unique')

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    params = parameters(CONFIG_NAME)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'lasher_train')
    network = tracker.network.eval()
    dataset = LasHeRDataset('trainingset', str(trace_dir / 'trace_sequences.txt'))
    seq_map = {seq.name: seq for seq in dataset.get_sequence_list()}
    state = {'active': None, 'scale': 1.0, 'messages': {}}
    handles = make_message_hooks(network, state)
    head_capture = {'output': None}

    def head_hook(_module, _inputs, output):
        head_capture['output'] = output
    handles.append(network.box_head.register_forward_hook(head_hook))

    rows = []
    started = time.time()
    for seq_name in names:
        sequence = seq_map[seq_name]
        trace_path = trace_dir / seq_name / 'frames.jsonl'
        trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        chosen = select_frames(trace, 1, 'onset-3-iou-0.5')
        tracker.initialize(read_pair(sequence, 0), {
            'init_bbox': sequence.ground_truth_rect[0], 'sequence_name': seq_name})
        for index, label in chosen:
            item = trace[index]
            frame_id = int(item['frame_id'])
            state_before = [float(v) for v in item['tracker_state_before']]
            gt = np.asarray(sequence.ground_truth_rect[frame_id], dtype=np.float64)
            image = read_pair(sequence, frame_id)
            crop, resize, mask = sample_target(image, state_before, params.search_factor,
                                                output_sz=params.search_size)
            search_nested = tracker.preprocessor.process(crop, mask, half=False)
            search = [search_nested.tensors[:, :3], search_nested.tensors[:, 3:]]
            gt_crop = transform_image_to_crop(
                torch.tensor(gt, dtype=torch.float32),
                torch.tensor(state_before, dtype=torch.float32), resize,
                torch.tensor([params.search_size, params.search_size]),
                normalize=True).numpy()
            masks = target_masks(gt_crop)

            def run(active=None, scale=1.0):
                state['active'], state['scale'], state['messages'] = active, scale, {}
                head_capture['output'] = None
                with torch.no_grad():
                    out = network(template=[tracker.z_dict1.tensors[:, :3],
                                            tracker.z_dict1.tensors[:, 3:]],
                                  search=search,
                                  ce_template_mask=tracker.box_mask_z,
                                  template_fg_masks=(tracker.template_fg_mask,
                                                     tracker.template_fg_mask))
                if head_capture['output'] is None:
                    raise RuntimeError('CENTER head output capture failed')
                metrics = response_metrics(out['score_map'], tracker.output_window,
                                           out['size_map'], out['offset_map'], masks, gt_crop)
                pred = predicted_box(network, tracker, out, state_before,
                                     image.shape, resize)
                metrics['iou'] = iou(pred, gt)
                metrics['center_error_px'] = math.hypot(
                    pred[0] + 0.5 * pred[2] - (gt[0] + 0.5 * gt[2]),
                    pred[1] + 0.5 * pred[3] - (gt[1] + 0.5 * gt[3]))
                return metrics, pred, state['messages']

            base, base_pred, base_messages = run()
            parity = float(np.max(np.abs(np.asarray(base_pred) -
                                         np.asarray(item['pred_box']))))
            if parity > 0.02:
                raise RuntimeError('fixed-state parity failed {} frame {}: {} px'.format(
                    seq_name, frame_id, parity))
            rows.append({'sequence': seq_name, 'frame': frame_id, 'state': label,
                         'site': 'baseline', 'scale': 1.0, 'message_delta_norm': 0.0,
                         'message_delta_ratio': 0.0, 'parity_max_px': parity, **base})
            for site in ('T', 'S'):
                for scale in scales[site]:
                    metrics, pred, messages = run(site, scale)
                    delta, ratio = message_delta(base_messages, messages, site)
                    rows.append({'sequence': seq_name, 'frame': frame_id, 'state': label,
                                 'site': site, 'scale': scale,
                                 'message_delta_norm': delta,
                                 'message_delta_ratio': ratio,
                                 'parity_max_px': parity, **metrics})
        print('completed {} selected={} elapsed_s={:.1f}'.format(
            seq_name, len(chosen), time.time() - started), flush=True)

    for handle in handles:
        handle.remove()
    fields = sorted({key for row in rows for key in row})
    with (output_dir / 'per_state.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'id': 'M1-MESSAGE-POSITION-LEVERAGE-20261012',
        'status': 'FIXED-STATE-DIAGNOSTIC-COMPLETE',
        'trace_dir': str(trace_dir), 'cohort_manifest': str(cohort_manifest),
        'cohort_manifest_sha256': sha256(cohort_manifest) if cohort_manifest.is_file() else 'NA',
        'trace_sequence_list_sha256': sha256(trace_dir / 'trace_sequences.txt'),
        'config_sha256': EXPECTED_CONFIG, 'checkpoint_sha256': EXPECTED_CHECKPOINT,
        'script_sha256': sha256(Path(__file__)), 'states': sum(r['site'] == 'baseline' for r in rows),
        'rows': len(rows), 'max_replay_parity_px': max(r['parity_max_px'] for r in rows),
        'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'elapsed_seconds': time.time() - started,
        'interpretation': 'T is loc9 post-softmax RGB<->TIR template-message scaling; S is block-10 post-softmax template-key/value contribution to search-query rows. Fixed-state sensitivities are not trained performance effects.',
        'scales': {'T': list(scales['T']), 'S': list(scales['S'])},
        'limitations': ['Train-side cohort not used in prior listed diagnostics; it may have been seen during M1 training.', 'GT is used only for selecting fixed states and post-hoc scoring.', 'No training, test-set tracking, or closed-loop intervention is performed.', 'Current working-tree replay is not archive-frame parity.']
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
