#!/usr/bin/env python3
"""Read-only capture of early/current template Value differences for ECLVR.

The wrapper observes V2TemplateFusion inputs and the current layer qkv weights.
It never writes to model tensors and does not change tracker state.  Outputs are
diagnostic evidence only, not a tracking benchmark.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from tools.template_route_probe_v1 import event_frames, read_frame, select_sequences


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _project_value(module, tokens):
    """Project tokens with the current fusion layer's own LN and Wv."""
    bsz, n_tokens, dim = tokens.shape
    normalized = module.norm(tokens)
    qkv = module.qkv(normalized)
    qkv = qkv.reshape(bsz, n_tokens, 3, module.num_heads, module.head_dim)
    # [B, H, N, D], matching V2TemplateFusion's attention layout.
    return qkv[:, :, 2].permute(0, 2, 1, 3).float()


def _capture_wrapper(module, location):
    original = module.forward

    def wrapped(self, x_rgb_template, x_tir_template, *args, **kwargs):
        early_context = kwargs.get('early_context')
        fg_masks = kwargs.get('fg_masks')
        capture = None
        if early_context is not None and len(early_context) == 4:
            early_rgb_t, early_tir_t, early_rgb_s, early_tir_s = early_context
            with torch.no_grad():
                current = torch.cat([
                    _project_value(self, x_rgb_template),
                    _project_value(self, x_tir_template)], dim=2)
                early = torch.cat([
                    _project_value(self, early_rgb_t),
                    _project_value(self, early_tir_t)], dim=2)
                delta = early - current
                early_rgb = F.normalize(early_rgb_t.float(), dim=-1, eps=1e-6)
                early_tir = F.normalize(early_tir_t.float(), dim=-1, eps=1e-6)
                early_rgb_s = F.normalize(early_rgb_s.float(), dim=-1, eps=1e-6)
                early_tir_s = F.normalize(early_tir_s.float(), dim=-1, eps=1e-6)
                evidence = torch.stack((
                    torch.amax(torch.matmul(early_rgb, early_rgb_s.transpose(1, 2)), dim=-1),
                    torch.amax(torch.matmul(early_tir, early_tir_s.transpose(1, 2)), dim=-1),
                    (early_rgb * early_tir).sum(dim=-1)), dim=-1)
                if fg_masks is None or len(fg_masks) != 2:
                    fg = torch.ones(
                        (x_rgb_template.shape[0], x_rgb_template.shape[1]),
                        device=x_rgb_template.device, dtype=torch.float32)
                    fg_source = 'uniform_fallback'
                else:
                    # The tracker passes one mask per modality. Keep both so
                    # source-side modality differences remain observable.
                    fg = torch.stack((fg_masks[0].float(), fg_masks[1].float()), dim=1)
                    fg_source = 'tracker_template_fg_masks'
                capture = {
                    'location': int(location),
                    'early_rgb_value': early[:, :, :x_rgb_template.shape[1]].cpu(),
                    'early_tir_value': early[:, :, x_rgb_template.shape[1]:].cpu(),
                    'current_rgb_value': current[:, :, :x_rgb_template.shape[1]].cpu(),
                    'current_tir_value': current[:, :, x_rgb_template.shape[1]:].cpu(),
                    'delta_rgb': delta[:, :, :x_rgb_template.shape[1]].cpu(),
                    'delta_tir': delta[:, :, x_rgb_template.shape[1]:].cpu(),
                    'evidence': evidence.cpu(),
                    'fg_masks': fg.cpu(),
                    'fg_source': fg_source,
                }
        self._eclvr_last_capture = capture
        return original(x_rgb_template, x_tir_template, *args, **kwargs)

    module.forward = types.MethodType(wrapped, module)


def install_wrappers(tracker):
    layers = tracker.network.backbone.tbsi_layers
    for index, layer in enumerate(layers):
        location = tracker.network.backbone.tbsi_loc[index]
        fusion = layer.v2_template_fusion
        _capture_wrapper(fusion, location)


def make_tracker(args):
    params = parameters(args.config)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(Path(args.checkpoint).resolve())
    tracker = TBSITrack(params, 'lasher_train')
    tracker.use_fp16 = False
    install_wrappers(tracker)
    return tracker, params


def flatten_stats(value, fg):
    # value [B,H,M,D], fg [B,M]. Weighted token summaries per sample.
    value = value.float()
    fg = fg.float().clamp(0, 1)
    token_norm = value.norm(dim=-1).mean(dim=1)
    denom_fg = fg.sum(dim=-1).clamp_min(1e-6)
    denom_bg = (1.0 - fg).sum(dim=-1).clamp_min(1e-6)
    fg_mean = (token_norm * fg).sum(dim=-1) / denom_fg
    bg_mean = (token_norm * (1.0 - fg)).sum(dim=-1) / denom_bg
    return float(fg_mean.mean()), float(bg_mean.mean())


def summarize(capture):
    rows = []
    for modality in ('rgb', 'tir'):
        early = capture['early_{}_value'.format(modality)].float()
        current = capture['current_{}_value'.format(modality)].float()
        delta = capture['delta_{}'.format(modality)].float()
        fg = capture['fg_masks'][:, 0 if modality == 'rgb' else 1]
        cos = F.cosine_similarity(early, current, dim=-1).mean().item()
        ratio = (delta.norm(dim=-1) /
                 current.norm(dim=-1).clamp_min(1e-6)).mean().item()
        fg_delta, bg_delta = flatten_stats(delta, fg)
        fg_current, bg_current = flatten_stats(current, fg)
        matrix = delta.reshape(-1, delta.shape[-1])
        singular = torch.linalg.svdvals(matrix)
        energy = singular.square()
        total = energy.sum().clamp_min(1e-12)
        r8 = min(8, int(singular.numel()))
        rank8_energy = float((energy[:r8].sum() / total).item())
        rows.append({
            'location': int(capture['location']),
            'modality': modality,
            'cos_early_current_mean': float(cos),
            'delta_over_current_norm_mean': float(ratio),
            'fg_delta_norm_mean': fg_delta,
            'bg_delta_norm_mean': bg_delta,
            'fg_current_norm_mean': fg_current,
            'bg_current_norm_mean': bg_current,
            'fg_bg_delta_ratio': float(fg_delta / max(bg_delta, 1e-8)),
            'delta_rank8_energy': rank8_energy,
            'fg_mask_source': capture['fg_source'],
        })
    return rows


def capture_event(tracker, out, event_id, metadata):
    arrays = {}
    summaries = []
    for layer in tracker.network.backbone.tbsi_layers:
        fusion = layer.v2_template_fusion
        capture = getattr(fusion, '_eclvr_last_capture', None)
        if capture is None:
            raise RuntimeError('missing ECLVR capture at TBSI location')
        location = capture['location']
        for key in ('early_rgb_value', 'early_tir_value',
                    'current_rgb_value', 'current_tir_value',
                    'delta_rgb', 'delta_tir', 'evidence', 'fg_masks'):
            arrays['l{}_{}'.format(location, key)] = capture[key][0].numpy()
        summaries.extend(summarize(capture))
    np.savez_compressed(out / '{}.npz'.format(event_id), **arrays)
    return summaries


def parity_check(args, sequences):
    # One-sequence serial parity. The wrapper only reads tensors and computes
    # CPU copies, so any nonzero box difference is an instrumentation failure.
    baseline, _ = make_tracker(args)
    instrumented, _ = make_tracker(args)
    seq = sequences[0]
    init = read_frame(seq, 0)
    info0 = {'init_bbox': seq.ground_truth_rect[0]}
    baseline.initialize(init, info0)
    instrumented.initialize(init, info0)
    max_diff = 0.0
    for frame_num in range(1, min(len(seq.frames[0]), args.parity_frames + 1)):
        image = read_frame(seq, frame_num)
        info = {'sequence_name': seq.name, 'frame_num': frame_num,
                'gt_bbox': seq.ground_truth_rect[frame_num]}
        a = baseline.track(image, info)['target_bbox']
        b = instrumented.track(image, info)['target_bbox']
        max_diff = max(max_diff, float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    return {'sequence': seq.name, 'frames': min(len(seq.frames[0]) - 1, args.parity_frames),
            'max_abs_box_diff': max_diff, 'passed': max_diff == 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--sequence-list', default=str(ROOT / 'experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt'))
    ap.add_argument('--output', required=True)
    ap.add_argument('--sequences', type=int, default=3)
    ap.add_argument('--parity-frames', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_TEMPLATE_UPDATE_MODE'] = 'off'
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = Path(args.checkpoint).resolve()
    sequence_list = Path(args.sequence_list).resolve()
    out = Path(args.output).resolve()
    if not checkpoint.is_file() or not sequence_list.is_file():
        raise FileNotFoundError('checkpoint or sequence list missing')
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing non-empty output {}'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    dataset = LasHeRDataset(split='trainingset', sequence_list_file=str(sequence_list)).get_sequence_list()
    sequences = select_sequences(dataset, args.sequences)
    if not sequences:
        raise RuntimeError('empty sequence selection')
    (out / 'selected_sequences.txt').write_text('\n'.join(s.name for s in sequences) + '\n')
    parity = parity_check(args, sequences)
    (out / 'parity.json').write_text(json.dumps(parity, indent=2) + '\n')
    if not parity['passed']:
        raise RuntimeError('instrumentation parity failed: {}'.format(parity))
    tracker, params = make_tracker(args)
    event_rows = []
    summary_rows = []
    event_index = 0
    for seq_index, seq in enumerate(sequences):
        print('[{}/{}] {}'.format(seq_index + 1, len(sequences), seq.name), flush=True)
        tracker.initialize(read_frame(seq, 0), {'init_bbox': seq.ground_truth_rect[0]})
        events = set(event_frames(seq))
        for frame_num in range(1, len(seq.frames[0])):
            state_before = list(tracker.state)
            image = read_frame(seq, frame_num)
            info = {'sequence_name': seq.name, 'frame_num': frame_num,
                    'gt_bbox': seq.ground_truth_rect[frame_num]}
            result = tracker.track(image, info)
            if frame_num not in events:
                continue
            event_id = 'ECLVR-%04d' % event_index
            summaries = capture_event(tracker, out, event_id, info)
            event_rows.append({'event_id': event_id, 'sequence': seq.name,
                               'frame': int(frame_num),
                               'state_before': [float(x) for x in state_before],
                               'pred_box': [float(x) for x in result['target_bbox']],
                               'gt_box_original': [float(x) for x in seq.ground_truth_rect[frame_num]]})
            for row in summaries:
                row.update({'event_id': event_id, 'sequence': seq.name, 'frame': int(frame_num)})
                summary_rows.append(row)
            event_index += 1
    with (out / 'events.jsonl').open('w') as handle:
        for row in event_rows:
            handle.write(json.dumps(row) + '\n')
    with (out / 'summary.jsonl').open('w') as handle:
        for row in summary_rows:
            handle.write(json.dumps(row) + '\n')
    manifest = {'config': args.config, 'checkpoint': str(checkpoint),
                'checkpoint_sha256': sha256(checkpoint),
                'sequence_list': str(sequence_list), 'sequences': len(sequences),
                'events': len(event_rows), 'fp16': False, 'template_update': 'off',
                'locations': [3, 6, 9], 'capture': 'read-only current-layer-Wv projection',
                'parity': parity, 'claim': 'mechanism diagnostics only'}
    (out / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
