#!/usr/bin/env python3
"""Paired train-side replay of original, foreground, and shifted MSHA queries."""
import copy
import argparse
import csv
import json
import os
import random
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

OUT = ROOT / 'output/experiments/M1_MSHA_QUERY_ROLE_FIXED_STATE_20261022'
SEQUENCE_LIST = ROOT / 'experiments/diagnostics/m1_center_readout_fresh_v1/sequence_list.txt'
EVENTS = ROOT / 'output/experiments/M1_CENTER_READOUT_FRESH_DIAG_20261010/analysis/per_event.csv'
CHECKPOINT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG = 'TBSI_E_MHSA_FULL_4090D_seed42'
ARMS = ('M1', 'QUERY_CONDITIONED', 'QUERY_SHUFFLED', 'EVIDENCE_SHUFFLED')


def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone(v) for v in value)
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    return copy.deepcopy(value)


def snapshot_rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def snapshot_tracker(tracker):
    names = ('state', 'prev_search', 'prev_temporal_tokens', 'frame_id', 'prev_head_feature',
             'prev_crop_xywh', 'prev_crop_transform', 'prev_head_frame_id', 'prev_head_sequence_id')
    return {name: clone(getattr(tracker, name)) for name in names if hasattr(tracker, name)}


def restore_tracker(tracker, state):
    for name, value in state.items():
        setattr(tracker, name, clone(value))


def read_frame(seq, index):
    rgb = cv2.cvtColor(cv2.imread(seq.frames[0][index]), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(seq.frames[1][index]), cv2.COLOR_BGR2RGB)
    return np.concatenate((rgb, tir), axis=2)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def load_events(smoke=False):
    rows = [r for r in csv.DictReader(EVENTS.open()) if r['label'] in ('severe_onset', 'matched_good')]
    if smoke:
        rows = [min((r for r in rows if r['label'] == label), key=lambda r: int(r['frame']))
                for label in ('severe_onset', 'matched_good')]
    selected = {}
    for row in rows:
        selected.setdefault(row['sequence'], {})[int(row['frame'])] = row['label']
    if not smoke and len(rows) != 28:
        raise ValueError('Expected 14 severe-onset and 14 matched-good events, got {}'.format(len(rows)))
    return selected


def set_collectors(tracker, cache):
    for layer_index, layer in enumerate(tracker.network.backbone.tbsi_layers):
        fusion = getattr(layer, 'v2_template_fusion', None)
        if fusion is None:
            continue
        fusion.analysis_enabled = True
        fusion.last_fg_cross_stats = {'mode': 'not_called'}
        fusion.last_query_conditioned_stats = {'mode': 'not_called'}

        def collect(layer_index=layer_index, **kwargs):
            masks = kwargs.get('fg_masks')
            if masks is None:
                return
            mask = torch.cat([masks[0], masks[1]], dim=1)[0].detach().float().cpu()
            feature = kwargs['final_output'][0].detach().float().cpu()
            cache[layer_index] = (feature, mask)

        fusion.output_analysis_collector = collect


def initialize_query_probe(tracker):
    """Install deterministic nonzero probe weights after loading protected M1."""
    modules = []
    for layer_index, layer in enumerate(tracker.network.backbone.tbsi_layers):
        fusion = getattr(layer, 'v2_template_fusion', None)
        if fusion is None or not getattr(fusion, 'query_conditioned_cross_value', False):
            continue
        modules.extend([
            (layer_index, fusion.query_conditioned_rgb_from_tir),
            (layer_index, fusion.query_conditioned_tir_from_rgb),
        ])
    if not modules:
        raise RuntimeError('query-conditioned modules were not constructed')
    with torch.no_grad():
        for layer_index, module in modules:
            generator = torch.Generator(device=module.down.weight.device)
            generator.manual_seed(20261024 + 17 * layer_index)
            module.down.weight.copy_(torch.randn(
                module.down.weight.shape, generator=generator,
                device=module.down.weight.device, dtype=module.down.weight.dtype) * 0.04)
            module.up.weight.copy_(torch.randn(
                module.up.weight.shape, generator=generator,
                device=module.up.weight.device, dtype=module.up.weight.dtype) * 0.04)
    return len(modules)


def feature_role_deltas(base, arm):
    rows = []
    for layer, (feat, mask) in arm.items():
        if layer not in base:
            continue
        base_feat, base_mask = base[layer]
        if feat.shape != base_feat.shape or mask.shape != base_mask.shape:
            raise ValueError('Feature or foreground-mask shape changed at layer {}'.format(layer))
        norm = (feat - base_feat).norm(dim=-1)
        mask = base_mask.clamp(0, 1)
        fg = float((norm * mask).sum() / mask.sum().clamp_min(1e-8))
        bg = float((norm * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1e-8))
        rows.append({'layer': layer, 'fg_query_delta_norm': fg, 'hard_bg_query_delta_norm': bg,
                     'fg_to_bg_delta_ratio': fg / max(bg, 1e-8)})
    return rows


def install_boundary_hooks(tracker, active_capture):
    handles = []

    def save(layer_name, boundary, value):
        if active_capture[0] is None:
            return
        if isinstance(value, (tuple, list)):
            value = torch.cat(list(value), dim=1)
        active_capture[0].setdefault(boundary, {})[layer_name] = (
            value.detach().float().cpu().clone())

    for index, layer in enumerate(tracker.network.backbone.tbsi_layers):
        fusion = getattr(layer, 'v2_template_fusion', None)
        if fusion is None:
            continue
        name = str(index)
        handles.append(fusion.proj.register_forward_pre_hook(
            lambda module, inputs, name=name: save(name, 'projection_input', inputs[0])))
        handles.append(fusion.proj.register_forward_hook(
            lambda module, inputs, output, name=name: save(name, 'projection_output', output)))
        handles.append(fusion.register_forward_hook(
            lambda module, inputs, output, name=name: save(name, 'fusion_return', output)))

    def center_input_hook(module, inputs):
        save('center', 'center_input', inputs[0])

    handles.append(tracker.network.box_head.register_forward_pre_hook(center_input_hook))
    return handles


def boundary_delta_rows(sequence, frame, role, condition, boundary, base, arm, masks):
    rows = []
    for layer, value in arm.items():
        if layer not in base:
            continue
        reference = base[layer]
        if value.shape != reference.shape:
            raise ValueError('{} shape mismatch at layer {}'.format(boundary, layer))
        delta = (value - reference).float()
        flat = delta.reshape(delta.shape[0], -1, delta.shape[-1]) if delta.ndim == 3 else None
        row = {
            'sequence': sequence, 'frame': frame, 'role': role, 'condition': condition,
            'boundary': boundary, 'layer': layer,
            'shape': json.dumps(list(value.shape)),
            'max_abs_delta': float(delta.abs().max()),
            'mean_l2_delta': float(delta.reshape(delta.shape[0], -1).norm(dim=-1).mean()),
            'fg_l2_delta': 'NA', 'bg_l2_delta': 'NA',
        }
        if flat is not None and layer in masks:
            mask = masks[layer].clamp(0, 1).reshape(1, -1, 1)
            token_norm = flat.norm(dim=-1)
            row['fg_l2_delta'] = float((token_norm * mask[..., 0]).sum() /
                                       mask.sum().clamp_min(1e-8))
            row['bg_l2_delta'] = float((token_norm * (1 - mask[..., 0])).sum() /
                                       (1 - mask).sum().clamp_min(1e-8))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(OUT))
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--cross-support-diagnostic', action='store_true')
    ap.add_argument('--fcsv-r-diagnostic', action='store_true')
    ap.add_argument('--receiver-centered-diagnostic', action='store_true')
    ap.add_argument('--fcsv-r-alpha', type=float, default=0.1)
    ap.add_argument('--fcsv-r-temperature', type=float, default=0.07)
    ap.add_argument('--config-name', default=CONFIG)
    ap.add_argument('--checkpoint-path', default=str(CHECKPOINT))
    ap.add_argument('--cross-support-layer', type=int, default=3)
    args = ap.parse_args()
    out_dir = Path(args.output).resolve()
    if out_dir.exists() and any(p.name != 'provenance' for p in out_dir.iterdir()):
        raise RuntimeError('refusing to overwrite non-empty output directory: {}'.format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_root = out_dir / 'trace'
    trace_root.mkdir(exist_ok=True)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_DISABLE_TRACKER_CACHE'] = '1'
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    events = load_events(smoke=args.smoke)
    sequences = LasHeRDataset('trainingset', str(SEQUENCE_LIST)).get_sequence_list()
    seqmap = {seq.name: seq for seq in sequences}
    if set(events) - set(seqmap):
        raise ValueError('Event sequences missing from fixed train-side manifest')

    config_name = args.config_name
    checkpoint_path = args.checkpoint_path
    params = parameters(config_name)
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(checkpoint_path)
    # All optional routes are explicitly disabled before the arm loop.  Each
    # diagnostic arm then enables exactly one route through its environment
    # variable, keeping the M1 native arm an identity replay.
    os.environ['TBSI_RECEIVER_CENTERED_VALUE_MODE'] = 'off'
    os.environ['TBSI_FCSV_R_MODE'] = 'off'
    if args.receiver_centered_diagnostic:
        arms = ('NATIVE', 'RECEIVER_CENTERED_CORRECT',
                'RECEIVER_CENTERED_SHIFTED', 'RECEIVER_CENTERED_ROW_SHUFFLED')
        baseline_arm = 'NATIVE'
        os.environ['TBSI_MSHA_QUERY_CONDITIONED_DISABLE'] = '1'
        os.environ['TBSI_MSHA_CROSS_SUPPORT_MODE'] = 'off'
    elif args.fcsv_r_diagnostic:
        arms = ('NATIVE', 'FCSV_R_MASK_ONLY',
                'FCSV_R_CONTINUOUS_CORRECT', 'FCSV_R_CONTINUOUS_SHIFTED')
        baseline_arm = 'NATIVE'
        os.environ['TBSI_MSHA_QUERY_CONDITIONED_CROSS_VALUE'] = '0'
        os.environ['TBSI_MSHA_CROSS_SUPPORT_MODE'] = 'off'
        os.environ['TBSI_FCSV_R_TEMPERATURE'] = str(args.fcsv_r_temperature)
        os.environ['TBSI_FCSV_R_ALPHA'] = str(args.fcsv_r_alpha)
    elif args.cross_support_diagnostic:
        arms = ('M1', 'ALIGNED_WINDOW', 'SHIFTED_WINDOW')
        baseline_arm = 'M1'
        os.environ['TBSI_MSHA_QUERY_CONDITIONED_CROSS_VALUE'] = '0'
        os.environ['TBSI_MSHA_CROSS_SUPPORT_MODE'] = 'off'
        os.environ['TBSI_MSHA_CROSS_SUPPORT_LAYER'] = str(args.cross_support_layer)
        os.environ['TBSI_MSHA_DIFF_DIAGNOSTICS_PATH'] = str(out_dir / 'attention_diagnostics.jsonl')
    else:
        arms = ARMS
        baseline_arm = 'M1'
        os.environ['TBSI_MSHA_QUERY_CONDITIONED_CROSS_VALUE'] = '1'
    tracker = TBSITrack(params, 'lasher_test')
    for fallback_index, layer in enumerate(tracker.network.backbone.tbsi_layers):
        fusion = getattr(layer, 'v2_template_fusion', None)
        if fusion is not None:
            fusion.path_effect_layer_index = (3, 6, 9)[fallback_index]
    probe_count = 0 if (
        args.cross_support_diagnostic or args.fcsv_r_diagnostic or
        args.receiver_centered_diagnostic
    ) else initialize_query_probe(tracker)
    output_rows, layer_rows, boundary_rows, branch_rows = [], [], [], []
    active_capture = [None]
    hook_handles = install_boundary_hooks(tracker, active_capture)

    for seq_name, frame_roles in sorted(events.items()):
        seq = seqmap[seq_name]
        tracker.initialize(read_frame(seq, 0), {
            'init_bbox': np.asarray(seq.ground_truth_rect[0], dtype=np.float32),
            'sequence_name': seq_name})
        previous_output = {}
        max_frame = max(frame_roles)
        for frame in range(1, max_frame + 1):
            frame_info = {'previous_output': previous_output, 'sequence_name': seq_name,
                          'frame_num': frame, 'gt_bbox': np.asarray(seq.ground_truth_rect[frame], dtype=np.float32)}
            image = read_frame(seq, frame)
            if frame not in frame_roles:
                os.environ.pop('TBSI_MSHA_FG_CROSS_MODE', None)
                os.environ['TBSI_MSHA_CROSS_SUPPORT_MODE'] = 'off'
                os.environ['TBSI_FCSV_R_MODE'] = 'off'
                previous_output = tracker.track(image, frame_info)
                continue

            role = frame_roles[frame]
            state0, rng0 = snapshot_tracker(tracker), snapshot_rng()
            arm_outputs, arm_features, arm_traces, arm_boundaries, arm_branch_stats = {}, {}, {}, {}, {}
            for arm in arms:
                restore_tracker(tracker, state0)
                restore_rng(rng0)
                os.environ.pop('TBSI_MSHA_FG_CROSS_MODE', None)
                if args.fcsv_r_diagnostic:
                    os.environ['TBSI_MSHA_QUERY_CONDITIONED_DISABLE'] = '1'
                    os.environ['TBSI_MSHA_CROSS_SUPPORT_MODE'] = 'off'
                    os.environ['TBSI_FCSV_R_MODE'] = {
                        'NATIVE': 'off',
                        'FCSV_R_MASK_ONLY': 'mask_only',
                        'FCSV_R_CONTINUOUS_CORRECT': 'continuous_correct',
                        'FCSV_R_CONTINUOUS_SHIFTED': 'continuous_shifted'}[arm]
                    os.environ['TBSI_RECEIVER_CENTERED_VALUE_MODE'] = 'off'
                elif args.receiver_centered_diagnostic:
                    os.environ['TBSI_MSHA_QUERY_CONDITIONED_DISABLE'] = '1'
                    os.environ['TBSI_MSHA_CROSS_SUPPORT_MODE'] = 'off'
                    os.environ['TBSI_FCSV_R_MODE'] = 'off'
                    os.environ['TBSI_RECEIVER_CENTERED_VALUE_MODE'] = {
                        'NATIVE': 'off',
                        'RECEIVER_CENTERED_CORRECT': 'correct',
                        'RECEIVER_CENTERED_SHIFTED': 'shifted',
                        'RECEIVER_CENTERED_ROW_SHUFFLED': 'row_shuffled',
                    }[arm]
                elif args.cross_support_diagnostic:
                    os.environ['TBSI_MSHA_QUERY_CONDITIONED_DISABLE'] = '1'
                    os.environ['TBSI_MSHA_CROSS_SUPPORT_MODE'] = {
                        'M1': 'off', 'ALIGNED_WINDOW': 'aligned_window',
                        'SHIFTED_WINDOW': 'shifted_window'}[arm]
                else:
                    os.environ['TBSI_MSHA_CROSS_SUPPORT_MODE'] = 'off'
                    os.environ['TBSI_FCSV_R_MODE'] = 'off'
                    os.environ['TBSI_RECEIVER_CENTERED_VALUE_MODE'] = 'off'
                    if arm == 'M1':
                        os.environ['TBSI_MSHA_QUERY_CONDITIONED_DISABLE'] = '1'
                        os.environ['TBSI_MSHA_QUERY_CONDITIONED_CONDITION'] = 'normal'
                    else:
                        os.environ.pop('TBSI_MSHA_QUERY_CONDITIONED_DISABLE', None)
                        os.environ['TBSI_MSHA_QUERY_CONDITIONED_CONDITION'] = {
                            'QUERY_CONDITIONED': 'normal',
                            'QUERY_SHUFFLED': 'query_shuffled',
                            'EVIDENCE_SHUFFLED': 'evidence_shuffled',
                        }[arm]
                os.environ['TBSI_MSHA_DIFF_DIAGNOSTIC_TAG'] = '{}:{}:{}:{}'.format(
                    seq_name, frame, role, arm)
                feature_cache = {}
                set_collectors(tracker, feature_cache)
                active_capture[0] = {}
                seq_trace_dir = trace_root / arm
                trace_file = seq_trace_dir / seq_name / 'frames.jsonl'
                trace_file.parent.mkdir(parents=True, exist_ok=True)
                tracker.trace_dir = str(seq_trace_dir)
                tracker.trace_file = str(trace_file)
                out = tracker.track(image, frame_info)
                arm_outputs[arm] = [float(v) for v in out['target_bbox']]
                arm_features[arm] = feature_cache
                arm_boundaries[arm] = active_capture[0]
                arm_branch_stats[arm] = [
                    {
                        'query_conditioned': copy.deepcopy(getattr(
                            layer.v2_template_fusion, 'last_query_conditioned_stats', {})),
                        'cross_support': copy.deepcopy(getattr(
                            layer.v2_template_fusion, 'last_cross_support_stats', {})),
                        'fcsv_r': copy.deepcopy(getattr(
                            layer.v2_template_fusion, 'last_fcsv_r_stats', {})),
                        'receiver_centered': copy.deepcopy(getattr(
                            layer.v2_template_fusion, 'last_receiver_centered_stats', {})),
                    }
                    for layer in tracker.network.backbone.tbsi_layers
                    if getattr(layer, 'v2_template_fusion', None) is not None]
                trace_lines = trace_file.read_text().splitlines()
                matching = [json.loads(line) for line in trace_lines
                            if int(json.loads(line).get('frame_id', -1)) == frame]
                if not matching:
                    raise RuntimeError('Missing trace row for {} frame {}'.format(arm, frame))
                arm_traces[arm] = matching[-1]
                if args.smoke:
                    layer_modes = [getattr(layer.v2_template_fusion, 'last_fg_cross_stats', {})
                                   for layer in tracker.network.backbone.tbsi_layers
                                   if getattr(layer, 'v2_template_fusion', None) is not None]
                    support_modes = [getattr(layer.v2_template_fusion, 'last_cross_support_stats', {})
                                     for layer in tracker.network.backbone.tbsi_layers
                                     if getattr(layer, 'v2_template_fusion', None) is not None]
                    print('[SMOKE-MODE] {} {} support={} role={}'.format(
                        arm, seq_name, support_modes, layer_modes), flush=True)

            restore_tracker(tracker, state0)
            restore_rng(rng0)
            os.environ.pop('TBSI_MSHA_FG_CROSS_MODE', None)
            os.environ.pop('TBSI_MSHA_QUERY_CONDITIONED_DISABLE', None)
            os.environ.pop('TBSI_MSHA_QUERY_CONDITIONED_CONDITION', None)
            os.environ.pop('TBSI_MSHA_DIFF_DIAGNOSTIC_TAG', None)
            os.environ['TBSI_MSHA_CROSS_SUPPORT_MODE'] = 'off'
            os.environ['TBSI_FCSV_R_MODE'] = 'off'
            os.environ['TBSI_RECEIVER_CENTERED_VALUE_MODE'] = 'off'
            # This baseline forward advances the tracker after the paired event.
            # Detach event collectors first so it cannot overwrite SHIFTED_QUERY captures.
            active_capture[0] = None
            for layer in tracker.network.backbone.tbsi_layers:
                fusion = getattr(layer, 'v2_template_fusion', None)
                if fusion is not None:
                    fusion.output_analysis_collector = None
            previous_output = tracker.track(image, frame_info)
            gt = [float(v) for v in seq.ground_truth_rect[frame]]
            base_diag = arm_traces[baseline_arm].get('peak_diagnostics', {})
            for arm in arms:
                trace = arm_traces[arm]
                diag = trace.get('peak_diagnostics', {})
                box = arm_outputs[arm]
                output_rows.append({
                    'sequence': seq_name, 'frame': frame, 'role': role, 'condition': arm,
                    'm1_trace_iou': float(iou(arm_outputs[baseline_arm], gt)),
                    'fixed_state_iou': float(iou(box, gt)),
                    'pred_x': box[0], 'pred_y': box[1], 'pred_w': box[2], 'pred_h': box[3],
                    'raw_gt_rank': diag.get('gt_rank_raw', ''),
                    'raw_gt_hard_bg_margin': diag.get('target_background_margin_raw', ''),
                    'raw_peak_xy': json.dumps(diag.get('raw_top1_xy')),
                    'target_rank_gt': diag.get('target_rank_score_gt', ''),
                    'target_rank_hard_bg': diag.get('target_rank_score_background', ''),
                    'target_rank_margin': diag.get('target_rank_score_margin', ''),
                    'target_rank_gt_rank': diag.get('target_rank_score_gt_rank', ''),
                    'state_before': json.dumps(trace.get('tracker_state_before')),
                    'crop_xywh': json.dumps(trace.get('current_crop_xywh')),
                })
                if arm != baseline_arm:
                    for item in feature_role_deltas(arm_features[baseline_arm], arm_features[arm]):
                        layer_rows.append({'sequence': seq_name, 'frame': frame, 'role': role,
                                           'condition': arm, **item})
                    for boundary in ('projection_input', 'projection_output', 'fusion_return'):
                        boundary_rows.extend(boundary_delta_rows(
                            seq_name, frame, role, arm, boundary,
                            arm_boundaries[baseline_arm].get(boundary, {}),
                            arm_boundaries[arm].get(boundary, {}),
                            {str(key): value[1] for key, value in arm_features[baseline_arm].items()}))
                    boundary_rows.extend(boundary_delta_rows(
                        seq_name, frame, role, arm, 'center_input',
                        arm_boundaries[baseline_arm].get('center_input', {}),
                        arm_boundaries[arm].get('center_input', {}), {}))
                for layer_index, stats in enumerate(arm_branch_stats[arm]):
                    branch_rows.append({
                        'sequence': seq_name, 'frame': frame, 'role': role,
                        'condition': arm, 'layer': layer_index,
                        'stats': json.dumps(stats, sort_keys=True),
                    })
            print('[EVENT] {} frame={} role={} {}'.format(
                seq_name, frame, role,
                '/'.join('{}={:.3f}'.format(a, output_rows[-len(arms) + i]['fixed_state_iou'])
                         for i, a in enumerate(arms))), flush=True)

    with (out_dir / 'center_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    with (out_dir / 'query_feature_deltas.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_rows[0]))
        writer.writeheader(); writer.writerows(layer_rows)
    with (out_dir / 'query_boundary_deltas.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(boundary_rows[0]))
        writer.writeheader(); writer.writerows(boundary_rows)
    with (out_dir / 'branch_stats.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(branch_rows[0]))
        writer.writeheader(); writer.writerows(branch_rows)
    for handle in hook_handles:
        handle.remove()
    manifest = {
            'experiment': ('M1-MSHA-RECEIVER-CENTERED-VALUE-FIXED28-20261026'
                           if args.receiver_centered_diagnostic else
                           'M1-MSHA-FCSV-R-FIXED28-20261026' if args.fcsv_r_diagnostic else
                           'M1-MSHA-CROSS-SUPPORT-FIXED28-L3-20261026'
                       if args.cross_support_diagnostic else
                       'M1-MSHA-QUERY-CONDITIONED-VALUE-FIXED-STATE-20261024'),
        'parent': config_name, 'checkpoint': str(checkpoint_path), 'cohort': str(SEQUENCE_LIST),
        'events': str(EVENTS), 'conditions': list(arms), 'precision': 'FP32', 'threads': 1,
        'query_conditioned_probe_modules': probe_count,
        'selection': 'pre-existing train-side cohort selected by hash before this MSHA study; 14 severe-onset and 14 matched-good fixed events',
        'control': ('M1 native support, aligned 3x3 cross-window, and equal-size half-grid shifted cross-window'
                    if args.cross_support_diagnostic else
                    'M1 disabled branch, query-shuffled condition, and evidence-shuffled condition; fixed M1 checkpoint with deterministic probe weights'),
        'smoke_only': bool(args.smoke),
        'cross_support_diagnostic': bool(args.cross_support_diagnostic),
        'fcsv_r_diagnostic': bool(args.fcsv_r_diagnostic),
        'receiver_centered_diagnostic': bool(args.receiver_centered_diagnostic),
        'receiver_centered_definition': (
            'Native QK attention is unchanged; early RGB/TIR template-search '
            'transport yields receiver-specific source distributions. A '
            'foreground-gated, cross-mass-weighted centered source Value '
            'residual is bounded by rho. shifted moves TIR search support; '
            'row_shuffled permutes receiver rows.'
            if args.receiver_centered_diagnostic else None),
        'cross_support_layer': args.cross_support_layer if args.cross_support_diagnostic else None,
        'cross_support_definition': ('At the selected layer, cross-modal QK support is restricted to a toroidal 3x3 template-token window; shifted arm offsets the window center by half-grid on each axis; self-attention remains native.' if args.cross_support_diagnostic else None),
    }
    (out_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
