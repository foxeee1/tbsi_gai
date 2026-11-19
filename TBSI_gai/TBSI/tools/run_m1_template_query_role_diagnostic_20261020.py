#!/usr/bin/env python3
"""Paired fixed-state diagnostic for foreground-vs-context template queries."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / 'tools/run_m1_template_query_stage_diagnostic_20261020.py'
SPEC = importlib.util.spec_from_file_location('m1_query_stage_base', str(BASE_PATH))
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
OUT = ROOT / 'output/experiments/M1_TEMPLATE_QUERY_ROLE_DIAGNOSTIC_20261020'
EVENTS = ROOT / 'output/experiments/M1_TEMPLATE_QUERY_STAGE_DIAGNOSTIC_20261020/validation_events.json'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def role_query_stats(rgb_t, tir_t, rgb_s, tir_s, fg_mask):
    rgb = F.normalize(rgb_t.float(), dim=-1, eps=1e-6)
    tir = F.normalize(tir_t.float(), dim=-1, eps=1e-6)
    rgb_search = F.normalize(rgb_s.float(), dim=-1, eps=1e-6)
    tir_search = F.normalize(tir_s.float(), dim=-1, eps=1e-6)
    fg = fg_mask.to(device=rgb.device, dtype=rgb.dtype).clamp(0, 1)
    if fg.ndim == 1:
        fg = fg.unsqueeze(0)

    def contrast(tokens, search, foreground):
        context = 1.0 - foreground
        fg_proto = F.normalize(
            (tokens * foreground.unsqueeze(-1)).sum(1) /
            foreground.sum(1, keepdim=True).clamp_min(1e-6), dim=-1, eps=1e-6)
        ctx_proto = F.normalize(
            (tokens * context.unsqueeze(-1)).sum(1) /
            context.sum(1, keepdim=True).clamp_min(1e-6), dim=-1, eps=1e-6)
        token_match = torch.amax(tokens @ search.transpose(1, 2), dim=-1)
        fg_match = torch.amax(fg_proto.unsqueeze(1) @ search.transpose(1, 2), dim=-1)
        ctx_match = torch.amax(ctx_proto.unsqueeze(1) @ search.transpose(1, 2), dim=-1)
        return foreground * (token_match - ctx_match) + context * (token_match - fg_match)

    agreement = (rgb * tir).sum(-1)[0]
    current_q = 0.5 * torch.amax(rgb @ rgb_search.transpose(1, 2), -1)[0] + \
        0.5 * torch.amax(tir @ tir_search.transpose(1, 2), -1)[0]
    role_q = 0.5 * contrast(rgb, rgb_search, fg) + \
        0.5 * contrast(tir, tir_search, fg)
    role_q = role_q[0]
    reliability = 0.5 * agreement + 0.5 * role_q
    participation = torch.sigmoid((reliability - 0.5) / 0.15).detach().cpu()
    fg = fg_mask.cpu().bool()
    bg = ~fg
    return {
        'query_match_mean': float(current_q.mean()),
        'query_fg_mean': float(current_q[fg].mean()),
        'query_bg_mean': float(current_q[bg].mean()),
        'target_context_query_mean': float(role_q.mean()),
        'target_context_query_fg_mean': float(role_q[fg].mean()),
        'target_context_query_bg_mean': float(role_q[bg].mean()),
        'reliability_mean': float(reliability.mean()),
        'participation_mean': float(participation.mean()),
        'participation_fg_mean': float(participation[fg].mean()),
        'participation_bg_mean': float(participation[bg].mean()),
        'participation_fg_bg_delta': float(participation[fg].mean() - participation[bg].mean()),
    }


def main():
    if sha256(base.CONFIG) != base.CONFIG_SHA256 or sha256(base.CHECKPOINT) != base.CHECKPOINT_SHA256:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    events = json.loads(EVENTS.read_text())
    if len(events) != 5 or any(event.get('role') != 'validation' for event in events):
        raise RuntimeError('expected the frozen five train-side validation events only')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    base.query_stats = role_query_stats
    names = sorted({event['sequence'] for event in events})
    list_path = OUT / 'sequence_list.txt'
    list_path.write_text('\n'.join(names) + '\n')
    dataset = base.LasHeRDataset('trainingset', str(list_path)).get_sequence_list()
    seqmap = {sequence.name: sequence for sequence in dataset}
    rows = []

    for index, event in enumerate(events, 1):
        sequence = seqmap[event['sequence']]
        off, bg = base.run_fixed_arm(event, sequence, 'off')
        os.environ['TBSI_TEMPLATE_UNCERTAINTY_THRESHOLD'] = '0.5'
        os.environ['TBSI_TEMPLATE_UNCERTAINTY_TEMPERATURE'] = '0.15'
        role, _ = base.run_fixed_arm(event, sequence, 'target_context_soft', bg)
        role['pair_id'] = off['pair_id'] = '{}:{}'.format(event['sequence'], event['frame'])
        for key, value in off.items():
            if key not in ('pair_id', 'sequence', 'frame', 'role', 'mode', 'state_before', 'gt_box', 'crop_xywh'):
                role['off_' + key] = value
                if isinstance(value, (int, float)) and isinstance(role.get(key), (int, float)):
                    role[key + '_delta'] = role[key] - value
        rows.extend((off, role))
        print('[fixed] {}/{} {} frame={} participation_fg-bg={:+.5f} margin_delta={:+.5f} rank={}->{}'.format(
            index, len(events), event['sequence'], event['frame'],
            role.get('loc9_participation_fg_bg_delta', float('nan')),
            role.get('gt_hard_bg_margin_delta', float('nan')),
            off['gt_rank'], role['gt_rank']), flush=True)

    columns = sorted({key for row in rows for key in row})
    with (OUT / 'per_event.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    selected = [row for row in rows if row['mode'] == 'target_context_soft']
    summary = {
        'protocol': {
            'parent': 'protected TBSI_E_MHSA_FULL_4090D_seed42 ep0015',
            'split': 'LasHeR trainingset, frozen identity_calibration_v2 validation events',
            'arms': ['off', 'target_context_soft tau=0.5 T=0.15'],
            'role_query': 'per-token max search cosine minus opposite-role prototype max search cosine, averaged RGB/TIR',
            'participation': 'sigmoid((0.5*cross_modal_agreement + 0.5*role_query - 0.5)/0.15)',
            'paired_controls': 'same template initialization, state, frame, crop, and off-arm hard-background cell',
            'excluded': 'all test split and training'} ,
        'n_validation_events': len(selected),
        'mean_margin_delta': float(np.mean([row['gt_hard_bg_margin_delta'] for row in selected])),
        'mean_rank_delta': float(np.mean([row['gt_rank_delta'] for row in selected])),
        'rank_improved': int(sum(row['gt_rank_delta'] < 0 for row in selected)),
        'top1_became_gt': int(sum(not row['off_top1_is_gt'] and row['top1_is_gt'] for row in selected)),
        'participation_fg_bg_delta_loc9_mean': float(np.mean([
            row['loc9_participation_fg_bg_delta'] for row in selected])),
        'query_match_fg_bg_delta_loc9_mean': float(np.mean([
            row['loc9_query_fg_mean'] - row['loc9_query_bg_mean'] for row in selected])),
        'target_context_query_fg_bg_delta_loc9_mean': float(np.mean([
            row['loc9_target_context_query_fg_mean'] - row['loc9_target_context_query_bg_mean']
            for row in selected])),
        'config_sha256': sha256(base.CONFIG),
        'checkpoint_sha256': sha256(base.CHECKPOINT),
        'validation_events_sha256': sha256(EVENTS),
        'validation_sequence_list_sha256': sha256(
            ROOT / 'experiments/diagnostics/m1_identity_calibration_v2/validation_sequences.txt'),
        'git_head': os.popen('git -C "{}" rev-parse HEAD'.format(ROOT)).read().strip(),
        'tbsi_layer_sha256': sha256(ROOT / 'lib/models/layers/tbsi_layer.py'),
        'runner_sha256': sha256(Path(__file__).resolve()),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
