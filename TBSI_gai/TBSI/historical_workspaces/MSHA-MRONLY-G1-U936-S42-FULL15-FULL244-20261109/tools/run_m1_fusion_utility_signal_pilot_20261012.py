#!/usr/bin/env python3
"""Fast fixed-crop test of whether pre-fusion signals predict CENTER utility."""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack

CONFIG_NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
CONFIG = ROOT / 'experiments/tbsi_track' / (CONFIG_NAME + '.yaml')
CHECKPOINT = ROOT / 'output/experiments' / CONFIG_NAME / 'checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG_SHA = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
CHECKPOINT_SHA = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def margin(score, target, background):
    target_mask = torch.as_tensor(target, dtype=torch.bool, device=score.device)
    background_mask = torch.as_tensor(background, dtype=torch.bool, device=score.device)
    return float(score[target_mask].max().item() - score[background_mask].max().item())


def rank(score, gt_xy):
    gx = min(15, max(0, int(float(gt_xy[0]))))
    gy = min(15, max(0, int(float(gt_xy[1]))))
    flat = score.reshape(-1)
    value = score[gy, gx]
    return int((flat > value).sum().item() + 1)


def attenuate(x, branch, target, strength):
    result = x.clone()
    channels = result.shape[1] // 2
    start = branch * channels
    flat = result[:, start:start + channels].flatten(2)
    selected = np.flatnonzero(target.reshape(-1))
    background = np.flatnonzero(~target.reshape(-1))
    index = torch.as_tensor(selected, dtype=torch.long, device=result.device)
    bg_index = torch.as_tensor(background, dtype=torch.long, device=result.device)
    mean = flat.index_select(2, bg_index).mean(2, keepdim=True)
    previous = flat.index_select(2, index)
    flat[:, :, index] = (1.0 - strength) * previous + strength * mean
    return result


def forward(network, tracker, x, target, background, gt_xy):
    with torch.no_grad():
        fused = network.tbsi_fuse_search(x)
        score, _bbox, _size, _offset = network.box_head(fused)
    score = (score * tracker.output_window)[0, 0].float()
    return {'margin': margin(score, target, background), 'gt_rank': rank(score, gt_xy),
            'peak': int(score.argmax().item())}


def corr(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < 1e-12 or np.std(y[valid]) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--signal-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--strength', type=float, default=0.5)
    parser.add_argument('--provenance-note', default='Sequence overlap must be audited from the frozen manifest.')
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing nonempty output directory: ' + str(output))
    if sha256(CONFIG) != CONFIG_SHA or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if not 0.0 < args.strength <= 1.0:
        raise RuntimeError('strength must be in (0, 1]')

    data = np.load(args.features)
    state_features = {key: data[key] for key in
                      ('concat', 'fused', 'target', 'background', 'gt_xy', 'sequence', 'frame', 'label')}
    signals = load_csv(args.signal_csv)
    keys = [(str(s), int(f)) for s, f in zip(state_features['sequence'], state_features['frame'])]
    signal_map = {(r['sequence'], int(r['frame'])): r for r in signals}
    if len(keys) != len(set(keys)) or any(key not in signal_map for key in keys):
        raise RuntimeError('feature/signal state keys do not match uniquely')

    os.environ['TBSI_INFER_FP16'] = '0'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    params = parameters(CONFIG_NAME)
    params.checkpoint = str(CHECKPOINT)
    params.debug = 0
    params.save_all_boxes = False
    tracker = TBSITrack(params, 'm1_fusion_utility_signal_pilot')
    network = tracker.network.eval()
    rows = []
    parity_max = 0.0
    started = time.time()
    for i, key in enumerate(keys):
        x = torch.as_tensor(state_features['concat'][i].transpose(2, 0, 1)[None],
                            dtype=torch.float32, device='cuda')
        target = state_features['target'][i].astype(bool)
        background = state_features['background'][i].astype(bool)
        gt_xy = state_features['gt_xy'][i]
        base = forward(network, tracker, x, target, background, gt_xy)
        saved_fused = state_features['fused'][i].astype(np.float32)
        # The archive stores half precision features; quantify that replay boundary.
        with torch.no_grad():
            fused = network.tbsi_fuse_search(x)[0].permute(1, 2, 0).float().cpu().numpy()
        parity_max = max(parity_max, float(np.max(np.abs(fused - saved_fused))))
        utility = {}
        for branch, name in enumerate(('rgb', 'tir')):
            target_arm = forward(network, tracker,
                                 attenuate(x, branch, target, args.strength),
                                 target, background, gt_xy)
            # Matched area control uses the first same-sized background cells.
            control_mask = np.zeros_like(target)
            control_mask.reshape(-1)[np.flatnonzero(background)[:int(target.sum())]] = True
            control_arm = forward(network, tracker,
                                  attenuate(x, branch, control_mask, args.strength),
                                  target, background, gt_xy)
            utility[name] = {
                'margin_loss_target': base['margin'] - target_arm['margin'],
                'margin_loss_control': base['margin'] - control_arm['margin'],
                'net_utility': control_arm['margin'] - target_arm['margin'],
                'rank_worsening_target': target_arm['gt_rank'] - base['gt_rank'],
                'rank_worsening_control': control_arm['gt_rank'] - base['gt_rank']}
        signal = signal_map[key]
        rows.append({
            'sequence': key[0], 'frame': key[1], 'label': str(state_features['label'][i]),
            'base_margin': base['margin'], 'base_gt_rank': base['gt_rank'],
            'rgb_utility': utility['rgb']['net_utility'],
            'tir_utility': utility['tir']['net_utility'],
            'utility_gap_rgb_minus_tir': utility['rgb']['net_utility'] - utility['tir']['net_utility'],
            'rgb_rank_worsening_net': utility['rgb']['rank_worsening_target'] - utility['rgb']['rank_worsening_control'],
            'tir_rank_worsening_net': utility['tir']['rank_worsening_target'] - utility['tir']['rank_worsening_control'],
            'energy_gap_rgb_minus_tir': float(signal['local_feature_energy_ratio_gap_rgb_minus_tir']),
            'template_support_gap_rgb_minus_tir': float(signal['template_support_gap_rgb_minus_tir']),
            'template_psr_gap_rgb_minus_tir': float(signal['template_support_psr_gap_rgb_minus_tir']),
            'conflict': float(signal['local_conflict_at_center_peak']),
            'label_severe': int(str(data['label'][i]) == 'severe_onset')})
        if (i + 1) % 10 == 0:
            print('processed {}/{}'.format(i + 1, len(keys)), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (output / 'per_state_utility.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    specs = {
        'energy_gap_rgb_minus_tir': 1,
        'template_support_gap_rgb_minus_tir': 1,
        'template_psr_gap_rgb_minus_tir': 1,
        'conflict': 0}
    summary = {'id': 'M1-FUSION-UTILITY-SIGNAL-PILOT-20261012',
               'status': 'DIAGNOSTIC-COMPLETE', 'n_states': len(rows),
               'n_sequences': len(set(row['sequence'] for row in rows)),
               'n_severe': sum(row['label_severe'] for row in rows),
               'intervention': 'fixed M1 crop; target-cell feature attenuation versus equal-area background control; CENTER Hann target-background margin',
               'strength': args.strength, 'candidate_signals': {},
               'utility_gap_mean': float(np.mean([r['utility_gap_rgb_minus_tir'] for r in rows])),
               'utility_gap_median': float(np.median([r['utility_gap_rgb_minus_tir'] for r in rows])),
               'utility_sign_consistency': float(np.mean([
                   np.sign(r['rgb_rank_worsening_net'] - r['tir_rank_worsening_net']) ==
                   np.sign(r['utility_gap_rgb_minus_tir']) for r in rows
                   if r['utility_gap_rgb_minus_tir'] != 0])),
               'replay_feature_max_abs_delta_vs_saved_fp16': parity_max}
    for signal_name, direction in specs.items():
        x = [r[signal_name] for r in rows]
        y = [r['utility_gap_rgb_minus_tir'] for r in rows]
        summary['candidate_signals'][signal_name] = {
            'pearson_r_with_utility_gap': corr(x, y),
            'directional_accuracy': float(np.mean([
                np.sign(float(a) * direction) == np.sign(float(b))
                for a, b in zip(x, y) if a != 0 and b != 0])) if any(a != 0 and b != 0 for a, b in zip(x, y)) else None}
    summary.update({'elapsed_seconds': time.time() - started,
                    'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
                    'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'config_sha256': sha256(CONFIG), 'checkpoint_sha256': sha256(CHECKPOINT),
                    'features_sha256': sha256(args.features), 'signal_csv_sha256': sha256(args.signal_csv),
                    'cohort_provenance': args.provenance_note,
                    'limitations': ['sequence overlap follows the supplied cohort provenance note',
                                    'utility is an offline controlled perturbation proxy, not a learned gate benefit',
                                    'CENTER target-background margin isolates selection evidence, not final IoU or closed-loop recovery']})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
