#!/usr/bin/env python3
"""Sequence-held-out linear probes for captured M1 search-stage features."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def fold_id(name):
    digest = hashlib.sha256(('m1-search-probe-20261009:' + name).encode()).hexdigest()
    return int(digest, 16) % 3


def fit_linear(features, positive, background, fit_indices):
    xs, ys = [], []
    for index in fit_indices:
        feat = torch.from_numpy(features[index].reshape(256, -1).astype(np.float32))
        target = torch.from_numpy(positive[index].reshape(-1))
        other = torch.from_numpy(background[index].reshape(-1))
        xs.append(feat[target])
        ys.append(torch.ones(int(target.sum())))
        bg_idx = torch.nonzero(other).reshape(-1)[::8]
        xs.append(feat[bg_idx])
        ys.append(torch.zeros(len(bg_idx)))
    x, y = torch.cat(xs), torch.cat(ys)
    mean = x.mean(0)
    std = x.std(0).clamp_min(.05)
    x = ((x - mean) / std).clamp(-8, 8)
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=.025)
    positive_weight = torch.tensor(float((y == 0).sum() / max(1, (y == 1).sum())))
    for _ in range(120):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ weight + bias, y,
                                                   pos_weight=positive_weight)
        loss = loss + .01 * weight.square().mean()
        loss.backward()
        optimizer.step()
    return mean.detach(), std.detach(), weight.detach(), float(bias.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    path = Path(args.features).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing nonempty output directory')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    data = np.load(str(path), allow_pickle=False)
    names = data['sequence'].astype(str)
    labels = data['label'].astype(str)
    unique = sorted(set(names.tolist()))
    folds = {name: fold_id(name) for name in unique}
    if any(sum(folds[name] == fold for name in unique) < 2 for fold in range(3)):
        raise RuntimeError('each held-out fold needs at least two independent sequences')
    rows = []
    for fold in range(3):
        fit_indices = [i for i, name in enumerate(names) if folds[name] != fold]
        dev_indices = [i for i, name in enumerate(names) if folds[name] == fold]
        for stage in ('rgb', 'tir', 'concat', 'fused'):
            feat = data[stage]
            mean, std, weight, bias = fit_linear(feat, data['target'],
                                                  data['background'], fit_indices)
            for index in dev_indices:
                x = torch.from_numpy(feat[index].reshape(256, -1).astype(np.float32))
                logits = (((x - mean) / std).clamp(-8, 8) @ weight + bias).reshape(16, 16)
                target = torch.from_numpy(data['target'][index])
                background = torch.from_numpy(data['background'][index])
                peak = int(logits.argmax().item())
                gx, gy = data['gt_xy'][index]
                rows.append({
                    'fold': fold, 'stage': stage, 'sequence': names[index],
                    'frame': int(data['frame'][index]), 'label': labels[index],
                    'target_bg_margin': float(logits[target].max() - logits[background].max()),
                    'peak_distance_tokens': math.hypot(peak % 16 + .5 - gx,
                                                       peak // 16 + .5 - gy),
                    'target_top': int(float(logits[target].max()) >= float(logits[background].max())),
                })
    with (output / 'probe_dev.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stages = {}
    for stage in ('rgb', 'tir', 'concat', 'fused'):
        stages[stage] = {}
        for label in sorted(set(labels.tolist())):
            subset = [row for row in rows if row['stage'] == stage and row['label'] == label]
            if subset:
                stages[stage][label] = {
                    'frames': len(subset), 'sequences': len(set(row['sequence'] for row in subset)),
                    'median_margin': float(np.median([row['target_bg_margin'] for row in subset])),
                    'median_peak_distance_tokens': float(np.median(
                        [row['peak_distance_tokens'] for row in subset])),
                    'target_top_fraction': float(np.mean([row['target_top'] for row in subset])),
                }
    summary = {
        'id': 'M1-SEARCH-STAGE-PROBE-20261009',
        'status': 'DIAGNOSTIC-COMPLETE',
        'features_file': str(path), 'folds': folds,
        'held_out_frames': len(names), 'train_steps_per_probe': 120,
        'probe_definition': '3-fold sequence-held-out linear 1x1 BCE readout; foreground GT-grid cells; every eighth background cell; fit-only normalization',
        'stages': stages,
        'limits': ['Probe tests decodability rather than original CENTER head use.',
                   'Input dimensions differ for concat; no equal-parameter causal claim.',
                   'GT-defined cells and sequence-level event sample are diagnostic only.'],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
