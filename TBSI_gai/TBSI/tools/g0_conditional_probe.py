#!/usr/bin/env python3
"""Frozen-feature conditional probes with sequence-isolated train/validation splits."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


GROUPS = ('L', 'L_plus_E', 'L_plus_Lcontrol', 'L_plus_Eshuffle')


def load_events(export_dir, labels_dir):
    labels = {}
    with (Path(labels_dir) / 'token_labels.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            labels[row['event_id']] = row
    events = []
    for event_id in sorted(labels):
        row = labels[event_id]
        if not row['rescue_eligible']:
            continue
        arrays = dict(np.load(Path(export_dir) / (event_id + '.npz')))
        row['arrays'] = arrays
        events.append(row)
    return events


def align(a, n=256):
    a = torch.as_tensor(a, dtype=torch.float32)
    if a.shape[0] == n:
        return a
    side = int(round(a.shape[0] ** 0.5))
    if side * side != a.shape[0]:
        raise ValueError('non-square token grid {}'.format(tuple(a.shape)))
    return torch.nn.functional.interpolate(a.T.reshape(1, a.shape[1], side, side),
                                           size=(16, 16), mode='bilinear', align_corners=False).reshape(a.shape[1], n).T


def make_inputs(events, seed):
    rng = random.Random(seed)
    by_seq = {}
    for event in events:
        by_seq.setdefault(event['sequence'], []).append(event)
    sequences = sorted(by_seq)
    rng.shuffle(sequences)
    split = max(1, int(round(len(sequences) * 0.75)))
    train_sequences = set(sequences[:split])
    train = [e for e in events if e['sequence'] in train_sequences]
    valid = [e for e in events if e['sequence'] not in train_sequences]

    def build(rows, shuffle_pool):
        prepared = []
        pool = list(shuffle_pool)
        rng.shuffle(pool)
        for index, event in enumerate(rows):
            ar = event['arrays']
            er = torch.cat([align(ar['early_rgb']), align(ar['early_tir'])], dim=1)
            late = torch.cat([align(ar['late_input_rgb']), align(ar['late_input_tir'])], dim=1)
            # Matched late control preserves dimensionality and per-event marginal scale.
            permutation = torch.roll(torch.arange(late.shape[0]), shifts=(index % max(late.shape[0], 1)))
            late_control = late[permutation]
            donor = pool[index % len(pool)]
            dar = donor['arrays']
            donor_early = torch.cat([align(dar['early_rgb']), align(dar['early_tir'])], dim=1)
            prepared.append({'event': event, 'inputs': {
                'L': late, 'L_plus_E': torch.cat([late, er], dim=1),
                'L_plus_Lcontrol': torch.cat([late, late_control], dim=1),
                'L_plus_Eshuffle': torch.cat([late, donor_early], dim=1)}})
        return prepared
    return build(train, train), build(valid, valid), sorted(train_sequences), sorted(set(sequences) - train_sequences)


class Probe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_group(rows, group, epochs, seed):
    torch.manual_seed(seed)
    x = torch.cat([r['inputs'][group] for r in rows], dim=0)
    y = torch.cat([torch.tensor(r['event']['target_mask'], dtype=torch.float32) for r in rows], dim=0)
    positive = y > 0.5
    negative_indices = torch.where(~positive)[0]
    positive_indices = torch.where(positive)[0]
    if not len(positive_indices) or not len(negative_indices):
        raise RuntimeError('missing positive or negative probe tokens')
    generator = torch.Generator().manual_seed(seed)
    neg_count = min(len(negative_indices), len(positive_indices) * 3)
    selected_neg = negative_indices[torch.randperm(len(negative_indices), generator=generator)[:neg_count]]
    selected = torch.cat([positive_indices, selected_neg])
    mean = x[selected].mean(dim=0)
    std = x[selected].std(dim=0).clamp_min(1e-6)
    x_train = (x[selected] - mean) / std
    model = Probe(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y[selected])
        loss.backward()
        optimizer.step()
    return model.eval(), mean, std


def average_precision(scores, labels):
    order = np.argsort(-scores)
    labels = np.asarray(labels)[order]
    positives = float(labels.sum())
    if positives == 0:
        return float('nan')
    precision = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    return float((precision * labels).sum() / positives)


def evaluate(model, mean, std, rows, group):
    result = []
    with torch.no_grad():
        for row in rows:
            x = (row['inputs'][group] - mean) / std
            scores = torch.sigmoid(model(x)).numpy()
            labels = np.asarray(row['event']['target_mask'], dtype=np.float32)
            result.append({'event_id': row['event']['event_id'], 'sequence': row['event']['sequence'],
                           'ap': average_precision(scores, labels),
                           'target_rank': int(np.where(np.argsort(-scores) == np.argmax(labels))[0][0] + 1)})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export-dir', required=True)
    ap.add_argument('--labels-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing non-empty output {}'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    events = load_events(args.export_dir, args.labels_dir)
    train, valid, train_sequences, valid_sequences = make_inputs(events, args.seed)
    if not train or not valid:
        raise RuntimeError('sequence-isolated split is empty')
    metrics = {'groups': {}, 'train_events': len(train), 'valid_events': len(valid),
               'train_sequences': train_sequences, 'valid_sequences': valid_sequences,
               'seed': args.seed, 'epochs': args.epochs}
    for group in GROUPS:
        model, mean, std = fit_group(train, group, args.epochs, args.seed)
        rows = evaluate(model, mean, std, valid, group)
        metrics['groups'][group] = {'events': rows, 'mean_ap': float(np.mean([r['ap'] for r in rows])),
                                    'median_ap': float(np.median([r['ap'] for r in rows]))}
    base = {r['event_id']: r for r in metrics['groups']['L']['events']}
    for group in GROUPS[1:]:
        for row in metrics['groups'][group]['events']:
            row['delta_ap_vs_L'] = row['ap'] - base[row['event_id']]['ap']
    metrics['paired'] = {}
    for group in GROUPS[1:]:
        deltas = [r['delta_ap_vs_L'] for r in metrics['groups'][group]['events']]
        metrics['paired'][group] = {'mean_delta_ap': float(np.mean(deltas)),
                                    'median_delta_ap': float(np.median(deltas)),
                                    'positive_event_fraction': float(np.mean(np.asarray(deltas) > 0))}
    (out / 'probe_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics['paired'], indent=2))


if __name__ == '__main__':
    main()
