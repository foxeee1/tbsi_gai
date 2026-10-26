#!/usr/bin/env python3
"""Summarize the existing M1 fixed-state cell-reachability audit.

This is an analysis-only artifact.  It intentionally does not rerun tracking,
train a model, or use GT to choose an online action.
"""
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'output/experiments/M1_CANDIDATE_RECALL_READOUT_20261010/analysis'
RUN = ROOT / 'output/experiments/M1_S1_CENTER_CROP_REACHABILITY_20261016_retry1'


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def mean(rows, key):
    if key == 'gt_cell_minus_top1_iou':
        vals = [float(r['gt_cell_iou']) - float(r['top1_iou']) for r in rows]
    else:
        vals = [float(r[key]) for r in rows]
    return sum(vals) / len(vals) if vals else None


def summarize(rows):
    out = {}
    for label in ('all', 'matched_good', 'pre_onset', 'severe_onset'):
        subset = rows if label == 'all' else [r for r in rows if r['state_label'] == label]
        item = {'events': len(subset), 'sequences': len({r['sequence'] for r in subset})}
        for k in (1, 3, 5, 8, 16, 32, 256):
            item[f'gt_cell_recall_at_{k}'] = mean(subset, f'gt_cell_recall_at_{k}')
            item[f'gt_support_recall_at_{k}'] = mean(subset, f'gt_support_recall_at_{k}')
            item[f'oracle_iou_at_{k}_mean'] = mean(subset, f'oracle_iou_at_{k}')
        for key in ('top1_iou', 'gt_cell_iou', 'gt_cell_minus_top1_iou',
                    'gt_cell_center_error_tokens', 'gt_3x3_best_iou',
                    'allmap_oracle_iou'):
            item[f'{key}_mean'] = mean(subset, key)
        out[label] = item
    return out


def main():
    if RUN.exists():
        raise RuntimeError(f'refusing existing output: {RUN}')
    rows = read_csv(SRC / 'per_event.csv')
    RUN.mkdir(parents=True)
    shutil.copy2(SRC / 'per_event.csv', RUN / 'per_event.csv')

    grouped = defaultdict(list)
    for row in rows:
        grouped[row['sequence']].append(row)
    seq_rows = []
    for seq, group in sorted(grouped.items()):
        item = {'sequence': seq, 'events': len(group)}
        for label in ('matched_good', 'pre_onset', 'severe_onset'):
            part = [r for r in group if r['state_label'] == label]
            item[f'{label}_events'] = len(part)
            item[f'{label}_top1_iou'] = mean(part, 'top1_iou')
            item[f'{label}_gt_cell_iou'] = mean(part, 'gt_cell_iou')
            item[f'{label}_center_oracle_gain'] = mean(part, 'gt_cell_minus_top1_iou')
            item[f'{label}_gt_cell_rank'] = mean(part, 'gt_cell_rank')
            item[f'{label}_gt_cell_recall_at_8'] = mean(part, 'gt_cell_recall_at_8')
        seq_rows.append(item)
    with (RUN / 'per_sequence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(seq_rows[0]))
        writer.writeheader(); writer.writerows(seq_rows)

    summary = {
        'id': 'M1-S1-CENTER-CROP-REACHABILITY-20261016',
        'status': 'DIAGNOSTIC-COMPLETE',
        'parent': 'Protected M1 TBSI_E_MHSA_FULL_4090D_seed42 epoch15',
        'source_audit': str(SRC.relative_to(ROOT)),
        'scope': 'M1 fixed-state train-side replay; 60 sequences, 123 states; no training and no closed-loop state update',
        'summary': summarize(rows),
        'decision_rule': {
            'center_route_if': 'GT-cell/top-k or center oracle leaves material IoU headroom, especially severe-onset',
            'stop_readout_if': 'center oracle is small and all-map oracle is close to top1',
            'deployment_warning': 'GT-cell and oracle selection are offline upper bounds, not deployable gains',
        },
        'limitations': [
            'Train-side cohort, not official full-244 test.',
            'The source audit replays M1 fixed states; it does not measure S1 closed-loop performance.',
            'Center oracle keeps each cell size/offset maps unchanged, so it isolates center selection only.',
            'Source audit reports medians; this derived artifact reports means for transparent aggregation.',
        ],
    }
    (RUN / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
