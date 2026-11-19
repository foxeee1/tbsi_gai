#!/usr/bin/env python3
"""Official paired metrics for the MR content-projection quick run."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation import get_dataset

METRICS = ('auc', 'precision', 'norm_precision', 'op50', 'op75', 'mean_iou', 'fps')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sequence_metrics(sequence, folder):
    folder = Path(folder)
    pred = np.loadtxt(str(folder / (sequence.name + '.txt')), ndmin=2)
    gt = np.asarray(sequence.ground_truth_rect, dtype=np.float64)
    times = np.loadtxt(str(folder / (sequence.name + '_time.txt')), ndmin=1)
    if pred.shape != gt.shape or not np.isfinite(pred).all():
        raise ValueError('prediction contract failed for ' + sequence.name)
    if times.shape[0] != gt.shape[0] or not np.isfinite(times).all() or (times <= 0).any():
        raise ValueError('timing contract failed for ' + sequence.name)
    visible = (torch.as_tensor(sequence.target_visible, dtype=torch.uint8)
               if sequence.target_visible is not None else None)
    overlap, center, norm, valid = calc_seq_err_robust(
        torch.as_tensor(pred, dtype=torch.float64), torch.as_tensor(gt, dtype=torch.float64),
        sequence.dataset, visible)
    overlap, center, norm, valid = [x.detach().cpu().numpy() for x in (overlap, center, norm, valid)]
    thresholds = np.arange(21, dtype=np.float64) * .05
    success = (overlap[:, None] > thresholds[None, :]).mean(axis=0) * 100.0
    return {
        'auc': float(success.mean()), 'precision': float((center <= 20).mean() * 100.0),
        'norm_precision': float((norm <= .2).mean() * 100.0),
        'op50': float(success[10]), 'op75': float(success[15]),
        'mean_iou': float(overlap.mean()), 'fps': float(len(times) / times.sum()),
        'frames': int(len(gt)), 'valid_frames': int(valid.sum()),
    }


def aggregate(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--list', required=True)
    ap.add_argument('--groups', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    names = [line.strip() for line in Path(args.list).read_text().splitlines() if line.strip()]
    wanted = set(names)
    dataset = {seq.name: seq for seq in get_dataset('lasher_test') if seq.name in wanted}
    if set(dataset) != wanted:
        raise ValueError('evaluation list does not match dataset: ' + repr(sorted(wanted - set(dataset))))
    groups = json.loads(Path(args.groups).read_text())['groups']
    control_rows, candidate_rows = {}, {}
    for name in names:
        control_rows[name] = sequence_metrics(dataset[name], args.control)
        candidate_rows[name] = sequence_metrics(dataset[name], args.candidate)
    rows = []
    for name in names:
        row = {'sequence': name}
        for key in METRICS:
            row['control_' + key] = control_rows[name][key]
            row['candidate_' + key] = candidate_rows[name][key]
            row['delta_' + key] = candidate_rows[name][key] - control_rows[name][key]
        row['frames'] = control_rows[name]['frames']
        row['group'] = next((label for label, members in groups.items() if name in members), 'unassigned')
        rows.append(row)
    with (out / 'per_sequence.csv').open('w', newline='') as f:
        fields = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    by_group = {'all_24': {'n': len(rows), 'control': aggregate([control_rows[n] for n in names]),
                           'candidate': aggregate([candidate_rows[n] for n in names])}}
    for label, members in groups.items():
        selected = [n for n in names if n in members]
        by_group[label] = {'n': len(selected), 'control': aggregate([control_rows[n] for n in selected]),
                           'candidate': aggregate([candidate_rows[n] for n in selected])}
    for item in by_group.values():
        item['delta'] = {key: item['candidate'][key] - item['control'][key] for key in METRICS}
    wins = {key: int(sum(r['delta_' + key] > 0 for r in rows)) for key in METRICS}
    ties = {key: int(sum(r['delta_' + key] == 0 for r in rows)) for key in METRICS}
    losses = {key: int(sum(r['delta_' + key] < 0 for r in rows)) for key in METRICS}
    summary = {'run': 'MSHA-MR-CONTENT-PROJECTION-QUICK-S42-20261114',
               'cohort': {'sequences': len(rows), 'names': names,
                          'manifest_sha256': sha(args.list), 'groups_sha256': sha(args.groups)},
               'groups': by_group, 'wins_ties_losses': {'wins': wins, 'ties': ties, 'losses': losses},
               'largest_gains_auc': sorted(rows, key=lambda r: r['delta_auc'], reverse=True)[:5],
               'largest_losses_auc': sorted(rows, key=lambda r: r['delta_auc'])[:5],
               'scope': 'single-seed 468-update quick screen; public SOT initialization; exploratory 24-sequence development cohort'}
    (out / 'paired_metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# MR content projection quick paired result', '',
             '两臂均从公共 SOT 预训练权重单阶段训练，seed42，59,904 pairs/468 successful updates；candidate 只开启模板背景 Value rank-1 投影（alpha=0.25）。评测为 FP32、threads=0、template update off 的预先固定 24 序列开发 cohort。',
             '', '| cohort | n | AUC control | AUC candidate | dAUC | Precision d | NPR d | OP50 d | OP75 d | mean IoU d |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, item in by_group.items():
        c, d = item['control'], item['delta']
        lines.append('| %s | %d | %.3f | %.3f | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f | %+.5f |' %
                     (label, item['n'], c['auc'], item['candidate']['auc'], d['auc'], d['precision'],
                      d['norm_precision'], d['op50'], d['op75'], d['mean_iou']))
    lines += ['', '## 判定', '',
              '- 该结果只支持当前 quick 合同下的机制/排序判断，不能表述为 full-data 或多 seed 稳健提升。',
              '- 逐序列明细见 `per_sequence.csv`；正负收益共存时，优先检查作用位置和后续 readout 敏感性，不据此直接晋级 full。',
              '', '## 资产', '', '- 训练 control/candidate、评测身份、输入 hash 和存储清理记录位于本 run 目录。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'all_24': by_group['all_24'], 'wins': wins, 'ties': ties, 'losses': losses}, indent=2))


if __name__ == '__main__':
    main()
