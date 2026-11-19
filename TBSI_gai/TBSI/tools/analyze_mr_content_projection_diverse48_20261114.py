#!/usr/bin/env python3
"""Recompute official LasHeR metrics for the pre-registered diverse48 cohort."""
import csv
import hashlib
import json
import statistics as stats
from pathlib import Path

import numpy as np
import torch

from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation import get_dataset


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output/experiments/MSHA-MR-CONTENT-PROJECTION-QUICK-S42-20261114'
CONTRACT = RUN / 'contracts/eval_content_projection_diverse48_20261114'
MANIFEST = CONTRACT.with_suffix('.txt')
CONTROL = RUN / 'eval_diverse48_control'
CANDIDATE = RUN / 'eval_diverse48_candidate'
ANALYSIS = RUN / 'analysis'
HISTORICAL = ROOT / 'output/experiments/MSHA-B0-U936-S43-FULL15-FULL244-20261112/analysis/two_seed_failure_20261113/per_sequence_two_seed.csv'
ATTRS = ['NO', 'PO', 'TO', 'HO', 'MB', 'LI', 'HI', 'AIV', 'LR', 'DEF', 'BC', 'SA', 'CM', 'TC', 'FL', 'OV', 'FM', 'SV', 'ARV']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_row(seq, arm_dir):
    pred = torch.tensor(np.loadtxt(str(arm_dir / (seq.name + '.txt')), ndmin=2), dtype=torch.float64)
    anno = torch.tensor(seq.ground_truth_rect, dtype=torch.float64)
    overlap, center, center_norm, valid = calc_seq_err_robust(pred, anno, seq.dataset)
    thresholds = torch.arange(0.0, 1.0 + 0.05, 0.05, dtype=torch.float64)
    auc = float(((overlap.view(-1, 1) > thresholds.view(1, -1)).float().mean(0).mean() * 100.0))
    times = np.loadtxt(str(arm_dir / (seq.name + '_time.txt')), ndmin=1)
    times = times[np.isfinite(times) & (times > 0)]
    fps = float(len(pred) / times.sum()) if len(times) else float('nan')
    return {
        'auc': auc,
        'precision': float((center <= 20.0).float().mean() * 100.0),
        'norm_precision': float((center_norm <= 0.20).float().mean() * 100.0),
        'op50': float((overlap > 0.50).float().mean() * 100.0),
        'op75': float((overlap > 0.75).float().mean() * 100.0),
        'mean_iou': float(overlap.mean()),
        'fps': fps,
        'frames': int(len(pred)),
    }


def mean(values):
    values = [float(v) for v in values if np.isfinite(v)]
    return float(stats.mean(values)) if values else float('nan')


def summarize(rows):
    out = {'n': len(rows)}
    for key in ['control_auc', 'candidate_auc', 'delta_auc', 'control_precision', 'candidate_precision',
                'delta_precision', 'control_norm_precision', 'candidate_norm_precision', 'delta_norm_precision',
                'control_op50', 'candidate_op50', 'delta_op50', 'control_op75', 'candidate_op75', 'delta_op75',
                'control_mean_iou', 'candidate_mean_iou', 'delta_mean_iou', 'control_fps', 'candidate_fps', 'delta_fps']:
        out[key] = mean([r[key] for r in rows])
    deltas = [r['delta_auc'] for r in rows]
    out.update({
        'wins': sum(x > 1e-9 for x in deltas),
        'ties': sum(abs(x) <= 1e-9 for x in deltas),
        'losses': sum(x < -1e-9 for x in deltas),
        'wins_05pp': sum(x > 0.5 for x in deltas),
        'ties_05pp': sum(abs(x) <= 0.5 for x in deltas),
        'losses_05pp': sum(x < -0.5 for x in deltas),
        'severe_losses_gt5pp': sum(x < -5.0 for x in deltas),
        'median_delta_auc': float(stats.median(deltas)) if deltas else float('nan'),
        'worst_delta_auc': min(deltas) if deltas else float('nan'),
        'best_delta_auc': max(deltas) if deltas else float('nan'),
    })
    positives = sorted((x for x in deltas if x > 0), reverse=True)
    out['top2_positive_fraction'] = float(sum(positives[:2]) / sum(positives)) if positives and sum(positives) else float('nan')
    out['mean_without_top2_positive'] = mean(sorted(deltas)[:-2]) if len(deltas) > 2 else float('nan')
    return out


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in MANIFEST.read_text().splitlines() if x.strip()]
    if len(names) != 48 or len(set(names)) != 48:
        raise RuntimeError('diverse48 manifest is not 48 unique sequences')
    for arm in [CONTROL, CANDIDATE]:
        missing = [n for n in names if not (arm / (n + '.txt')).exists() or not (arm / (n + '_time.txt')).exists()]
        if missing:
            raise RuntimeError('missing predictions for %s: %s' % (arm.name, missing))

    historical = {r['sequence']: r for r in csv.DictReader(HISTORICAL.open())}
    dataset = get_dataset('lasher_test')
    by_name = {s.name: s for s in dataset}
    if set(names) - set(by_name):
        raise RuntimeError('manifest sequence absent from LasHeR test dataset')
    rows = []
    for name in names:
        seq = by_name[name]
        c = metric_row(seq, CONTROL)
        q = metric_row(seq, CANDIDATE)
        h = historical[name]
        row = {'sequence': name, 'group': h['cross_seed_group'], 'frames': c['frames']}
        for attr in ATTRS:
            row['attr_' + attr] = int(h['attr_' + attr])
        for key in ['auc', 'precision', 'norm_precision', 'op50', 'op75', 'mean_iou', 'fps']:
            row['control_' + key] = c[key]
            row['candidate_' + key] = q[key]
            row['delta_' + key] = q[key] - c[key]
        rows.append(row)

    fields = list(rows[0])
    with (ANALYSIS / 'diverse48_per_sequence.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    old_rows = list(csv.DictReader((ANALYSIS / 'per_sequence.csv').open()))
    old_names = {r['sequence'] for r in old_rows}
    cohorts = {'all_48': rows, 'original_24': [r for r in rows if r['sequence'] in old_names],
               'added_24': [r for r in rows if r['sequence'] not in old_names]}
    for group in sorted({r['group'] for r in rows}):
        cohorts[group] = [r for r in rows if r['group'] == group]
    attr_summary = {}
    for attr in ATTRS:
        subset = [r for r in rows if r['attr_' + attr]]
        attr_summary[attr] = summarize(subset) if subset else {'n': 0}
    summary = {
        'run': RUN.name,
        'manifest': str(MANIFEST),
        'manifest_sha256': sha(MANIFEST),
        'scope': 'single-seed public-SOT 468-update quick; 48-sequence diverse development expansion; not full244 or holdout',
        'cohorts': {name: summarize(subset) for name, subset in cohorts.items()},
        'attributes': attr_summary,
        'severe_loss_sequences': [r['sequence'] for r in rows if r['delta_auc'] < -5.0],
        'large_gain_sequences': [r['sequence'] for r in sorted(rows, key=lambda x: x['delta_auc'], reverse=True)[:5]],
        'large_loss_sequences': [r['sequence'] for r in sorted(rows, key=lambda x: x['delta_auc'])[:5]],
        'inputs': {'historical_csv_sha256': sha(HISTORICAL), 'control_identity_sha256': sha(CONTROL / 'identity.json'),
                   'candidate_identity_sha256': sha(CANDIDATE / 'identity.json')},
        'descriptive_thresholds': {'tie_pp': 1e-9, 'band_pp': 0.5, 'severe_loss_pp': -5.0},
    }
    (ANALYSIS / 'diverse48_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    lines = ['# MR Content Projection：多样化 48 序列扩展评测', '',
             '两臂使用同一公共 SOT 初始化、seed42、468 successful updates；candidate 仅开启 `bg_protect` Content Projection（alpha=0.25）。推理为 FP32、threads=0、template update off。该 cohort 是预注册的开发扩展，不是 full244，也不是独立 holdout。', '',
             '| cohort | n | control AUC | candidate AUC | ΔAUC | 胜/平/负 | ±0.5pp 胜/平/负 | 中位 Δ | >5pp 回退 |',
             '|---|---:|---:|---:|---:|---|---|---:|---:|']
    for name in ['all_48', 'original_24', 'added_24', 'consistent_mr_gain', 'consistent_mr_regression', 'seed_direction_flip', 'one_seed_tie']:
        s = summary['cohorts'].get(name)
        if not s:
            continue
        lines.append('| {} | {} | {:.3f} | {:.3f} | {:+.3f} | {}/{}/{} | {}/{}/{} | {:+.3f} | {} |'.format(
            name, s['n'], s['control_auc'], s['candidate_auc'], s['delta_auc'], s['wins'], s['ties'], s['losses'],
            s['wins_05pp'], s['ties_05pp'], s['losses_05pp'], s['median_delta_auc'], s['severe_losses_gt5pp']))
    lines += ['', '## 结论', '',
              '- `all_48` 的总体均值、逐序列胜负和尾部同时报告；不能仅用原 24 条的方向推断扩展集。',
              '- 属性统计是描述性分层，属性可重叠且来自历史标签，不能作为因果退化证明。',
              '- 只有当新增 24 条不出现明显方向反转、目标分层收益和补集损伤可接受、严重回退减少时，才有理由把该机制列入 full 候选；本报告本身不自动晋级。', '',
              '详细逐序列指标见 `diverse48_per_sequence.csv`，机器可读汇总见 `diverse48_summary.json`。']
    (ANALYSIS / 'DIVERSE48_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
