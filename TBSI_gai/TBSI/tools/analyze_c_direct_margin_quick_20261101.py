#!/usr/bin/env python3
"""Paired analysis for the C direct localization margin quick run."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import analyze_early_gate_p1_v2 as base

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'EG_C_DIRECT_MARGIN_QUICK_RETRY1_S42_20261101'
RUN = ROOT / 'output/experiments' / RUN_ID
P1 = ROOT / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def freeze(args):
    names = [line.strip() for line in Path(args.list).read_text().splitlines() if line.strip()]
    manifest = json.loads((P1 / 'data_manifest.json').read_text())
    frames = {item['sequence']: item['frames'] for item in manifest['test_sequences']}
    if len(names) != 96 or len(set(names)) != 96 or not set(names) <= set(frames):
        raise ValueError('Expected fixed screen96_max2000 manifest')
    rows = []
    for name in names:
        metrics, _ = base.read_result(args.control_test, name, frames[name])
        rows.append({'sequence': name, **metrics})
    ranked = sorted(rows, key=lambda row: (row['auc'], row['sequence']))
    groups = {'low_target': [row['sequence'] for row in ranked[:32]],
              'middle': [row['sequence'] for row in ranked[32:64]],
              'normal_protection': [row['sequence'] for row in ranked[64:]]}
    write_json(RUN / 'control_groups.json', {'definition': 'Fresh C control AUC tertiles, frozen before candidate test',
        'control_test': str(Path(args.control_test).resolve()), 'groups': groups, 'control_metrics': rows})
    print(json.dumps({key: len(value) for key, value in groups.items()}))


def compare(args):
    pairing = base.validate_pair(args.control_train, args.candidate_train, RUN / 'analysis/pairing.json')
    ctrl_cal = json.loads((Path(args.control_train) / 'direct_margin_calibration.json').read_text())
    cand_cal = json.loads((Path(args.candidate_train) / 'direct_margin_calibration.json').read_text())
    if ctrl_cal['microbatch_sha256'] != cand_cal['microbatch_sha256']:
        raise ValueError('Calibration inputs differ')
    for key in ['task_score_grad_norms', 'margin_score_grad_norms', 'valid_fractions', 'q_means']:
        if not np.allclose(ctrl_cal[key], cand_cal[key], rtol=1e-6, atol=1e-9):
            raise ValueError('Common-initialization direct-margin calibration differs: ' + key)
    if cand_cal['target_margin_to_task_gradient_ratio'] != 0.1 or ctrl_cal['target_margin_to_task_gradient_ratio'] != 0.0:
        raise ValueError('Unexpected direct-margin gradient-ratio contract')
    group_info = json.loads((RUN / 'control_groups.json').read_text())
    membership = {name: group for group, names in group_info['groups'].items() for name in names}
    manifest = json.loads((P1 / 'data_manifest.json').read_text())
    entries = {item['sequence']: item for item in manifest['test_sequences']}
    names = [line.strip() for line in Path(args.list).read_text().splitlines() if line.strip()]
    rows = []
    for name in names:
        item = entries[name]
        control, ciou = base.read_result(args.control_test, name, item['frames'])
        candidate, diou = base.read_result(args.candidate_test, name, item['frames'])
        row = {'sequence': name, 'frames': item['frames'], 'group': membership[name]}
        for key, value in control.items():
            row[key + '_control'] = value
            row[key + '_candidate'] = candidate[key]
            row[key + '_delta'] = candidate[key] - value
        row['recovered_from_control_failure_frames'] = int(((ciou < .1) & (diou >= .5)).sum())
        row['new_failure_from_control_good_frames'] = int(((ciou >= .5) & (diou < .1)).sum())
        rows.append(row)
    out = RUN / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'per_sequence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    expected_lambda = (0.1 * cand_cal['task_score_grad_norm_median'] /
                       cand_cal['margin_score_grad_norm_median'])
    if not np.isclose(cand_cal['effective_loss_weight'], expected_lambda, rtol=1e-8, atol=1e-12):
        raise ValueError('Candidate lambda differs from frozen gradient-ratio calibration')
    if ctrl_cal['effective_loss_weight'] != 0.0:
        raise ValueError('Control direct-margin weight must be zero')
    summary = {'pairing': pairing, 'sequences': len(rows), 'frames': sum(row['frames'] for row in rows),
               'all_evaluated': base.summarize(rows, None),
               **{key: base.summarize(rows, set(value)) for key, value in group_info['groups'].items()},
               'direct_margin_calibration': {
                   'control': ctrl_cal, 'candidate': cand_cal,
                   'calibration_gradients_equal': True,
                   'candidate_lambda_verified': True},
               'scope': 'single seed; 468 successful updates; fixed, repeatedly inspected screen96 development cohort'}
    write_json(out / 'summary.json', summary)
    metrics = summary['all_evaluated']['metrics']
    labels = [('auc', 'AUC'), ('precision', 'Precision'), ('norm_precision', 'Norm Precision'),
              ('op50', 'OP50'), ('op75', 'OP75')]
    lines = ['# C + 直接 Hann 定位 margin：快速配对结果', '',
             '| 指标 | C control | C + margin | 差值 (pp) |', '|---|---:|---:|---:|']
    for key, label in labels:
        val = metrics[key]
        lines.append('| %s | %.4f | %.4f | %+.4f |' %
                      (label, val['control'], val['candidate'], val['delta']))
    lines += ['', '范围：%d 条序列、%d 帧。AUC 序列 bootstrap 90%% 区间为 [%.4f, %.4f] pp；不代表训练 seed 方差。' %
              (summary['sequences'], summary['frames'], *summary['all_evaluated']['sequence_bootstrap_90pct']),
              '配对状态相同：%s；增强输入相同：%s；成功更新：%s；校准输入 hash 相同：%s。' %
              (pairing['shared_state_equal'], pairing['augmented_input_equal'],
               pairing['successful_updates'],
               ctrl_cal['microbatch_sha256'] == cand_cal['microbatch_sha256']),
              '校准有效 GT 比率 control/candidate：%.4f / %.4f；目标梯度比：%.3f；有效 lambda：%.8g / %.8g。' %
              (float(np.mean(ctrl_cal['valid_fractions'])), float(np.mean(cand_cal['valid_fractions'])),
               cand_cal['target_margin_to_task_gradient_ratio'], ctrl_cal['effective_loss_weight'],
               cand_cal['effective_loss_weight']), '',
              '该结果是单 seed、468 updates 的开发集筛选。loss 作用于 CENTER score map，可经 head/backbone 学习；总分变化不能单独证明条件 gate 的消息因果效用。']
    for group in ['low_target', 'middle', 'normal_protection']:
        lines += ['', '## ' + group, '', '| n | AUC delta | PR | NPR | OP50 | OP75 | 去前三收益 |',
                  '|---:|---:|---:|---:|---:|---:|---:|']
        item = summary[group]
        m = item['metrics']
        lines.append('| %d | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f |' %
                      (item['n'], item['auc_delta'], m['precision']['delta'], m['norm_precision']['delta'],
                       m['op50']['delta'], m['op75']['delta'], item['mean_without_top3_gains']))
    lines += ['', '## 决策', '',
              '若增益只集中在少数序列、去前三后转负或正常保护组明显退化，不晋级。若五项方向一致且长尾可接受，只支持下一步机制归因，不等于修复 MSHA 传播。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'metrics': metrics, 'groups': {key: summary[key]['auc_delta'] for key in
        ['low_target', 'middle', 'normal_protection']}, 'pairing': pairing}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--control-test', required=True)
    freeze_parser.add_argument('--list', required=True)
    freeze_parser.set_defaults(func=freeze)
    compare_parser = sub.add_parser('compare')
    compare_parser.add_argument('--control-train', required=True)
    compare_parser.add_argument('--candidate-train', required=True)
    compare_parser.add_argument('--control-test', required=True)
    compare_parser.add_argument('--candidate-test', required=True)
    compare_parser.add_argument('--list', required=True)
    compare_parser.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
