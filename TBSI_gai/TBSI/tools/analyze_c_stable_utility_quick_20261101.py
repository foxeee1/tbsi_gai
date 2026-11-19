#!/usr/bin/env python3
"""Paired screen analysis for the C stable-utility quick experiment."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import analyze_early_gate_p1_v2 as base

RUN_ID = os.environ.get('C_STABLE_UTILITY_RUN_ID', 'EG_C_STABLE_UTILITY_QUICK_S42_20261101')
RUN = ROOT / 'output/experiments' / RUN_ID
P1 = ROOT / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3'


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def freeze(args):
    names = [x.strip() for x in Path(args.list).read_text().splitlines() if x.strip()]
    manifest = json.loads((P1 / 'data_manifest.json').read_text())
    frames = {x['sequence']: x['frames'] for x in manifest['test_sequences']}
    if len(names) != 96 or len(set(names)) != 96 or not set(names) <= set(frames):
        raise ValueError('Expected the fixed 96-sequence screen96_max2000 cohort')
    rows = []
    for name in names:
        result, _ = base.read_result(args.control_test, name, frames[name])
        rows.append({'sequence': name, **result})
    ranked = sorted(rows, key=lambda row: (row['auc'], row['sequence']))
    groups = {'low_target': [row['sequence'] for row in ranked[:32]],
              'middle': [row['sequence'] for row in ranked[32:64]],
              'normal_protection': [row['sequence'] for row in ranked[64:]]}
    write_json(RUN / 'control_groups.json', {
        'control_test': str(Path(args.control_test).resolve()),
        'control_identity_sha256': base.digest(Path(args.control_test) / 'identity.json'),
        'cohort_list': str(Path(args.list).resolve()),
        'cohort_sha256': base.digest(args.list),
        'definition': 'AUC tertiles of fresh C-control predictions on fixed screen96; frozen before candidate evaluation',
        'groups': groups, 'control_metrics': rows})
    print(json.dumps({'groups': {key: len(value) for key, value in groups.items()}}, indent=2))


def compare(args):
    pair = base.validate_pair(args.control_train, args.candidate_train, RUN / 'analysis/pairing.json')
    names = [x.strip() for x in Path(args.list).read_text().splitlines() if x.strip()]
    group_data = json.loads((RUN / 'control_groups.json').read_text())
    membership = {name: key for key, values in group_data['groups'].items() for name in values}
    manifest = json.loads((P1 / 'data_manifest.json').read_text())
    entries = {x['sequence']: x for x in manifest['test_sequences']}
    rows = []
    for name in names:
        item = entries[name]
        control, ciou = base.read_result(args.control_test, name, item['frames'])
        candidate, diou = base.read_result(args.candidate_test, name, item['frames'])
        row = {'sequence': name, 'frames': item['frames'], 'group': membership[name]}
        for key in control:
            row[key + '_control'] = control[key]
            row[key + '_candidate'] = candidate[key]
            row[key + '_delta'] = candidate[key] - control[key]
        row['recovered_from_control_failure_frames'] = int(((ciou < .1) & (diou >= .5)).sum())
        row['new_failure_from_control_good_frames'] = int(((ciou >= .5) & (diou < .1)).sum())
        rows.append(row)
    analysis_dir = RUN / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / 'per_sequence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'pairing': pair, 'sequences': len(rows), 'frames': sum(r['frames'] for r in rows),
               'all_evaluated': base.summarize(rows, None),
               **{key: base.summarize(rows, set(values)) for key, values in group_data['groups'].items()},
               'scope': 'single seed; 468 successful updates; fixed development screen96_max2000; sequence bootstrap is not seed uncertainty'}
    write_json(analysis_dir / 'summary.json', summary)
    metric_labels = [('auc', 'AUC'), ('precision', 'Precision'),
                     ('norm_precision', 'Norm Precision'), ('op50', 'OP50'), ('op75', 'OP75')]
    overall = summary['all_evaluated']['metrics']
    report = ['# C + 稳定来源效用 quick 配对结果', '',
              '| 指标 | C control | C + stable utility | 差值 (pp) |',
              '|---|---:|---:|---:|']
    for key, label in metric_labels:
        value = overall[key]
        report.append('| %s | %.4f | %.4f | %+.4f |' %
                      (label, value['control'], value['candidate'], value['delta']))
    report += ['',
               '## 闭环合同', '',
               '可观察问题：C 的 source gate 已按来源 token 调整 cross-Value 幅度，但任务损失没有直接要求高 utility 来源获得较高 gate。待证伪假设是：在同一训练 crop 上，中心峰 margin 与目标框峰 margin 都认为更有用的来源 token，若被辅助 loss 推向更高 source log-gain，可以改善闭环定位。',
               '唯一主变量：稳定排序 loss 系数 0 对比校准到 rank/task gate-gradient ratio 0.1；C 架构、响应图、source gate、Value/QK、head、数据顺序、优化器、训练进度与评测完全相同。',
               '目标的具体含义：在 Hann 加权 CENTER 图上计算中心邻域峰和 GT 框内峰各自相对框外峰的 log-margin，再对 `[B,64,2]` source log-gain 求局部一阶导；从平均效用导数的高/低四分位取 token pair，只保留两种 margin 导数都认为高组更有用的 pair，优化 `relu(0.01 - (gain_high - gain_low))`。',
               '这监督的是当前训练 crop 上的局部敏感度排序，不是消息有限替换实验，也不等价于自由跟踪中某条消息对后续定位的因果贡献。', '',
               '配对合同通过：公共预训练初态哈希一致，完整 state 哈希一致；3744 个 batch、59904 个样本的增强输入 hash 无 mismatch；两臂各 468 successful updates、覆盖 977 条训练序列、0 次 skip/overflow retry。', '',
               '96 条固定序列，%d 帧；配对 90%% AUC 序列区间：`[%.4f, %.4f]` pp。' %
               (summary['frames'], *summary['all_evaluated']['sequence_bootstrap_90pct']),
               '该区间只反映当前 checkpoint 的序列差异，不估计训练 seed 方差。', '',
               '## 结论', '',
               '本轮不晋级长训：总体五项指标均为负，AUC 序列区间跨 0；normal protection 组 AUC 为 -0.8219 pp，去掉最大三项收益后总体均值为 -1.2413 pp。',
               '候选 stable-pair 覆盖率约 99.5%，有效 GT 比例 100%；稳定性筛选几乎没有移除排序 pair。候选校准后的有效 loss 系数约 0.001203，rank/task gate 梯度比约 0.1，监督路径非零且配对正确。因此当前负结果不应归因于没有梯度、没跑够 updates 或两臂数据不一致；但单 seed、468 updates 的短预算仍不能证明长训方向或 seed 稳健性。', '',
               '## 分层与序列', '',
               '| Control AUC 冻结分层 | n | AUC 差值 (pp) | Precision | Norm Precision | OP50 | OP75 | 去掉组内前三收益 |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, label in [('low_target', '低目标组'), ('middle', '中间组'),
                       ('normal_protection', '正常保护组')]:
        group = summary[key]
        metrics = group['metrics']
        report.append('| %s | %d | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f |' %
                      (label, group['n'], group['auc_delta'], metrics['precision']['delta'],
                       metrics['norm_precision']['delta'], metrics['op50']['delta'],
                       metrics['op75']['delta'], group['mean_without_top3_gains']))
    top_gains = sorted(rows, key=lambda row: row['auc_delta'], reverse=True)[:3]
    top_losses = sorted(rows, key=lambda row: row['auc_delta'])[:3]
    report += ['',
               '总体为 51 胜、1 平、44 负；9 条序列下降超过 5 pp，7 条超过 10 pp，最差十分位均值 -13.55 pp。正收益中 70.8% 集中在前三条。',
               '最大三项收益：' + '；'.join('%s %+0.2f pp' % (row['sequence'], row['auc_delta'])
                                           for row in top_gains) + '。',
               '最大三项退化：' + '；'.join('%s %+0.2f pp' % (row['sequence'], row['auc_delta'])
                                           for row in top_losses) + '。',
               '低目标组 AUC 点估计 +0.271 pp，但中位数为 -0.225 pp、去掉组内前三后为 -2.873 pp，且 Precision -0.584 pp；说明组内平均正值被少数序列驱动。正常保护组五项指标一致退化，AUC -0.822 pp、failure fraction 增加 1.034 pp，是需要保护的反向信号。', '',
               '## 证据解释', '',
               '已排除：共同初态、训练样本顺序/增强输入、更新数或明显梯度断路造成的 control/candidate 比较混杂。',
               '暂不支持：在当前定义与预算下，给 C 加“中心峰与框峰排序一致”的来源增益排序 loss 能带来整体定位收益；也暂不支持以本结果声称减少了 MSHA 错误传播。',
               '尚未区分：两种 margin 的导数排序为何几乎完全一致；loss 是否改变了 gate 排序但伤害正常序列；局部一阶敏感度与有限幅度消息效用之间的差异；短预算与 seed 方差是否改变方向。当前证据不能把退化归因到任一项。', '',
               '## 诊断', '',
               '- 对照与候选共同初态及增强输入配对：%s。' %
               ('通过' if pair['shared_state_equal'] and pair['augmented_input_equal'] else '不通过'),
               '- 稳定排序比例和效用梯度见两臂 `utility_calibration.json` 与 `updates.jsonl`。',
               '- 本结果是短预算开发筛选，不是 full15 或论文确认。', '',
               '## 下一步', '',
               '当前建议：不做本候选 full train/test，也不把它作为涨点方案保留。下一步先用已存的 utility calibration / updates 核验稳定 pair 统计是否按 token、层、模态聚合正确；若确认统计真实，再做一个训练前的固定 crop 诊断，比较局部梯度排序与有限幅度逐来源 gate 干预对 CENTER margin 的排序一致率。只有证据显示有足够数量的冲突/低置信 pair，才允许下一轮把“稳定筛选阈值”作为唯一模型变量，重新校准 loss 强度并配对 quick。不要把本轮负值归因于训练方差，也不要将其解释为“效用监督普遍无效”。']
    (analysis_dir / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({'all_evaluated': summary['all_evaluated'],
                      'groups': {k: v['auc_delta'] for k, v in summary.items()
                                 if k in group_data['groups']}}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze')
    p.add_argument('--control-test', required=True)
    p.add_argument('--list', required=True)
    p.set_defaults(func=freeze)
    p = sub.add_parser('compare')
    p.add_argument('--control-train', required=True)
    p.add_argument('--candidate-train', required=True)
    p.add_argument('--control-test', required=True)
    p.add_argument('--candidate-test', required=True)
    p.add_argument('--list', required=True)
    p.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
