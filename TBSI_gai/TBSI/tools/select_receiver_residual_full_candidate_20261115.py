#!/usr/bin/env python3
"""Select one full-run candidate after the locked quick pipeline finishes.

The selection preferences are recorded before reading candidate evaluation.
This process only waits and writes a decision/configuration; it never trains.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output/experiments/MSHA-MR-RESIDUAL-TARGET-DIRECTION-QUICK-S42-20261115'
PARENT = ROOT / 'output/experiments/MSHA-MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def summarize(rows):
    values = sorted(r['delta_auc'] for r in rows)
    n = len(values)
    median = (values[(n - 1) // 2] + values[n // 2]) / 2
    k = (n + 9) // 10
    wins = sum(x > .25 for x in values)
    losses = sum(x < -.25 for x in values)
    return dict(auc=sum(r['candidate_auc'] for r in rows)/n,
                delta_auc=sum(values)/n, median=median,
                wins=sum(x > 1e-9 for x in values),
                ties=sum(abs(x) <= 1e-9 for x in values),
                losses=sum(x < -1e-9 for x in values),
                practical_wins=wins, practical_ties=n-wins-losses,
                practical_losses=losses, practical_netwins=wins-losses,
                worst_ceil10pct=sum(values[:k])/k,
                mean_without_top3=sum(values[:-3])/(n-3))


def register(decision, report, snapshot):
    identity = 'MR-RECEIVER-RESIDUAL-FULL-CANDIDATE-SELECTION-20261115'
    csvpath = ROOT/'experiments/experiment_asset_registry.csv'
    with csvpath.open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    selected = decision['selected_stats']
    row = {k: 'NA' for k in fields}
    row.update(ID=identity, Date='20261115',
               Parent='MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-CANDIDATE; MR-RESIDUAL-TARGET-DIRECTION-S42-20261115-CANDIDATE_RETRY1',
               CommitOrSnapshot=str(snapshot), Config=decision['full_candidate_config'], Seed='42',
               Dataset='Locked screen96_max2000:96seq/54666frames; quick train977/59904pairs/468updates; no new training or inference in this selection',
               EvalScope='Offline selection between two declared residual variants; same no-residual reference; full15/full244 prepared but NOT launched',
               AUC=str(selected['auc']), PR=str(decision['selected_metrics']['precision']),
               NPR=str(decision['selected_metrics']['npr']), OP50=str(decision['selected_metrics']['op50']),
               OP75=str(decision['selected_metrics']['op75']),
               AssumptionOrHypothesis='Choose target-direction loss only if mean AUC improves without worse practical netwins,tail or top3-excluded mean against common no-residual reference; otherwise original residual.',
               Conclusion=decision['reason'], Status='ONE-CANDIDATE-SELECTED / FULL-NOT-LAUNCHED / EXPLORATORY',
               Evidence=str(report), Notes='Metrics in this row describe the selected quick checkpoint,not full training. Dynamic Early gate+MR; historical fixed-g1 MR is not a matched control. Selection rule was saved before current quick evaluation.')
    matches = [i for i, r in enumerate(rows) if r['ID'] == identity]
    assert len(matches) <= 1
    if matches:
        rows[matches[0]] = row
    else:
        rows.append(row)
    temporary = csvpath.with_suffix('.selection.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csvpath)
    mdpath = ROOT/'experiments/实验资产数据库.md'
    md = mdpath.read_text()
    begin, end = '<!-- '+identity+' BEGIN -->', '<!-- '+identity+' END -->'
    block = begin+'\n### '+identity+'\n\n'+''.join('- **'+k+'**：'+row[k]+'\n' for k in fields)+end
    if begin in md:
        i = md.index(begin)
        j = md.index(end, i) + len(end)
        md = md[:i]+block+md[j:]
    else:
        md += '\n\n'+block+'\n'
    mdpath.write_text(md)


def select():
    protocol = read(RUN/'FULL_CANDIDATE_SELECTION_PROTOCOL.json')
    parent_summary = read(PARENT/'analysis/summary.json')
    original_rows = list(csv.DictReader((PARENT/'analysis/per_sequence.csv').open()))
    original = summarize([{k: float(r[k]) for k in ['candidate_auc', 'delta_auc']} for r in original_rows])
    pipeline = read(RUN/'pipeline_status.json')
    current, checks, mechanism = None, {}, None
    if pipeline.get('status') == 'COMPLETE':
        summary = read(RUN/'analysis/summary.json')
        assert summary['pairing']['all_checks_pass']
        rows = list(csv.DictReader((RUN/'analysis/per_sequence.csv').open()))
        assert len(rows) == 96
        current = summarize([dict(candidate_auc=float(r['candidate_auc']),
                                  delta_auc=float(r['candidate_minus_no_residual_auc'])) for r in rows])
        checks = dict(auc_improves=current['auc'] > original['auc'],
                      practical_netwins_not_worse=current['practical_netwins'] >= original['practical_netwins'],
                      tail_not_worse=current['worst_ceil10pct'] >= original['worst_ceil10pct'],
                      top3_excluded_mean_not_worse=current['mean_without_top3'] >= original['mean_without_top3'])
        mechanism = summary['mechanism']
    else:
        assert pipeline.get('status') == 'FAILED'
    use_loss = bool(checks) and all(checks.values())
    name = 'MR Receiver Message Residual + Target Direction Supervision' if use_loss else 'MR Receiver Message Residual（无辅助 loss）'
    config_name = 'MR-RECEIVER-RESIDUAL-DIRECTION-FULL15-S42-20261115' if use_loss else 'MR-RECEIVER-RESIDUAL-FULL15-S42-20261115'
    config = ROOT/'experiments/tbsi_track'/(config_name+'.yaml')
    control_config = ROOT/'experiments/tbsi_track/MR-RECEIVER-RESIDUAL-FULL15-CONTROL-S42-20261115.yaml'
    source_config = ROOT/'experiments/tbsi_track/MR-RESIDUAL-TARGET-DIRECTION-QUICK-S42-RETRY1-20261115.yaml' if use_loss else ROOT/'experiments/tbsi_track/MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-candidate.yaml'
    source_control = ROOT/'experiments/tbsi_track/MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-control.yaml'
    for source, target in [(source_config, config), (source_control, control_config)]:
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            shutil.copy2(source, target)
    reason = ('方向监督同时满足预先记录的均值、实用净胜数、尾部和去前三增益后的均值偏好，选方向监督版本。'
              if use_loss else '方向监督未同时满足预先记录的稳定性偏好，选原接收端消息残差；这不证明辅助监督普遍无效。')
    if current is None:
        reason = '方向监督快训/评测未完整完成，无法作有效性能排名；选已完成配对训练与96条测试的原接收端消息残差。'
    decision = dict(unique_candidate=name, selected_with_target_direction_supervision=use_loss,
                    reason=reason, checks=checks, original=original, direction_supervised=current,
                    selected_stats=current if use_loss else original,
                    selected_metrics=summary['candidate'] if use_loss else parent_summary['candidate'],
                    common_reference_auc=parent_summary['control']['auc'],
                    mechanism=mechanism, protocol=protocol, full_candidate_config=str(config),
                    full_control_config=str(control_config), full_launched=False,
                    full_checkpoint='Explicit endpoint u7020/epoch15; fresh public SOT,not quick continuation',
                    selection_scope='Repeated-use screen96 development cohort,single seed; not proof of full244 or cross-seed improvement')
    dest = RUN/'full_candidate_selection'
    dest.mkdir(exist_ok=True)
    shutil.copy2(__file__, dest/Path(__file__).name)
    write(dest/'decision.json', decision)
    lines = ['# 今晚全量实验唯一候选', '', '**'+name+'**', '', reason, '',
             '两组都相对同一个无接收端残差的动态 Early gate＋MR quick 对照。', '',
             '| quick版本 | AUC | ΔAUC pp | 中位数 pp | ±0.25pp 胜/平/负 | 最差10%均值 pp | 去前三增益均值 pp |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label, stats in [('原残差', original), ('残差＋方向监督', current)]:
        if stats is None:
            lines.append('| '+label+' | 未完整完成 | NA | NA | NA | NA | NA |')
        else:
            lines.append(f'| {label} | {stats["auc"]:.4f} | {stats["delta_auc"]:+.4f} | {stats["median"]:+.4f} | {stats["practical_wins"]}/{stats["practical_ties"]}/{stats["practical_losses"]} | {stats["worst_ceil10pct"]:+.4f} | {stats["mean_without_top3"]:+.4f} |')
    lines += ['', '## 全量合同', '',
              '唯一方法候选加同条件无残差对照，均从公共 SOT 权重 fresh 单阶段训练；seed42，15 epochs、7020 成功更新、898560 pairs，batch16×accum8，worker4独占，原15epoch schedule不压缩。固定 epoch15，不从 quick 权重续训。',
              '全部244条序列，FP32、threads0、在线模板更新关闭。先核对 full 的初态、前3744个增强输入及训练数值前缀，再比较结果。历史固定g=1 MR仅列附加参照。',
              '候选配置：`'+str(config)+'`。对照配置：`'+str(control_config)+'`。当前仅准备，未启动全量。', '',
              '## 论文证据边界', '',
              '问题是 MR 已分配的跨模态消息未必适合当前接收模板。残差分支读取接收token和实际聚合消息，以受限的小幅内容修改补充来源分配。若选方向监督版，其唯一追加变量是训练侧方向目标，推理无GT。',
              '已有固定crop诊断支持消息可以影响定位，但原残差方向并非一致有用。全量实验检验学习后任务效用与收益分布；仅凭这轮短训不能宣称解决错误传播、TO/AIV退化或多数序列稳定提升。',
              '报告完整244逐序列胜负、±0.25pp净胜、尾部、集中度、历史翻转/回归组及属性分析；只支持实际成立的属性或行为范围。']
    report = dest/'REPORT.md'
    report.write_text('\n'.join(lines)+'\n')
    register(decision, report, dest/Path(__file__).name)
    write(RUN/'full_candidate_selection_status.json', dict(status='COMPLETE', unique_candidate=name, report=str(report), time=time.time()))
    print(json.dumps({k:v for k,v in decision.items() if k not in ['mechanism', 'protocol']}, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    assert str(ROOT.resolve()).startswith('/root/autodl-tmp/')
    write(RUN/'full_candidate_selection_status.json', dict(status='WAITING', pid=os.getpid(), time=time.time()))
    try:
        while True:
            status = read(RUN/'pipeline_status.json')
            if status.get('status') in ['COMPLETE', 'FAILED']:
                break
            if not args.wait:
                raise RuntimeError('Quick pipeline is not complete; no final candidate selected yet')
            time.sleep(10)
        select()
    except Exception as exc:
        write(RUN/'full_candidate_selection_status.json', dict(status='FAILED', error=str(exc), time=time.time()))
        raise


if __name__ == '__main__':
    main()
