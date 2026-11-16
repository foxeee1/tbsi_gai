"""Offline audit of existing paired quick results; no model execution."""
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/reports/mr_quick_tradeoffs_20261114'
RUNS = {
    'Agreement delta gate': 'MSHA-MR-AGREEMENT-DELTA-GATE-QUICK-S42-20261113',
    'Selective gate retry1': 'MSHA-MR-SELECTIVE-GATE-QUICK-RETRY1-S42-20261113',
    'Local residual': 'MSHA-MR-LOCAL-RESIDUAL-QUICK-S42-20261113-RETRY2',
    'Orthogonal direction': 'MSHA-MR-ORTHO-DIRECTION-QUICK-S42-20261113',
    'Radial amplitude': 'MSHA-MR-RADIAL-AMPLITUDE-QUICK-S42-20261113-RETRY1',
    'Content projection': 'MSHA-MR-CONTENT-PROJECTION-QUICK-S42-20261114',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, inputs, detailed = [], {}, []
    for label, run in RUNS.items():
        p = ROOT / 'output/experiments' / run / 'analysis/per_sequence.csv'
        inputs[str(p.relative_to(ROOT))] = digest(p)
        rows = list(csv.DictReader(p.open()))
        factor = 1 if 'control_auc' in rows[0] else 100
        changes = [float(r['delta_auc']) * factor for r in rows]
        control = [float(r.get('control_auc', r.get('auc'))) * factor for r in rows]
        positive = sorted((v for v in changes if v > 0), reverse=True)
        row = dict(label=label, run=run, n=len(rows), control_auc=st.mean(control),
                   candidate_auc=st.mean(control)+st.mean(changes), delta_auc=st.mean(changes),
                   median=st.median(changes), wins=sum(v > 1e-9 for v in changes),
                   ties=sum(abs(v) <= 1e-9 for v in changes), losses=sum(v < -1e-9 for v in changes),
                   wins_05=sum(v > .5 for v in changes), ties_05=sum(abs(v) <= .5 for v in changes),
                   losses_05=sum(v < -.5 for v in changes), severe_losses_5=sum(v < -5 for v in changes),
                   worst=min(changes), worst3_mean=st.mean(sorted(changes)[:3]),
                   top2_gross_positive_fraction=sum(positive[:2])/sum(positive),
                   mean_without_top2=st.mean(sorted(changes)[:-2]))
        row['groups'] = {
            g: dict(n=sum(r['group'] == g for r in rows),
                    mean=st.mean(float(r['delta_auc'])*factor for r in rows if r['group'] == g))
            for g in sorted({r['group'] for r in rows})}
        summaries.append(row)
        detailed.extend(dict(method=label, sequence=r['sequence'], group=r['group'], delta_auc=d)
                        for r, d in zip(rows, changes))
    agreement = ROOT / 'output/experiments' / RUNS['Agreement delta gate']
    traces = []
    for arm in ['control', 'candidate']:
        p = agreement / arm / 'batches.jsonl'
        inputs[str(p.relative_to(ROOT))] = digest(p)
        traces.append([json.loads(line) for line in p.open()])
    pairing = dict(records=[len(t) for t in traces],
                   matched=traces[0] == traces[1],
                   note='Full JSON records compared, including sample identities and augmented-input hashes.')
    if len(traces[0]) != 3744 or not pairing['matched']:
        raise RuntimeError('Agreement paired trace differs; do not rank its utility.')
    summary = dict(units='AUC percentage points; sequence macro',
                   tolerances={'tie_pp': 1e-9, 'descriptive_band_pp': .5,
                               'severe_loss_pp': -5},
                   warning='0.5/5 pp are retrospective descriptive cutoffs, not significance tests or preregistered gates.',
                   agreement_pairing=pairing, methods=summaries)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    with (OUT / 'per_sequence_comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)
    fields = [k for k in summaries[0] if k != 'groups']
    with (OUT / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in summaries)
    table = ['| 方法 | control AUC | ΔAUC | 胜/平/负 | ±0.5pp后 | 中位数 | 下降>5pp条数 | 去掉最大两条增益后的均值 |',
             '|---|---:|---:|---|---|---:|---:|---:|']
    for r in summaries:
        table.append('| {label} | {control_auc:.3f} | {delta_auc:+.3f} | {wins}/{ties}/{losses} | {wins_05}/{ties_05}/{losses_05} | {median:+.3f} | {severe_losses_5} | {mean_without_top2:+.3f} |'.format(**r))
    report = '''# MR quick：退化目标、胜负覆盖与 full 候选复核

本次只分析已有 CSV、合同、输入记录和源码，没有训练、tracker 推理或新 benchmark。所有 quick 均为 seed42、468 updates、59,904 pairs，在同24条开发序列上评测；各行首先对照本行 control，不能当作同一个 MR parent 的统一排行榜。

可以把目标收敛为独立定义的退化条件内有效、补集损伤有限，不需要所有序列上涨。但“胜−负”必须与平均增益和严重回退联合评价。Orthogonal full 对 matched control 为126胜/2平/116负，平均却为−0.77995pp；±0.5pp后51/144/49。数量优势没有解决尾部损失。

## 已有 quick 的重新汇总

{table}

±0.5pp与−5pp是本次事后统一描述口径，不是显著性判断，也不是旧实验的预注册晋级线。剔除最大两条只作集中度敏感性分析，不是新的总体估计。原 Orthogonal quick 报告写14/0/10，但保存CSV实际为11/1/12（1e−9pp容差）；原报告保留，本次以CSV为准。

## 唯一优先复核方向

若以逐序列胜负覆盖为首要目标，Agreement Delta Gate 是下一次匹配 quick 的优先方向，而非现成的 full 晋级者。其ΔAUC+0.842pp、17胜7负；±0.5pp后9胜9平6负，中位+0.241pp。最大两条占正增益75.28%，去掉它们后−0.962pp；carcominginlight −26.298pp。历史回归/收益/翻转分层均值分别+3.417/+2.259/−3.151pp。这并未解决翻转，也不是普遍稳定提升。

其算子是 δ'ij=clip(ej,3,0,1)δij；它只约束新增来源score修正，未直接拒绝低一致性来源的原始attention消息。相似度高不等于可靠，不能据此宣称退化感知。不同来源的乘数也可能改变有效score差值，不保证实际message或定位扰动变小。

匹配合同支持：Agreement两臂402个共有初态tensor hash一致，3744条增强输入/采样记录逐条一致，468成功更新、零overflow。限制：YAML是dynamic Early gate，optimizer_membership列出三层early_gate_scale；runner没有G1归零冻结逻辑。当前源码pair分支scale初值0.1。因此它不能当作历史最强MR-only g=1的严格单变量验证。不同quick的control AUC为38.804–41.358，也不能把全部差异叫seed噪声。

Selective Gate retry1 是数值收益最强的备选证据：+4.227pp，14胜10负，±0.5pp后10胜9平5负；去掉两大增益仍+1.367pp，最差−6.121pp。它不应因gate没学就被认定没用；但gate输出层仍为weight=0、bias=4，实际是固定gate＋可学习残差，不能归因学习式选择。直接对照也是带orthogonal residual的control，不是MR-only。它若继续，应先按固定动作重新定义假设与对照；本轮不同时启动这条线。

## 怎样定义退化与论文主张

“47两边回归”“134方向翻转”是历史MR−B0符号分层，不是视觉退化标签。seed43初态未匹配，不能把翻转全归因seed。现有属性回查只有BC、HO等是小幅同向线索；存在样本少、增益集中和属性重叠，不能选涨得最多的标签直接写成确认性结论。

建议预先定义一个帧内事件：目标仍位于当前crop内，至少一模态目标可辨，另一模态的遮挡/低对比或相似干扰使目标与干扰候选支持冲突。用独立画面/GT规则或盲标定义，并纳入该条件下成功及失败样本；遮挡、低对比、干扰应分开记，不能合并所有挑战后事后挑有利结果。先在训练侧事件上判断最常见、可重复的一类，再锁定测试条件。框误差本身只能刻画定位表型，不能确认模态退化首因。

论文证据链可限定为：该退化使无条件帧内交互传播有害信息（待同state/crop验证）→约束MR新增重分配减少无收益改写→条件内定位/胜率改善，同时报告补集、长尾与速度。当前资产尚未证明此因果链。

## 最小后续 quick（建议，未启动）

只保留Agreement方向，先在历史G1语义上建立新配对run：公共SOT单阶段fresh optimizer；两臂共有初态、增强输入、data/schedule一致；三层Early scale显式0且冻结；candidate只开agreement delta gate。完成既定468更新，在预先固定的更大代表性开发cohort上评测（建议96条，包含目前24条并单独报告其余72条；新72条按固定随机规则选，不能按候选涨跌选）。这些仍来自已多次使用的test244，不称独立holdout。若现有checkpoint、代码快照可靠且可用，可先扩评既有Agreement两臂节省训练，但其dynamic合同结果仍须与G1新配对分开。

新退化条件在看该轮候选结果前锁定。共同报告ΔAUC、±0.5pp胜负、中位、最差尾部/严重回退、最大两条集中度，以及目标条件与补集。若其余72条方向、目标条件收益和尾部同时支持，再选作唯一full15/full244候选；full仍从公共SOT独立训练。468→7020的方向可靠性目前未校准，quick改善只是投入full验证的依据，不能保证full上涨。不要求完整机制证明才准有界quick，也不自动因小样本区间跨零淘汰。

结论：支持下一步单变量检验；当前没有可据现有证据直接确认晋级的组。

复算命令：`python tools/summarize_mr_quick_tradeoffs_20261114.py`。输入hash及脚本hash见provenance.json。
'''.format(table='\n'.join(table))
    (OUT / 'REPORT.md').write_text(report)
    (OUT / 'provenance.json').write_text(json.dumps(dict(inputs=inputs, script_sha256=digest(Path(__file__)),
         operation='CPU offline CSV/JSON aggregation only'), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
