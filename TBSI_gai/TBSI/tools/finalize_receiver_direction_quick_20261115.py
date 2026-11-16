#!/usr/bin/env python3
"""Official per-sequence metrics, concentration and fixed-crop mechanism summary."""
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from run_receiver_direction_quick_20261115 import RUN,PARENT,CFG,LIST,record
from run_dual_message_p1 import sha,write_json
from analyze_receiver_message_residual_quick_20261115 import metrics,METRICS
from analyze_receiver_residual_boxes_attributes_20261115 import write_csv,summarize
from lib.test.evaluation import get_dataset

def main():
    torch.set_num_threads(1)
    pairing=json.loads((RUN/'pairing.json').read_text());assert pairing['all_checks_pass']
    dataset={s.name:s for s in get_dataset('lasher_test')};names=[n for n in LIST.read_text().splitlines() if n.strip()]
    metadata={r['sequence']:r for r in csv.DictReader((PARENT/'analysis/box_attribute_audit_20261115/per_sequence.csv').open())}
    rows=[]
    for name in names:
        parent=metrics(PARENT/'eval_candidate',dataset[name]);cand=metrics(RUN/'eval_candidate',dataset[name]);base=metrics(PARENT/'eval_control',dataset[name]);m=metadata[name]
        r=dict(sequence=name,frames=parent['frames'],attributes=m['attributes'],history_group=m['history_group'])
        for key in METRICS:
            r['control_'+key]=parent[key];r['candidate_'+key]=cand[key];r['delta_'+key]=cand[key]-parent[key]
            r['candidate_minus_no_residual_'+key]=cand[key]-base[key]
        rows.append(r)
    dest=RUN/'analysis';dest.mkdir(exist_ok=True);write_csv(dest/'per_sequence.csv',rows)
    control={k:float(np.mean([r['control_'+k] for r in rows])) for k in METRICS};candidate={k:float(np.mean([r['candidate_'+k] for r in rows])) for k in METRICS}
    stats=summarize(rows);groups=[]
    for key in ['SA','LI','LR','BC','TO_or_AIV','seed_direction_flip','consistent_mr_regression','consistent_mr_gain','one_seed_tie']:
        rr=[r for r in rows if (key in r['attributes'].split() if key in ['SA','LI','LR','BC'] else bool(set(r['attributes'].split())&{'TO','AIV'}) if key=='TO_or_AIV' else r['history_group']==key)]
        if rr:groups.append(dict(group=key,**summarize(rr)))
    write_csv(dest/'attribute_history_summary.csv',groups)
    mechanism=json.loads((RUN/'mechanism/summary.json').read_text());training=json.loads((RUN/'candidate_retry1/completed.json').read_text());evaluation=json.loads((RUN/'eval_candidate/completed.json').read_text())
    summary=dict(control=control,candidate=candidate,delta={k:candidate[k]-control[k] for k in METRICS},stats=stats,groups=groups,mechanism=mechanism,
        pairing=pairing,training=training,evaluation=evaluation,checkpoint_sha256=sha(RUN/'candidate_retry1/checkpoint_u0468.pth.tar'))
    write_json(dest/'summary.json',summary)
    text=['# 残差目标方向监督：配对快速训练测试','',
        '唯一变量为训练侧辅助方向监督，架构/输入不变。GT训练时生成分离梯度教师，辅助梯度只更新六个残差预测器；推理不使用GT、不增加新参数。',
        '共同公共预训练初态与全部3744增强microbatch核验通过，因此复用上一组无辅助loss的残差候选作为直接parent。两组均为468更新、59904pairs、977训练序列；不是历史MR g=1严格缩短复现。','',
        '| 指标 | 无辅助监督parent | 方向监督candidate | 差值pp |','|---|---:|---:|---:|']
    for k in METRICS:text.append(f'| {k} | {control[k]:.4f} | {candidate[k]:.4f} | {candidate[k]-control[k]:+.4f} |')
    text+=['',f'AUC逐序列胜/平/负 {stats["wins"]}/{stats["ties"]}/{stats["losses"]}；中位数 {stats["median"]:+.4f}pp；±0.25pp胜/平/负 {stats["practical_wins_gt_025"]}/{stats["practical_ties_le_025"]}/{stats["practical_losses_lt_m025"]}。',
        f'序列bootstrap95% {stats["sequence_bootstrap95"]}；最差10%均值 {stats["worst_ceil10pct"]:+.4f}pp；去前三增益后均值 {stats["mean_without_top3"]:+.4f}pp；前三占正收益 {stats["positive_gain_top3_fraction"]:.4f}。',
        '', '## 固定crop机制','', '```json',json.dumps(mechanism['arms'],ensure_ascii=False,indent=2),'```','',
        '只有目标方向改善与跟踪效用同时成立，才能支持后续方法确认；方向改善而AUC失败，说明监督可驱动但任务收益未成立；两者均无支持时，不继续同类loss盲搜。原始逐序列/属性/历史组和机制读数完整保留。',
        '', '固定96开发集、单seed、短训，结果仅为探索性。属性重叠且是序列级，没有宣称跨seed稳定或自动晋级full。']
    (dest/'REPORT.md').write_text('\n'.join(text)+'\n')
    conclusion=f'468updates,96seq. AUC {candidate["auc"]:.4f} vs direct residual parent {control["auc"]:.4f},delta {candidate["auc"]-control["auc"]:+.4f}pp;WTL {stats["wins"]}/{stats["ties"]}/{stats["losses"]};median {stats["median"]:+.4f}pp;all3744batchhashes/commoninit match. Fixedcrop task/margin favorable counts parent/candidate: '+json.dumps({a:{k:v[k]['favorable'] for k in ['task','margin']} for a,v in mechanism['arms'].items()})
    record(RUN/'candidate_retry1',CFG,42,'candidate','QUICK-TRAIN-EVAL-COMPLETE / EXPLORATORY / NO-FULL-PROMOTION',conclusion,str(RUN/'candidate_retry1/checkpoint_u0468.pth.tar'),training['gpu_wall_hours'])
    # Complete metric fields and evidence in both synchronized registries.
    p=ROOT/'experiments/experiment_asset_registry.csv'
    with p.open(newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rr=list(reader)
    identity='MR-RESIDUAL-TARGET-DIRECTION-S42-20261115-CANDIDATE_RETRY1'
    row=next(r for r in rr if r['ID']==identity)
    row.update(AUC=str(candidate['auc']),PR=str(candidate['precision']),NPR=str(candidate['npr']),OP50=str(candidate['op50']),OP75=str(candidate['op75']),FPS=str(evaluation['throughput_fps']),Evidence=str(dest/'REPORT.md'),Checkpoint=str(RUN/'candidate_retry1/checkpoint_u0468.pth.tar')+' SHA256 '+summary['checkpoint_sha256'])
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
    md=ROOT/'experiments/实验资产数据库.md';mt=md.read_text();begin='<!-- '+identity+' BEGIN -->';end='<!-- '+identity+' END -->';a=mt.index(begin);b=mt.index(end,a)+len(end)
    block=begin+'\n### '+identity+'\n\n'+''.join('- **'+k+'**：'+row[k]+'\n' for k in fields)+end;md.write_text(mt[:a]+block+mt[b:])
    print(json.dumps({k:summary[k] for k in ['control','candidate','delta','stats']},indent=2))

if __name__=='__main__':main()
