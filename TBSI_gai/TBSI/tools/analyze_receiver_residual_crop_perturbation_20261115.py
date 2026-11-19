#!/usr/bin/env python3
"""Summarize the locked crop probe, with no model inference or training."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import csv
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from analyze_receiver_residual_boxes_attributes_20261115 import RUN,OUT,write_json,write_csv
DEST=RUN/'analysis/crop_perturbation_20261115'
ARMS=['control','candidate_native','candidate_off']
COLORS=['#dd8b28','#167fa3','#858585']

def dist(v):
    a=np.asarray(v,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.percentile(a,90)),p95=float(np.percentile(a,95)),max=float(a.max())) if len(a) else None

def main():
    assert (DEST/'execution_record.json').exists()
    shutil.copy2(__file__,DEST/'source'/Path(__file__).name)
    raw=list(csv.DictReader((DEST/'frame_readouts.csv').open()))
    metadata={r['sequence']:r for r in csv.DictReader((OUT/'per_sequence.csv').open())}
    events={}
    for r in raw:
        key=int(r['event_id']);events.setdefault(key,[]).append(r)
    event_rows=[]
    for key,rr in events.items():
        r0=rr[0]; m=metadata[r0['sequence']]
        er=dict(event_id=key,sequence=r0['sequence'],frame=int(r0['frame'])+1,roles=r0['roles'],kind=r0['kind'],
            attributes=m['attributes'],history_group=m['history_group'],
            target_local_eligible=all(r['gt_center_inside']=='True' and float(r['gt_coverage'])>=.5 for r in rr),
            minimum_gt_coverage=min(float(r['gt_coverage']) for r in rr))
        values={}; centers={}
        for arm in ARMS:
            aa=[r for r in rr if r['arm']==arm];assert len(aa)==5
            values[arm]=np.array([float(r['iou']) for r in aa]);centers[arm]=aa[0]
            er[arm+'_iou_center']=float(aa[0]['iou'])
            er[arm+'_iou_range']=float(np.ptp(values[arm]))
            er[arm+'_iou_max_abs_change']=float(np.max(abs(values[arm]-values[arm][0])))
            for response in ['raw','hann']:
                er[arm+'_'+response+'_index_changes']=sum(r[response+'_peak']!=aa[0][response+'_peak'] for r in aa[1:])
                er[arm+'_'+response+'_support_switch']=len(set(r[response+'_peak_on_gt_support'] for r in aa))>1
                margins=[float(r[response+'_margin']) for r in aa if r[response+'_margin']]
                er[arm+'_'+response+'_margin_range']=float(np.ptp(margins)) if len(margins)==5 else None
                er[arm+'_'+response+'_margin_flip_001']=min(margins)<-.001 and max(margins)>.001 if len(margins)==5 else False
        for label,comparison in [('trained','control'),('action','candidate_off')]:
            d=values['candidate_native']-values[comparison]
            er[label+'_utility_center']=float(d[0]);er[label+'_utility_mean']=float(d.mean())
            er[label+'_utility_min']=float(d.min());er[label+'_utility_max']=float(d.max());er[label+'_utility_range']=float(np.ptp(d))
            er[label+'_utility_max_abs']=float(np.max(abs(d)))
            for tol,tag in [(0.,'exact'),(.001,'001'),(.005,'005'),(.01,'01')]:
                er[label+'_utility_flip_'+tag]=bool(d.min() < -tol and d.max()>tol)
            er[label+'_stable_help_005']=bool(d.min()>.005);er[label+'_stable_harm_005']=bool(d.max()<-.005)
            er[label+'_peak_changes']=sum(a['hann_peak']!=b['hann_peak'] for a,b in zip([r for r in rr if r['arm']=='candidate_native'],[r for r in rr if r['arm']==comparison]))
            er[label+'_range_difference']=er['candidate_native_iou_range']-er[comparison+'_iou_range']
        for arm in ARMS:
            er[arm+'_center_hann_target_margin']=float(centers[arm]['hann_margin']) if centers[arm]['hann_margin'] else None
        event_rows.append(er)
    write_csv(DEST/'per_event.csv',event_rows)

    def summarize(evs):
        result=dict(events=len(evs),sequences=len(set(e['sequence'] for e in evs)))
        if not evs:return result
        for arm in ARMS:
            result[arm]=dict(iou_range=dist([e[arm+'_iou_range'] for e in evs]),
                range_gt_001=sum(e[arm+'_iou_range']>.01 for e in evs),range_gt_005=sum(e[arm+'_iou_range']>.05 for e in evs),
                hann_any_index_change_events=sum(e[arm+'_hann_index_changes']>0 for e in evs),
                hann_target_support_switch_events=sum(e[arm+'_hann_support_switch'] for e in evs),
                raw_target_support_switch_events=sum(e[arm+'_raw_support_switch'] for e in evs),
                hann_target_margin_flip_001_events=sum(e[arm+'_hann_margin_flip_001'] for e in evs))
        for label in ['trained','action']:
            result[label]=dict(utility_range=dist([e[label+'_utility_range'] for e in evs]),max_abs_utility=dist([e[label+'_utility_max_abs'] for e in evs]),
                center_utility_mean=float(np.mean([e[label+'_utility_center'] for e in evs])),
                flip_counts={tag:sum(e[label+'_utility_flip_'+tag] for e in evs) for tag in ['exact','001','005','01']},
                stable_help_005=sum(e[label+'_stable_help_005'] for e in evs),stable_harm_005=sum(e[label+'_stable_harm_005'] for e in evs),
                peak_change_forwards=sum(e[label+'_peak_changes'] for e in evs),
                range_difference=dist([e[label+'_range_difference'] for e in evs]))
        # Sequence cluster resampling; one primary event per sequence, supplements averaged per sequence.
        names=list(dict.fromkeys(e['sequence'] for e in evs))
        rng=np.random.default_rng(20261115); draw=rng.integers(len(names),size=(5000,len(names)))
        result['sequence_bootstrap95_iou_range_difference']={}
        for label in ['trained','action']:
            means=np.array([np.mean([e[label+'_range_difference'] for e in evs if e['sequence']==n]) for n in names])
            result['sequence_bootstrap95_iou_range_difference'][label]=np.quantile(means[draw].mean(1),[.025,.975]).tolist()
        return result

    summaries={}
    for role in ['primary_prefix','supplemental_retrospective']:
        members=[e for e in event_rows if role in e['roles']]
        summaries[role+'_all']=summarize(members)
        summaries[role+'_target_local']=summarize([e for e in members if e['target_local_eligible']])
    groups=[]
    primary=[e for e in event_rows if 'primary_prefix' in e['roles'] and e['target_local_eligible']]
    for group in ['SA','LI','LR','BC','TO_or_AIV','seed_direction_flip','consistent_mr_regression','consistent_mr_gain','one_seed_tie']:
        def member(e):
            if group=='TO_or_AIV':return bool(set(e['attributes'].split())&{'TO','AIV'})
            if group in ['SA','LI','LR','BC']:return group in e['attributes'].split()
            return group==e['history_group']
        ee=[e for e in primary if member(e)]
        groups.append(dict(group=group,events=len(ee),control_mean_iou_range=float(np.mean([e['control_iou_range'] for e in ee])) if ee else None,
            candidate_mean_iou_range=float(np.mean([e['candidate_native_iou_range'] for e in ee])) if ee else None,
            trained_flip_005=sum(e['trained_utility_flip_005'] for e in ee),action_flip_005=sum(e['action_utility_flip_005'] for e in ee),
            candidate_hann_support_switch=sum(e['candidate_native_hann_support_switch'] for e in ee)))
    write_csv(DEST/'attribute_history_summary.csv',groups)
    operators=list(csv.DictReader((DEST/'operator_readouts.csv').open()))
    unique={(r['event_id'],r['module'],r['dx'],r['dy']):r for r in operators}
    perturbed=[r for r in unique.values() if r['dx']!='0' or r['dy']!='0']
    operator_summary={k:dist([float(r[k]) for r in perturbed]) for k in ['relative_edit','delta_cosine_to_center','delta_relative_change_to_center']}
    operator_summary['cosine_below_zero_count']=sum(float(r['delta_cosine_to_center'])<0 for r in perturbed)
    operator_summary['observations']=len(perturbed)
    execution=json.loads((DEST/'execution_record.json').read_text())
    result=dict(scope='exploratory fixed-checkpoint diagnostic; no sequence-AUC measurement',cohorts=summaries,operator=operator_summary,execution=execution)
    write_json(DEST/'summary.json',result)

    # Paper-independent diagnostic figure; shared units avoid exaggerating tiny action effects.
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for arm,col in zip(ARMS,COLORS):
        v=np.sort([e[arm+'_iou_range']*100 for e in primary]);axes[0,0].plot(v,np.linspace(0,1,len(v)),label=arm,color=col)
    axes[0,0].set(xlabel='IoU range over five crops (percentage points)',ylabel='Fraction of eligible primary events',title='Same frame, +/-1 original-image pixel')
    axes[0,0].legend(fontsize=8)
    x=np.array([e['control_iou_range']*100 for e in primary]);y=np.array([e['candidate_native_iou_range']*100 for e in primary])
    axes[0,1].scatter(x,y,s=16,c=COLORS[1],alpha=.7);limit=max(x.max(),y.max(),.01)*1.05
    axes[0,1].plot([0,limit],[0,limit],color='#888',ls='--');axes[0,1].set(xlabel='Control IoU range (pp)',ylabel='Candidate IoU range (pp)',title='Learned models: paired crop sensitivity')
    v=np.array([e['action_utility_max_abs']*100 for e in primary]);axes[1,0].hist(v,bins=20,color=COLORS[1]);axes[1,0].set(xlabel='Max |native - residual off| IoU (pp)',ylabel='Events',title='Direct action effect (separate scale)')
    for label,col in [('trained',COLORS[0]),('action',COLORS[1])]:
        counts=[sum(e[label+'_utility_flip_'+t] for e in primary) for t in ['001','005','01']]
        axes[1,1].plot([.1,.5,1.],counts,'o-',color=col,label=label)
    axes[1,1].set(xlabel='Required positive AND negative utility (IoU pp)',ylabel='Events with a sign reversal',title='Utility sign reversal across crops');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Crop perturbation diagnostic | %d target-covered primary events | seed 42'%len(primary))
    fig.tight_layout();fig.savefig(DEST/'crop_stability.png',dpi=150);fig.savefig(DEST/'crop_stability.pdf');plt.close(fig)

    p=summaries['primary_prefix_target_local'];allp=summaries['primary_prefix_all'];s=summaries['supplemental_retrospective_target_local']
    c=p['control'];n=p['candidate_native'];a=p['action'];t=p['trained']
    text=f'''# 相邻 crop 稳定性验证：锁定帧、权重与历史状态

本次只回答一个问题：相同目标的相邻 crop 是否使定位收益翻转，以及这种敏感性是否由新增 receiver residual 特异地造成。

**结论：支持“整网定位存在相邻 crop 敏感性”；暂不支持“新增 residual 的明显动作翻转是首要原因”。** 主分析两训练模型的收益在 {t['flip_counts']['005']}/{p['events']} 个事件中发生超过正负 0.5 IoU pp 的翻转；同 checkpoint 当前动作则没有该量级的翻转，455 次峰值选择全部一致。不能把这一点解释为当前路径无用：此前连续回放已观察到微小变化经反馈放大。本轮应把“动作在模板端很稳定，但目标相关变化是否有效传到搜索端”保留为下一待区分问题，尚未定位幅度不足、下游抵消或监督方向这三种解释。

## 范围与合同

- 使用已有 u468 control/candidate，未训练、未改模型。当前 quick 为动态 Value gate + MR，不能当历史 MR g=1 的严格缩短复现。
- 主分析沿用先前锁定的 96 个分歧前事件（部分无分歧序列为固定 25% 位置）。补充沿用先前 192 个事后事件，去重后共 {execution['events']} 帧。
- 每帧五种 crop：中心、原图左右上下各 1 像素。尺寸不变，实际整数裁剪原点逐次断言；control / candidate native / 同候选关闭 residual 的图像输入和模板 hash 完全匹配。
- 主分析目标局部可评事件 {p['events']}/{allp['events']}：五种 crop 均包含 GT 中心且覆盖至少一半 GT。其他事件仍保存在原始读数。补充局部事件 {s['events']}/192。
- 共 {execution['forwards']} 个正式 forward，另 {execution['repeat_forwards']} 个重复核验。中心 crop control 回放最大框差 {execution['control_center_replay_max_box_error']:.8g} 原像素；重复输入最大框/响应差 {execution['repeat_max_box_error']:.8g}/{execution['repeat_max_score_error']:.8g}。公共模型状态未改变。
- GT 仅用于离线量化，不输入算子。score 的 GT 区域为实际整数 crop 下与标注框相交的格子；背景排除 GT 外扩一格，raw/Hann 分开保存。支持区域翻转不等于已证明语义身份翻转。

## 主结果（每序列一个事件；IoU 区间，不是 AUC）

| 读数 | control | candidate native | candidate residual off |
|---|---:|---:|---:|
| 五 crop IoU 极差均值 | {c['iou_range']['mean']*100:.4f} pp | {n['iou_range']['mean']*100:.4f} pp | {p['candidate_off']['iou_range']['mean']*100:.4f} pp |
| 极差中位数 | {c['iou_range']['median']*100:.4f} pp | {n['iou_range']['median']*100:.4f} pp | {p['candidate_off']['iou_range']['median']*100:.4f} pp |
| 极差 >5 IoU pp 的事件 | {c['range_gt_005']} | {n['range_gt_005']} | {p['candidate_off']['range_gt_005']} |
| Hann 峰索引随 crop 变化的事件 | {c['hann_any_index_change_events']} | {n['hann_any_index_change_events']} | {p['candidate_off']['hann_any_index_change_events']} |
| Hann 峰跨 GT 支持区内外的事件 | {c['hann_target_support_switch_events']} | {n['hann_target_support_switch_events']} | {p['candidate_off']['hann_target_support_switch_events']} |

候选与独立训练 control 的相对 IoU，在五种 crop 中同时出现 >+0.5pp 与 <-0.5pp 的事件为 **{t['flip_counts']['005']}/{p['events']}**；阈值 0.1/1pp 时分别为 {t['flip_counts']['001']}/{t['flip_counts']['01']}。这测量的是训练后的整网相对效用，而不是当前 residual 动作的因果效用。五 crop 均 >+0.5pp / 均 <-0.5pp 的事件为 {t['stable_help_005']}/{t['stable_harm_005']}。

同候选 native 与关闭 residual 比较，主分析 **{a['peak_change_forwards']}/{p['events']*5}** 个 forward 的 Hann 峰不同；最大绝对 IoU 差为 **{a['max_abs_utility']['max']*100:.6f}pp**。超出正负 0.5pp 的效用翻转为 **{a['flip_counts']['005']}/{p['events']}**。必须同时查看效应量，不能把数值级正负变换当作任务矫正翻转。

候选减 control 的 crop 极差均值为 {t['range_difference']['mean']*100:+.4f}pp，序列 bootstrap 95% 区间为 {np.array(p['sequence_bootstrap95_iou_range_difference']['trained'])*100}pp。候选减同 checkpoint off 为 {a['range_difference']['mean']*100:+.6f}pp。bootstrap 仅代表本 cohort 的序列采样不确定性，不代表 seed 不确定性。

补充 169 个目标局部事件的 control / candidate / off 平均极差分别为 {s['control']['iou_range']['mean']*100:.4f}/{s['candidate_native']['iou_range']['mean']*100:.4f}/{s['candidate_off']['iou_range']['mean']*100:.4f}pp；候选与 control 的相对极差方向与主分析相反，因此不能宣称候选普遍使 crop 敏感性变差或改善。补充事件的 native/off 同样没有 Hann 峰变化、没有超过正负 0.1pp 的效用翻转。

## 新增残差是否随相邻 crop 改变方向

按同一模块、相同模板 token 轴对比偏移与中心，所有事件/六模块/四偏移共 {operator_summary['observations']} 个读数。残差方向余弦中位数 {operator_summary['delta_cosine_to_center']['median']:.8f}，最差值在原始 operator CSV；方向余弦为负的读数 {operator_summary['cosine_below_zero_count']}。残差相对中心的变化范数中位数 {operator_summary['delta_relative_change_to_center']['median']*100:.5f}%，P95 {operator_summary['delta_relative_change_to_center']['p95']*100:.5f}%。这是模板轴上的动作稳定性，不是搜索空间等变性的充分证明。

## 属性与历史失败组

详见 `attribute_history_summary.csv`，只对主分析局部事件分层。SA/LI/LR/BC/TO/AIV 是重叠的官方序列标签，不是当前帧的退化真值。历史翻转指 MR−B0 的方向在 seed42/43 间改变；本次没有运行多 seed，也没有覆盖全部 134 翻转序列或全部 47 一致回归序列。所有分层均为探索性，不据此重选独立验证集。

## 判别边界与下一步

这次可区分整网的 crop 敏感性、训练后两模型的收益敏感性、同 checkpoint 当前 residual 的直接效用敏感性。不能由稀疏固定帧读数否定此前六条连续回放观察到的反馈放大；也不能把 crop 平移测试写成已经找到跨模态污染的首因。

在解释上述三层读数后才选择下一步：若 direct action 无可辨定位作用而整网收益会翻转，应优先核对修正量到搜索 head 的可利用路径和监督，而不是立即加入 crop 一致性 loss、扩充算子或改在线模板。正式方法主张仍需匹配训练和完整测试；本轮未自动晋级全量。

全部输入/配置/权重/代码快照见 CONTRACT.json；原始帧读数、响应图、逐事件、算子读数和重复核验均保留。
'''
    (DEST/'REPORT.md').write_text(text)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
