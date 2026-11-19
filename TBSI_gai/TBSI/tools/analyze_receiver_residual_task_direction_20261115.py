#!/usr/bin/env python3
"""Analyze frozen residual direction and loss-to-head transmission."""
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
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from analyze_receiver_residual_boxes_attributes_20261115 import RUN,OUT,write_json,write_csv
DEST=RUN/'analysis/task_direction_20261115'

def stats(v):
    a=np.asarray(v,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),p05=float(np.percentile(a,5)),p95=float(np.percentile(a,95))) if len(a) else None

def main():
    assert (DEST/'execution_record.json').exists()
    shutil.copy2(Path(__file__).resolve(),DEST/'source'/Path(__file__).name)
    rows=list(csv.DictReader((DEST/'frame_readouts.csv').open()));grads=list(csv.DictReader((DEST/'gradients.csv').open()))
    check=list(csv.DictReader((DEST/'checks.csv').open()));transfer=list(csv.DictReader((DEST/'transfer_readouts.csv').open()))
    meta={r['sequence']:r for r in csv.DictReader((OUT/'per_sequence.csv').open())}
    by={(int(r['event_id']),r['precision'],r['mode'],float(r['scale'])):r for r in rows}
    eligible=sorted(set(int(r['event_id']) for r in rows if r['target_local_eligible']=='True'))
    ids=sorted(set(int(r['event_id']) for r in rows));assert len(ids)==96 and len(eligible)==91
    check_summary={k:max(float(r['value']) for r in check if r['check']==k) for k in sorted(set(r['check'] for r in check))}
    assert check_summary['historical_high_native_box_max_error']<.001,check_summary
    assert check_summary['frozen1_native_score_max_error']<1e-6,check_summary
    assert check_summary['gradient0_off_score_max_error']<1e-6,check_summary
    per=[];finite=[]
    for event in eligible:
        zero=by[event,'highest','learned',0.];m=meta[zero['sequence']]
        r=dict(event_id=event,sequence=zero['sequence'],frame=int(zero['frame']),attributes=m['attributes'],history_group=m['history_group'])
        for loss in ['total','box','focal']:
            gg=[g for g in grads if int(g['event_id'])==event and g['loss']==loss];assert len(gg)==6
            directional=sum(float(g['directional_derivative']) for g in gg)
            oracle=sum(float(g['matched_oracle_derivative']) for g in gg)
            r[loss+'_derivative']=directional;r[loss+'_oracle_derivative']=oracle
            r[loss+'_alignment_efficiency']=-directional/max(-oracle,1e-30)
            r[loss+'_derivative_cancellation_ratio']=abs(directional)/max(sum(abs(float(g['directional_derivative'])) for g in gg),1e-30)
        r['box_focal_derivative_conflict']=r['box_derivative']*r['focal_derivative']<0
        for k in [1.,4.,16.,64.]:
            a=by[event,'highest','learned',k];b=by[event,'highest','learned',-k]
            o=by.get((event,'highest','gt_oracle',k))
            r['learned_loss_delta_'+str(int(k))]=float(a['total_loss'])-float(zero['total_loss'])
            r['learned_iou_delta_'+str(int(k))]=float(a['iou'])-float(zero['iou'])
            r['negative_loss_delta_'+str(int(k))]=float(b['total_loss'])-float(zero['total_loss'])
            if o:
                r['oracle_loss_delta_'+str(int(k))]=float(o['total_loss'])-float(zero['total_loss'])
                r['oracle_iou_delta_'+str(int(k))]=float(o['iou'])-float(zero['iou'])
            if k in [4.,16.]:
                fd=(float(a['total_loss'])-float(b['total_loss']))/(2*k)
                stable=a['raw_peak']==b['raw_peak']==zero['raw_peak']
                finite.append(dict(event_id=event,sequence=r['sequence'],scale=k,raw_peak_stable=stable,autograd_derivative=r['total_derivative'],
                    finite_difference_derivative=fd,relative_error=abs(fd-r['total_derivative'])/max(abs(fd),abs(r['total_derivative']),1e-12),
                    sign_agreement=fd*r['total_derivative']>0,above_floor=abs(fd)>1e-7 and abs(r['total_derivative'])>1e-7))
        per.append(r)
    write_csv(DEST/'per_event.csv',per);write_csv(DEST/'finite_difference_checks.csv',finite)
    comparisons=[]
    for precision,mode,scales in [('highest','learned',[-64.,-16.,-4.,-1.,1.,4.,16.,64.]),('highest','token_roll',[16.,64.]),('highest','gt_oracle',[1.,16.,64.]),('high','native',[1.]),('high','learned',[16.])]:
        for scale in scales:
            aa=[by[event,precision,mode,scale] for event in eligible]
            zz=[by[event,precision,'learned',0.] for event in eligible]
            d={k:np.array([float(a[k])-float(z[k]) for a,z in zip(aa,zz)]) for k in ['total_loss','box_loss','focal_loss','iou','raw_margin','hann_margin']}
            result=dict(precision=precision,mode=mode,scale=scale,events=len(aa),
                **{k+'_mean_delta':float(v.mean()) for k,v in d.items()},total_loss_median_delta=float(np.median(d['total_loss'])),iou_median_delta=float(np.median(d['iou'])),
                loss_decrease_gt1e4=int((d['total_loss']< -1e-4).sum()),loss_increase_gt1e4=int((d['total_loss']>1e-4).sum()),
                loss_decrease_gt1e3=int((d['total_loss']< -1e-3).sum()),loss_increase_gt1e3=int((d['total_loss']>1e-3).sum()),
                iou_wins_gt005=int((d['iou']>.005).sum()),iou_losses_lt_m005=int((d['iou']<-.005).sum()),
                hann_peak_changes=sum(a['hann_peak']!=z['hann_peak'] for a,z in zip(aa,zz)),raw_peak_changes=sum(a['raw_peak']!=z['raw_peak'] for a,z in zip(aa,zz)),
                max_abs_iou_delta=float(abs(d['iou']).max()),max_abs_loss_delta=float(abs(d['total_loss']).max()))
            comparisons.append(result)
    write_csv(DEST/'scale_comparisons.csv',comparisons)
    finite_summary={}
    for scale in [4.,16.]:
        ff=[r for r in finite if r['scale']==scale and r['raw_peak_stable']]
        valid=[r for r in ff if r['above_floor']]
        finite_summary[str(int(scale))]=dict(stable_peak_events=len(ff),above_floor_events=len(valid),sign_agreement=sum(r['sign_agreement'] for r in valid),
            relative_error=stats([r['relative_error'] for r in valid]),correlation=float(np.corrcoef([r['autograd_derivative'] for r in valid],[r['finite_difference_derivative'] for r in valid])[0,1]) if len(valid)>1 else None)
    alignment={}
    for loss in ['total','box','focal']:
        ds=np.array([r[loss+'_derivative'] for r in per])
        alignment[loss]=dict(derivative=stats(ds),favorable_lt_m1e7=int((ds< -1e-7).sum()),unfavorable_gt1e7=int((ds>1e-7).sum()),
            near_zero=int((abs(ds)<=1e-7).sum()),alignment_efficiency=stats([r[loss+'_alignment_efficiency'] for r in per]),
            derivative_cancellation_ratio=stats([r[loss+'_derivative_cancellation_ratio'] for r in per]))
    modules=[]
    for name in sorted(set(g['module'] for g in grads)):
        gg=[g for g in grads if g['module']==name and g['loss']=='total'];ds=np.array([float(g['directional_derivative']) for g in gg])
        modules.append(dict(module=name,events=len(gg),negative_gradient_cosine_mean=float(np.mean([float(g['negative_gradient_cosine']) for g in gg])),
            favorable_lt_m1e7=int((ds< -1e-7).sum()),unfavorable_gt1e7=int((ds>1e-7).sum()),mean_derivative=float(ds.mean()),
            mean_relative_edit=float(np.mean([float(g['delta_over_message']) for g in gg]))))
    write_csv(DEST/'module_alignment.csv',modules)
    groups=[]
    for group in ['SA','LI','LR','BC','TO_or_AIV','seed_direction_flip','consistent_mr_regression','consistent_mr_gain','one_seed_tie']:
        def member(r):
            if group=='TO_or_AIV':return bool(set(r['attributes'].split())&{'TO','AIV'})
            if group in ['SA','LI','LR','BC']:return group in r['attributes'].split()
            return r['history_group']==group
        rr=[r for r in per if member(r)]
        groups.append(dict(group=group,events=len(rr),favorable=sum(r['total_derivative']< -1e-7 for r in rr),unfavorable=sum(r['total_derivative']>1e-7 for r in rr),
            alignment_efficiency_mean=float(np.mean([r['total_alignment_efficiency'] for r in rr])) if rr else None,
            learned64_iou_mean_delta=float(np.mean([r['learned_iou_delta_64'] for r in rr])) if rr else None,
            oracle64_iou_mean_delta=float(np.mean([r['oracle_iou_delta_64'] for r in rr])) if rr else None))
    write_csv(DEST/'attribute_history_summary.csv',groups)
    precision={}
    for mode,scale in [('native',1.),('learned',0.),('learned',16.)]:
        aa=[by[e,'highest',mode,scale] for e in eligible];bb=[by[e,'high',mode,scale] for e in eligible]
        precision[mode+'_'+str(int(scale))]=dict(iou_difference=stats([float(a['iou'])-float(b['iou']) for a,b in zip(aa,bb)]),
            hann_peak_changes=sum(a['hann_peak']!=b['hann_peak'] for a,b in zip(aa,bb)),raw_peak_changes=sum(a['raw_peak']!=b['raw_peak'] for a,b in zip(aa,bb)))
    transfer_summary={}
    for scale in [1.,4.,16.,64.]:
        rr=[r for r in transfer if float(r['scale'])==scale and r['target_local_eligible']=='True']
        transfer_summary[str(int(scale))]={k:stats([float(r[k]) for r in rr]) for k in ['search_feature_relative_change','score_l2_change','score_max_abs_change']}
    execution=json.loads((DEST/'execution_record.json').read_text())
    rng=np.random.default_rng(20261115);sample=rng.integers(len(per),size=(5000,len(per)))
    intervals={k:np.quantile(np.asarray([r[k] for r in per])[sample].mean(1),[.025,.975]).tolist() for k in ['total_alignment_efficiency','learned_iou_delta_64','oracle_iou_delta_64']}
    summary=dict(events=96,eligible=91,checks=check_summary,alignment=alignment,finite_difference=finite_summary,exploratory_sequence_bootstrap95=intervals,
        box_focal_derivative_conflicts=sum(r['box_focal_derivative_conflict'] for r in per),comparisons=comparisons,precision=precision,transfer=transfer_summary,execution=execution)
    write_json(DEST/'summary.json',summary)
    # A four-panel diagnostic with quantities kept in their actual units.
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].hist([r['total_alignment_efficiency'] for r in per],bins=20,color='#237d9e');axes[0,0].axvline(0,color='#777',ls='--')
    axes[0,0].set(xlabel='Learned-direction alignment to loss descent',ylabel='Events',title='Positive favors lower GT task loss')
    ff=[r for r in finite if r['scale']==16. and r['raw_peak_stable']]
    x=np.array([r['autograd_derivative'] for r in ff]);y=np.array([r['finite_difference_derivative'] for r in ff])
    axes[0,1].scatter(x,y,s=15,alpha=.7);lo=min(x.min(),y.min());hi=max(x.max(),y.max());axes[0,1].plot([lo,hi],[lo,hi],'--',color='#777')
    axes[0,1].set(xlabel='Autograd loss derivative',ylabel='Finite difference (+/-16)',title='Same raw peak only')
    for mode,col in [('learned','#237d9e'),('token_roll','#999'),('gt_oracle','#d58a29')]:
        cc=sorted([r for r in comparisons if r['precision']=='highest' and r['mode']==mode and r['scale']>0],key=lambda r:r['scale'])
        axes[1,0].plot([r['scale'] for r in cc],[r['total_loss_mean_delta'] for r in cc],'o-',color=col,label=mode)
        axes[1,1].plot([r['scale'] for r in cc],[r['iou_mean_delta']*100 for r in cc],'o-',color=col,label=mode)
    for ax in axes[1]:ax.set_xscale('log',base=4);ax.set_xlabel('Frozen edit strength (native = 1)');ax.axhline(0,color='#777',ls='--');ax.legend(fontsize=8)
    axes[1,0].set(ylabel='Mean GT task loss change',title='GT oracle: diagnostic only')
    axes[1,1].set(ylabel='Mean Hann-decoded IoU change (pp)',title='Fixed-crop IoU, not sequence AUC')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Learned message direction | 91 locked target-covered events | no training')
    fig.tight_layout();fig.savefig(DEST/'task_direction.png',dpi=150);fig.savefig(DEST/'task_direction.pdf');plt.close(fig)
    a=alignment['total'];look={(r['precision'],r['mode'],r['scale']):r for r in comparisons}
    native=look['highest','learned',1.];amp=look['highest','learned',64.];oracle=look['highest','gt_oracle',64.];roll=look['highest','token_roll',64.]
    lines=['# 残差任务方向与搜索端传递验证','',
        '问题：固定权重、模板、帧与 crop，已学残差是否指向正确目标，还是方向有用但幅度不足？本轮仅做离线干预，无训练、无模型代码更改。','',
        '**判别结论：暂不支持“方向已经有用，只需放大幅度”。支持“这个接口仍有传递有利变化的能力，但当前已学动作没有多数目标一致性”的后续单变量检验。** 这里的“没有多数一致性”仅针对所选checkpoint和91个固定事件；不是否定训练路径作用或此前连续回放的反馈放大。','',
        '## 合同与工具核验','',
        '- 沿用原96主事件；91个满足前次目标局部合同，其余5个保留推理但不算目标损失/梯度。不是独立验证集。',
        '- 同一u468候选。主分析使用 highest FP32、关闭 CUDA/cuDNN TF32；high/native/off/16 单列以核对历史数值口径。所有参数冻结，0 optimizer steps。',
        '- 先记录native六处残差，再冻结这些向量；改变alpha只移动同一向量，不重新预测动作。alpha0为off，alpha1与native逐框/响应核对。反向、token轴roll13与GT负梯度只用于诊断。',
        '- 原损失形式为配置权重的GIoU+L1+focal，raw peak用于训练式框读数；实际定位另用Hann解码。标签使用整数裁剪GT几何；这是eval帧离线任务效用，不是训练增强回放。GT梯度方向不进入可部署方法。',
        f'- high/native最大历史框误差 {check_summary["historical_high_native_box_max_error"]:.9g}px；frozen1与native最大score差 {check_summary["frozen1_native_score_max_error"]:.9g}；gradient0与off最大score差 {check_summary["gradient0_off_score_max_error"]:.9g}。',
        '- 主损失在输出上使用float64归约，网络仍FP32；导数以同raw峰的有限差分核验，原始检查见 finite_difference_checks.csv。','',
        '## 学到的方向','',
        f'- 总损失导数：有利 {a["favorable_lt_m1e7"]}/91，不利 {a["unfavorable_gt1e7"]}/91，接近零 {a["near_zero"]}/91（±1e-7描述性阈值）。',
        f'- 相对同修改范数GT下降方向的对齐效率均值 {a["alignment_efficiency"]["mean"]:.8f}，中位数 {a["alignment_efficiency"]["median"]:.8f}。正值有利，负值不利，绝对大小衡量一阶对齐，不是模块性能。',
        f'- box与focal的总方向导数异号 {summary["box_focal_derivative_conflicts"]}/91。各模块读数见 module_alignment.csv；不能仅由冲突推出原训练loss有错。',
        f'- ±16有限差分核验：同raw峰 {finite_summary["16"]["stable_peak_events"]} 事件，超过导数底限 {finite_summary["16"]["above_floor_events"]} 个，其中符号一致 {finite_summary["16"]["sign_agreement"]} 个；相关性 {finite_summary["16"]["correlation"]}。完整误差亦保留，不能只引用方向符号。','',
        '## 冻结方向后的幅度干预','',
        '| 动作 | 强度 | 平均任务loss变化 | 平均IoU变化（pp） | IoU胜/负（±0.5pp） | Hann换峰事件 |',
        '|---|---:|---:|---:|---:|---:|']
    for mode in ['learned','token_roll','gt_oracle']:
        for r in comparisons:
            if r['precision']!='highest' or r['mode']!=mode or r['scale'] not in [1.,16.,64.,-64.]:continue
            lines.append(f'| {mode} | {r["scale"]:g} | {r["total_loss_mean_delta"]:+.8f} | {r["iou_mean_delta"]*100:+.6f} | {r["iou_wins_gt005"]}/{r["iou_losses_lt_m005"]} | {r["hann_peak_changes"]} |')
    lines+=['','GT oracle在每个模块保持与已学残差相同的范数，方向使用该帧GT loss梯度，因此它是可达性参照，绝非可部署候选；不能把它的平均改善写成方法涨点。',
        '', '## 数值与属性边界','',
        f'- 原high口径native与最高FP32口径native的Hann峰不同 {precision["native_1"]["hann_peak_changes"]}/91；IoU差异均值 {precision["native_1"]["iou_difference"]["mean"]*100:+.6f}pp。这是精度因素，不与模型作用混为一谈。',
        '- attribute_history_summary.csv覆盖SA/LI/LR/BC/TO或AIV与历史翻转/回归/改善。标签重叠且是序列级；不是逐帧退化标注。没有新增seed。',
        '- 修正能改变搜索特征/score，不自动意味着目标方向正确；score传递很弱不自动证明接口结构无能力。见transfer_readouts.csv与GT范数匹配参照。',
        '', '## 判别结论与下一步边界','',
        '1. **已排除本次工具的断图/不可达解释。** 参数虽冻结，但零叶子到head的梯度存在；同修改范数的GT下降方向可实际降低任务loss。native1/off与归因路径parity为0，±16有限差分上超过底限的88个事件符号88/88一致，相关性0.999965。不能据此证明任意可学习模块都能学到该方向。',
        '2. **暂不支持幅度单独解释。** 已学方向放大64倍的平均IoU变化仅+0.000176pp，91个事件无Hann换峰，无超过±0.5pp的胜负；损失改善16个、恶化17个（±1e-4）。反向也未获得统一改善。固定方向token-roll仅用于诊断，没有据此证实条件无用。',
        '3. **支持方向利用率不足的研究假设，尚未区分输入/表达与训练监督。** 42有利/46不利/3近零，GT下降方向对齐效率均值近0；相同范数的GT方向在64倍时91/91降低loss（均超过1e-3），Hann IoU平均+0.230799pp，中位数+0.115374pp，±0.5pp胜负6/0。GT方向刻意依赖当前样本标注，其优势只是接口可达性参照。',
        '4. **数值精度确实掩盖原强度读数，但不是统一有用方向的证据。** 主分析最高FP32下，原强度对搜索融合特征的相对变化均值约1.90e-6，score最大改变量均值约6.63e-7；high口径的细小变化不能直接当语义效用。提高精度后仍未得到多数有利方向。',
        '5. **下一优先问题**：训练时的任务监督能否使同一残差结构学到“提高GT目标响应、降低非目标响应”的方向。应保持算子/输入/初始化/数据预算不变，再检验一项训练侧效用约束；GT只用于训练，不能把本次逐帧oracle作为部署输入。尚未授权或启动该训练，也尚未断言应采用哪一种loss。',
        '',f'探索性序列bootstrap95%：对齐效率均值 {intervals["total_alignment_efficiency"]}；learned64 IoU均值变化（非pp）{intervals["learned_iou_delta_64"]}；oracle64 {intervals["oracle_iou_delta_64"]}。只度量这些固定序列事件的不确定性，不代替seed。',
        '', '历史翻转组本次可分析45条：19有利、25不利、1近零；历史一致回归18条：8有利、10不利。一致改善25条：14有利、9不利、2近零。LI18条为12/6，BC26条为8/16；均是重叠、探索性序列分层，不能按当前结果宣布属性特异机制。',
        '', '完整命令、配置/权重hash、代码快照、输入hash、数值检查和原始读数均在本目录；状态校验通过，模型权重及统计量无变化。']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
