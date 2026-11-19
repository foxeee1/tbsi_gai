#!/usr/bin/env python3
"""Consolidate measured evidence and sync the two asset registries."""
import csv
import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from analyze_receiver_residual_boxes_attributes_20261115 import OUT,RUN,write_json,write_csv,summarize,sha

def register(record):
    csvpath=ROOT/'experiments/experiment_asset_registry.csv';mdpath=ROOT/'experiments/实验资产数据库.md'
    existing=list(csv.DictReader(csvpath.open()));fields=list(existing[0]);identity=record['ID']
    assert not any(r['ID']==identity for r in existing),'duplicate asset '+identity
    row={k:record.get(k,'NA') for k in fields}
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=fields);w.writerow(row)
    block='\n\n<!-- '+identity+' BEGIN -->\n### '+identity+'\n\n'+'\n'.join('- **'+k+'**：'+str(row[k]) for k in fields)+'\n<!-- '+identity+' END -->\n'
    # Preserve the exact old bytes in a local evidence backup before paired appends.
    backup=OUT/'registry_before_finalization';backup.mkdir(exist_ok=True)
    for p in [csvpath,mdpath]:
        if not (backup/p.name).exists():shutil.copy2(p,backup/p.name)
    with csvpath.open('a') as f:f.write(stream.getvalue())
    with mdpath.open('a') as f:f.write(block)

def f(v):return 'NA' if v is None else '%+.3f'%v

def main():
    rows=list(csv.DictReader((OUT/'per_sequence.csv').open()))
    typed=[dict(sequence=r['sequence'],control_auc=float(r['control_auc']),candidate_auc=float(r['candidate_auc']),delta_auc=float(r['delta_auc'])) for r in rows]
    summary=json.loads((OUT/'summary.json').read_text());a=summary['attributes']
    # For n<=3, removal of the top3 has no remaining sequences; report NA.
    for s in a.values():
        if s['n']<=3:s['mean_without_top3']=None
    write_json(OUT/'summary.json',summary)
    atr=list(csv.DictReader((OUT/'attributes.csv').open()))
    for r in atr:
        if int(r['n'])<=3:r['mean_without_top3']='NA'
    write_csv(OUT/'attributes.csv',atr)
    fixed=json.loads((OUT/'fixed_crop/summary.json').read_text());prefix=json.loads((OUT/'fixed_crop_prefix/summary.json').read_text())
    loop=json.loads((OUT/'closed_loop_six/summary.json').read_text())
    split=json.loads((OUT/'trajectory_split/summary.json').read_text())
    fullrows={r['sequence']:r for r in rows}
    for r in rows:
        r['dominant_auc_component']=max(['rescue','harm','both_good','both_bad','rest'],key=lambda c:abs(float(r[c+'_auc_contribution_pp'])))
    write_csv(OUT/'per_sequence.csv',rows)
    # Independent conservation checks for the analysis tool itself.
    assert abs(sum(v['delta_auc_contribution_pp'] for v in summary['decomposition'].values())-summary['overall']['delta_auc'])<1e-9
    assert len(list((OUT/'sequences').glob('*.png')))==len(rows)==96
    assert all(s['state_unchanged'] for s in [fixed,prefix])
    total_invalid=sum(int(r['invalid_gt_frames']) for r in rows)
    checks=dict(sequence_count=96,per_sequence_official_auc_reproduction='all96 errors <1e-8 pp',decomposition_sums_to_overall=True,
        invalid_gt_frames=total_invalid,handling='invalid GT contributes0 AUC under official all-frame denominator; excluded from geometric/failure statistics and event selection',
        initialization_frames=96,annotated_noninitial_frames=sum(int(r['frames']) for r in rows)-96-total_invalid,
        visualization_count=96,one_panel_per_sequence=True,paired_augmented_microbatches=summary['pairing'],
        fixed_crop_replay_max_error=max(fixed['control_replay_max_abs_box_error'],prefix['control_replay_max_abs_box_error']))
    write_json(OUT/'analysis_checks.json',checks)
    snapshot=OUT/'source/audit_code.tar.gz'
    if not snapshot.exists():
        with tarfile.open(str(snapshot),'w:gz') as tar:
            for folder in ['lib','tools']:
                for p in sorted((ROOT/folder).rglob('*.py')):
                    if '__pycache__' not in str(p):tar.add(str(p),arcname=str(p.relative_to(ROOT)),recursive=False)
            for p in (ROOT/'experiments/tbsi_track').glob('MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-*.yaml'):
                tar.add(str(p),arcname=str(p.relative_to(ROOT)),recursive=False)
    snap=str(snapshot.relative_to(ROOT))+' SHA256 '+sha(snapshot)
    text=['# MR 接收端消息残差：逐序列、属性与学习后机制复核','',
        '本轮结论：总体 quick 均值提高，但尚未建立“接收端条件 → 目标局部消息修正 → 稳定定位”的机制闭环。最需要先修正的是对照身份与已学动作的可辨识性，而不是继续增加算子复杂度。',
        '', '## 数据与口径', '',
        '全部96条已评测序列、54666帧，seed42、468成功更新、公共SOT单阶段、FP32、模板更新关闭。逐序列AUC由独立NumPy实现计算，96条均复现已有官方口径到1e-8 pp以内。共同初态相同、3744个增强microbatch的输入与采样元数据完全相同。',
        '', '存在%d帧无效标注，官方AUC保留所有帧作分母、无效帧贡献0；几何/失锁统计排除无效GT与初始化帧。未凭0面积标注推断遮挡。本地官方属性是序列级0/1，不提供退化发生时刻，不能据此宣称逐帧退化识别。'%total_invalid,
        '', '**对照身份更正**：quick配置使用动态Value gate；历史最强MR-only g=1的实际gate固定为1。当前只能称“动态gate版本MR上的配对quick”，不能称严格复现历史最强MR的短训前缀，也不能把历史MR/B0与本轮candidate混成三臂因果比较。历史分组仅作为外部已有标签连接。',
        '', '## 核心算子与信息流', '',
        r'在第3/6/9个融合位置，每层双向各1个模块，共6个；之前“2个模块”只适用于单个V2TemplateFusion。真实路径是 template joint attention → MR relation重新分配 → 聚合cross message → 本算子 → 加self message与共享输出投影 → 更新模板token → 后续ViT将变化传给search → CENTER head。tbsi_e分支不运行旧TBSI的显式search-to-template/template-to-search桥。',
        '', r'\[m_i=\sum_j\widetilde A_{ij}V_j,\quad h_i=\mathrm{GELU}(U_q\mathrm{LN}(t_i)\odot U_m\mathrm{LN}(m_i)),\]',
        r'\[r_i=W_oh_i+b,\quad\widehat m_i=m_i+0.1\,\max(\mathrm{RMS}(m_i),10^{-6})\tanh(r_i).\]',
        '', '上式补全旧审查报告省略的tanh与RMS下界。算子使用16维乘性交互，逐token输出通道修正；没有显式邻域、来源候选或当前搜索位置图。输入template/message可间接带搜索上下文，不能说完全没有搜索信息；但新增分支不能直接检查写入后哪个search位置受益。attention与MR排序在受干预层同输入时不直接改动，后续层可能改变。',
        '', '聚合是多对一映射：不同来源内容若得到相同m_i，后续残差无法再区分其来源。LayerNorm又去除了输入的整体幅度，readout bias允许与接收端内容无关的修正。因此它有内容变换能力，但没有由结构保证的局部目标识别、退化识别或拒绝有害写入能力。这不是代码错误，也不能只凭该表达能力分析断言方法无效。',
        '', '## 总体稳定性与框误差分解', '',
        '|指标|结果|','|---|---:|',
        '|AUC control → candidate|55.1986 → 56.4796|',
        '|均值 / 中位数 ΔAUC|+1.2809 / +0.0361 pp|',
        '|精确胜/平/负|49 / 0 / 47|',
        '|以0.25pp作描述容差的胜/近似平/负|32 / 32 / 32|',
        '|双侧各裁10%的均值|+0.2726 pp|',
        '|移除最大3个收益后的均值|+0.3730 pp|',
        '|最差ceil(10%)=10条均值|−4.5337 pp|',
        '|大于5pp收益 / 小于−5pp退化|9 / 4|',
        '', '前三大收益占净增益71.8%，占所有正收益48.7%；两个分母不能混用。旧报告的−4.9299pp取最差floor(10%)=9条，本轮明确采用ceil得到10条。序列bootstrap95%区间[+0.158,+2.601]仅描述本checkpoint开发序列的不确定性，不是跨seed稳健性。',
        '', '|帧类别（仅有效非首帧）|帧数|对总体ΔAUC贡献(pp)|','|---|---:|---:|']
    for key,label in [('rescue','control IoU<.1，candidate≥.5'),('harm','candidate IoU<.1，control≥.5'),('both_good','双方IoU≥.5'),('both_bad','双方IoU<.1'),('rest','其余')]:
        d=summary['decomposition'][key];text.append('|%s|%d|%+.4f|'%(label,d['total_frames'],d['delta_auc_contribution_pp']))
    text+=['', '全部长失败段（IoU<.1持续至少10帧）累计从14884降至13150帧，减少1734帧（11.65%）。这说明当前均值收益偏向失锁时长差异；双方跟得好的大多数帧几乎没有净收益。不能把“control失败而candidate成功”自动写为candidate从自身失败中恢复，也不能把crop出界自动归因于模态融合。',
        '', '## 属性与历史失败组', '',
        '|切片|n|均值ΔAUC|中位数|胜/负|移除最大3条收益|','|---|---:|---:|---:|---:|---:|']
    for key in ['SA','LI','BC','LR','TO','TO_or_AIV','AIV','OV']:
        d=a[key];text.append('|%s|%d|%s|%s|%d/%d|%s|'%(key,d['n'],f(d['delta_auc']),f(d['median']),d['wins'],d['losses'],f(d['mean_without_top3'])))
    text+=['', 'SA与LI值得作后续预先锁定的诊断切片；SA的中位数依然很小，LI的区间跨零，均不足以直接写成已确认属性优势。BC/LR上负例偏多，提示干扰/小目标的边界需要重点保护。AIV只有2条、OV只有3条，不能据大均值收口论文。TO/AIV合并31条的中位数为负、去前三收益后转负，当前不支持以此作为主要性能主张。属性高度共现，不能把SA、CM、SV的同一批收益计算成多个独立证据。',
        '', '|历史seed42/43分组在本96条中的交集|n|本轮ΔAUC|中位数|胜/负|去前三收益|','|---|---:|---:|---:|---:|---:|']
    for key in ['consistent_mr_regression','seed_direction_flip','consistent_mr_gain']:
        d=summary['history_groups'][key];text.append('|%s|%d|%s|%s|%d/%d|%s|'%(key,d['n'],f(d['delta_auc']),f(d['median']),d['wins'],d['losses'],f(d['mean_without_top3'])))
    text+=['', '历史“两边均回归47条”在本轮仅覆盖18条；“方向翻转134条”仅覆盖50条。历史翻转定义为MR−B0的AUC在seed42与seed43异号，并非物理退化属性或本轮candidate的跨seed翻转。本轮并未验证全部47/134条，也不能从上述单seed结果宣称已消除翻转。',
        '', '## 代表性框对照', '',
        '|序列|ΔAUC|control/candidate 长失败帧|GT中心在各自crop外的比例|官方属性|','|---|---:|---:|---:|---|']
    selected=sorted(rows,key=lambda r:float(r['delta_auc']))[:4]+sorted(rows,key=lambda r:float(r['delta_auc']),reverse=True)[:5]
    for r in selected:
        text.append('|[%s](sequences/%s.png)|%+.3f|%s/%s|%.1f%%/%.1f%%|%s|'%(r['sequence'],r['sequence'],float(r['delta_auc']),r['control_long_failure_frames'],r['candidate_long_failure_frames'],100*float(r['control_gt_center_outside_crop_fraction']),100*float(r['candidate_gt_center_outside_crop_fraction']),r['attributes']))
    text+=['', '人工查看代表性画面可见：rightbike中control转向邻近目标，candidate继续跟住标注对象；leftboy2jointhe4在多人场景中出现明显轨迹分叉。反向的girldownstairfromlight中两臂都曾失锁，control较早回到GT，而candidate较晚恢复；darktreesboy中candidate框明显膨胀、覆盖到目标之外。这里描述可见现象，不据序列名推断退化，不把视觉案例当成精确干扰物身份标注。',
        '', '## 学习后实际动作', '',
        '做两套独立配对：96条×2事件帧=192帧，以及96条各1个首次连续3帧框分歧之前的帧=96帧。每帧使用相同原图、首帧模板和control前一帧crop；分别对candidate做native/off/bias-only/receiver-roll，再加载control比较。control单帧重放误差最大6.11e-5像素，参数/BN状态未改变。',
        '', '|读数|192事件帧|96分歧前帧|','|---|---:|---:|',
        '|修正范数/消息范数|0.003459%|0.003461%|',
        '|空间公共分量占修正能量（模块/帧平均）|92.44%|92.40%|',
        '|bias-only修正范数/完整修正范数|67.67%|67.64%|',
        '|关闭残差后的平均IoU变化|−0.00000147|+0.00000173|',
        '|关闭/只保留bias/receiver-roll的Hann峰改变|均0/192|均0/96|',
        '', 'bias比率是范数比，不是可加的因果贡献率。空间公共能量对每个模块调用分别计算再平均，不是把所有张量合并后的能量占比。前景/背景仅用首帧框映射的token中心作几何诊断，不是分割真值。',
        '', 'learned readout确实非零、未出现tanh饱和。因此不能说分支没学或代码断路；但当前读数不支持“它已学会空间选择性的接收端矫正”。不能仅因即时改动小淘汰可学习候选，同样不能仅因最终均值上涨证明局部机制。',
        '', '## 六序列连续轨迹路径核验', '',
        '补做已有结果中最大3收益、最大3退化序列的candidate同checkpoint原生重放与关闭残差重放。选样回顾性，仅检验路径必要性；off不是重训parent。',
        '', '|序列|原存AUC|native重放AUC|off AUC|off−native|native逐框最大复现误差(px)|','|---|---:|---:|---:|---:|---:|']
    for r in loop['rows']:
        text.append('|%s|%.4f|%.4f|%.4f|%+.4f|%.8g|'%(r['sequence'],r['saved_native_auc'],r['replay_native_auc'],r['off_auc'],r['off_minus_native_auc'],r['native_replay_max_box_error']))
    text+=['', '原生重放已逐框复现（最大误差约5e-10像素），可以确认这条残差路径对所选部分连续轨迹有实质影响：两条最大收益序列关闭后大幅退化，三条退化序列关闭后回升，rightbike则保留原收益。故不能把即时改动小解释为分支无用或仅影响训练路径；同样不能把全部quick收益归因为推理期残差。六条仅是回顾性路径必要性诊断，不替代匹配重训。',
        '', '## 微小动作如何形成大幅轨迹变化', '',
        '补做6条序列首次整数crop不同和首次持续IoU分歧处的2×2核验：真实native/off历史state × 当前native/off动作，共10事件40次forward。相同state下动作开关单帧IoU差最大约7.5e-5、定位峰均不改变；换另一真实state，在4个大分歧点IoU变化为0.519、0.758、0.784、0.772。',
        '', '首次整数crop不同发生于第251/41/11/72/170/55帧（顺序同六序列表）。前5条在前一预测框仅约0.0016–0.0067像素差时，即跨过sample_target的round/ceil采样边界。它为微小动作→crop输入不同→轨迹差异放大提供直接证据，但不能认定整数采样是全部失败的唯一首因。完整时间线与动作/state配对见trajectory_split/。',
        '', '## 建议的论文闭环与唯一最小后续', '',
        '问题应聚焦“跨模态模板消息写入缺乏局部可用性约束，目标/干扰混淆时小差异经定位crop反馈形成长失锁”。本轮已观察到长失锁收益/伤害，但仍未证实其首因就是跨模态消息。避免写成在线模板被污染：本tracker首帧模板固定，跨帧状态主要由预测crop反馈。',
        '', '第一步先对齐真正MR g=1的公开预训练单阶段合同；动态gate quick不能替代该对照。下一轮必须先在固定同一帧/状态下，对所有臂施加相同的±1原像素crop平移，测GT附近响应、非GT峰、框误差、修正的一致性。这是评测诊断变量，不修改正式tracker，不涉及回滚或模板更新。结构上若仍要一个最小可证伪候选，只在现有readout输出、tanh之前减去模板token维均值：',
        r'\[r_i^c=r_i-\frac1N\sum_k r_k,\qquad\widehat m_i=m_i+0.1\,\mathrm{RMS}(m_i)\tanh(r_i^c).\]',
        '', '这是一个明确的空间对比约束：token无关的bias和公共readout分量被严格消掉，迫使修正来自token之间的差异。唯一主变量是新增分支输出的token均值投影，位置/rank/上界/loss/数据/进度不变；现阶段仅是待验证建议，未修改模型或启动训练。它与历史图平滑/常量保持不同：这里直接移除新增修正的公共模式，而不是平滑消息或仅保证常量不漂移。',
        '', '该候选只区分空间公共修正与token差异修正，不能由92%公共能量直接断言公共分量有害，也不能保证更稳定。移除后可能损失有效全局信息；当前动作很小，削弱它可能只得到无信息结果。应先用固定crop微扰检验局部效用，再决定是否匹配学习。若微扰复核证实同一目标在邻近crop下修正翻转，才单独考虑训练侧GT对齐的目标区域等变一致性loss：把两种crop的目标区域输出映回共同原图坐标，只约束可见目标附近响应，避免raw输出强制相同或把背景伪峰一起蒸馏。结构去公共分量与该loss必须分开实验；均尚未实现或启动。',
        '', '验证不应只看总AUC：预先固定完整测试范围与SA/LI诊断切片、BC/LR保护切片；报告中位数、0.25pp容差胜负、裁尾均值、去前三收益、最差10条、长失锁时长和新增严重退化。SA/LI切片需在未用于挑选该假设的序列/配对第二seed上确认，属性只用于报告而非按测试标签路由。闭环证据至少区分参数共享训练效应、条件是否被实际使用、同crop定位效用和连续跟踪效用。',
        '', '当前可写的有边界结论是“本quick观察到长期失锁减少、同时仍有新增伤害，局部矫正机制尚未证实”；当前不能写“已解决退化感知/模态可靠性排序/多数序列稳定提升”。',
        '', '## 完整产物', '',
        '- [96条可筛选图册](index.html)：逐序列RGB/TIR三事件框与IoU曲线，可按序列/属性检索、按收益或退化排序。',
        '- [逐序列数值与属性](per_sequence.csv)，[官方属性切片](attributes.csv)，[逐帧预测/标注/几何误差](per_frame.csv.gz)。',
        '- [帧类别分解](frame_decomposition.csv)，[分析校验](analysis_checks.json)，[事件清单](events.json)。',
        '- `fixed_crop/`、`fixed_crop_prefix/`：锁定事件、逐帧干预、六模块原始读数、环境与checkpoint校验。',
        '- `closed_loop_six/`：原生重放/off逐帧框、命令、环境、逐序列AUC。',
        '- `source/audit_code.tar.gz`：本轮分析代码快照；原训练代码快照如缺失，不用当前快照冒充原训练provenance。',
        '', '全部96条明细如下；dominant component仅按观测AUC贡献归类，不是失败根因标签。',
        '', '|序列|ΔAUC|历史组|主要数值贡献类别|属性|','|---|---:|---|---|---|']
    for r in sorted(rows,key=lambda x:float(x['delta_auc'])):
        text.append('|[%s](sequences/%s.png)|%+.3f|%s|%s|%s|'%(r['sequence'],r['sequence'],float(r['delta_auc']),r['history_group'],r['dominant_auc_component'],r['attributes']))
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
    base=dict(Date='20261115',Seed=42,Parent='MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-CANDIDATE; matched dynamic-gate control',CommitOrSnapshot=snap,
        Config='MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-control/candidate YAML; same explicit u468 checkpoints',
        Checkpoint='control26e9a6673b16e06f707619d80c3ffc081d0af77fe016d8fbc33aa9170b21b644; candidate c9e90631e2f43a468769ff2c6403b03a3a7f3cf4c012cf47949db34b3671c33d',
        Evidence=str((OUT/'REPORT.md').relative_to(ROOT)),Notes='Exploratory post-hoc single-seed; official sequence-level overlapping attributes; no training/model change; current quick is not historical MR g=1.')
    record=dict(base);record.update(ID='MR-RECEIVER-BOX-ATTRIBUTE-AUDIT-S42-20261115',Dataset='All96 fixed screen96 /54666frames; invalidGT37; seed42/43 historical labels joined; no exclusions',
        EvalScope='Read-only per-frame boxes; all96 panels; sequence attributes and geometric crop reconstruction; optimizer steps0',AUC=56.479563868814346,PR=69.00359252095222,NPR=65.39899652202924,OP50=67.99693301320076,OP75=48.26187855067352,
        GPU='CPU-only; reads outputs from1x RTX4090D',HardwareConfig='Python3.8.20; NumPy'+np.__version__+'; single process; no GPU inference',GPUHours=0,
        AssumptionOrHypothesis='Distinguish broad box-quality benefit from concentrated sustained-failure differences; test whether official attributes support scoped claims.',
        Conclusion='DeltaAUC+1.280923; median+.036127;49/0/47; tolerance.25pp32/32/32. Rescue+1.671331/harm-.525586/bothgood+.026462pp. SA/LI exploratory;TOorAIV31 mediannegative and remove-top3negative. History regress18/47 and flip50/134 only; not all failures fixed.',Status='DIAGNOSTIC-COMPLETE / STABILITY-MIXED / ATTRIBUTES-EXPLORATORY');register(record)
    record=dict(base);record.update(ID='MR-RECEIVER-LEARNED-ACTION-AUDIT-S42-20261115',Dataset='All96;192 retrospective event frames plus96 pre-disagreement frames; identical control-previous-state crop',
        EvalScope='Candidate native/off/bias-only/receiver-roll pluscontrol;1440 one-step forwards;6modules;optimizer steps0',GPU='1x NVIDIA GeForce RTX4090D24GiB',
        HardwareConfig='Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8;FP32 high;threads1;no workers;temporal/update disabled',GPUHours=(fixed['wall_seconds']+prefix['wall_seconds'])/3600,
        AssumptionOrHypothesis='Distinguish learned receiver/target-specific action from token-common correction and training-path effects, including frames before trajectory disagreement.',
        Conclusion='All288 fixed-crop comparisons have0 Hann-peak changes;edit norm~.00346%;common correction energy~92.4%;bias/fullnorm~67.6%;meanIoU change~1e-6;nonzero readout,no saturation,state unchanged. Does not establish local semantic repair;separate6-sequence replay confirms substantial continuous-trajectory path effects.',Status='DIAGNOSTIC-COMPLETE / LOCAL-CAUSAL-CLAIM-UNRESOLVED');register(record)
    record=dict(base);record.update(ID='MR-RECEIVER-SIX-REPLAY-AUDIT-S42-20261115',Dataset='Retrospective top3 gains and bottom3 losses from96;'+str(sum(r['frames'] for r in loop['rows']))+'frames per arm',
        EvalScope='Candidate same checkpoint native/off closed-loop;12 sequence replays;no optimizer steps',GPU='1x NVIDIA GeForce RTX4090D24GiB',HardwareConfig='Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8;FP32 high;threads1;no workers',GPUHours=loop['seconds']/3600,
        AssumptionOrHypothesis='Check whether small instantaneous residual edits accumulate into the largest native trajectory gains/losses, with native saved-result reproducibility control.',
        Conclusion='Per-sequence same-checkpoint off-minus-native AUC: '+'; '.join(r['sequence']+':%+.6f'%r['off_minus_native_auc'] for r in loop['rows'])+'; native replay maxboxerror '+str(max(r['native_replay_max_box_error'] for r in loop['rows']))+'px. Off is not retrained control; selected6 only.',Status='DIAGNOSTIC-COMPLETE / PATH-NECESSITY-TESTED-SELECTED6');register(record)
    record=dict(base);record.update(ID='MR-RECEIVER-CROP-TRAJECTORY-SPLIT-S42-20261115',Dataset='Same6 retrospective sequences;10events at first crop difference or sustained IoU gap',
        EvalScope='2actual priorstates x2 operatorstates;40 samecheckpoint forwards;no training',GPU='1x NVIDIA GeForce RTX4090D24GiB',HardwareConfig='Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8;FP32 high;threads1',GPUHours=split['seconds']/3600,
        AssumptionOrHypothesis='Distinguish current operator action from already diverged crop input;quantify integercrop boundary crossing.',
        Conclusion='All10events have0 same-state operator peakchanges and IoU change<7.5e-5;actual state switch at major gaps changesIoU .519/.758/.784/.772. First integercrop boundary crossed with priorboxdifference .0016-.0067px in5 sequences. Supports feedback amplification;not sole cause claim.',Status='DIAGNOSTIC-COMPLETE / STATE-ACTION-SEPARATED / FEEDBACK-AMPLIFICATION-OBSERVED');register(record)
    # Retain a transparent record of the initial analysis-tool failure, repaired before conclusions.
    record=dict(base);record.update(ID='MR-RECEIVER-BOX-AUDIT-TOOL-REPAIR-20261115',Dataset='First attempted all96 trajectory analysis stopped at ballshootatthebasket3times',EvalScope='CPU analysis only; no prediction/model mutation',GPU='CPU-only',HardwareConfig='Python3.8.20/NumPy',GPUHours=0,
        AssumptionOrHypothesis='Verify GT validity handling matches official all-frame AUC denominator.',Conclusion='Initial tool incorrectly assumed all GT width/height positive; stopped on invalid GT. Repaired with explicit valid mask;37 invalid GTframes reported;all96 official AUC match <1e-8pp. Not a method failure.',Status='FAILED-ANALYSIS-TOOL / REPAIRED');register(record)
    print(json.dumps(dict(report=str(OUT/'REPORT.md'),checks=checks,code_snapshot=snap,assets_added=5),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
