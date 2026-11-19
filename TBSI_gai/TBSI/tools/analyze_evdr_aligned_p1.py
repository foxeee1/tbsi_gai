#!/usr/bin/env python3
"""Two fixed comparisons for the single EVDR receiver-transport iteration."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
import torch
import analyze_early_gate_p1_v2 as analysis
from record_early_gate_p1_v2 import register
from run_early_gate_p1_v2 import sha, write_json

RUN = ROOT/'experiments/early_gate_closed_loop_20261026/evdr_scope_v2_20261026'
P1 = RUN.parent/'p1_sot_v3'
TRAIN = ROOT/'output/experiments'
TEST = ROOT/'output/test/early_gate_p1_sot_v3'
ALIGNED_TRAIN = TRAIN/'EG_EVDR_ALIGNED_P1_SOT_S42_20261026'
ALIGNED_TEST = ROOT/'output/test/evdr_aligned_p1/aligned_u0468_screen96_max2000_20261026'


def main():
    torch.set_num_threads(1)
    if not (ALIGNED_TRAIN/'completed.json').exists() or not (ALIGNED_TEST/'completed.json').exists():
        raise RuntimeError('The fixed training/evaluation endpoint is incomplete')
    for arm in ['control','candidate']:
        train = TRAIN/('EG_EVDR_P1_SOT_V3_'+arm.upper()+'_S42_20261026')
        output = RUN/('analysis_vs_'+('early_gate' if arm=='control' else 'evdr_v1'))
        analysis.compare(SimpleNamespace(control=str(TEST/(arm+'_u0468_screen96_max2000_20261026')),
              candidate=str(ALIGNED_TEST), list=str(P1/'screen96_max2000.txt'), output=str(output),
              train_control=str(train), train_candidate=str(ALIGNED_TRAIN)))
    primary=json.loads((RUN/'analysis_vs_early_gate/summary.json').read_text())
    mechanism=json.loads((RUN/'analysis_vs_evdr_v1/summary.json').read_text())
    old_rows=list(csv.DictReader((P1/'analysis_u0468_screen96_max2000/per_sequence.csv').open()))
    new_rows=list(csv.DictReader((RUN/'analysis_vs_early_gate/per_sequence.csv').open()))
    old={r['sequence']:r for r in old_rows}
    transitions=[]
    for row in new_rows:
        prior=old[row['sequence']]
        transitions.append({'sequence':row['sequence'],'group':row['group'],
            'evdr_v1_minus_control_auc':float(prior['auc_delta']),
            'aligned_minus_control_auc':float(row['auc_delta']),
            'aligned_minus_v1_auc':float(row['auc_delta'])-float(prior['auc_delta']),
            'old_large_loss':float(prior['auc_delta']) < -5,
            'current_large_loss':float(row['auc_delta']) < -5,
            'old_new_failure_frames':int(prior['new_failure_from_control_good_frames']),
            'current_new_failure_frames':int(row['new_failure_from_control_good_frames']),
            'current_recovered_control_failure_frames':int(row['recovered_from_control_failure_frames'])})
    transitions.sort(key=lambda r:r['aligned_minus_control_auc'])
    write_json(RUN/'failure_transitions.json',{'scope':'retrospective development diagnosis; not independently validated subgroups',
         'old_large_loss_count':sum(r['old_large_loss'] for r in transitions),
         'current_large_loss_count':sum(r['current_large_loss'] for r in transitions),
         'old_losses_remaining_large':sum(r['old_large_loss'] and r['current_large_loss'] for r in transitions),
         'new_large_losses':sum(not r['old_large_loss'] and r['current_large_loss'] for r in transitions),
         'rows':transitions})
    paired=json.loads((RUN/'analysis_vs_early_gate/pairing.json').read_text())
    done=json.loads((ALIGNED_TRAIN/'completed.json').read_text())
    evaluated=json.loads((ALIGNED_TEST/'completed.json').read_text())
    overall=primary['all_evaluated']; low=primary['low_target']; normal=primary['normal_protection']
    hints=primary.get('screen_followup_checks',{})
    # These are the existing protocol's practical development tolerances,
    # never a permanent method admission gate or a full-training permission.
    if hints and all(hints.values()):
        decision='SUPPORTS-BROADER-CONFIRMATION / NO-AUTOMATIC-FULL-PROMOTION'
        next_step='Supports evaluating the fixed234 checkpoint, unused sequences and a matched second seed before considering a fresh public-SOT full run. Quick/full direction remains uncalibrated.'
    elif overall['auc_delta']<0 and normal['auc_delta']<-.25:
        decision='PAIRED-UTILITY-NEGATIVE / NORMAL-HARM / NO-FULL-PROMOTION'
        next_step='Current budget does not support receiver transport as a solution to normal-sequence harm. Do not add another operator or jointly tune step/layer/loss; distinguish common optimization/progress/feedback conditions using retained checkpoints.'
    else:
        decision='EXPLORATORY-MIXED / MECHANISM-UNRESOLVED / NO-FULL-PROMOTION'
        next_step='Inspect progress and tail concentration without selecting a best checkpoint. This development result alone does not justify full training or a universal mechanism verdict.'
    times=[]
    for p in ALIGNED_TEST.glob('*_time.txt'):
        times.extend(float(s) for s in p.read_text().split())
    fps=(len(times)/sum(times)) if times and sum(times)>0 else None
    checkpoint=ALIGNED_TRAIN/'checkpoint_u0468.pth.tar'
    register('EG-EVDR-ALIGNED-P1-SOT-S42-20261026',{
        'AUC':overall['metrics']['auc']['candidate'],'PR':overall['metrics']['precision']['candidate'],
        'NPR':overall['metrics']['norm_precision']['candidate'],'OP50':overall['metrics']['op50']['candidate'],
        'OP75':overall['metrics']['op75']['candidate'],'FPS':fps if fps is not None else 'NA',
        'GPUHours':done['gpu_wall_hours']+evaluated['seconds']/3600.,
        'Checkpoint':str(checkpoint)+' SHA256 '+sha(checkpoint)+'; successful_updates468,epoch1.0',
        'Conclusion':'468-update paired development result: aligned-Early gate overall ΔAUC%+.6f pp,low%+.6f,normal%+.6f; aligned-EVDR-v1 ΔAUC%+.6f. %s'%(overall['auc_delta'],low['auc_delta'],normal['auc_delta'],mechanism['all_evaluated']['auc_delta'],next_step),
        'Status':'SCREEN-COMPLETE / '+decision,'Evidence':str(RUN)+'; '+str(ALIGNED_TRAIN)+'; '+str(ALIGNED_TEST),
        'Notes':'Common initialization/input pairing verified: %d augmented batches,%d samples,%s successful updates. Single seed42; fixed96<=2000-frame development cohort; bootstrap is sequence uncertainty, not seed uncertainty. FPS=sum actual frame count/sum native tracker times, including initialization. No quick/full direction claim or continuation.'%(paired['batches'],paired['samples'],paired['successful_updates'])})
    lines=['# EVDR receiver运输：固定终点结果','',
       '从公共SOT独立单阶段训练，468成功更新、59904pairs、977/977序列覆盖。共同初态与%d个增强输入batch完全配对；训练状态不从quick或tracker继承。'%paired['batches'],'',
       '|对照/分组|n|ΔAUC(pp)|90%序列区间|>5pp损失|',
       '|---|---:|---:|---|---:|']
    for label,s in [('vs Early gate/总体',overall),('vs Early gate/低组',low),('vs Early gate/中组',primary['middle']),('vs Early gate/正常组',normal),('vs EVDR-v1/总体',mechanism['all_evaluated'])]:
        lines.append('|%s|%d|%+.4f|[%+.4f,%+.4f]|%d|'%(label,s['n'],s['auc_delta'],*s['sequence_bootstrap_90pct'],s['losses_over_5pp']))
    lines+=['','候选AUC %.6f、PR %.6f、NPR %.6f、OP50 %.6f、OP75 %.6f。'%(overall['metrics']['auc']['candidate'],overall['metrics']['precision']['candidate'],overall['metrics']['norm_precision']['candidate'],overall['metrics']['op50']['candidate'],overall['metrics']['op75']['candidate']),
       '', '总体逐序列中位数%+.4fpp，10%%trimmed%+.4fpp，去掉3条最大收益后%+.4fpp，最差10%%均值%+.4fpp；胜%d/负%d。'%(overall['median'],overall['trimmed_mean_10pct'],overall['mean_without_top3_gains'],overall['worst_decile_mean'],overall['wins'],overall['losses']),
       '', '决定：`'+decision+'`。'+next_step,
       '', '证据边界：固定96条≤2000帧开发cohort；本轮事件选择为回顾性诊断。不是独立验证或full244；单seed序列bootstrap不能代表seed方差。当前不宣称quick/full方向一致，不宣称方向优于幅度或receiver条件必要，不宣称新颖性。',
       '', '问题—假设—单变量—反证及实现检查见PROTOCOL.md、temporal.json、common_crop.json、operator_readouts.json、contract_checks.json。该结果仅适用于上述数据/训练进度；已排除输入/初始化不配对，尚未排除训练进度、优化协同与反馈闭环解释。',
       '', '训练GPU墙时%.6fh，测试%.6fh；原生跟踪计时FPS %s。完整指标、逐序列与输入核对见analysis_vs_early_gate/和analysis_vs_evdr_v1/。'%(done['gpu_wall_hours'],evaluated['seconds']/3600.,str(fps))]
    lines += ['', '|旧EVDR-v1大损失序列|v1−control AUC(pp)|aligned−control AUC(pp)|aligned−v1 AUC(pp)|',
              '|---|---:|---:|---:|']
    for row in sorted([r for r in transitions if r['old_large_loss']],key=lambda r:r['evdr_v1_minus_control_auc']):
        lines.append('|%s|%+.4f|%+.4f|%+.4f|'%(row['sequence'],row['evdr_v1_minus_control_auc'],row['aligned_minus_control_auc'],row['aligned_minus_v1_auc']))
    new_losses=[r for r in transitions if r['current_large_loss'] and not r['old_large_loss']]
    lines += ['', '新增>5pp损失（相对Early gate）序列：'+(', '.join('%s(%+.3f)'%(r['sequence'],r['aligned_minus_control_auc']) for r in new_losses) if new_losses else '无')+'。旧9条与新增损失按同一个既定5pp描述界限报告，不作独立验证。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write_json(RUN/'decision.json',{'status':decision,'next_step':next_step,'aligned_vs_early_gate':overall['auc_delta'],
         'low_delta':low['auc_delta'],'normal_delta':normal['auc_delta'],'aligned_vs_evdr_v1':mechanism['all_evaluated']['auc_delta'],
         'fixed_endpoint':468,'full_training_launched':False,'quick_full_direction_calibrated':False})
    contract_path=RUN.parent/'run_contract.json'
    contract=json.loads(contract_path.read_text());contract['status']='EVDR_ALIGNED_SCREEN_COMPLETE'
    contract['current_run']['state']=decision;contract['current_run']['result_evidence']='evdr_scope_v2_20261026/RESULTS.md'
    contract['evidence_axes']['utility']=json.loads((RUN/'decision.json').read_text())
    write_json(contract_path,contract)
    readme=ROOT/'README_当前研究.md'
    text=readme.read_text()
    previous='当前运行：EG_EVDR_ALIGNED_P1_SOT_S42_20261026，公共SOT单阶段fresh optimizer，468成功更新，worker4单任务。完成后配对同一96与冻结分组；不得把回顾性事件样本当独立验证。'
    text=text.replace(previous,'本轮EG_EVDR_ALIGNED_P1_SOT_S42_20261026已完成468步及同96配对测试；总体ΔAUC%+.4fpp，低组%+.4fpp，正常组%+.4fpp（均相对Early gate）。状态：%s。完整结论见evdr_scope_v2_20261026/RESULTS.md。'%(overall['auc_delta'],low['auc_delta'],normal['auc_delta'],decision))
    readme.write_text(text)
    plan_path=RUN.parent/'PLAN.md'
    with plan_path.open('a') as f:
        f.write('\n## 固定终点完成\n\n%s；总体/低组/正常ΔAUC为%+.4f/%+.4f/%+.4fpp。%s\n'%(decision,overall['auc_delta'],low['auc_delta'],normal['auc_delta'],next_step))
    print(json.dumps({'decision':decision,'next_step':next_step},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
