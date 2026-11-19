#!/usr/bin/env python3
"""Paired original-MR / historical-CBMR / normalization-refinement analysis."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
import analyze_mr_cbmr_quick_20261115 as base

RUN=ROOT/'output/experiments/MSHA-MR-CYCLE-DIFFERENCE-QUICK-S42-20261115'
OLD=ROOT/'output/experiments/MSHA-MR-CBMR-QUICK-S42-20261115'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def pairing():
    a=[json.loads(x) for x in (OLD/'control/batches.jsonl').read_text().splitlines()]
    b=[json.loads(x) for x in (RUN/'candidate/batches.jsonl').read_text().splitlines()]
    control_init=json.loads((OLD/'control/initial_state.json').read_text())
    candidate_init=json.loads((RUN/'candidate/initial_state.json').read_text())
    checks=dict(batch_count=len(a)==len(b)==3744,
        every_input_hash=len(a)==len(b) and all(x['input_sha256']==y['input_sha256'] for x,y in zip(a,b)),
        every_box_hash=len(a)==len(b) and all(x['template_box_sha256']==y['template_box_sha256'] for x,y in zip(a,b)),
        every_sample=len(a)==len(b) and all(x['samples']==y['samples'] for x,y in zip(a,b)),
        common_init=control_init['shared_state_sha256']==candidate_init['matched_parent_state_sha256'],
        optimizer=json.loads((OLD/'control/optimizer_contract.json').read_text())==json.loads((RUN/'candidate/optimizer_contract.json').read_text()),
        disabled_parity=json.loads((RUN/'contracts/implementation_checks.json').read_text())['disabled_before_after_bitwise'])
    checks['all_checks_pass']=all(checks.values())
    (RUN/'contracts/training_pairing.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not checks['all_checks_pass']:
        raise RuntimeError('Confounded pair; no efficacy ranking: '+repr(checks))
    return checks


def main():
    checks=pairing()
    link=RUN/'eval_control'
    if not link.exists():
        link.symlink_to(OLD/'eval_control',target_is_directory=True)
    if link.resolve()!=(OLD/'eval_control').resolve():
        raise RuntimeError('Wrong reused control path')
    base.RUN=RUN
    base.LIST=RUN/'contracts/eval_flip_gain_regression_24.txt'
    base.main()
    path=RUN/'analysis/summary.json'
    summary=json.loads(path.read_text())
    old_summary=json.loads((OLD/'analysis/summary.json').read_text())
    rows=list(csv.DictReader((RUN/'analysis/per_sequence.csv').open()))
    old_rows={r['sequence']:r for r in csv.DictReader((OLD/'analysis/per_sequence.csv').open())}
    summary['checkpoint_sha256']=sha(RUN/'candidate/checkpoint_u0468.pth.tar')
    summary['training_pairing']=checks
    summary['historical_cbmr']=old_summary['candidate']
    summary['difference_minus_cbmr']={k:summary['candidate'][k]-old_summary['candidate'][k]
                                     for k in summary['candidate']}
    summary['strata']={}
    for group,names in summary['groups']['groups'].items():
        values=np.array([float(r['delta_auc']) for r in rows if r['sequence'] in names])
        summary['strata'][group]=dict(mean_pp=float(values.mean()),median_pp=float(np.median(values)),
            wins=int((values>1e-9).sum()),ties=int((abs(values)<=1e-9).sum()),losses=int((values<-1e-9).sum()))
    for row in rows:
        row['historical_cbmr_auc']=float(old_rows[row['sequence']]['candidate_auc'])
        row['difference_minus_cbmr_auc']=float(row['candidate_auc'])-row['historical_cbmr_auc']
    diff=np.array([r['difference_minus_cbmr_auc'] for r in rows])
    summary['sequence_vs_cbmr']=dict(mean_pp=float(diff.mean()),median_pp=float(np.median(diff)),
        wins=int((diff>1e-9).sum()),ties=int((abs(diff)<=1e-9).sum()),losses=int((diff<-1e-9).sum()))
    gains=sorted([float(r['delta_auc']) for r in rows if float(r['delta_auc'])>0],reverse=True)
    summary['tail']={'severe_losses_lt_minus5pp':[r['sequence'] for r in rows if float(r['delta_auc']) < -5],
        'top3_fraction_of_positive_gain':sum(gains[:3])/sum(gains) if gains else None,
        'mean_excluding_largest_gain_and_loss_pp':float(np.sort([float(r['delta_auc']) for r in rows])[1:-1].mean())}
    path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (RUN/'analysis/per_sequence.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=['# MR cycle-difference normalization quick result','','Matched public SOT fresh seed42 /468 successful updates; original24 exploratory cohort. All3744 augmented input/box/sample hashes, common init and optimizer contracts match. Reused historical MR control is justified by before/after disabled bitwise parity. No full/seed robustness claim.','',
        '| metric | MR control | historical CBMR | difference | difference-MR pp |','|---|---:|---:|---:|---:|']
    for k in ('auc','precision','npr','op50','op75'):
        report.append('| %s | %.4f | %.4f | %.4f | %+.4f |' % (k,summary['control'][k],summary['historical_cbmr'][k],summary['candidate'][k],summary['candidate_minus_control'][k]))
    report+=['','## Historical strata','','| stratum | mean AUC delta pp | median pp | W/T/L |','|---|---:|---:|---|']
    for k,v in summary['strata'].items():
        report.append('| %s | %+.4f | %+.4f | %d/%d/%d |' % (k,v['mean_pp'],v['median_pp'],v['wins'],v['ties'],v['losses']))
    report+=['','## Per sequence','','| sequence | MR AUC | CBMR AUC | difference AUC | delta vs MR pp |','|---|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda x:float(x['delta_auc'])):
        report.append('| %s | %.4f | %.4f | %.4f | %+.4f |' % (r['sequence'],float(r['control_auc']),r['historical_cbmr_auc'],float(r['candidate_auc']),float(r['delta_auc'])))
    report+=['','## Interpretation limits','',
        'Normalization changes preserve constant messages by construction. This does not repair dense graph discrimination or establish a reliable degradation detector. Any quick AUC change includes learned network differences and tracking feedback; same-crop diagnostics do not replace retraining. No full promotion on these development results alone.',
        '',json.dumps({'auc_stats':summary['auc_delta_stats'],'tail':summary['tail'],'sequence_vs_cbmr':summary['sequence_vs_cbmr']},indent=2)]
    (RUN/'analysis/REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
