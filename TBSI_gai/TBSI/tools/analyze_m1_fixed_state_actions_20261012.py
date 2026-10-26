#!/usr/bin/env python3
"""Sequence-paired analysis of fixed-state action replay with frozen split."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

def read(path):
    with open(path,newline='') as f: return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--actions',required=True); ap.add_argument('--screen',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--seed',type=int,default=3407); ap.add_argument('--bootstrap',type=int,default=50000); a=ap.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    actions=read(a.actions); screen=read(a.screen)
    split={r['sequence']:r['split'] for r in screen}
    if len(split)!=72 or set(split.values())!={'train','heldout'}: raise RuntimeError('expected frozen 48/24 train-heldout split')
    seqs=sorted(set(r['sequence'] for r in actions))
    if len(seqs)!=72 or not set(seqs).issubset(split): raise RuntimeError('action/split cohort mismatch')
    arms=list(dict.fromkeys(r['arm'] for r in actions)); grouped=defaultdict(list)
    for r in actions: grouped[(r['arm'],r['sequence'])].append(float(r['iou']))
    seq_mean={(arm,s):float(np.mean(grouped[(arm,s)])) for arm in arms for s in seqs}
    rng=np.random.default_rng(a.seed); scopes={'all':seqs,'train':[s for s in seqs if split[s]=='train'],'heldout':[s for s in seqs if split[s]=='heldout']}
    rows=[]
    for scope,names in scopes.items():
        for arm in arms:
            values=np.asarray([seq_mean[(arm,s)] for s in names])
            row={'scope':scope,'arm':arm,'n_sequences':len(names),'n_states':sum(len(grouped[(arm,s)]) for s in names),'mean_sequence_iou':float(values.mean()),'median_sequence_iou':float(np.median(values))}
            for control in ('no_op','fixed_tir','random'):
                if control not in arms: continue
                diff=np.asarray([seq_mean[(arm,s)]-seq_mean[(control,s)] for s in names])
                boot=np.asarray([rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(a.bootstrap)])
                prefix='vs_'+control
                row[prefix+'_delta_iou']=float(diff.mean())
                row[prefix+'_ci95_low']=float(np.quantile(boot,.025)); row[prefix+'_ci95_high']=float(np.quantile(boot,.975))
                row[prefix+'_wins']=int((diff>0).sum()); row[prefix+'_losses']=int((diff<0).sum()); row[prefix+'_ties']=int((diff==0).sum())
            rows.append(row)
    with (out/'paired_by_split.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # Label slices are state-level descriptive checks; no inferential claim is attached.
    labels=defaultdict(list)
    for r in actions:
        labels[(split[r['sequence']],r['label'],r['arm'])].append(float(r['iou']))
    label_rows=[{'split':s,'label':l,'arm':arm,'n_states':len(v),'mean_iou':float(np.mean(v))} for (s,l,arm),v in labels.items()]
    with (out/'label_slices_descriptive.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(label_rows[0])); w.writeheader(); w.writerows(label_rows)
    geometry=[]
    for scope,names in scopes.items():
        keep=set(names)
        for arm in arms:
            chosen=[r for r in actions if r['arm']==arm and r['sequence'] in keep]
            geometry.append({'scope':scope,'arm':arm,'n_states':len(chosen),
                'mean_peak_distance_tokens':float(np.mean([float(r['peak_distance_tokens']) for r in chosen])),
                'mean_decoded_center_error_px':float(np.mean([float(r['decoded_center_error_px']) for r in chosen])),
                'mean_decoded_size_rel_l1':float(np.mean([float(r['decoded_size_rel_l1']) for r in chosen])),
                'hann_peak_gt_cell_rate':float(np.mean([int(r['hann_peak_is_gt_cell']) for r in chosen]))})
    with (out/'geometry_by_split.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(geometry[0])); w.writeheader(); w.writerows(geometry)
    meta={'n_sequences':len(seqs),'calibration_split':{'train':sum(v=='train' for v in split.values()),'heldout':sum(v=='heldout' for v in split.values())},'primary_scope':'heldout','primary_estimand':'mean of per-sequence mean fixed-state decoded IoU difference','bootstrap_unit':'sequence','bootstrap_replicates':a.bootstrap,'seed':a.seed,'limitations':['fixed-state diagnostics, not closed-loop tracking','heldout cohort has 24 sequences and 44 states; selected diagnostic states are not population-representative','label slices are descriptive only','single random-direction assignment realization']}
    (out/'analysis_protocol.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
