#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np

ARMS=('m1','fixed_rgb','fixed_tir','s1','s2','s3','s1s2','s1s2s3')
METRICS=('auc','precision','norm_precision','op50','op75')
FLOW=('gt_rank_hann','peak_distance_tokens','score_entropy','rgb_norm','tir_norm','fused_norm','branch_cosine','iou')
def load(path,name):
    with (path/name).open(newline='') as f: return list(csv.DictReader(f))
def boot(x):
    a=np.asarray(x,float); rng=np.random.RandomState(20261012)
    return [float(v) for v in np.percentile(rng.choice(a,(5000,len(a)),replace=True).mean(1),[2.5,97.5])]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root-dir',required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args()
    root,out=Path(a.root_dir).resolve(),Path(a.output_dir).resolve()
    if out.exists() and any(out.iterdir()): raise RuntimeError('nonempty output')
    dirs={'m1':'m1_full20_retry','fixed_rgb':'fixed_rgb_full20','fixed_tir':'fixed_tir_full20','s1':'s1_full20','s2':'s2_full20','s3':'s3_full20','s1s2':'s1s2_full20','s1s2s3':'s1s2s3_full20'}
    data={k:{r['sequence']:r for r in load(root/v,'per_sequence.csv')} for k,v in dirs.items()}
    names=sorted(data['m1']); assert len(names)==20
    table=[]
    for arm in ARMS:
        item={'arm':arm,'n_sequences':20,'n_info_samples':193}
        for metric in METRICS:
            vals=np.array([float(data[arm][s][metric]) for s in names]); item[metric]=float(vals.mean())
            if arm!='m1':
                d=vals-np.array([float(data['m1'][s][metric]) for s in names]); item[metric+'_delta']=float(d.mean()); item[metric+'_delta_ci95']=boot(d)
                if metric=='auc': item.update(auc_wins=int((d>0).sum()),auc_losses=int((d<0).sum()),auc_ties=int((d==0).sum()))
        item['fps_sequence_macro']=float(np.mean([float(data[arm][s]['fps']) for s in names]))
        inf=load(root/dirs[arm],'information_flow.csv')
        for metric in FLOW: item[metric+'_sequence_macro']=float(np.mean([float(r[metric]) for r in inf]))
        table.append(item)
    result={'id':'M1-SINGLE-SIGNAL-NORMAL20-20261012','scope':'20 frozen normal sequences with archived M1 AUC>=60 and zero severe-run frames','table':table}
    out.mkdir(parents=True); (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=[k for k,v in table[0].items() if not isinstance(v,list)]
    with (out/'eight_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:r.get(k,'') for k in fields} for r in table)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
