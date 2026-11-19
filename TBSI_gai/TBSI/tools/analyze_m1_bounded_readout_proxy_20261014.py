#!/usr/bin/env python3
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'experiments/manifests/m1_normal20_lowauc62_union_20261013.txt'
LOW = set((ROOT / 'output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/matched_m1_full62/sequence_list.txt').read_text().split())
NORMAL = set((ROOT / 'experiments/manifests/m1_normal20_control_20261012.txt').read_text().split())
M1 = ROOT / 'output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_9904'
CAND = ROOT / 'output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_9915'
OUT = ROOT / 'output/experiments/M1_BOUNDED_READOUT_PROXY_20261014'

def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[0]+a[2],b[0]+b[2]); y2=min(a[1]+a[3],b[1]+b[3])
    inter=max(0.,x2-x1)*max(0.,y2-y1); union=a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union > 0 else 0.

def metrics(root, seq):
    pred=np.loadtxt(root/(seq+'.txt'),delimiter='\t',ndmin=2).astype(float)
    gt=np.loadtxt(ROOT/'data/lasher/testingset'/seq/'init.txt',delimiter=',',ndmin=2).astype(float)
    n=min(len(pred),len(gt)); pred=pred[:n]; gt=gt[:n]
    ov=np.asarray([iou(pred[i],gt[i]) for i in range(n)])
    pc=np.column_stack((pred[:,0]+pred[:,2]/2,pred[:,1]+pred[:,3]/2)); gc=np.column_stack((gt[:,0]+gt[:,2]/2,gt[:,1]+gt[:,3]/2))
    d=np.linalg.norm(pc-gc,axis=1); diag=np.sqrt(gt[:,2]*gt[:,3])
    return {'AUC':float(np.mean([(ov>=t).mean() for t in np.linspace(0,1,101)])*100),'PR':float((d<=20).mean()*100),'NPR':float((d/np.maximum(diag,1e-9)<=.2).mean()*100),'OP50':float((ov>=.5).mean()*100),'OP75':float((ov>=.75).mean()*100),'frames':n}

def main():
    seqs=[x.strip() for x in MANIFEST.read_text().splitlines() if x.strip()]
    rows=[]
    for s in seqs:
        a=metrics(M1,s); b=metrics(CAND,s)
        rows.append({'sequence':s,'cohort':'lowauc62' if s in LOW else 'normal20','M1_AUC':a['AUC'],'PRED_AUC':b['AUC'],**{('delta_'+k):b[k]-a[k] for k in ('AUC','PR','NPR','OP50','OP75')}})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'per_sequence.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for group in ('normal20','lowauc62','union82'):
        sub=rows if group=='union82' else [r for r in rows if r['cohort']==group]
        summary[group]={'n':len(sub)}
        for k in ('AUC','PR','NPR','OP50','OP75'):
            d=np.array([r['delta_'+k] for r in sub]); summary[group][k]={'M1':float(np.mean([r['M1_AUC'] for r in sub])) if k=='AUC' else None,'delta':float(d.mean()),'wins':int((d>0).sum()),'losses':int((d<0).sum()),'ties':int((d==0).sum())}
    (OUT/'summary.json').write_text(json.dumps({'id':'M1-BOUNDED-READOUT-PROXY-20261014','parent':'TBSI_E_MHSA_FULL_4090D_seed42_9904','candidate':'TBSI_E_MHSA_FULL_4090D_seed42_9915','manifest':str(MANIFEST),'summary':summary},indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
