#!/usr/bin/env python3
"""Fit a tiny sequence-held-out candidate/no-op ranker on fixed-state utility."""
import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/experiments/M1_PRECENTER_GATED_RESIDUAL_PROXY_20261016/fixed_state_audit'
OUT=ROOT/'output/experiments/M1_PRECENTER_GATED_RESIDUAL_PROXY_20261016/center_ranker_oof'
FEATURES=['rank','x','y','score','margin','center_dist','size_w','size_h','offset_x','offset_y']

def fit(x,y,alpha=10.):
    mu=x.mean(0); sd=x.std(0); sd[sd<1e-6]=1.; z=(x-mu)/sd
    A=np.c_[np.ones(len(z)),z]; w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]),A.T@y)
    return {'mean':mu.tolist(),'scale':sd.tolist(),'weight':w.tolist()}
def pred(m,x):
    z=(x-np.asarray(m['mean']))/np.asarray(m['scale']); return np.c_[np.ones(len(z)),z]@np.asarray(m['weight'])
def split(seq):
    return int(hashlib.sha256(('center-ranker-oof:'+seq).encode()).hexdigest(),16)%3 != 0

def main():
    with (RUN/'candidate_rows.csv').open() as f: rows=list(csv.DictReader(f))
    rows=[r for r in rows if r['arm']=='TBSI_E_MHSA_FULL_4090D_seed42']
    seqs=sorted({r['sequence'] for r in rows}); train={s for s in seqs if split(s)}; test=set(seqs)-train
    x=np.asarray([[float(r[k]) for k in FEATURES] for r in rows]); y=np.asarray([float(r['utility']) for r in rows]); tr=np.asarray([r['sequence'] in train for r in rows])
    model=fit(x[tr],y[tr]); scores=pred(model,x)
    groups={}
    for i,r in enumerate(rows): groups.setdefault((r['sequence'],r['frame']),[]).append((float(scores[i]),float(r['utility']),r))
    selected=[]
    for (seq,frame),items in groups.items():
        if seq not in test: continue
        best=max(items,key=lambda z:z[0]); chosen=best[1] if best[0]>0 else 0.
        selected.append({'sequence':seq,'frame':frame,'state_label':best[2]['state_label'],'predicted_utility':best[0],'selected_utility':chosen,'oracle_utility':max([0.]+[v[1] for v in items]),'abstain':int(best[0]<=0),'rank':best[2]['rank']})
    seqnames=sorted({r['sequence'] for r in selected}); per={s:float(np.mean([r['selected_utility'] for r in selected if r['sequence']==s])) for s in seqnames}; oracle={s:float(np.mean([r['oracle_utility'] for r in selected if r['sequence']==s])) for s in seqnames}; vals=np.asarray(list(per.values())); ors=np.asarray(list(oracle.values()))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'heldout_selection.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0])); w.writeheader(); w.writerows(selected)
    summary={'id':'M1-CENTER-TOPK-RANKER-OOF-20261016','train_sequences':len(train),'heldout_sequences':len(test),'heldout_states':len(selected),'selected_sequence_macro_utility':float(vals.mean()),'oracle_sequence_macro_utility':float(ors.mean()),'oracle_capture':float(vals.mean()/ors.mean()) if ors.mean()>0 else 0.,'positive_sequences':int((vals>0).sum()),'negative_sequences':int((vals<0).sum()),'abstention_rate':float(np.mean([r['abstain'] for r in selected])),'model':model,'features':FEATURES,'limitations':['fixed-state utility only','single deterministic sequence split','not closed-loop tracking','no tracker integration']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
