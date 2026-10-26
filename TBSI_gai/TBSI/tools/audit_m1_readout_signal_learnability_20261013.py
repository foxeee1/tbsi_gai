#!/usr/bin/env python3
"""Nested, sequence-held-out audit of signals for top-K readout selection."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

def ridge_fit(x,y,alpha):
    mu=x.mean(0); sd=x.std(0); sd[sd<1e-6]=1.; z=np.clip((x-mu)/sd,-8,8.); ym=float(y.mean())
    w=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-ym))
    return mu,sd,ym,w
def ridge_pred(m,x):
    mu,sd,ym,w=m; return float(ym)+np.clip((x-mu)/sd,-8,8)@w
def rankcorr(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b));
    return float(np.corrcoef(ra,rb)[0,1]) if len(a)>1 and np.std(ra)>0 and np.std(rb)>0 else 0.

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--candidates',required=True); ap.add_argument('--split-summary',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--alpha',type=float,default=10.); args=ap.parse_args()
    out=Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()): raise RuntimeError('refusing nonempty output')
    data=np.load(args.features,allow_pickle=False); cand=list(csv.DictReader(open(args.candidates))); split=json.load(open(args.split_summary)); fit=set(split['fit_sequences']); held=set(split['heldout_sequence_names'])
    templates_r=data['template_rgb'].astype(np.float32); templates_t=data['template_tir'].astype(np.float32); fused=data['fused'].astype(np.float32); rgb=data['rgb'].astype(np.float32); tir=data['tir'].astype(np.float32)
    keys={(str(data['sequence'][i]),int(data['frame'][i])):i for i in range(len(data['sequence']))}
    by={}
    for r in cand: by.setdefault((r['sequence'],int(r['frame'])),[]).append(r)
    groups={'response':0,'fusion':5,'template':9,'state':11}; rows=[]; xgroups={k:[] for k in groups}; ys=[]; meta_rows=[]
    for key,items in by.items():
        i=keys[key]; f=fused[i]; rfeat=rgb[i]; tfeat=tir[i];
        f_norm=np.linalg.norm(f,axis=2); r_norm=np.linalg.norm(rfeat,axis=2); t_norm=np.linalg.norm(tfeat,axis=2);
        tr=templates_r[i].mean(0); tt=templates_t[i].mean(0); trn=np.linalg.norm(tr)+1e-6; ttn=np.linalg.norm(tt)+1e-6
        # State-independent signals are available for all candidates; template and geometry are appended below.
        for r in items:
            rank=int(r['rank']); idx=int(r['candidate_index']); x,y=idx%16,idx//16; fv=f[y,x]; rv=rfeat[y,x]; tv=tfeat[y,x]
            def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-6))
            # The last two base slots are deployment constants retained for stable dimensional groups.
            response=np.array([float(r['rank'])/8.,x/15.,y/15.,float(r['candidate_index'])/255.,0.],float)
            fusion=np.r_[response,float(f_norm[y,x]),float(r_norm[y,x]),float(t_norm[y,x]),cos(rv,tv),float(np.linalg.norm(rv-tv))]
            template=np.r_[fusion,cos(fv,tr),cos(fv,tt),cos(rv,tr),cos(tv,tt)]
            state=np.r_[template,float(np.mean(f_norm)),float(np.std(f_norm)),float(np.linalg.norm(f.mean((0,1)))),float(np.mean(data['template_fg'][i]))]
            for name,vec in [('response',response),('fusion',fusion),('template',template),('state',state)]: xgroups[name].append(vec)
            ys.append(float(r['utility'])); meta_rows.append({'sequence':key[0],'frame':key[1],'rank':rank,'utility':float(r['utility']),'split':'fit' if key[0] in fit else 'heldout'})
    y=np.asarray(ys); train=np.asarray([r['split']=='fit' for r in meta_rows]); heldmask=~train; results=[]; predictions=[]
    for name,start in groups.items():
        x=np.asarray(xgroups[name]); model=ridge_fit(x[train],y[train],args.alpha); pred=np.asarray([ridge_pred(model,v) for v in x]);
        states={}
        for j,r in enumerate(meta_rows):
            if not heldmask[j]: continue
            states.setdefault((r['sequence'],r['frame']),[]).append((pred[j],r['utility'],r))
        selected=[]; oracle=[]; actual_pred=[]
        for key,items in states.items():
            b=max(items,key=lambda z:z[0]); selected.append(b[1] if b[0]>0 else 0.); oracle.append(max([0.]+[z[1] for z in items])); actual_pred.append(b[0])
        seqs=sorted(set(k[0] for k in states)); sm=[]; om=[]
        for s in seqs:
            sm.append(float(np.mean([selected[j] for j,k in enumerate(states) if k[0]==s]))); om.append(float(np.mean([oracle[j] for j,k in enumerate(states) if k[0]==s])))
        rmse=float(np.sqrt(np.mean((pred[heldmask]-y[heldmask])**2))); results.append({'model':name,'features':int(x.shape[1]),'heldout_rmse':rmse,'heldout_rankcorr':rankcorr(pred[heldmask],y[heldmask]),'selected_mean_utility':float(np.mean(selected)),'selected_sequence_macro_utility':float(np.mean(sm)),'oracle_sequence_macro_utility':float(np.mean(om)),'oracle_fraction':float(np.mean(sm)/np.mean(om)) if np.mean(om)>0 else 0.,'harmful_rate':float(np.mean(np.asarray(selected)<-1e-4)),'abstention_rate':float(np.mean(np.asarray(actual_pred)<=0))})
        predictions.append({'model':name,'heldout_pred':pred[heldmask].tolist(),'heldout_utility':y[heldmask].tolist()})
    out.mkdir(parents=True,exist_ok=True); (out/'summary.json').write_text(json.dumps({'id':'M1-READOUT-SIGNAL-LEARNABILITY-20261013','status':'SIGNAL-AUDIT-COMPLETE','fit_sequences':sorted(fit),'heldout_sequences':sorted(held),'models':results,'alpha':args.alpha,'limitations':['fixed-state utility only','same 24-sequence cohort as prior action-space diagnostics','candidate top-K action is response override with M1 regression maps']},indent=2)+'\n')
    with (out/'nested_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    with (out/'candidate_features.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(meta_rows[0])); w.writeheader(); w.writerows(meta_rows)
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
