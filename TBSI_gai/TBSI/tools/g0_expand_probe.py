#!/usr/bin/env python3
"""Run fixed two-seed, six-condition probes on old G0 train and fresh validation events."""
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch import nn

CONDITIONS=('L_zero','L_E','L_Lcopy','L_E_spatial','L_E_event','XY_only')

def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def align(a):
    a=torch.tensor(a,dtype=torch.float32)
    if tuple(a.shape)==(256,768): return a
    raise ValueError('expected [256,768], got {}'.format(tuple(a.shape)))
def base_inputs(row, npz):
    L=torch.cat([align(npz['late_input_rgb']),align(npz['late_input_tir'])],1)
    E=torch.cat([align(npz['early_rgb']),align(npz['early_tir'])],1)
    y=torch.tensor(row['target_mask'],dtype=torch.float32)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 16), torch.linspace(-1, 1, 16))
    xy=torch.stack([yy, xx],-1).reshape(256,2)
    return L,E,y,xy
def make_condition(L,E,xy,cond,spatial,event_donor=None):
    if cond=='L_zero': X=torch.zeros_like(E)
    elif cond=='L_E': X=E
    elif cond=='L_Lcopy': X=L.clone()
    elif cond=='L_E_spatial': X=E[spatial]
    elif cond=='L_E_event': X=event_donor
    elif cond=='XY_only': X=torch.zeros_like(E)
    else: raise ValueError(cond)
    if cond=='XY_only': return torch.cat([xy.repeat(1,768),torch.zeros_like(E)],1),xy
    return torch.cat([L,X],1),xy if cond=='XY_only' else torch.zeros((256,2))
class Probe(nn.Module):
    def __init__(self,d=3072): super().__init__(); self.net=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,x): return self.net(x).squeeze(-1)
def ap(scores,labels):
    order=np.argsort(-scores); y=np.asarray(labels)[order]; n=float(y.sum())
    if n==0:return float('nan')
    return float((np.cumsum(y)/np.arange(1,len(y)+1)*y).sum()/n)
def metrics(scores,y):
    order=np.argsort(-scores); positives=np.where(y>0.5)[0]; ranks=[int(np.where(order==p)[0][0]+1) for p in positives]
    return {'ap_full_grid':ap(scores,y),'best_positive_rank':min(ranks) if ranks else None,'positive_recall_at_5':float(np.mean(np.isin(positives,order[:5]))) if len(positives) else None,'positive_recall_at_10':float(np.mean(np.isin(positives,order[:10]))) if len(positives) else None}
def prepare(rows, feature_dir, condition, split_events, seed, donors, train=False):
    rng=random.Random(seed); out=[]; pool=list(split_events); rng.shuffle(pool)
    for i,row in enumerate(rows):
        z=np.load(Path(feature_dir)/(row['event_id']+'.npz')); L,E,y,xy=base_inputs(row,z)
        perm=torch.tensor(list(range(256))); random.Random(seed+991).shuffle(perm.tolist())
        # fixed derangement, generated once for each run and shared by conditions/seeds
        perm=torch.roll(torch.arange(256),shifts=37)
        donor=pool[(i+1)%len(pool)]; dz=np.load(Path(feature_dir)/(donor['event_id']+'.npz'))
        donorE=torch.cat([align(dz['early_rgb']),align(dz['early_tir'])],1)
        X,_=make_condition(L,E,xy,condition,perm,donorE)
        out.append((X,y,row))
    return out
def fit(train, seed, steps=640):
    torch.manual_seed(seed); X=torch.cat([x for x,_,_ in train]); y=torch.cat([y for _,y,_ in train]);
    pos=torch.where(y>0.5)[0]; neg=torch.where(y<=0.5)[0]; g=torch.Generator().manual_seed(seed); k=min(len(neg),len(pos)*3); sel=torch.cat([pos,neg[torch.randperm(len(neg),generator=g)[:k]]])
    mean=X[sel].mean(0); std=X[sel].std(0).clamp_min(1e-6); model=Probe(); opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss=nn.BCEWithLogitsLoss()
    for _ in range(steps): opt.zero_grad(); v=loss(model((X[sel]-mean)/std),y[sel]); v.backward(); opt.step()
    return model.eval(),mean,std
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--old-export',required=True); ap.add_argument('--old-labels',required=True); ap.add_argument('--fresh-export',required=True); ap.add_argument('--output',required=True); ap.add_argument('--steps',type=int,default=640); args=ap.parse_args()
    out=Path(args.output).resolve();
    if out.exists() and any(out.iterdir()): raise RuntimeError('refusing non-empty output {}'.format(out))
    out.mkdir(parents=True)
    manifest=json.loads(Path(args.manifest).read_text()); old=read_jsonl(Path(args.old_labels)/'token_labels.jsonl'); fresh=read_jsonl(Path(args.fresh_export) / 'events.jsonl')
    train=[r for r in old if r['sequence'] in set(manifest['old_train_sequences']) and r['rescue_eligible']]; valid=[r for r in fresh if r['target_in_crop'] and r['target_token_count']>0]
    if len(train)<20 or len(valid)<20: raise RuntimeError('insufficient train/valid events')
    all_results=[]
    for seed in manifest['probe_seeds']:
      for cond in CONDITIONS:
        tr=prepare(train,args.old_export,cond,train,seed,train); va=prepare(valid,args.fresh_export+'/features',cond,valid,seed,valid)
        model,mean,std=fit(tr,seed,args.steps); rows=[]
        with torch.no_grad():
          for X,y,r in va:
            m=metrics(torch.sigmoid(model((X-mean)/std)).numpy(),y.numpy()); m.update({'seed':seed,'condition':cond,'event_id':r['event_id'],'sequence':r['sequence']}); rows.append(m)
        all_results.extend(rows); print(seed,cond,len(rows),flush=True)
    (out/'per_event.jsonl').write_text('\n'.join(json.dumps(x) for x in all_results)+'\n')
    summary={'conditions':list(CONDITIONS),'seeds':manifest['probe_seeds'],'train_events':len(train),'valid_events':len(valid),'steps':args.steps}
    for seed in manifest['probe_seeds']:
      by={(x['seed'],x['condition']):x for x in []}; summary[str(seed)]={}
      rows=[x for x in all_results if x['seed']==seed]
      for cond in CONDITIONS: summary[str(seed)][cond]={'mean_ap':float(np.mean([x['ap_full_grid'] for x in rows if x['condition']==cond])),'mean_best_rank':float(np.mean([x['best_positive_rank'] for x in rows if x['condition']==cond]))}
      base={x['event_id']:x['ap_full_grid'] for x in rows if x['condition']=='L_zero'}
      for cond in CONDITIONS[1:]:
        ds=[x['ap_full_grid']-base[x['event_id']] for x in rows if x['condition']==cond]
        summary[str(seed)][cond]['mean_delta_vs_L_zero']=float(np.mean(ds)); summary[str(seed)][cond]['positive_event_fraction']=float(np.mean(np.array(ds)>0))
    (out/'probe_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
