#!/usr/bin/env python3
"""Sequence-held-out selector for frozen fused/RGB/TIR candidates."""
import csv, hashlib, json, math, platform, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/experiments/M1_S1_MODALITY_CANDIDATE_AUDIT_20261015_retry2"
OUT_DEFAULT = ROOT / "output/experiments/M1_S1_CANDIDATE_SELECTOR_OOF_20261015"
CONFIG = ROOT / "experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml"
CHECKPOINT = ROOT / "output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def features(rows):
    arms = ("m1_fused", "rgb_only", "tir_only")
    x=[]
    for r in rows:
        b=np.asarray(r["m1_fused_box"],float); bw=max(b[2],1e-3); bh=max(b[3],1e-3)
        vals=[]
        for a in arms:
            q=np.asarray(r[a+"_box"],float)
            vals += [(q[0]-b[0])/bw,(q[1]-b[1])/bh,q[2]/bw,q[3]/bh,
                     math.log(max(q[2],1e-3)/bw),math.log(max(q[3],1e-3)/bh)]
        rgb=np.asarray(r["rgb_only_box"],float); tir=np.asarray(r["tir_only_box"],float)
        vals += [(rgb[0]-tir[0])/bw,(rgb[1]-tir[1])/bh,
                 (rgb[2]-tir[2])/bw,(rgb[3]-tir[3])/bh]
        x.append([1.0]+vals)
    return np.asarray(x,float)

def fit(x,y,alpha=10.0):
    mu=x[:,1:].mean(0); sd=x[:,1:].std(0); sd[sd<1e-6]=1
    z=(x[:,1:]-mu)/sd; zz=np.concatenate([np.ones((len(z),1)),z],1)
    return mu,sd,np.linalg.solve(zz.T@zz+alpha*np.eye(zz.shape[1]),zz.T@y)

def pred(m,x):
    mu,sd,w=m; z=(x[:,1:]-mu)/sd; return np.c_[np.ones((len(z),1)),z]@w

def ci(vals, groups, seed=3407, n=10000):
    d={}
    for v,g in zip(vals,groups): d.setdefault(g,[]).append(v)
    means=np.array([np.mean(v) for v in d.values()]); rng=np.random.RandomState(seed)
    s=rng.randint(0,len(means),(n,len(means))); q=means[s].mean(1)
    return [float(np.quantile(q,.025)),float(np.quantile(q,.975))]

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--source',default=str(SOURCE)); ap.add_argument('--evidence',required=True); ap.add_argument('--output',default=str(OUT_DEFAULT)); args=ap.parse_args()
    src=Path(args.source); out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refusing nonempty output '+str(out))
    rows=list(csv.DictReader((src/'per_state_candidates.csv').open()))
    evidence={(r['sequence'],r['frame']):r for r in csv.DictReader(Path(args.evidence).open())}
    if set(evidence)!={(r['sequence'],r['frame']) for r in rows}: raise RuntimeError('evidence keys differ')
    for r in rows:
        for a in ('m1_fused','rgb_only','tir_only'):
            r[a+'_box']=[float(r[a+'_box_'+n]) for n in ('x','y','w','h')]
        e=evidence[(r['sequence'],r['frame'])]
        for a in ('m1_fused','rgb_only','tir_only'):
            for n in ('peak_score','top2_margin','entropy','peak_x','peak_y'):
                r[a+'_'+n]=float(e[a+'_'+n])
    x0=features(rows); extra=[]
    for r in rows:
        vals=[]
        for a in ('m1_fused','rgb_only','tir_only'):
            vals += [r[a+'_peak_score'],r[a+'_top2_margin'],r[a+'_entropy'],r[a+'_peak_x']/15.0,r[a+'_peak_y']/15.0]
        vals += [r['rgb_only_peak_score']-r['tir_only_peak_score'],r['rgb_only_top2_margin']-r['tir_only_top2_margin'],r['rgb_only_entropy']-r['tir_only_entropy']]
        extra.append(vals)
    x=np.c_[x0,np.asarray(extra,float)]; y=np.asarray([[float(r[a+'_iou']) for a in ('m1_fused','rgb_only','tir_only')] for r in rows]); groups=np.asarray([r['sequence'] for r in rows])
    out.mkdir(parents=True)
    selected=[]
    for held in sorted(set(groups)):
        te=groups==held; tr=~te
        models=[fit(x[tr],y[tr,j]) for j in range(3)]
        scores=np.column_stack([pred(m,x[te]) for m in models])
        # A small margin protects fused when the selector is uncertain.
        chosen=np.argmax(scores,1)
        for k,idx in enumerate(np.where(te)[0]):
            chosen[k]=0 if scores[k,chosen[k]] < scores[k,0]+0.01 else chosen[k]
            selected.append((idx,int(chosen[k]),scores[k]))
    selected.sort(); pred_arm=np.array([a for _,a,_ in selected]); pred_score=np.array([s for _,_,s in selected]);
    actual=np.asarray([[float(r[a+'_iou']) for a in ('m1_fused','rgb_only','tir_only')] for r in rows])
    sel_iou=actual[np.arange(len(rows)),pred_arm]; fused=actual[:,0]; oracle=actual.max(1)
    result={'states':len(rows),'sequences':len(set(groups)),'features':int(x.shape[1]),
            'arms':{},'selector':{},'provenance':{}}
    for name,v in [('m1_fused',fused),('selector',sel_iou),('oracle',oracle)]:
        d=v-fused
        result['arms'][name]={'mean_iou':float(v.mean()),'median_iou':float(np.median(v)),
          'mean_delta_vs_fused':float(d.mean()),'median_delta_vs_fused':float(np.median(d)),
          'positive_fraction':float(np.mean(d>1e-12)),'negative_fraction':float(np.mean(d<-1e-12)),
          'sequence_cluster_ci_delta':ci(d,groups)}
    result['selector']={'chosen_counts':{a:int(np.sum(pred_arm==j)) for j,a in enumerate(('fused','rgb','tir'))},
                        'matches_oracle':float(np.mean(pred_arm==np.argmax(actual,1))),
                        'selector_beats_fused_states':int(np.sum(sel_iou>fused+1e-12)),
                        'selector_hurts_states':int(np.sum(sel_iou<fused-1e-12))}
    out.mkdir(exist_ok=True)
    with (out/'per_state_selector.csv').open('w',newline='') as f:
        fields=['sequence','frame','selected_arm','selected_iou','fused_iou','rgb_iou','tir_iou','oracle_iou','delta_vs_fused','pred_fused','pred_rgb','pred_tir']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i,r in enumerate(rows):
            w.writerow({'sequence':r['sequence'],'frame':r['frame'],'selected_arm':('fused','rgb','tir')[pred_arm[i]],'selected_iou':sel_iou[i], 'fused_iou':fused[i], 'rgb_iou':actual[i,1], 'tir_iou':actual[i,2], 'oracle_iou':oracle[i], 'delta_vs_fused':sel_iou[i]-fused[i], 'pred_fused':pred_score[i,0], 'pred_rgb':pred_score[i,1], 'pred_tir':pred_score[i,2]})
    by={}
    for i,r in enumerate(rows): by.setdefault(r['sequence'],[]).append(i)
    with (out/'per_sequence_selector.csv').open('w',newline='') as f:
        fields=['sequence','states','fused_mean_iou','selector_mean_iou','oracle_mean_iou','selector_delta_vs_fused','oracle_delta_vs_fused','selected_fused','selected_rgb','selected_tir']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for s,ix in sorted(by.items()):
            ix=np.asarray(ix); counts=[int(np.sum(pred_arm[ix]==j)) for j in range(3)]
            w.writerow({'sequence':s,'states':len(ix),'fused_mean_iou':fused[ix].mean(),'selector_mean_iou':sel_iou[ix].mean(),'oracle_mean_iou':oracle[ix].mean(),'selector_delta_vs_fused':(sel_iou[ix]-fused[ix]).mean(),'oracle_delta_vs_fused':(oracle[ix]-fused[ix]).mean(),'selected_fused':counts[0],'selected_rgb':counts[1],'selected_tir':counts[2]})
    result['provenance']={'source':str(src),'evidence':str(args.evidence),'source_metrics_sha256':sha(src/'metrics.json'),'config_sha256':sha(CONFIG),'checkpoint_sha256':sha(CHECKPOINT),'runner_sha256':sha(Path(__file__)),'protocol':'sequence leave-one-out ridge IoU prediction from candidate geometry plus frozen response evidence; 0.01 fused margin; no GT at selection time','status':'DIAGNOSTIC-COMPLETE'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); (out/'manifest.json').write_text(json.dumps(result['provenance'],indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
