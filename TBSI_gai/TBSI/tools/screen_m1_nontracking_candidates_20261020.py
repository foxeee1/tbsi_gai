#!/usr/bin/env python3
"""Training-free signal screen for fourteen non-tracking AMCM hypotheses."""
import csv, json, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack

MANIFEST = ROOT/'output/experiments/M1_PRECISION_SCREEN_MANIFEST_20261020/manifest.json'
CKPT = ROOT/'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
CFG = 'TBSI_E_MHSA_FULL_4090D_seed42'
OUT = ROOT/'output/experiments/M1_NONTRACKING_CANDIDATE_SCREEN_20261020'

def read_pair(seq, frame, seqmap):
    s=seqmap[seq]
    a=cv2.cvtColor(cv2.imread(s.frames[0][frame]),cv2.COLOR_BGR2RGB)
    b=cv2.cvtColor(cv2.imread(s.frames[1][frame]),cv2.COLOR_BGR2RGB)
    return np.concatenate((a,b),2)

def auc(scores, mask):
    p=np.asarray(scores)[mask].ravel(); n=np.asarray(scores)[~mask].ravel()
    if not len(p) or not len(n): return float('nan')
    # Exact pairwise AUC; ties receive half credit.
    return float(((p[:,None] > n[None,:]).sum() + 0.5*(p[:,None] == n[None,:]).sum())/(len(p)*len(n)))

def event_mask(event, size=16):
    x,y,w,h=map(float,event['crop_xywh']); gx,gy,gw,gh=map(float,event['gt_box'])
    x1,x2=(gx-x)*size/w,(gx+gw-x)*size/w; y1,y2=(gy-y)*size/w,(gy+gh-y)*size/w
    yy,xx=np.mgrid[:size,:size]
    m=((xx+.5>=x1)&(xx+.5<=x2)&(yy+.5>=y1)&(yy+.5<=y2))
    if not m.any(): m[int(np.clip((y1+y2)/2,0,size-1)),int(np.clip((x1+x2)/2,0,size-1))]=True
    return m

def summarize(name, score, mask):
    score=np.asarray(score,float); target=score[mask]; bg=score[~mask]
    return {'candidate':name,'auc':auc(score,mask),'target_mean':float(target.mean()),
            'background_mean':float(bg.mean()),'target_bg_margin':float(target.mean()-bg.mean()),
            'target_bg_ratio':float(abs(target).mean()/(abs(bg).mean()+1e-8)),
            'target_max_bg_max_margin':float(target.max()-bg.max())}

def candidate_maps(rgb, tir, trgb, ttir):
    rgbn=F.normalize(rgb,dim=1); tirn=F.normalize(tir,dim=1)
    pr=F.normalize(trgb,dim=1)[:,:,None,None]; pt=F.normalize(ttir,dim=1)[:,:,None,None]
    sr=(rgbn*pr).sum(1)[0]; st=(tirn*pt).sum(1)[0]
    mean_sim=0.5*(sr+st); max_sim=torch.maximum(sr,st)
    agree=(rgbn*tirn).sum(1)[0]; disagreement=1-agree
    # Minimal, parameter-free proxies. They expose the spatial signal each
    # paper family would consume; they do not reproduce the full networks.
    avg3=F.avg_pool2d(mean_sim[None,None],3,1,1)[0,0]
    avg5=F.avg_pool2d(mean_sim[None,None],5,1,2)[0,0]
    local_detail=(mean_sim-avg3).abs()
    return {
        'cmx': mean_sim-0.5*disagreement,
        'film': mean_sim,
        'mmtm': torch.full_like(mean_sim, 0.5*(rgbn.mean()+tirn.mean())),
        'seafusion': max_sim + 0.25*local_detail,
        'cddfuse': mean_sim*torch.clamp(agree,min=0),
        'didfuse': mean_sim + 0.25*local_detail,
        'swinfusion': 0.5*(avg3+avg5),
        'gmu': max_sim,
        'cbam': torch.sigmoid(rgb.abs().mean(1)[0]+tir.abs().mean(1)[0]),
        'sknet': torch.maximum(avg3,avg5),
        'dynamic_conv': avg3 + 0.5*(avg3-avg5).abs(),
        'tokenfusion': max_sim,
        'focal_modulation': 0.5*(avg3+avg5)+0.25*local_detail,
        'crossvit': torch.logsumexp(torch.stack((sr,st),0),0),
    }

def main():
    os.environ['TBSI_CHECKPOINT_OVERRIDE']=str(CKPT); os.environ['TBSI_INFER_FP16']='0'
    manifest=json.loads(MANIFEST.read_text())['selection']
    names=sorted({e['sequence'] for e in manifest}); out=OUT; out.mkdir(parents=True,exist_ok=True)
    lp=out/'sequence_list.txt'; lp.write_text('\n'.join(names)+'\n')
    seqmap={s.name:s for s in LasHeRDataset('testingset',str(lp)).get_sequence_list()}
    p=parameters(CFG); p.debug=0; p.save_all_boxes=False; p.checkpoint=str(CKPT)
    tracker=TBSITrack(p,'m1_candidate_screen')
    captured={}
    def bb(_m,_i,o):
        if isinstance(o, dict):
            o = o.get('feat', o.get('backbone_feat', next(iter(o.values()))))
            if isinstance(o, dict):
                o = o.get('feat', o.get('tokens', o.get('x', next(iter(o.values())))))
        if isinstance(o,(list,tuple)):
            o = o[0] if torch.is_tensor(o[0]) else o[-1]
        captured['backbone']=o
    def fi(_m,i): captured['fuse_in']=i[0].detach().float().cpu()
    hs=[tracker.network.backbone.register_forward_hook(bb),tracker.network.tbsi_fuse_search.register_forward_pre_hook(fi)]
    rows=[]
    for idx,event in enumerate(manifest,1):
        first=read_pair(event['sequence'],0,seqmap)
        tracker.initialize(first,{'init_bbox':np.asarray(seqmap[event['sequence']].ground_truth_rect[0]),'sequence_name':event['sequence']})
        tracker.state=list(map(float,event['tracker_state_before']))
        captured.clear(); tracker.track(read_pair(event['sequence'],int(event['frame']),seqmap),{'gt_bbox':event['gt_box'],'sequence_name':event['sequence'],'frame_num':event['frame']})
        cat=captured['backbone'].detach().float()
        rgb=cat[:,64:320,:].transpose(1,2).reshape(1,-1,16,16); tir=cat[:,-256:,:].transpose(1,2).reshape(1,-1,16,16)
        trgb=cat[:,:64,:].mean(1); ttir=cat[:,320:384,:].mean(1)
        rgbn=F.normalize(rgb,dim=1); tirn=F.normalize(tir,dim=1); pr=F.normalize(trgb,dim=1)[:,:,None,None]; pt=F.normalize(ttir,dim=1)[:,:,None,None]
        mask=event_mask(event)
        for name,score in candidate_maps(rgb,tir,trgb,ttir).items():
            row={'sequence':event['sequence'],'frame':event['frame'],'role':event['role'],'m1_auc':event['m1_auc']}
            row.update(summarize(name,score.detach().cpu().numpy(),mask)); rows.append(row)
        if idx%5==0: print('[{}/{}] {}'.format(idx,len(manifest),event['sequence']),flush=True)
    for h in hs:h.remove()
    with (out/'per_event_candidate.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    candidate_names=['cmx','film','mmtm','seafusion','cddfuse','didfuse','swinfusion','gmu','cbam','sknet','dynamic_conv','tokenfusion','focal_modulation','crossvit']
    for name in candidate_names:
        rr=[r for r in rows if r['candidate']==name]
        for role in ('failure','control','all'):
            q=rr if role=='all' else [r for r in rr if r['role']==role]
            summary.append({'candidate':name,'role':role,'events':len(q),
                            'mean_auc':float(np.mean([float(r['auc']) for r in q])),
                            'mean_margin':float(np.mean([float(r['target_bg_margin']) for r in q])),
                            'mean_target_bg_ratio':float(np.mean([float(r['target_bg_ratio']) for r in q])),
                            'auc_above_chance':sum(float(r['auc'])>0.5 for r in q)})
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (out/'protocol.json').write_text(json.dumps({'scope':'training-free M1 fixed crop/state/template; 20 failures + 5 controls','candidate_definitions':{'cmx':'mean template similarity minus cross-modal disagreement','film':'mean template similarity','mmtm':'constant channel-statistics proxy','seafusion':'winner similarity plus local detail','cddfuse':'similarity times cross-modal agreement','didfuse':'common similarity plus local detail','swinfusion':'3x3/5x5 context average','gmu':'per-location modality winner','cbam':'channel-mean spatial attention','sknet':'multi-scale max','dynamic_conv':'adaptive local-scale proxy','tokenfusion':'token importance as modality winner','focal_modulation':'multi-scale focal context','crossvit':'log-sum-exp cross-modal token score'},'warning':'signal screen only; no candidate module was trained or injected into tbsi_fuse_search'},indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
