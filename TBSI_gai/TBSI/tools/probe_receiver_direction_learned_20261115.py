#!/usr/bin/env python3
"""After-training fixed-crop direction comparison; no optimizer or GT model input."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['TBSI_INFER_FP16']='0'
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from run_dual_message_p1 import setup,config,state_hash,environment,sha
from run_receiver_direction_quick_20261115 import RUN,PARENT,CFG
from probe_receiver_residual_task_direction_20261115 import Injection,target_losses
from probe_receiver_residual_crop_perturbation_20261115 import geometry,response_stats
from analyze_receiver_residual_boxes_attributes_20261115 import DATA,frames,box_stats,write_csv,write_json
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.box_ops import clip_box

def raw_margin(out,target):
    boxes=target.detach();b,_,h,w=out['score_map'].shape
    positive=generate_heatmap(boxes[None],h*16,16)[-1].to(boxes.device).eq(1)
    yy,xx=torch.meshgrid((torch.arange(h,device=boxes.device)+.5)/h,(torch.arange(w,device=boxes.device)+.5)/w,indexing='ij')
    x,y,bw,bh=boxes.unbind(-1)
    dilated=((xx[None]>=x[:,None,None]-1/w)&(xx[None]<=x[:,None,None]+bw[:,None,None]+1/w)&(yy[None]>=y[:,None,None]-1/h)&(yy[None]<=y[:,None,None]+bh[:,None,None]+1/h))
    negative=~dilated & ~positive
    assert positive.any() and negative.any()
    logscore=out['score_map'][:,0].double().clamp_min(1e-6).log()
    return (logscore.masked_fill(~positive,-torch.inf).flatten(1).amax(-1)-logscore.masked_fill(~negative,-torch.inf).flatten(1).amax(-1)).mean()

def main():
    setup(42);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.time();dest=RUN/'mechanism';dest.mkdir(exist_ok=False)
    manifest=json.loads((PARENT/'analysis/task_direction_20261115/manifest_locked.json').read_text())
    write_json(dest/'manifest_locked.json',manifest)
    cfg=config(CFG)
    checkpoints={'parent':PARENT/'candidate_final/checkpoint_u0468.pth.tar','candidate':RUN/'candidate_retry1/checkpoint_u0468.pth.tar'}
    tr=TBSITrack(SimpleNamespace(cfg=cfg,checkpoint=str(checkpoints['candidate']),template_factor=2.,template_size=128,search_factor=4.,search_size=256,debug=0,save_all_boxes=False),'lasher_test')
    for p in tr.network.parameters():p.requires_grad_(False)
    inj=Injection(tr.network);rows=[];gradrows=[];checks={};window=tr.output_window.cpu().numpy().reshape(16,16)
    for arm,checkpoint in checkpoints.items():
        state=torch.load(str(checkpoint),map_location='cpu')['net'];tr.network.load_state_dict(state,strict=True);before=state_hash(tr.network.state_dict());del state
        for idx,e in enumerate(manifest):
            paths=frames(e['sequence']);g=np.loadtxt(DATA/'testingset'/e['sequence']/'init.txt',delimiter=',',ndmin=2);f=e['frame'];gt=g[f]
            def image(k):return np.concatenate([cv2.cvtColor(cv2.imread(str(p[k])),cv2.COLOR_BGR2RGB) for p in paths],axis=2)
            tr.initialize(image(0),dict(init_bbox=g[0].tolist()));im=image(f);tl,side=geometry(e['crop_state'])
            patch,resize,mask=sample_target(im,e['crop_state'],4.,output_sz=256);search=tr.preprocessor.process(patch,mask,half=False).tensors
            target=torch.tensor(np.r_[(gt[:2]-tl)/side,gt[2:]/side],device='cuda',dtype=torch.float32)[None]
            kw=dict(template=[tr.z_dict1.tensors[:,:3],tr.z_dict1.tensors[:,3:]],search=[search[:,:3],search[:,3:]],prev_search=None,prev_tokens=None,ce_template_mask=tr.box_mask_z,
                template_boxes=None if tr.template_boxes is None else torch.stack((tr.template_boxes,tr.template_boxes),1))
            context=dict(arm=arm,event_id=e['event_id'],sequence=e['sequence'],frame=f+1,eligible=e['target_local_eligible'])
            for mode in ['native','learned']:
                inj.mode=mode;inj.scale=0.
                with torch.no_grad():out=tr.network(**kw)
                score=out['score_map'].cpu().numpy().reshape(16,16)
                decoded=tr.network.box_head.cal_bbox(out['score_map']*tr.output_window,out['size_map'],out['offset_map']).view(-1,4).mean(0)*side
                tr.state=list(e['crop_state']);box=np.array(clip_box(tr.map_box_back(decoded,resize),*im.shape[:2],margin=10))
                losses=target_losses(out,target,cfg) if e['target_local_eligible'] else {}
                rows.append(dict(context,mode='native' if mode=='native' else 'off',iou=float(box_stats(box[None],gt[None])['iou'][0]),
                    task_loss=float(losses['total']) if losses else None,raw_margin_training_definition=float(raw_margin(out,target)) if losses else None,
                    box=json.dumps(box.tolist()),**response_stats(score,window,gt,tl,side)))
            if e['target_local_eligible']:
                inj.mode='gradient';inj.leaves={}
                with torch.enable_grad():
                    out=tr.network(**kw);losses=target_losses(out,target,cfg);q=raw_margin(out,target)
                    names=[n for n,m in inj.mods];leaves=[inj.leaves[n] for n in names]
                    for objective,loss in [('task',losses['total']),('margin',-q)]:
                        grads=torch.autograd.grad(loss,leaves,retain_graph=objective=='task')
                        derivative=0.;oracle=0.
                        for name,g in zip(names,grads):
                            delta=inj.reference[name].double();gg=g.double();dd=float((delta*gg).sum());dn=float(delta.norm());gn=float(gg.norm());derivative+=dd;oracle-=dn*gn
                            gradrows.append(dict(context,objective=objective,module=name,derivative=dd,negative_gradient_cosine=-dd/max(dn*gn,1e-30),relative_edit=dn/max(inj.message_norms[name],1e-30)))
                        rows[-1][objective+'_directional_derivative']=derivative;rows[-1][objective+'_alignment_efficiency']=-derivative/max(-oracle,1e-30)
                del grads,out,losses,leaves
            else:
                for objective in ['task','margin']:
                    rows[-1][objective+'_directional_derivative']=None;rows[-1][objective+'_alignment_efficiency']=None
            # Native row carries NA derivative columns too, keeping rectangular CSV.
            for objective in ['task','margin']:
                rows[-2][objective+'_directional_derivative']=None;rows[-2][objective+'_alignment_efficiency']=None
            inj.reference={};inj.leaves={}
            if (idx+1)%24==0:print(json.dumps(dict(arm=arm,events=idx+1,seconds=time.time()-started)),flush=True)
        checks[arm]=before==state_hash(tr.network.state_dict());assert checks[arm]
    # Normalize column order to the first row explicitly (off rows were assigned columns first).
    keys=list(rows[0]);rows=[{k:r.get(k) for k in keys} for r in rows]
    write_csv(dest/'frame_readouts.csv',rows);write_csv(dest/'gradients.csv',gradrows)
    summary={}
    for arm in checkpoints:
        rr=[r for r in rows if r['arm']==arm and r['mode']=='off' and r['eligible']]
        native={(r['event_id']):r for r in rows if r['arm']==arm and r['mode']=='native'}
        ss=dict(events=len(rr),direct_hann_peak_changes=sum(native[r['event_id']]['hann_peak']!=r['hann_peak'] for r in rr),
            mean_direct_iou_delta=float(np.mean([native[r['event_id']]['iou']-r['iou'] for r in rr])))
        for obj in ['task','margin']:
            ds=np.array([r[obj+'_directional_derivative'] for r in rr]);ss[obj]=dict(favorable=int((ds< -1e-7).sum()),unfavorable=int((ds>1e-7).sum()),near_zero=int((abs(ds)<=1e-7).sum()),
                alignment_efficiency_mean=float(np.mean([r[obj+'_alignment_efficiency'] for r in rr])))
        summary[arm]=ss
    write_json(dest/'summary.json',dict(arms=summary,state_unchanged=checks,optimizer_steps=0,seconds=time.time()-started,environment=environment(),checkpoint_sha256={k:sha(v) for k,v in checkpoints.items()}))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
