#!/usr/bin/env python3
"""Trained-checkpoint interventions at locked frame/crop states. Never trains."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['TBSI_INFER_FP16']='0'
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from run_dual_message_p1 import config,setup,environment,sha
from analyze_receiver_residual_boxes_attributes_20261115 import OUT,RUN,DATA,frames,box_stats,write_csv,write_json
from lib.test.tracker.tbsi_track import TBSITrack
from lib.models.layers.receiver_conditioned_message_residual import ReceiverConditionedMessageResidual

class Probe:
    def __init__(self,tracker):
        self.tracker=tracker;self.mode='native';self.rows=[];self.context={};self.head=None
        self.mods=[(n,m) for n,m in tracker.network.named_modules() if isinstance(m,ReceiverConditionedMessageResidual)]
        assert len(self.mods)==6,len(self.mods)
        self.handles=[m.register_forward_hook(self.hook(n)) for n,m in self.mods]
        self.handles.append(tracker.network.box_head.register_forward_hook(self.save_head))

    def save_head(self,module,args,out):
        self.head=out[0].detach().float().cpu().numpy().reshape(-1)

    def hook(self,name):
        def run(mod,args,out):
            q,raw=args;b,h,n,dh=raw.shape
            msg=raw.transpose(1,2).reshape(b,n,h*dh).float()
            delta=out.transpose(1,2).reshape_as(msg).float()-msg
            scale=msg.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
            bias=mod.max_delta*mod.readout.bias.tanh().view(1,1,-1)*scale
            if self.mode=='off':return raw
            if self.mode=='bias_only':return (msg+bias).to(raw.dtype).reshape(b,n,h,dh).transpose(1,2)
            if self.mode=='receiver_roll':
                u=mod.receiver_proj(mod.receiver_norm(q.float().roll(13,dims=1)))
                v=mod.message_proj(mod.message_norm(msg))
                edit=mod.max_delta*torch.tanh(mod.readout(F.gelu(u*v)))*scale
                return (msg+edit).to(raw.dtype).reshape(b,n,h,dh).transpose(1,2)
            if self.mode=='native':
                foreground=self.tracker.c1_template_mask.to(msg.device)
                # c1 mask is only for diagnostics, never fed into the operator.
                energy=delta.square().sum().clamp_min(1e-30)
                radial=(delta*msg).sum(-1,keepdim=True)/msg.square().sum(-1,keepdim=True).clamp_min(1e-30)*msg
                spatial_common=delta.mean(1,keepdim=True).expand_as(delta)
                latent=F.gelu(mod.receiver_proj(mod.receiver_norm(q.float()))*mod.message_proj(mod.message_norm(msg)))
                readout=mod.readout(latent)
                rel=delta.norm(dim=-1)/msg.norm(dim=-1).clamp_min(1e-12)
                self.rows.append(dict(self.context,module=name,relative_edit=float(delta.norm()/msg.norm().clamp_min(1e-12)),
                    radial_energy_fraction=float(radial.square().sum()/energy),spatial_common_energy_fraction=float(spatial_common.square().sum()/energy),
                    bias_to_full_delta_norm=float(bias.norm()/delta.norm().clamp_min(1e-12)),
                    no_bias_difference_fraction=float((mod.max_delta*(readout-mod.readout.bias).tanh()*scale-delta).norm()/delta.norm().clamp_min(1e-12)),
                    fg_relative_edit=float(rel[:,foreground].mean()) if foreground.any() else 0.,
                    bg_relative_edit=float(rel[:,~foreground].mean()) if (~foreground).any() else 0.,
                    saturation_abs_pre_tanh_gt2=float((readout.abs()>2).float().mean())))
            return out
        return run

def main():
    setup(42);started=time.time();prefix='--prefix' in sys.argv
    dest=OUT/('fixed_crop_prefix' if prefix else 'fixed_crop');dest.mkdir(exist_ok=True)
    shutil.copy2(__file__,dest/Path(__file__).name)
    assert not (dest/'summary.json').exists(),'Refuse overwriting complete diagnosis'
    events=json.loads((OUT/'events.json').read_text());selected=[]
    names=list(dict.fromkeys(e['sequence'] for e in events))
    for name in names:
        ee=[e for e in events if e['sequence']==name]
        if prefix:
            preds=[np.loadtxt(RUN/a/(name+'.txt'),ndmin=2) for a in ['eval_control','eval_candidate']]
            g=np.loadtxt(DATA/'testingset'/name/'init.txt',delimiter=',',ndmin=2)
            valid=(g[:,2:]>0).all(1)
            p0,p1=preds;centers=[p[:,:2]+p[:,2:]/2 for p in preds]
            center_gap=np.linalg.norm((centers[1]-centers[0])/np.maximum((p0[:,2:]+p1[:,2:])/2,1e-6),axis=1)
            size_gap=np.max(abs(np.log(np.maximum(p1[:,2:],1e-6)/np.maximum(p0[:,2:],1e-6))),axis=1)
            separated=((center_gap>.1)|(size_gap>.1))&valid
            starts=np.flatnonzero(np.convolve(separated.astype(int),np.ones(3,dtype=int),mode='valid')==3)
            f=max(1,int(starts[0])-1) if len(starts) else max(1,int(len(g)*.25))
            while not valid[f]:f+=1
            selected.append(dict(sequence=name,frame=f,kind='before_first_3f_box_disagreement' if len(starts) else 'no_disagreement_uniform25',crop_state=p0[f-1].tolist()))
            continue
        chosen=[next((e for e in ee if e['kind']=='first_gap'),next(e for e in ee if e['kind']=='peak_15f'))]
        chosen+= [next(e for e in ee if e['frame']!=chosen[0]['frame'])]
        selected+=chosen
    write_json(dest/'events_locked.json',selected)
    cfgpath=ROOT/'experiments/tbsi_track/MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-candidate.yaml'
    cfg=config(cfgpath)
    assert not cfg.MODEL.TEMPORAL_TOKENS
    params=SimpleNamespace(cfg=cfg,checkpoint=str(RUN/'candidate_final/checkpoint_u0468.pth.tar'),
        template_factor=cfg.TEST.TEMPLATE_FACTOR,template_size=cfg.TEST.TEMPLATE_SIZE,
        search_factor=cfg.TEST.SEARCH_FACTOR,search_size=cfg.TEST.SEARCH_SIZE,debug=0,save_all_boxes=False)
    tracker=TBSITrack(params,'lasher_test');probe=Probe(tracker)
    initial={n:t.detach().cpu().clone() for n,t in tracker.network.state_dict().items()}
    ckpt=torch.load(str(RUN/'control_final3/checkpoint_u0468.pth.tar'),map_location='cpu')['net']
    rows=[];states={};parities=[]
    for arm,state in [('candidate',initial),('control',ckpt)]:
        miss,unexpected=tracker.network.load_state_dict(state,strict=False)
        if arm=='control':assert not unexpected and all('receiver_message_residual' in n for n in miss)
        else:assert not miss and not unexpected
        for idx,name in enumerate(names):
            pp=frames(name);gt=np.loadtxt(DATA/'testingset'/name/'init.txt',delimiter=',',ndmin=2)
            def image(frame):
                a=[cv2.imread(str(p[frame])) for p in pp];assert all(x is not None for x in a)
                return np.concatenate([cv2.cvtColor(x,cv2.COLOR_BGR2RGB) for x in a],axis=2)
            tracker.initialize(image(0),dict(init_bbox=gt[0].tolist()))
            for e in [e for e in selected if e['sequence']==name]:
                f=e['frame'];im=image(f)
                modes=['native','off','bias_only','receiver_roll'] if arm=='candidate' else ['off']
                for mode in modes:
                    probe.mode=mode;probe.context=dict(sequence=name,frame=f,kind=e['kind'])
                    tracker.state=list(e['crop_state']);tracker.frame_id=f-1
                    result=tracker.track(im);box=np.asarray(result['target_bbox']);stats=box_stats(box[None],gt[f:f+1])
                    head=probe.head;assert head is not None
                    peak=int(np.argmax(head*tracker.output_window.cpu().numpy().reshape(-1)))
                    rows.append(dict(sequence=name,frame=f,kind=e['kind'],arm=arm,mode=mode,iou=float(stats['iou'][0]),center_norm=float(stats['cn'][0]),hann_peak=peak,
                        box=json.dumps(box.tolist()),crop_state=json.dumps(e['crop_state']),head=json.dumps(head.tolist())))
                    if arm=='control':
                        saved=np.loadtxt(RUN/'eval_control'/(name+'.txt'),ndmin=2)[f]
                        parities.append(float(np.max(abs(box-saved))))
            if (idx+1)%12==0:print(json.dumps(dict(arm=arm,done=idx+1,total=len(names),seconds=time.time()-started)),flush=True)
    # No parameter or running-stat mutation may occur after inference; compare to loaded control.
    state_unchanged=all(torch.equal(t.cpu(),ckpt[n]) for n,t in tracker.network.state_dict().items() if n in ckpt)
    assert state_unchanged
    write_csv(dest/'frame_interventions.csv',rows);write_csv(dest/'operator_readouts.csv',probe.rows)
    result=dict(environment=environment(),frames=len(selected),sequences=len(names),modules=len(probe.mods),optimizer_steps=0,
        scope='Same trained candidate off/bias-only/receiver-roll, fixed control-previous-state crop; joint-training control has different learned weights; no counterfactual retraining.',
        state_unchanged=state_unchanged,control_replay_max_abs_box_error=max(parities),control_replay_median_abs_error=float(np.median(parities)),
        checkpoints={a:sha(RUN/b/'checkpoint_u0468.pth.tar') for a,b in [('control','control_final3'),('candidate','candidate_final')]},
        intervention_summary={},operator_means={},wall_seconds=time.time()-started)
    natives={(r['sequence'],r['frame']):r for r in rows if r['arm']=='candidate' and r['mode']=='native'}
    for arm,mode in [('candidate','native'),('candidate','off'),('candidate','bias_only'),('candidate','receiver_roll'),('control','off')]:
        rr=[r for r in rows if r['arm']==arm and r['mode']==mode];delta=np.array([r['iou']-natives[(r['sequence'],r['frame'])]['iou'] for r in rr])
        grouped=[np.mean([r['iou']-natives[(r['sequence'],r['frame'])]['iou'] for r in rr if r['sequence']==s]) for s in names]
        result['intervention_summary'][arm+'_'+mode]=dict(mean_iou=float(np.mean([r['iou'] for r in rr])),delta_from_native_frame_mean=float(delta.mean()),
            delta_sequence_macro_mean=float(np.mean(grouped)),wins=int((delta>1e-6).sum()),losses=int((delta< -1e-6).sum()),ties=int((abs(delta)<=1e-6).sum()),
            hann_peak_changes=sum(r['hann_peak']!=natives[(r['sequence'],r['frame'])]['hann_peak'] for r in rr))
    for key in ['relative_edit','radial_energy_fraction','spatial_common_energy_fraction','bias_to_full_delta_norm','no_bias_difference_fraction','fg_relative_edit','bg_relative_edit','saturation_abs_pre_tanh_gt2']:
        v=[r[key] for r in probe.rows];result['operator_means'][key]=dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)))
    write_json(dest/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
