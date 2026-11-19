#!/usr/bin/env python3
"""Frozen learned-message direction versus target-loss gradient; never trains."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['TBSI_INFER_FP16']='0'
import csv
import hashlib
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from run_dual_message_p1 import setup,config,sha,state_hash,environment
from analyze_receiver_residual_boxes_attributes_20261115 import RUN,OUT,DATA,frames,box_stats,write_json,write_csv
from probe_receiver_residual_crop_perturbation_20261115 import geometry,response_stats
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target
from lib.models.layers.receiver_conditioned_message_residual import ReceiverConditionedMessageResidual
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.focal_loss import FocalLoss
from lib.utils.box_ops import box_cxcywh_to_xyxy,box_xywh_to_xyxy,giou_loss,clip_box

DEST=RUN/'analysis/task_direction_20261115'
PREVIOUS=RUN/'analysis/crop_perturbation_20261115'
SCALES=[0.,1.,4.,16.,64.,-1.,-4.,-16.,-64.]

class Injection:
    def __init__(self,network):
        self.mods=[(n,m) for n,m in network.named_modules() if isinstance(m,ReceiverConditionedMessageResidual)]
        assert len(self.mods)==6
        self.mode='native';self.scale=1.;self.reference={};self.vectors={};self.leaves={};self.message_norms={};self.feature=None
        self.handles=[m.register_forward_hook(self.hook(n)) for n,m in self.mods]
        self.handles.append(network.tbsi_fuse_search.register_forward_hook(self.save_feature))
    def save_feature(self,m,args,out):self.feature=out.detach()
    def hook(self,name):
        def run(mod,args,out):
            raw=args[1]
            if self.mode=='native':
                self.reference[name]=(out-raw).detach().clone()
                self.message_norms[name]=float(raw.detach().norm())
                return out
            if self.mode=='gradient':
                leaf=torch.zeros_like(raw,requires_grad=True);self.leaves[name]=leaf
                return raw+leaf
            delta=self.reference[name] if self.mode=='learned' else self.vectors[name]
            return raw+self.scale*delta
        return run

def target_losses(out,target,cfg):
    # Actor-equivalent primary loss form, evaluated on exact integer evaluation crop geometry.
    pred=box_cxcywh_to_xyxy(out['pred_boxes'].double()).reshape(-1,4)
    truth=box_xywh_to_xyxy(target.double()).clamp(0,1)
    giou,_=giou_loss(pred,truth);l1=F.l1_loss(pred,truth)
    gaussian=generate_heatmap(target[None],256,16)[-1][:,None].double()
    assert int((gaussian==1).sum())==1,'GT center must map to valid score-grid positive'
    focal=FocalLoss()(out['score_map'].double(),gaussian)
    box=cfg.TRAIN.GIOU_WEIGHT*giou+cfg.TRAIN.L1_WEIGHT*l1
    total=box+focal
    assert all(torch.isfinite(t) for t in [total,box,focal])
    return dict(total=total,box=box,focal=focal,giou=giou,l1=l1)

def main():
    setup(42);start=time.time();DEST.mkdir(parents=True,exist_ok=False);(DEST/'source').mkdir()
    prior=list(csv.DictReader((PREVIOUS/'per_event.csv').open()))
    eligible={int(e['event_id']):e['target_local_eligible']=='True' for e in prior}
    manifest=[dict(e,target_local_eligible=eligible[e['event_id']]) for e in json.loads((PREVIOUS/'manifest_locked.json').read_text()) if 'primary_prefix' in e['roles']]
    assert len(manifest)==96 and sum(e['target_local_eligible'] for e in manifest)==91
    write_json(DEST/'manifest_locked.json',manifest)
    cfgpath=ROOT/'experiments/tbsi_track/MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-candidate.yaml'
    checkpoint=RUN/'candidate_final/checkpoint_u0468.pth.tar'
    with tarfile.open(DEST/'source/code_snapshot.tar.gz','w:gz') as tar:
        for folder in ['lib/models','lib/test/tracker','lib/test/utils','lib/train/data','lib/config','lib/utils']:
            for p in sorted((ROOT/folder).rglob('*.py')):tar.add(p,arcname=str(p.relative_to(ROOT)))
        for p in [Path(__file__).resolve(),cfgpath,ROOT/'tools/run_dual_message_p1.py',ROOT/'tools/analyze_receiver_residual_boxes_attributes_20261115.py',ROOT/'tools/probe_receiver_residual_crop_perturbation_20261115.py']:
            tar.add(p,arcname=str(p.relative_to(ROOT)))
    write_json(DEST/'CONTRACT.json',dict(question='Does the learned residual point toward lower target loss at a fixed crop, or is useful direction merely too weak downstream?',
        optimizer_steps=0,training=False,model_code_changed=False,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),config=str(cfgpath),config_sha256=sha(cfgpath),
        source_snapshot_sha256=sha(DEST/'source/code_snapshot.tar.gz'),manifest_sha256=sha(DEST/'manifest_locked.json'),data=str(DATA/'testingset'),
        selection='Verbatim96 previously locked primary events; previously defined91 target-covered events eligible for target loss/gradients. No selection by present output.',
        learned_path='Capture native delta at all6 modules once per precision/frame. Frozen additive raw + alpha*captured_delta. Alpha0 is off; alpha1 must reproduce native before attribution. QK may change downstream.',
        scales=SCALES,controls=['template-token roll13 with exact per-module learned delta norm at16/64','negative total-loss gradient with exact per-module learned delta norm at1/16/64; GT oracle only'],
        precision='Primary highest FP32 matmul, CUDA/cuDNN TF32 disabled. Separate historical high native/off/16 diagnostic. Loss reduction in float64 to reduce subtraction error; network FP32.',
        loss='2-style weights from cfg GIOU_WEIGHT and L1_WEIGHT, plus existing FocalLoss. Raw head pred_boxes use unwindowed peak as in training; additionally evaluate actual Hann-decoded original-image IoU. GT coordinates use exact integer crop, not a replay of training augmentation.',
        gradients='Freeze every parameter; add zero-valued leaf at each cross-message output. Compute total/box/focal gradients at residual-off; directional derivative dot(grad, captured native delta), cosine with negative gradient, per-module and summed. No optimizer or parameter .grad.',
        derivative_check='Compare total directional derivative to centered finite differences at +/-4 and +/-16 only where raw peak stays the same. Report mismatch; do not assume autograd alone proves utility.',
        practical_thresholds='IoU +/-0.005 =0.5pp; loss decreases reported at1e-4 and1e-3, sign with1e-7 descriptive; no amplitude selected as a winning method.',
        cautions=['Single trained candidate; differences from control due joint training not attributed here.','GT oracle is an upper-bound diagnostic and is neither a trained candidate nor a deployable direction.','Short frozen scale sweeps cannot establish full trajectory gains or a beneficial new loss.','Current dynamic Value-gate quick is not historical strongest MR g=1 reproduction.']))
    cfg=config(cfgpath);assert not cfg.MODEL.TEMPORAL_TOKENS
    tr=TBSITrack(SimpleNamespace(cfg=cfg,checkpoint=str(checkpoint),template_factor=2.,template_size=128,search_factor=4.,search_size=256,debug=0,save_all_boxes=False),'lasher_test')
    for p in tr.network.parameters():p.requires_grad_(False)
    before=state_hash(tr.network.state_dict());inj=Injection(tr.network)
    historical_cudnn_tf32=torch.backends.cudnn.allow_tf32
    rows=[];gradrows=[];checks=[];features=[];prior_rows=list(csv.DictReader((PREVIOUS/'frame_readouts.csv').open()))
    prior_native={int(r['event_id']):r for r in prior_rows if r['arm']=='candidate_native' and r['dx']==r['dy']=='0'}
    input_hashes=[];window=tr.output_window.detach().cpu().numpy().reshape(16,16)
    for idx,e in enumerate(manifest):
        paths=frames(e['sequence']);g=np.loadtxt(DATA/'testingset'/e['sequence']/'init.txt',delimiter=',',ndmin=2);f=e['frame'];gt=g[f]
        def image(k):return np.concatenate([cv2.cvtColor(cv2.imread(str(p[k])),cv2.COLOR_BGR2RGB) for p in paths],axis=2)
        tr.initialize(image(0),dict(init_bbox=g[0].tolist()));im=image(f);state=e['crop_state'];tl,side=geometry(state)
        patch,resize,mask=sample_target(im,state,4.,output_sz=256)
        patchhash=hashlib.sha256(patch.tobytes()).hexdigest();assert patchhash==prior_native[e['event_id']]['patch_sha256']
        tensor=tr.preprocessor.process(patch,mask,half=False).tensors
        target=torch.tensor(np.r_[(gt[:2]-tl)/side,gt[2:]/side],device='cuda',dtype=torch.float32)[None]
        kwargs=dict(template=[tr.z_dict1.tensors[:,:3],tr.z_dict1.tensors[:,3:]],search=[tensor[:,:3],tensor[:,3:]],
            prev_search=None,prev_tokens=None,ce_template_mask=tr.box_mask_z,
            template_boxes=None if tr.template_boxes is None else torch.stack((tr.template_boxes,tr.template_boxes),1))
        context=dict(event_id=e['event_id'],sequence=e['sequence'],frame=f+1,target_local_eligible=e['target_local_eligible'])
        input_hashes.append(dict(context,patch_sha256=patchhash,template_tensor_sha256=hashlib.sha256(tr.z_dict1.tensors.cpu().numpy().tobytes()).hexdigest()))
        def forward(precision,mode,scale=1.):
            inj.mode=mode;inj.scale=scale
            with torch.no_grad():out=tr.network(**kwargs)
            response=out['score_map']*tr.output_window
            decoded=tr.network.box_head.cal_bbox(response,out['size_map'],out['offset_map']).view(-1,4)
            pb=decoded.mean(0)*side
            tr.state=list(state)
            box=np.array(clip_box(tr.map_box_back(pb,resize),*im.shape[:2],margin=10))
            stats=box_stats(box[None],gt[None]);score=out['score_map'].detach().cpu().numpy().reshape(16,16)
            losses=target_losses(out,target,cfg) if e['target_local_eligible'] else {}
            row=dict(context,precision=precision,mode=mode,scale=scale,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,iou=float(stats['iou'][0]),center_norm=float(stats['cn'][0]),
                box=json.dumps(box.tolist()),**{k+'_loss':float(losses[k]) if k in losses else None for k in ['total','box','focal','giou','l1']},
                **response_stats(score,window,gt,tl,side))
            rows.append(row)
            return out,row,inj.feature.clone()
        # Separate precision diagnostic; preserve original setup semantics before comparison.
        torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=historical_cudnn_tf32
        high_native,hr,_=forward('high','native')
        error=float(np.max(abs(np.array(json.loads(hr['box']))-np.array(json.loads(prior_native[e['event_id']]['box'])))))
        checks.append(dict(context,check='historical_high_native_box_max_error',value=error))
        for scale in [0.,16.]:forward('high','learned',scale)
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        out,nr,nfeat=forward('highest','native')
        native_boxes=out['pred_boxes'].detach().clone();native_scores=out['score_map'].detach().clone()
        offout,offrow,offfeat=forward('highest','learned',0.)
        for scale in SCALES[1:]:
            fo,fr,feat=forward('highest','learned',scale)
            features.append(dict(context,mode='learned',scale=scale,search_feature_relative_change=float((feat-offfeat).norm()/offfeat.norm()),
                score_l2_change=float((fo['score_map']-offout['score_map']).norm()),
                score_max_abs_change=float((fo['score_map']-offout['score_map']).abs().max())))
            if scale==1.:
                checks.append(dict(context,check='frozen1_native_score_max_error',value=float((fo['score_map']-native_scores).abs().max())))
                checks.append(dict(context,check='frozen1_native_pred_box_max_error',value=float((fo['pred_boxes']-native_boxes).abs().max())))
        # Norm-matched token-axis control. Axis2 is template token, not heads/channel.
        inj.vectors={n:d.roll(13,dims=2) for n,d in inj.reference.items()}
        for scale in [16.,64.]:forward('highest','token_roll',scale)
        if e['target_local_eligible']:
            inj.mode='gradient';inj.leaves={}
            with torch.enable_grad():
                gradout=tr.network(**kwargs);losses=target_losses(gradout,target,cfg)
                names=[n for n,_ in inj.mods];leaves=[inj.leaves[n] for n in names]
                grads={}
                for loss in ['box','focal','total']:
                    gg=torch.autograd.grad(losses[loss],leaves,retain_graph=(loss!='total'),allow_unused=False)
                    grads[loss]={n:t.detach() for n,t in zip(names,gg)}
            checks.append(dict(context,check='gradient0_off_score_max_error',value=float((gradout['score_map'].detach()-offout['score_map']).abs().max())))
            for loss in ['total','box','focal']:
                for name in names:
                    d=inj.reference[name].double();gg=grads[loss][name].double();dn=float(d.norm());gn=float(gg.norm())
                    gradrows.append(dict(context,loss=loss,module=name,delta_norm=dn,gradient_norm=gn,message_norm=inj.message_norms[name],
                        delta_over_message=dn/max(inj.message_norms[name],1e-30),directional_derivative=float((d*gg).sum()),
                        negative_gradient_cosine=float(-(d*gg).sum()/max(dn*gn,1e-30)),
                        matched_oracle_derivative=-dn*gn))
            inj.vectors={n:-grads['total'][n]/grads['total'][n].norm().clamp_min(1e-30)*inj.reference[n].norm() for n in names}
            for scale in [1.,16.,64.]:forward('highest','gt_oracle',scale)
            for n in names:
                checks.append(dict(context,check='oracle_norm_relative_error_'+n,value=float(abs(inj.vectors[n].norm()/inj.reference[n].norm()-1))))
            del grads,gradout,losses,leaves,gg
        inj.reference={};inj.vectors={};inj.leaves={}
        del high_native,out,fo,offout
        if (idx+1)%12==0:print(json.dumps(dict(done=idx+1,total=96,forwards=len(rows),seconds=time.time()-start)),flush=True)
    assert before==state_hash(tr.network.state_dict())
    assert all(p.grad is None for p in tr.network.parameters())
    write_csv(DEST/'frame_readouts.csv',rows);write_csv(DEST/'gradients.csv',gradrows);write_csv(DEST/'checks.csv',checks);write_csv(DEST/'transfer_readouts.csv',features)
    write_json(DEST/'input_hashes.json',input_hashes)
    write_json(DEST/'execution_record.json',dict(environment=environment(),events=96,loss_gradient_events=91,inference_forwards=len(rows),gradient_forwards=91,
        optimizer_steps=0,state_unchanged=True,parameters_frozen=True,parameter_grads_none=True,all_patches_match_previous=True,
        wall_seconds=time.time()-start,gpu_hours_exclusive=(time.time()-start)/3600,working_directory=str(ROOT),
        command='python tools/probe_receiver_residual_task_direction_20261115.py'))
    print(json.dumps(dict(completed=True,seconds=time.time()-start,forwards=len(rows)),indent=2),flush=True)

if __name__=='__main__':main()
