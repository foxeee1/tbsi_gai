#!/usr/bin/env python3
"""Frozen-relation source-content interventions on historical full MR."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['TBSI_INFER_FP16']='0'
import json
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import audit_mr_local_transmission_20261115 as base

RUN=base.RUN/'source_content_full'
FULL=ROOT/'output/experiments/MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109'


class SourceAudit(base.Audit):
    def __init__(self,net):
        super().__init__(net)
        self.source_selection=None
        self.control='native'
        self.relations={}
        self.contract_errors=[]
        self.allocator_rows=[]
        self.raw_scores={}
        self.context={}
        for layer,host in enumerate(self.hosts):
            for direction in ('rgb','tir'):
                def score(module,args,out,layer=layer,direction=direction):
                    self.raw_scores[(layer,direction)]=.25*torch.tanh(out.squeeze(-1).detach())
                self.handles.append(getattr(host,'mass_reallocator_'+direction).readout.register_forward_hook(score))
                def relation(module,args,out,layer=layer,direction=direction):
                    if self.control=='native' and self.source_selection is None:
                        original=args[0];v=args[2]
                        score=self.raw_scores[(layer,direction)]
                        centered=score-score.mean(-1,keepdim=True)
                        cond0=original/original.sum(-1,keepdim=True).clamp_min(1e-20)
                        cond1=out/out.sum(-1,keepdim=True).clamp_min(1e-20)
                        message0=original@v;message1=out@v
                        dv=v-v.mean(-2,keepdim=True)
                        self.allocator_rows.append(dict(self.context,layer=layer,receiver=direction,
                            raw_score_abs_mean=float(score.abs().mean()),raw_score_rms=float(score.square().mean().sqrt()),
                            source_centered_score_rms=float(centered.square().mean().sqrt()),
                            centered_score_energy_fraction=float(centered.square().sum()/score.square().sum().clamp_min(1e-20)),
                            conditional_relation_l1=float((cond1-cond0).abs().sum(-1).mean()),
                            message_relative_edit=float((message1-message0).norm()/message0.norm().clamp_min(1e-20)),
                            source_global_detail_energy_fraction=float(dv.square().sum()/v.square().sum().clamp_min(1e-20)),
                            conserved_mass_max_error=float((out.sum(-1)-original.sum(-1)).abs().max()),
                            zero_sum_identity_error=float((((out-original)@v)-((out-original)@dv)).abs().max())))
                    if self.control=='mr_off':
                        out=args[0]
                    self.relations[(layer,direction)]=out.detach().clone()
                    return out
                self.handles.append(getattr(host,'mass_reallocator_'+direction).register_forward_hook(relation))
            def edit(module,args,out,layer=layer):
                if self.control=='cross_off':
                    return tuple(torch.zeros_like(x) for x in out)
                result=list(out)
                for idx,direction in enumerate(('rgb','tir')):
                    v=self.values[layer][:,:,64:] if direction=='rgb' else self.values[layer][:,:,:64]
                    relation=self.relations[(layer,direction)]
                    error=float((relation@v-out[idx]).abs().max())
                    assert error<3e-6,error
                    self.contract_errors.append(error)
                    if self.source_selection is None or self.source_selection[:2]!=(layer,direction):
                        continue
                    region,action=self.source_selection[2:]
                    mask=self.region_masks[direction][region]
                    if action=='drop_contrast':
                        delta=torch.zeros_like(v)
                        delta[:,:,mask,:]=v.mean(-2,keepdim=True)-v[:,:,mask,:].mean(-2,keepdim=True)
                    else:
                        delta=base.action_delta(v,mask,'drop_detail' if action=='drop_detail' else 'shuffle_detail')
                    self.last_source_delta=delta
                    self.last_message_delta=relation@delta
                    result[idx]=out[idx]+self.last_message_delta
                return tuple(result)
            self.handles.append(host.cbmr.register_forward_hook(edit))


def main():
    global RUN
    p=argparse.ArgumentParser();p.add_argument('--controls-only',action='store_true');args=p.parse_args()
    if args.controls_only:RUN=RUN.parent/'global_controls_full'
    if (RUN/'completed.json').exists():raise RuntimeError('Refusing overwrite')
    RUN.mkdir(parents=True,exist_ok=True)
    base.setup(42)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=base.config(FULL/'contracts/mr_only_g1.yaml')
    cfg.MODEL.BACKBONE.CROSS_MASS_PRESERVING_RELATION=False
    cfg.MODEL.BACKBONE.MASS_REDIST_ENABLE=True
    cfg.MODEL.BACKBONE.CBMR_ENABLE=True;cfg.MODEL.BACKBONE.CBMR_TEMPLATE_BOXES=True
    tracker=base.TBSITrack(SimpleNamespace(cfg=cfg,checkpoint=str(FULL/'train/checkpoint_u7020.pth.tar'),
        template_factor=2.,template_size=128,search_factor=4.,search_size=256,debug=0,save_all_boxes=False),'lasher_test')
    for p in tracker.network.parameters():p.requires_grad_(False)
    audit=SourceAudit(tracker.network)
    state=base.state_hash(tracker.network.state_dict())
    reference=json.loads((base.RUN/'full_reference.json').read_text())['rows']
    manifest=json.loads((base.RUN/'frame_manifest.json').read_text())
    rows=[];start=time.time()
    for index,entry in enumerate(manifest):
        image=base.read_image(entry['search_paths'])
        entry['image_height'],entry['image_width']=image.shape[:2]
        tracker.initialize(base.read_image(entry['template_paths']),dict(init_bbox=entry['init_box']))
        audit.region_masks={key:base.masks(tracker.template_boxes[0]) for key in ('rgb','tir')}
        patch,resize,mask=base.sample_target(image,entry['crop_state'],4.,output_sz=256)
        search=tracker.preprocessor.process(patch,mask,half=False).tensors
        context=base.target_context(entry,resize,256)
        audit.context=dict(sample=index,split=entry['split'],sequence=entry['sequence'],frame=entry['frame'])
        with torch.no_grad():
            audit.source_selection=None
            audit.control='native'
            out=audit.forward(tracker,search)
            assert base.digest_tensor(search)+':'+base.digest_tensor(tracker.z_dict1.tensors)==reference[index]['input_sha256']
            assert all(base.digest_tensor(out[k])==v for k,v in reference[index]['outputs'].items())
            negatives=out['score_map'].flatten().masked_fill(context['exclusion'],-torch.inf)
            context['negative_index']=int(negatives.argmax()) if torch.isfinite(negatives).any() else None
            measured_base=base.measures(tracker,out,entry,resize,context)
            stages={k:v.clone() for k,v in audit.stages.items()}
            messages={k:v.clone() for k,v in audit.messages.items()}
            if args.controls_only:
                for mode in ('mr_off','cross_off'):
                    audit.control=mode
                    out=audit.forward(tracker,search)
                    value=base.measures(tracker,out,entry,resize,context)
                    rows.append(dict(sample=index,split=entry['split'],sequence=entry['sequence'],frame=entry['frame'],
                        center_inside=context['center_inside'],action=mode,
                        final_search_relative_delta=float((audit.stages['final_search']-stages['final_search']).norm()/stages['final_search'].norm()),
                        score_relative_delta=float((audit.stages['score']-stages['score']).norm()/stages['score'].norm()),
                        iou_delta=value['iou']-measured_base['iou'],
                        margin_delta=value['margin']-measured_base['margin'] if measured_base['margin'] is not None else None,
                        loss_delta=value['loss']-measured_base['loss'] if measured_base['loss'] is not None else None,
                        hann_peak_changed=value['hann_peak']!=measured_base['hann_peak'],
                        box_max_pixel_delta=float(np.max(np.abs(np.array(json.loads(value['box']))-json.loads(measured_base['box']))))))
                audit.control='native'
            for layer in range(3):
                if args.controls_only:break
                for direction in ('rgb','tir'):
                    for region in base.REGIONS:
                        if int(audit.region_masks[direction][region].sum())<2:continue
                        for action in ('drop_detail','shuffle_detail','drop_contrast'):
                            audit.source_selection=(layer,direction,region,action)
                            out=audit.forward(tracker,search)
                            value=base.measures(tracker,out,entry,resize,context)
                            sd,md=audit.last_source_delta,audit.last_message_delta
                            rows.append(dict(sample=index,split=entry['split'],sequence=entry['sequence'],frame=entry['frame'],
                                center_inside=context['center_inside'],layer=layer,receiver=direction,source_region=region,action=action,
                                source_edit_norm=float(sd.norm()),message_edit_norm=float(md.norm()),
                                message_edit_relative=float(md.norm()/messages[(layer,direction)].norm().clamp_min(1e-20)),
                                final_search_relative_delta=float((audit.stages['final_search']-stages['final_search']).norm()/stages['final_search'].norm()),
                                score_relative_delta=float((audit.stages['score']-stages['score']).norm()/stages['score'].norm()),
                                iou_delta=value['iou']-measured_base['iou'],
                                margin_delta=value['margin']-measured_base['margin'] if measured_base['margin'] is not None else None,
                                loss_delta=value['loss']-measured_base['loss'] if measured_base['loss'] is not None else None,
                                hann_peak_changed=value['hann_peak']!=measured_base['hann_peak'],
                                box_max_pixel_delta=float(np.max(np.abs(np.array(json.loads(value['box']))-json.loads(measured_base['box']))))))
        if (index+1)%4==0:
            base.csv_write(RUN/'interventions.csv',rows)
            print(json.dumps(dict(samples=index+1,interventions=len(rows),wall_seconds=time.time()-start)),flush=True)
    assert base.state_hash(tracker.network.state_dict())==state
    base.csv_write(RUN/'allocator.csv',audit.allocator_rows)
    base.write_json(RUN/'completed.json',dict(environment=base.environment(),samples=len(manifest),interventions=len(rows),
        wall_seconds=time.time()-start,gpu_wall_hours=(time.time()-start)/3600,state_unchanged=True,
        actual_relation_times_v_max_error=max(audit.contract_errors),all72_original_workspace_parity=True,optimizer_steps=0))


if __name__=='__main__':main()
