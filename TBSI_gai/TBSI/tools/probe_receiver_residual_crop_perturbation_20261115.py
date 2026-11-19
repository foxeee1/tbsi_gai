#!/usr/bin/env python3
"""Locked +/-1 original-pixel crop intervention; no training or tracker changes."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TBSI_INFER_FP16'] = '0'
import csv
import hashlib
import importlib
import json
import math
import shutil
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
from run_dual_message_p1 import setup, config, environment, sha, state_hash
from analyze_receiver_residual_boxes_attributes_20261115 import OUT, RUN, DATA, frames, box_stats, write_json, write_csv
from lib.test.tracker.tbsi_track import TBSITrack
from lib.models.layers.receiver_conditioned_message_residual import ReceiverConditionedMessageResidual

DEST = RUN/'analysis/crop_perturbation_20261115'
OFFSETS = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]

def geometry(state):
    x,y,w,h = state
    side = math.ceil(math.sqrt(w*h)*4.)
    return np.array([round(x+w/2-side/2), round(y+h/2-side/2)]), side

def support(gt, tl, side):
    """Any-overlap GT cells; background excludes a one-cell expanded GT box."""
    yy,xx = np.mgrid[:16,:16]
    xy = np.stack([xx,yy],-1)*side/16 + tl
    cell = side/16
    fg = ((xy+cell > gt[:2]) & (xy < gt[:2]+gt[2:])).all(-1)
    bg = ~((xy+cell > gt[:2]-cell) & (xy < gt[:2]+gt[2:]+cell)).all(-1)
    inter = np.maximum(0, np.minimum(gt[:2]+gt[2:], tl+side)-np.maximum(gt[:2],tl)).prod()
    center = gt[:2]+gt[2:]/2
    return fg,bg,float(inter/gt[2:].prod()),bool(((center>=tl)&(center<tl+side)).all())

def response_stats(score, window, gt, tl, side):
    fg,bg,coverage,inside = support(gt,tl,side)
    result = dict(gt_coverage=coverage, gt_center_inside=inside, fg_cells=int(fg.sum()), bg_cells=int(bg.sum()))
    for label, response in [('raw',score), ('hann',score*window)]:
        peak = int(response.argmax()); y,x = divmod(peak,16)
        result[label+'_peak'] = peak
        result[label+'_peak_on_gt_support'] = bool(fg[y,x])
        result[label+'_peak_original_x'] = float(tl[0]+(x+.5)*side/16)
        result[label+'_peak_original_y'] = float(tl[1]+(y+.5)*side/16)
        result[label+'_target_max'] = float(response[fg].max()) if fg.any() else None
        result[label+'_background_max'] = float(response[bg].max()) if bg.any() else None
        result[label+'_margin'] = float(response[fg].max()-response[bg].max()) if fg.any() and bg.any() else None
    return result

class ActionProbe:
    def __init__(self, tracker):
        self.mode='native'; self.context={}; self.score=None; self.rows=[]; self.reference={}
        self.mods=[(n,m) for n,m in tracker.network.named_modules() if isinstance(m,ReceiverConditionedMessageResidual)]
        assert len(self.mods)==6
        self.handles=[m.register_forward_hook(self.hook(n)) for n,m in self.mods]
        self.handles.append(tracker.network.box_head.register_forward_hook(self.head))

    def head(self, module, args, out):
        self.score=out[0].detach().float().cpu().numpy().reshape(16,16)

    def hook(self, name):
        def run(module,args,out):
            if self.mode=='off': return args[1]
            msg=args[1].detach().float(); delta=(out.float()-msg).cpu().numpy().reshape(-1)
            key=(self.context['event_id'],name)
            if self.context['dx']==0 and self.context['dy']==0:
                self.reference[key]=delta.copy()
            ref=self.reference[key]; norm=float(np.linalg.norm(delta)); refnorm=float(np.linalg.norm(ref))
            self.rows.append(dict(self.context,module=name,
                relative_edit=norm/max(float(msg.norm()),1e-30),
                delta_cosine_to_center=float(np.dot(delta,ref)/max(norm*refnorm,1e-30)),
                delta_relative_change_to_center=float(np.linalg.norm(delta-ref)/max(refnorm,1e-30))))
            return out
        return run

def main():
    setup(42); start=time.time()
    DEST.mkdir(parents=True,exist_ok=False)
    (DEST/'source').mkdir()
    # Selection is inherited verbatim from completed diagnostics, before new forward outputs.
    primary=json.loads((OUT/'fixed_crop_prefix/events_locked.json').read_text())
    supplemental=json.loads((OUT/'fixed_crop/events_locked.json').read_text())
    names=list(dict.fromkeys(e['sequence'] for e in primary)); events={}
    for role,source in [('primary_prefix',primary),('supplemental_retrospective',supplemental)]:
        for e in source:
            key=(e['sequence'],e['frame'])
            if key in events:
                assert np.allclose(events[key]['crop_state'],e['crop_state'],rtol=0,atol=0)
                events[key]['roles'].append(role); continue
            events[key]=dict(e,roles=[role],event_id=len(events))
    selected=list(events.values())
    write_json(DEST/'manifest_locked.json', selected)
    cp={a:RUN/b/'checkpoint_u0468.pth.tar' for a,b in [('candidate','candidate_final'),('control','control_final3')]}
    configs={a:ROOT/('experiments/tbsi_track/MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-'+a+'.yaml') for a in cp}
    with tarfile.open(DEST/'source/code_snapshot.tar.gz','w:gz') as tar:
        for folder in ['lib/models','lib/test/tracker','lib/test/utils','lib/train/data','lib/config','lib/utils']:
            for p in sorted((ROOT/folder).rglob('*.py')): tar.add(p,arcname=str(p.relative_to(ROOT)))
        for p in [Path(__file__).resolve(),ROOT/'tools/run_dual_message_p1.py',ROOT/'tools/analyze_receiver_residual_boxes_attributes_20261115.py',*configs.values()]:
            tar.add(p,arcname=str(p.relative_to(ROOT)))
    contract=dict(question='Does a +/-1 original-image-pixel crop translation reverse target-local utility, and is that sensitivity specific to the learned residual?',
        optimizer_steps=0, training=False, model_code_changes=False,
        checkpoint_paths={a:str(p) for a,p in cp.items()},checkpoints_sha256={a:sha(p) for a,p in cp.items()},
        config_paths={a:str(p) for a,p in configs.items()},configs_sha256={a:sha(p) for a,p in configs.items()},
        manifest_sha256=sha(DEST/'manifest_locked.json'),source_snapshot_sha256=sha(DEST/'source/code_snapshot.tar.gz'),
        input_manifests={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'fixed_crop_prefix/events_locked.json',OUT/'fixed_crop/events_locked.json']},
        dataset=str(DATA/'testingset'),offsets_original_pixels=OFFSETS,sequences=len(names),events=len(selected),primary_events=len(primary),
        arms=['control','candidate_native','candidate_off'],
        selection='All previously locked 96 prefix events, plus previously locked 192 retrospective events, deduplicated by sequence/frame. Not an independent validation cohort.',
        geometry='Translate previous box x/y by integer offset with unchanged width/height; assert actual rounded integer crop origin moves exactly by requested offset and side is unchanged. GT stays in original coordinates.',
        target_region='16x16 score cells overlapping actual GT box in integer crop geometry; background outside GT expanded by one score cell. Report raw and Hann separately. Peak positions use cell centers, not decoded offset.',
        valid_local_utility='GT center inside all five crops and GT coverage >=0.5 on all five; all events remain in raw outputs.',
        meaningful_utility_flip='Both positive and negative candidate-minus-comparator IoU differences beyond +/-0.005 across five crops; descriptive 0.5 IoU percentage-point tolerance, also report +/-0.001 and exact signs.',
        repeat_floor='Repeat centered crop for first eight manifest sequences, each arm; no optimizer; same template.',
        warnings=['Current quick control is dynamic Value gate + MR, not historical MR g=1.',
                  'Candidate vs control includes differences learned during joint training; native vs off isolates the current inference branch only.',
                  'Sparse-frame IoU is not sequence AUC or full-test performance; attributes are overlapping sequence tags, not per-frame degradation labels.'])
    write_json(DEST/'CONTRACT.json',contract)
    cfg=config(configs['candidate']); assert not cfg.MODEL.TEMPORAL_TOKENS
    params=SimpleNamespace(cfg=cfg,checkpoint=str(cp['candidate']),template_factor=2.,template_size=128,search_factor=4.,search_size=256,debug=0,save_all_boxes=False)
    tracker=TBSITrack(params,'lasher_test'); probe=ActionProbe(tracker)
    candidate={n:t.detach().cpu().clone() for n,t in tracker.network.state_dict().items()}
    control=torch.load(str(cp['control']),map_location='cpu')['net']
    tracker_module=importlib.import_module('lib.test.tracker.tbsi_track')
    original_sample=tracker_module.sample_target; sampled={}
    def traced_sample(*args,**kwargs):
        result=original_sample(*args,**kwargs)
        sampled['patch_sha256']=hashlib.sha256(result[0].tobytes()).hexdigest()
        sampled['padding_mask_sha256']=hashlib.sha256(result[2].tobytes()).hexdigest()
        return result
    tracker_module.sample_target=traced_sample
    rows=[]; maps=[]; inputs={}; templates={}; repeats=[]; parities=[]; state_checks={}; image_hashes={}
    window=tracker.output_window.detach().cpu().numpy().reshape(16,16)
    for arm,weights in [('candidate',candidate),('control',control)]:
        missing,unexpected=tracker.network.load_state_dict(weights,strict=False)
        assert not unexpected and (not missing if arm=='candidate' else all('receiver_message_residual' in n for n in missing))
        before=state_hash(tracker.network.state_dict())
        for idx,name in enumerate(names):
            paths=frames(name); gt=np.loadtxt(DATA/'testingset'/name/'init.txt',delimiter=',',ndmin=2)
            def image(f):
                parts=[cv2.imread(str(p[f])) for p in paths];assert all(p is not None for p in parts)
                im=np.concatenate([cv2.cvtColor(p,cv2.COLOR_BGR2RGB) for p in parts],axis=2)
                image_hashes[name+':'+str(f)]=hashlib.sha256(im.tobytes()).hexdigest()
                return im
            tracker.initialize(image(0),dict(init_bbox=gt[0].tolist()))
            template_hash=hashlib.sha256(tracker.z_dict1.tensors.detach().cpu().numpy().tobytes()).hexdigest()
            if name in templates: assert templates[name]==template_hash
            templates[name]=template_hash
            for event in [e for e in selected if e['sequence']==name]:
                f=event['frame']; im=image(f); g=gt[f]; assert np.isfinite(g).all() and (g[2:]>0).all()
                base=list(event['crop_state']);base_tl,base_side=geometry(base)
                for dx,dy in OFFSETS:
                    state=[base[0]+dx,base[1]+dy,*base[2:]];tl,side=geometry(state)
                    assert side==base_side and np.array_equal(tl-base_tl,[dx,dy]),(name,f,dx,dy)
                    for mode in (['native','off'] if arm=='candidate' else ['off']):
                        probe.mode=mode;probe.context=dict(event_id=event['event_id'],sequence=name,frame=f,dx=dx,dy=dy)
                        tracker.state=list(state);tracker.frame_id=f-1
                        box=np.asarray(tracker.track(im)['target_bbox']);score=probe.score.copy()
                        input_key=(event['event_id'],dx,dy)
                        value=(sampled['patch_sha256'],sampled['padding_mask_sha256'])
                        if input_key in inputs: assert inputs[input_key]==value
                        inputs[input_key]=value
                        s=box_stats(box[None],g[None]); label='candidate_'+mode if arm=='candidate' else 'control'
                        row=dict(probe.context,roles='|'.join(event['roles']),kind=event['kind'],arm=label,iou=float(s['iou'][0]),center_norm=float(s['cn'][0]),
                            box=json.dumps(box.tolist()),gt=json.dumps(g.tolist()),crop_state=json.dumps(state),crop_tl_x=int(tl[0]),crop_tl_y=int(tl[1]),crop_side=side,
                            patch_sha256=value[0],padding_mask_sha256=value[1],template_tensor_sha256=template_hash,
                            **response_stats(score,window,g,tl,side))
                        rows.append(row);maps.append(score)
                        if arm=='control' and dx==dy==0:
                            stored=np.loadtxt(RUN/'eval_control'/(name+'.txt'),ndmin=2)[f]
                            parities.append(float(np.max(abs(stored-box))))
                        if idx<8 and 'primary_prefix' in event['roles'] and dx==dy==0:
                            tracker.state=list(state);tracker.frame_id=f-1
                            again=np.asarray(tracker.track(im)['target_bbox'])
                            repeats.append(dict(sequence=name,arm=label,max_abs_box_difference=float(np.max(abs(again-box))),
                                max_abs_score_difference=float(np.max(abs(probe.score-score)))))
            if (idx+1)%12==0:
                print(json.dumps(dict(arm=arm,sequences=idx+1,total=len(names),forwards=len(rows),seconds=time.time()-start)),flush=True)
        state_checks[arm]=(state_hash(tracker.network.state_dict())==before); assert state_checks[arm]
        probe.reference.clear()
    tracker_module.sample_target=original_sample
    write_csv(DEST/'frame_readouts.csv',rows);write_csv(DEST/'operator_readouts.csv',probe.rows)
    write_csv(DEST/'repeat_checks.csv',repeats)
    np.savez_compressed(DEST/'score_maps.npz',raw=np.stack(maps),window=window)
    write_json(DEST/'input_image_hashes.json',image_hashes)
    result=dict(environment=environment(),sequences=len(names),events=len(selected),forwards=len(rows),repeat_forwards=len(repeats),optimizer_steps=0,
        state_unchanged=state_checks,common_inputs_equal=True,common_templates_equal=True,exact_integer_translation_verified=True,
        control_center_replay_max_box_error=max(parities),repeat_max_box_error=max(r['max_abs_box_difference'] for r in repeats),
        repeat_max_score_error=max(r['max_abs_score_difference'] for r in repeats),wall_seconds=time.time()-start,
        estimated_gpu_hours_single_exclusive_task=(time.time()-start)/3600,
        command='python tools/probe_receiver_residual_crop_perturbation_20261115.py',working_directory=str(ROOT))
    write_json(DEST/'execution_record.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
