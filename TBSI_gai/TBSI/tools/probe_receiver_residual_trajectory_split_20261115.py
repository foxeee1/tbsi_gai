#!/usr/bin/env python3
"""Separate crop-state and operator effects at the six observed trajectory splits."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['TBSI_INFER_FP16']='0'
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from run_dual_message_p1 import config,setup,environment
from analyze_receiver_residual_boxes_attributes_20261115 import OUT,RUN,DATA,frames,box_stats,crop_stats,runs,write_json,write_csv
from lib.test.tracker.tbsi_track import TBSITrack
from lib.models.layers.receiver_conditioned_message_residual import ReceiverConditionedMessageResidual

def main():
    setup(42);start=time.time();dest=OUT/'trajectory_split';dest.mkdir(exist_ok=True);shutil.copy2(__file__,dest/Path(__file__).name)
    assert not (dest/'summary.json').exists()
    prior=OUT/'closed_loop_six';names=json.loads((prior/'CONTRACT.json').read_text())['names'];manifest=[];timeline=[]
    for name in names:
        g=np.loadtxt(DATA/'testingset'/name/'init.txt',delimiter=',',ndmin=2)
        a,b=[np.loadtxt(prior/(name+'_'+m+'.txt'),ndmin=2) for m in ['native_replay','residual_off']]
        ca,cb=[crop_stats(np.vstack([p[0],p[:-1]]),g) for p in [a,b]]
        difference=(ca[2]!=cb[2]).any(1)|(ca[3]!=cb[3]);where=np.flatnonzero(difference);first=int(where[0]) if len(where) else -1
        io=[box_stats(p,g)['iou'] for p in [a,b]];rr=runs(abs(io[0]-io[1])>.3,5);gap=rr[0][0] if rr else -1
        pick={first:'first_integer_crop_difference',gap:'first_5f_iou_gap'}
        for frame,kind in pick.items():
            if frame<1:continue
            manifest.append(dict(sequence=name,frame=frame,kind=kind,native_prev=a[frame-1].tolist(),off_prev=b[frame-1].tolist(),
                native_crop_tl=ca[2][frame].tolist(),off_crop_tl=cb[2][frame].tolist(),native_crop_side=float(ca[3][frame]),off_crop_side=float(cb[3][frame])))
        timeline.append(dict(sequence=name,first_integer_crop_difference_frame=first+1 if first>=0 else -1,first_5f_iou_gap_frame=gap+1 if gap>=0 else -1,
            fraction_different_integer_crops=float(difference[1:].mean()),initial_state_delta_at_first_crop=float(np.max(abs(a[first-1]-b[first-1]))) if first>0 else None))
    write_json(dest/'manifest_locked.json',manifest);write_csv(dest/'split_timeline.csv',timeline)
    cfg=config(ROOT/'experiments/tbsi_track/MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-candidate.yaml')
    params=SimpleNamespace(cfg=cfg,checkpoint=str(RUN/'candidate_final/checkpoint_u0468.pth.tar'),template_factor=2.,template_size=128,search_factor=4.,search_size=256,debug=0,save_all_boxes=False)
    tr=TBSITrack(params,'lasher_test');mode=['native'];score=[None]
    handles=[m.register_forward_hook(lambda m,args,out:args[1] if mode[0]=='off' else out) for m in tr.network.modules() if isinstance(m,ReceiverConditionedMessageResidual)]
    def head(mod,args,out):score[0]=(out[0]*tr.output_window).detach().cpu().numpy().reshape(-1)
    handles.append(tr.network.box_head.register_forward_hook(head));rows=[]
    for name in names:
        paths=frames(name);g=np.loadtxt(DATA/'testingset'/name/'init.txt',delimiter=',',ndmin=2)
        def image(f):return np.concatenate([cv2.cvtColor(cv2.imread(str(p[f])),cv2.COLOR_BGR2RGB) for p in paths],axis=2)
        tr.initialize(image(0),dict(init_bbox=g[0].tolist()))
        for e in [e for e in manifest if e['sequence']==name]:
            f=e['frame'];im=image(f)
            for state in ['native','off']:
                for action in ['native','off']:
                    mode[0]=action;tr.state=list(e[state+'_prev']);tr.frame_id=f-1
                    box=np.asarray(tr.track(im)['target_bbox']);s=box_stats(box[None],g[f:f+1]);peak=int(np.argmax(score[0]))
                    rows.append(dict(sequence=name,frame=f+1,kind=e['kind'],crop_state_arm=state,operator=action,iou=float(s['iou'][0]),hann_peak=peak,box=json.dumps(box.tolist())))
    write_csv(dest/'factorial_readouts.csv',rows)
    result=dict(timeline=timeline,events=len(manifest),forwards=len(rows),optimizer_steps=0,environment=environment(),seconds=time.time()-start,
        scope='retrospective selected6;same checkpoint;2 actual prior states x2 operator states;integer crop differs is a geometric fact,not proof it uniquely causes tracking failure')
    write_json(dest/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
