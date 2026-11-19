#!/usr/bin/env python3
"""Retrospective path-necessity check on three largest gains/losses; no training."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['TBSI_INFER_FP16']='0'
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
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from run_dual_message_p1 import setup,config,environment,sha
from analyze_receiver_residual_boxes_attributes_20261115 import OUT,RUN,DATA,frames,box_stats,write_json,write_csv
from lib.test.tracker.tbsi_track import TBSITrack
from lib.models.layers.receiver_conditioned_message_residual import ReceiverConditionedMessageResidual

def main():
    setup(42);start=time.time();dest=OUT/'closed_loop_six';dest.mkdir(exist_ok=True)
    assert not (dest/'summary.json').exists()
    shutil.copy2(__file__,dest/Path(__file__).name)
    rr=list(csv.DictReader((OUT/'per_sequence.csv').open()));rr.sort(key=lambda r:float(r['delta_auc']))
    names=[r['sequence'] for r in rr[:3]+rr[-3:]]
    write_json(dest/'CONTRACT.json',dict(names=names,selection='top3 gains and bottom3 losses in existing96, retrospective path-necessity only; no independent performance claim',
        modes=['native_replay','residual_off'],checkpoint='candidate_final u468; same checkpoint both arms',optimizer_steps=0))
    cfg=config(ROOT/'experiments/tbsi_track/MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-candidate.yaml')
    p=SimpleNamespace(cfg=cfg,checkpoint=str(RUN/'candidate_final/checkpoint_u0468.pth.tar'),template_factor=cfg.TEST.TEMPLATE_FACTOR,template_size=cfg.TEST.TEMPLATE_SIZE,
        search_factor=cfg.TEST.SEARCH_FACTOR,search_size=cfg.TEST.SEARCH_SIZE,debug=0,save_all_boxes=False)
    tr=TBSITrack(p,'lasher_test');mode=['native_replay']
    handles=[m.register_forward_hook(lambda mod,args,out:args[1] if mode[0]=='residual_off' else out) for m in tr.network.modules() if isinstance(m,ReceiverConditionedMessageResidual)]
    assert len(handles)==6
    result=[]
    for name in names:
        paths=frames(name);g=np.loadtxt(DATA/'testingset'/name/'init.txt',delimiter=',',ndmin=2);old=np.loadtxt(RUN/'eval_candidate'/(name+'.txt'),ndmin=2)
        def image(f):
            im=[cv2.imread(str(paths[k][f])) for k in (0,1)];assert all(x is not None for x in im)
            return np.concatenate([cv2.cvtColor(x,cv2.COLOR_BGR2RGB) for x in im],axis=2)
        predictions={}
        for action in ['native_replay','residual_off']:
            mode[0]=action;tr.initialize(image(0),dict(init_bbox=g[0].tolist()));boxes=[g[0].tolist()]
            for f in range(1,len(g)):boxes.append(tr.track(image(f))['target_bbox'])
            boxes=np.asarray(boxes);predictions[action]=boxes;np.savetxt(dest/(name+'_'+action+'.txt'),boxes,fmt='%.9f',delimiter='\t')
            print(json.dumps(dict(sequence=name,mode=action,frames=len(g),seconds=time.time()-start)),flush=True)
        native=box_stats(predictions['native_replay'],g);off=box_stats(predictions['residual_off'],g);hist=box_stats(old,g)
        result.append(dict(sequence=name,frames=len(g),saved_native_auc=float(hist['auc'].mean()),replay_native_auc=float(native['auc'].mean()),off_auc=float(off['auc'].mean()),
            off_minus_native_auc=float((off['auc']-native['auc']).mean()),native_replay_max_box_error=float(abs(predictions['native_replay']-old).max()),
            off_minus_native_mean_abs_box_delta=float(abs(predictions['residual_off']-predictions['native_replay']).mean())))
        write_csv(dest/'per_sequence.csv',result)
    write_json(dest/'summary.json',dict(rows=result,environment=environment(),checkpoint_sha256=sha(RUN/'candidate_final/checkpoint_u0468.pth.tar'),optimizer_steps=0,
        seconds=time.time()-start,scope='selected6 path necessity; off replay is not retrained control; native reproducibility must be assessed before interpreting differences'))

if __name__=='__main__':main()
