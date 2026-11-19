#!/usr/bin/env python3
"""Trace learned-CBMR interventions from template residual to search and head."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from audit_mr_cbmr_20261115 import Intervention, config, setup, TBSITrack, ROOT, PARENT, OUT


def main():
    setup(42)
    destination=OUT/'transmission_v2.csv'
    if destination.exists():
        raise RuntimeError('Refusing overwrite')
    cfg=config(ROOT/'experiments/tbsi_track/MR-CBMR-QUICK-S42-20261115-candidate.yaml')
    tracker=TBSITrack(SimpleNamespace(cfg=cfg,checkpoint=str(PARENT/'candidate/checkpoint_u0468.pth.tar'),
        template_factor=cfg.TEST.TEMPLATE_FACTOR,template_size=cfg.TEST.TEMPLATE_SIZE,
        search_factor=cfg.TEST.SEARCH_FACTOR,search_size=cfg.TEST.SEARCH_SIZE,
        debug=0,save_all_boxes=False),'lasher_test')
    intervention=Intervention();intervention.install(tracker.network)
    manifest=json.loads((OUT/'frame_manifest.json').read_text())
    from lib.test.evaluation import get_dataset
    sequences={x.name:x for x in get_dataset('lasher_test')}
    collected={}
    counters={}
    def capture(name,split=False):
        def hook(module,args,output):
            if isinstance(output,tuple):
                for k,x in enumerate(output):
                    collected[name+'/'+str(k)]=x.detach().float().clone()
            else:
                idx=counters.get(name,0);counters[name]=idx+1
                value=output.detach().float()
                if split:
                    collected[name+'/'+str(idx)+'/template']=value[:,:64].clone()
                    collected[name+'/'+str(idx)+'/search']=value[:,64:].clone()
                else:
                    collected[name+'/'+str(idx)]=value.clone()
        return hook
    handles=[]
    for i,host in enumerate(tracker.network.backbone.tbsi_layers):
        handles.append(host.v2_template_fusion.register_forward_hook(capture('fusion'+str(i))))
        handles.append(host.v2_template_fusion.cbmr.register_forward_hook(capture('message'+str(i))))
    for i in (4,7,10,11):
        handles.append(tracker.network.backbone.blocks[i].register_forward_hook(capture('vit'+str(i),split=True)))
    def capture_head(module,args,output):
        score, bbox, size, offset = output
        for key,value in (('score_map',score),('size_map',size),('offset_map',offset)):
            collected['head/'+key]=value.detach().float().clone()
        collected['head/hann_response']=(score*tracker.output_window).detach().float().clone()
    # Tracker calls network.forward directly, bypassing hooks on network.__call__.
    handles.append(tracker.network.box_head.register_forward_hook(capture_head))
    def image(seq,t):
        return np.concatenate([cv2.cvtColor(cv2.imread(seq.frames[k][t]),cv2.COLOR_BGR2RGB) for k in (0,1)],axis=2)
    rows=[]
    for name in dict.fromkeys(x['sequence'] for x in manifest):
        seq=sequences[name];tracker.initialize(image(seq,0),seq.init_info())
        intervention.prepare(torch.stack((tracker.template_boxes,tracker.template_boxes),dim=1))
        for entry in [x for x in manifest if x['sequence']==name]:
            im=image(seq,entry['frame']);pair={}
            for mode in ('off','native'):
                intervention.mode=mode;intervention.call_index={n:0 for n,_ in intervention.layers}
                collected={};counters={}
                tracker.state=list(entry['crop_state']);tracker.frame_id=entry['frame']-1
                with torch.inference_mode():tracker.track(im)
                assert 'head/score_map' in collected, 'Head instrumentation did not execute'
                pair[mode]=collected
            for key,a in pair['off'].items():
                b=pair['native'][key];delta=b-a
                rows.append(dict(sequence=name,frame=entry['frame'],stage=key,
                    reference_norm=float(a.norm()),delta_norm=float(delta.norm()),
                    relative_delta=float(delta.norm()/a.norm().clamp_min(1e-9)),max_abs=float(delta.abs().max())))
        print(name,flush=True)
    with destination.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={key:{field:float(np.mean([r[field] for r in rows if r['stage']==key]))
                  for field in ('reference_norm','delta_norm','relative_delta','max_abs')}
             for key in sorted(set(r['stage'] for r in rows))}
    (OUT/'transmission_summary_v2.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
