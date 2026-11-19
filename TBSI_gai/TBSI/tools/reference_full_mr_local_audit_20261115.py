#!/usr/bin/env python3
"""Read-only original-workspace MR reference; no historical admission tools."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['TBSI_INFER_FP16']='0'
import sys
import json
import hashlib
import time
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
FULL=ROOT/'output/experiments/MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109'
OUT=ROOT/'output/experiments/MSHA-MR-LOCAL-TRANSMISSION-AUDIT-S42-20261115'
sys.path.insert(0,str(FULL/'workspace'))
from lib.models.tbsi_track import build_tbsi_track
from lib.config.tbsi_track.config import cfg,update_config_from_file
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    started=time.time()
    cv2.setNumThreads(0);torch.set_num_threads(1);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    update_config_from_file(str(FULL/'contracts/mr_only_g1.yaml'))
    net=build_tbsi_track(cfg,training=False).cuda().eval()
    state=torch.load(FULL/'train/checkpoint_u7020.pth.tar',map_location='cpu')['net']
    net.load_state_dict(state,strict=True)
    gates={n:float(v) for n,v in state.items() if n.endswith('early_gate_scale')}
    assert len(gates)==3 and all(v==0 for v in gates.values())
    del state
    pre=Preprocessor();rows=[]
    def image(paths):
        return np.concatenate([cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB) for p in paths],axis=2)
    for index,entry in enumerate(json.loads((OUT/'frame_manifest.json').read_text())):
        z,_,zm=sample_target(image(entry['template_paths']),entry['init_box'],2,128)
        x,_,xm=sample_target(image(entry['search_paths']),entry['crop_state'],4,256)
        z=pre.process(z,zm,half=False).tensors;x=pre.process(x,xm,half=False).tensors
        with torch.no_grad():
            out=net.forward(template=[z[:,:3],z[:,3:]],search=[x[:,:3],x[:,3:]])
        rows.append(dict(sample=index,input_sha256=digest(x)+':'+digest(z),
            outputs={key:digest(out[key]) for key in ('score_map','size_map','offset_map','backbone_feat')}))
    (OUT/'full_reference.json').write_text(json.dumps(dict(rows=rows,gate_scales=gates,
        source_workspace=str(FULL/'workspace'),config=str(FULL/'contracts/mr_only_g1.yaml'),
        precision='highest FP32;TF32 disabled both matmul and cudnn',wall_seconds=time.time()-started,
        gpu_wall_hours=(time.time()-started)/3600),indent=2)+'\n')
    print('Original-workspace reference complete:',len(rows),flush=True)


if __name__=='__main__':main()
