#!/usr/bin/env python3
import csv, json, hashlib, os, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
FEATURES=ROOT/'output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/selected_features.npz'
SOURCE=ROOT/'output/experiments/M1_S1_MODALITY_CANDIDATE_AUDIT_20261015_retry3/per_state_candidates.csv'
CONFIG=ROOT/'experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml'
CHECKPOINT=ROOT/'output/experiments/M1_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
CHECKPOINT=ROOT/'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import argparse
 ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); a=ap.parse_args(); out=Path(a.output)
 if out.exists() and any(out.iterdir()): raise RuntimeError('nonempty output')
 out.mkdir(parents=True)
 os.environ['TBSI_INFER_FP16']='0'; os.environ['TBSI_CHECKPOINT_OVERRIDE']=str(CHECKPOINT)
 torch.set_num_threads(1); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
 from lib.test.parameter.tbsi_track import parameters
 from lib.test.tracker.tbsi_track import TBSITrack
 d=np.load(str(FEATURES),allow_pickle=False); rows=list(csv.DictReader(SOURCE.open()))
 assert len(rows)==144 and len(d['fused'])==144
 params=parameters('TBSI_E_MHSA_FULL_4090D_seed42'); params.debug=0; params.save_all_boxes=False; params.checkpoint=str(CHECKPOINT)
 tracker=TBSITrack(params,'response_evidence'); net=tracker.network.eval(); arms=('m1_fused','rgb_only','tir_only'); arr={'m1_fused':d['fused'],'rgb_only':d['rgb'],'tir_only':d['tir']}
 output=[]
 with torch.inference_mode():
  for i,r in enumerate(rows):
   item={'sequence':r['sequence'],'frame':r['frame']}
   for arm in arms:
    feat=torch.as_tensor(arr[arm][i].transpose(2,0,1)[None],dtype=torch.float32,device='cuda')
    score,_,_,_=net.box_head(feat); resp=(tracker.output_window*score).flatten().float()
    vals=torch.topk(resp, k=2).values; prob=torch.softmax(resp,dim=0); ent=-(prob*torch.log(prob.clamp_min(1e-12))).sum()/np.log(resp.numel())
    idx=int(torch.argmax(resp).item()); item[arm+'_peak_score']=float(vals[0]); item[arm+'_top2_margin']=float(vals[0]-vals[1]); item[arm+'_entropy']=float(ent); item[arm+'_peak_x']=idx%16; item[arm+'_peak_y']=idx//16
   output.append(item)
 fields=list(output[0].keys())
 with (out/'response_evidence.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(output)
 manifest={'status':'COMPLETE','source':str(SOURCE),'features':str(FEATURES),'source_sha256':sha(SOURCE),'features_sha256':sha(FEATURES),'config_sha256':sha(CONFIG),'checkpoint_sha256':sha(CHECKPOINT),'runner_sha256':sha(Path(__file__)),'states':len(output),'protocol':'frozen M1 CENTER response evidence from each cached candidate feature; no GT used'}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
