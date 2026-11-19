#!/usr/bin/env python3
"""Verify fresh-sequence tracker boxes with diagnostic export disabled/enabled."""
import argparse, json, os, random, sys
from pathlib import Path
import cv2, numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from tools.g0_expand_export import read_frame
def build(args, diagnostic):
 p=parameters(args.config); p.debug=0; p.save_all_boxes=False; p.checkpoint=args.checkpoint
 t=TBSITrack(p,'lasher_train'); t.diagnostic=diagnostic; t.diagnostic_early_block=2; t.diagnostic_late_block=9; return t
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default='TBSI_E_MHSA_FULL_4090D_seed42'); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--output',required=True); ap.add_argument('--sequences',type=int,default=3); args=ap.parse_args()
 os.environ['TBSI_INFER_FP16']='0'; random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
 out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
 m=json.loads(Path(args.manifest).read_text()); ds=LasHeRDataset(split='trainingset',sequence_list_file=m['sequence_list']).get_sequence_list(); by={x.name:x for x in ds}; names=m['fresh_sequences'][:args.sequences]; a=build(args,False); b=build(args,True); rows=[]
 for name in names:
  seq=by[name]; a.initialize(read_frame(seq,0),{'init_bbox':seq.ground_truth_rect[0]}); b.initialize(read_frame(seq,0),{'init_bbox':seq.ground_truth_rect[0]})
  for fn in range(1,len(seq.frames[0])):
   image=read_frame(seq,fn); info={'sequence_name':seq.name,'frame_num':fn,'gt_bbox':seq.ground_truth_rect[fn]}; box_a=a.track(image,info)['target_bbox']; box_b=b.track(image,info)['target_bbox'];
   if fn in (1,len(seq.frames[0])-1): rows.append({'sequence':name,'frame':fn,'max_abs_box_diff':float(np.max(np.abs(np.asarray(box_a)-np.asarray(box_b))))})
 result={'checked_sequences':names,'events':rows,'max_abs_box_diff':max([x['max_abs_box_diff'] for x in rows] or [0.0]),'passed':all(x['max_abs_box_diff']==0.0 for x in rows)}; Path(args.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); raise SystemExit(0 if result['passed'] else 2)
if __name__=='__main__': main()
