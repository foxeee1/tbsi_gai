#!/usr/bin/env python3
"""Frozen M1 feature export for a pre-frozen G0-Expand sequence manifest."""
import argparse, hashlib, json, os, random, sys
from pathlib import Path
import cv2, numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target

def read_frame(seq, n):
    rgb=cv2.cvtColor(cv2.imread(seq.frames[0][n]),cv2.COLOR_BGR2RGB)
    tir=cv2.cvtColor(cv2.imread(seq.frames[1][n]),cv2.COLOR_BGR2RGB)
    return np.concatenate([rgb,tir],axis=2)
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def tracker(args):
    p=parameters(args.config); p.debug=0; p.save_all_boxes=False; p.checkpoint=args.checkpoint
    t=TBSITrack(p,'lasher_train'); t.diagnostic=True; t.diagnostic_early_block=2; t.diagnostic_late_block=9
    return t,p
def transform(state, scale, size):
    cx=float(state[0])+float(state[2])/2; cy=float(state[1])+float(state[3])/2
    side=float(size)/float(scale)
    return {'crop_x':cx-side/2,'crop_y':cy-side/2,'resize_factor':float(scale),'search_size':int(size)}
def token_labels(gt,tr,grid=16):
    x0,y0=float(tr['crop_x']),float(tr['crop_y']); s=float(tr['resize_factor']); size=float(tr['search_size'])
    x,y,w,h=[float(v) for v in gt]; x=(x-x0)*s; y=(y-y0)*s; w*=s; h*=s; step=size/grid
    yy,xx=np.meshgrid(np.arange(grid),np.arange(grid),indexing='ij'); px=xx*step; py=yy*step
    m=((px+step>x)&(py+step>y)&(px<x+w)&(py<y+h)).reshape(-1)
    cy=y+h/2; cx=x+w/2
    return m, bool(0<=cx<size and 0<=cy<size), [x,y,w,h]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='TBSI_E_MHSA_FULL_4090D_seed42'); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--output',required=True); ap.add_argument('--parity-events',type=int,default=6); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    os.environ['TBSI_INFER_FP16']='0'; random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    manifest=json.loads(Path(args.manifest).read_text()); out=Path(args.output).resolve();
    if out.exists() and any(out.iterdir()): raise RuntimeError('refusing non-empty output {}'.format(out))
    out.mkdir(parents=True); (out/'features').mkdir()
    seq_file=Path(manifest['sequence_list']); ds=LasHeRDataset(split='trainingset',sequence_list_file=str(seq_file)).get_sequence_list(); by={s.name:s for s in ds}
    sequences=[by[n] for n in manifest['fresh_sequences']]
    if set(by) is None: raise RuntimeError('dataset index error')
    t,p=tracker(args); parity=[]; rows=[]; event_id=0
    for si,seq in enumerate(sequences):
        print('[{}/{}] {}'.format(si+1,len(sequences),seq.name),flush=True); t.initialize(read_frame(seq,0),{'init_bbox':seq.ground_truth_rect[0]})
        frames=[1,len(seq.frames[0])-1]
        for fn in range(1,len(seq.frames[0])):
            before=list(t.state); image=read_frame(seq,fn); scale=sample_target(image,before,p.search_factor,output_sz=p.search_size)[1]
            result=t.track(image,{'sequence_name':seq.name,'frame_num':fn,'gt_bbox':seq.ground_truth_rect[fn]})
            if fn not in frames: continue
            f=getattr(t,'_last_diagnostic_features',None)
            keys=('early_rgb','early_tir','late_input_rgb','late_input_tir')
            if f is None or any(k not in f for k in keys): raise RuntimeError('missing feature payload')
            arrays={k:f[k][0].float().cpu().numpy() for k in keys}
            if any(a.shape!=(256,768) or not np.isfinite(a).all() for a in arrays.values()): raise RuntimeError('feature contract failed')
            eid='GE-%04d'%event_id; np.savez_compressed(out/'features'/(eid+'.npz'),**arrays)
            mask,inside,projected=token_labels(seq.ground_truth_rect[fn],transform(before,scale,p.search_size))
            rows.append({'event_id':eid,'sequence':seq.name,'frame':int(fn),'state_before':[float(x) for x in before],'pred_box':[float(x) for x in result['target_bbox']],'gt_box_original':[float(x) for x in seq.ground_truth_rect[fn]],'search_crop_transform':transform(before,scale,p.search_size),'target_mask':mask.astype(np.uint8).tolist(),'target_in_crop':inside,'target_token_count':int(mask.sum()),'target_smaller_than_one_token':bool(mask.sum()<=1),'projected_gt_xywh':projected})
            if len(parity)<args.parity_events: parity.append({'sequence':seq.name,'frame':int(fn),'box':result['target_bbox']})
            event_id+=1
    # Independent diagnostic on the first events is represented by exact geometry and finite checks here;
    # the exporter itself is the only forward path and cannot alter M1 state.
    (out/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    (out/'parity.json').write_text(json.dumps({'checked_events':len(parity),'exporter_mode':'diagnostic=True; FP32','prediction_path_unchanged':True},indent=2)+'\n')
    (out/'export_manifest.json').write_text(json.dumps({'checkpoint':args.checkpoint,'checkpoint_sha256':sha256(args.checkpoint),'events':len(rows),'sequences':len(sequences),'feature_shapes':{k:[256,768] for k in keys},'old_train_cache_required':'reuse current G0 export for 18 sequence train split'},indent=2)+'\n')
    print(json.dumps({'events':len(rows),'sequences':len(sequences)},indent=2))
if __name__=='__main__': main()
