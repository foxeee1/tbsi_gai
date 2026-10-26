#!/usr/bin/env python3
"""Top-K readout reachability diagnostic for frozen M1 states."""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box

NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
CONFIG = ROOT / 'experiments/tbsi_track' / (NAME + '.yaml')
CHECKPOINT = ROOT / 'output/experiments' / NAME / 'checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG_SHA = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
CHECKPOINT_SHA = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def iou(a, b):
    ix = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0],b[0]))
    iy = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1],b[1]))
    inter = ix*iy; union = a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union > 0 else 0.

def decode_response(network, tracker, response, size, offset, state, shape, resize):
    box = network.box_head.cal_bbox(response, size, offset).view(-1, 4).mean(0)
    box = box * (tracker.params.search_size / resize)
    cx, cy = state[0]+.5*state[2], state[1]+.5*state[3]
    half = .5*tracker.params.search_size/resize
    xywh = [float(box[0]+cx-half-.5*box[2]), float(box[1]+cy-half-.5*box[3]),
            float(box[2]), float(box[3])]
    return clip_box(xywh, shape[0], shape[1], margin=10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True); ap.add_argument('--trace-dir', required=True)
    ap.add_argument('--output-dir', required=True); ap.add_argument('--topk', default='1,4,8,16')
    args = ap.parse_args()
    outdir = Path(args.output_dir).resolve()
    if outdir.exists() and any(outdir.iterdir()): raise RuntimeError('refusing nonempty output')
    if sha256(CONFIG) != CONFIG_SHA or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('protected M1 hash mismatch')
    topks = [int(x) for x in args.topk.split(',')]
    data = np.load(args.features, allow_pickle=False)
    required = {'fused','sequence','frame','gt_xy','label'}
    if not required.issubset(set(data.files)): raise RuntimeError('missing feature fields')
    trace = {}
    for p in Path(args.trace_dir).glob('*/frames.jsonl'):
        for line in p.read_text().splitlines():
            if line.strip():
                r=json.loads(line); trace[(p.parent.name,int(r['frame_id']))]=r
    seqs = sorted(set(str(v) for v in data['sequence']))
    list_path = outdir.parent/'topk_sequence_list.txt'; list_path.parent.mkdir(parents=True,exist_ok=True)
    list_path.write_text('\n'.join(seqs)+'\n')
    seqmap={s.name:s for s in LasHeRDataset('trainingset',str(list_path)).get_sequence_list()}
    os.environ['TBSI_INFER_FP16']='0'; torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    params=parameters(NAME); params.debug=0; params.save_all_boxes=False; params.checkpoint=str(CHECKPOINT)
    tracker=TBSITrack(params,'m1_readout_topk'); network=tracker.network.eval(); rows=[]
    with torch.no_grad():
        for i in range(len(data['sequence'])):
            seq=str(data['sequence'][i]); frame=int(data['frame'][i]); tr=trace[(seq,frame)]
            state=np.asarray(tr['tracker_state_before'],dtype=np.float64); gt=np.asarray(tr['gt_box'],dtype=np.float64)
            rgb=cv2.imread(seqmap[seq].frames[0][frame],cv2.IMREAD_COLOR); tir=cv2.imread(seqmap[seq].frames[1][frame],cv2.IMREAD_COLOR)
            image=np.concatenate((cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB),cv2.cvtColor(tir,cv2.COLOR_BGR2RGB)),2)
            _,resize,_=sample_target(image,state,params.search_factor,output_sz=params.search_size)
            fused=torch.as_tensor(data['fused'][i].transpose(2,0,1)[None],dtype=torch.float32,device='cuda')
            score,_,size,offset=network.box_head(fused); response=tracker.output_window*score
            base_box=decode_response(network,tracker,response,size,offset,state,image.shape,resize); base=iou(base_box,gt)
            flat=response.flatten(1)[0]; order=torch.argsort(flat,descending=True).tolist()
            gx,gy=int(data['gt_xy'][i,0]),int(data['gt_xy'][i,1]); gx=max(0,min(15,gx)); gy=max(0,min(15,gy))
            gt_cx,gt_cy=gt[0]+.5*gt[2],gt[1]+.5*gt[3]; scx,scy=state[0]+.5*state[2],state[1]+.5*state[3]
            cx=(gt_cx-scx)*resize/params.search_size+.5; cy=(gt_cy-scy)*resize/params.search_size+.5
            gt_wh=torch.tensor([gt[2]*resize/params.search_size,gt[3]*resize/params.search_size],device='cuda',dtype=size.dtype)
            actions={'base':(base,0,0,0,0)}
            for k in topks:
                cand=order[:k]
                def evaluate(use_gt_reg):
                    best=-1.; best_cell=-1
                    for idx in cand:
                        x,y=idx%16,idx//16; resp=response.clone(); resp[:,:,y,x]=flat.max()+1e-3
                        off=offset; sz=size
                        if use_gt_reg:
                            off=offset.clone(); off[:,:,y,x]=torch.tensor([cx*16-x,cy*16-y],device='cuda',dtype=offset.dtype)
                            sz=size.clone(); sz[:,:,y,x]=gt_wh
                        val=iou(decode_response(network,tracker,resp,sz,off,state,image.shape,resize),gt)
                        if val>best: best=float(val); best_cell=idx
                    return best,best_cell
                cur,curcell=evaluate(False); reg,regcell=evaluate(True)
                gt_idx=gy*16+gx
                def one(idx,use_gt_reg):
                    x,y=idx%16,idx//16; resp=response.clone(); resp[:,:,y,x]=flat.max()+1e-3; off=offset; sz=size
                    if use_gt_reg:
                        off=offset.clone(); off[:,:,y,x]=torch.tensor([cx*16-x,cy*16-y],device='cuda',dtype=offset.dtype)
                        sz=size.clone(); sz[:,:,y,x]=gt_wh
                    return iou(decode_response(network,tracker,resp,sz,off,state,image.shape,resize),gt)
                gtcur=one(gt_idx,False); gtreg=one(gt_idx,True)
                rows.append({'sequence':seq,'frame':frame,'label':str(data['label'][i]),'topk':k,'base_iou':base,
                    'topk_current_reg_iou':cur,'topk_gt_reg_iou':reg,'gt_cell_current_reg_iou':gtcur,
                    'gt_cell_gt_reg_iou':gtreg,'topk_contains_gt':int(gt_idx in cand),'topk_current_cell':curcell,'topk_gt_reg_cell':regcell})
            if (i+1)%16==0: print('states={}/{}'.format(i+1,len(data['sequence'])),flush=True)
    outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/'per_state.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=[]
    for k in topks:
        sub=[r for r in rows if r['topk']==k]; seqnames=sorted(set(r['sequence'] for r in sub))
        def macro(field): return float(np.mean([np.mean([float(r[field]) for r in sub if r['sequence']==s]) for s in seqnames]))
        summary.append({'topk':k,'states':len(sub),'sequences':len(seqnames),'base_iou':macro('base_iou'),
            'topk_current_reg_delta':macro('topk_current_reg_iou')-macro('base_iou'),
            'topk_gt_reg_delta':macro('topk_gt_reg_iou')-macro('base_iou'),
            'gt_cell_current_reg_delta':macro('gt_cell_current_reg_iou')-macro('base_iou'),
            'gt_cell_gt_reg_delta':macro('gt_cell_gt_reg_iou')-macro('base_iou'),
            'topk_contains_gt_rate':float(np.mean([int(r['topk_contains_gt']) for r in sub]))})
    with (outdir/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    meta={'id':'M1-READOUT-TOPK-REACHABILITY-20261013','parent':NAME,'topks':topks,'states':len(data['sequence']),'sequences':len(seqs),
        'config_sha256':sha256(CONFIG),'checkpoint_sha256':sha256(CHECKPOINT),'features_sha256':sha256(Path(args.features)),
        'estimand':'fixed-state decoded-IoU top-K reachability; oracle selection among candidate cells','status':'TOPK-REPLAY-COMPLETE',
        'limitations':['top-K best action is an offline oracle','GT regression values are not deployable','fixed-state replay is not closed-loop tracking']}
    (outdir/'summary.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps({'meta':meta,'summary':summary},indent=2))

if __name__=='__main__': main()
