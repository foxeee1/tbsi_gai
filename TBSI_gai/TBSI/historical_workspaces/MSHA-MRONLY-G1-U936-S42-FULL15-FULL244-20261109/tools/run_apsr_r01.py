#!/usr/bin/env python3
"""APSR-v0 R0/R1 frozen-memory retrieval diagnostics.

Both policies use the same S0 predicted-crop FIFO memory and one-step
counterfactual forwards.  R0 uses ROI global-pool cosine; R1 uses symmetric
bidirectional token matching.  No policy prediction is propagated.
"""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from lib.test.evaluation.datasets import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.train.data.processing_utils import sample_target

def box(v):
    if torch.is_tensor(v): v=v.detach().float().cpu().reshape(-1).tolist()
    return np.nan_to_num(np.asarray(v,dtype=np.float64).reshape(4), nan=0., posinf=0., neginf=0.)
def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ix=max(0.,min(ax+max(0.,aw),bx+max(0.,bw))-max(ax,bx)); iy=max(0.,min(ay+max(0.,ah),by+max(0.,bh))-max(ay,by))
    un=max(0.,aw)*max(0.,ah)+max(0.,bw)*max(0.,bh)-ix*iy
    return float(ix*iy/un) if un else 0.
def clone(x):
    if x is None: return None
    if torch.is_tensor(x): return x.detach().clone()
    if isinstance(x,list): return [clone(v) for v in x]
    if isinstance(x,tuple): return tuple(clone(v) for v in x)
    if isinstance(x,dict): return {k:clone(v) for k,v in x.items()}
    return copy.deepcopy(x)
def names(args):
    ns=[x.strip() for x in open(args.sequence_list,encoding='utf-8') if x.strip()]
    if args.max_sequences: ns=ns[:args.max_sequences]
    if args.split=='train':
        ds=LasHeRDataset(split='testingset',sequence_list_file=None)
        ds.base_path=os.path.join(ds.env_settings.lasher_path,'trainingset'); ds.sequence_list=ns
        got={s.name:s for s in ds.get_sequence_list()}; return [got[n] for n in ns if n in got]
    got={s.name:s for s in get_dataset('lasher_test')}; return [got[n] for n in ns if n in got]
def template(tracker,image,b):
    b=box(b); b[2:]=np.maximum(b[2:],1.)
    p,rf,m=sample_target(image,b.tolist(),tracker.params.template_factor,output_sz=tracker.params.template_size)
    return tracker.preprocessor.process(p,m,half=False),float(rf)
def rgbt(v,i):
    a=cv2.cvtColor(cv2.imread(v),cv2.COLOR_BGR2RGB); b=cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2RGB)
    return np.concatenate([a,b],axis=2)
def roi_tokens(tokens, b, crop_side, resize_factor, side, out=8):
    # tokens: [1, side*side, D], bbox in original-image coordinates.
    t=tokens.transpose(1,2).reshape(1,tokens.shape[-1],side,side)
    cx=b[0]+b[2]/2.; cy=b[1]+b[3]/2.; x0=cx-crop_side/2.; y0=cy-crop_side/2.
    x=(b[0]-x0)*resize_factor; y=(b[1]-y0)*resize_factor; w=b[2]*resize_factor; h=b[3]*resize_factor
    # feature coordinates are image-crop coordinates scaled by side / crop output.
    scale=float(side)/float(side*16)
    try:
        from torchvision.ops import roi_align
        r=roi_align(t,torch.tensor([[0,x,y,x+w,y+h]],device=t.device,dtype=t.dtype),output_size=(out,out),spatial_scale=scale,aligned=True)
    except Exception:
        # deterministic fallback for environments without torchvision ops.
        xx=max(0,min(side-1,int((x+w/2)*scale))); yy=max(0,min(side-1,int((y+h/2)*scale)))
        r=t[:,:,yy:yy+1,xx:xx+1].expand(-1,-1,out,out)
    return r.flatten(2).transpose(1,2)
def scores(q,k,method):
    # q/k: [1,64,D]. Return cosine score for one modality.
    q=F.normalize(q.float(),dim=-1); k=F.normalize(k.float(),dim=-1)
    if method=='r0': return float((q.mean(1)*k.mean(1)).sum(-1).item())
    sim=torch.matmul(q,k.transpose(1,2))
    return float(0.5*(sim.max(-1).values.mean()+sim.max(-2).values.mean()).item())
def run_seq(seq,tr,method):
    frames=[rgbt(seq.frames[0][i],seq.frames[1][i]) for i in range(len(seq.frames[0]))]
    gt=np.asarray(seq.ground_truth_rect,dtype=np.float64); tr.initialize(frames[0],seq.init_info())
    _, init_rf, _ = sample_target(frames[0], gt[0].tolist(), tr.params.template_factor, output_sz=tr.params.template_size)
    slots=[{'slot':0,'bbox':gt[0].tolist(),'template':tr.z_dict1,'rf':float(init_rf)}]
    anchor_state=box(tr.state); prev_s=clone(tr.prev_search); prev_t=clone(tr.prev_temporal_tokens); anchor_id=tr.frame_id
    captured=[]
    def hook(_m,_inp,out): captured.append(out.detach())
    h=tr.network.backbone.blocks[3].register_forward_hook(hook)
    rows=[]
    try:
      for t in range(len(frames)-1):
        if t>0:
          anchor_state=box(tr.state); prev_s=clone(tr.prev_search); prev_t=clone(tr.prev_temporal_tokens); anchor_id=tr.frame_id
        if len(slots)==4:
          info=seq.frame_info(t+1); info.update(sequence_name=seq.name,frame_num=t+1,gt_bbox=gt[t+1])
          slot_iou={}; slot_keys=[]; query=None
          _, search_rf, _ = sample_target(frames[t+1], anchor_state.tolist(), tr.params.search_factor, output_sz=tr.params.search_size)
          for ent in slots:
            tr.state=anchor_state.tolist(); tr.prev_search=clone(prev_s); tr.prev_temporal_tokens=clone(prev_t); tr.frame_id=anchor_id; tr.z_dict1=ent['template']; captured.clear()
            out=tr.track(frames[t+1],info); slot_iou[str(ent['slot'])]=iou(box(out['target_bbox']),gt[t+1])
            if len(captured)<2: raise RuntimeError('block-3 hook did not capture RGB/TIR outputs')
            # Hook is called once per modality; 64 template and 256 search tokens.
            feats=[captured[0],captured[1]]
            keys=[]
            for f in feats: keys.append(roi_tokens(f[:,:64,:],box(ent['bbox']),float(tr.params.template_size)/float(ent['rf']),float(ent['rf']),8,8))
            slot_keys.append(keys)
            if ent['slot']==0:
                # Search crop geometry is recovered from the track input state.
                query=[roi_tokens(f[:,64:,:],anchor_state,float(tr.params.search_size)/float(search_rf),float(search_rf),16,8) for f in feats]
          if query is None: raise RuntimeError('active slot missing')
          rscore=[]; modality_scores=[[],[]]
          for j in range(len(slots)):
            sr=scores(query[0],slot_keys[j][0],method); st=scores(query[1],slot_keys[j][1],method)
            modality_scores[0].append(sr); modality_scores[1].append(st); rscore.append((sr+st)/2.)
          best=max(slot_iou,key=slot_iou.get); chosen=str(slots[int(np.argmax(rscore))]['slot'])
          cand_slots={str(slots[int(np.argmax(modality_scores[m]))]['slot']) for m in (0,1)}
          rows.append({'sequence':seq.name,'frame':int(t),'method':method,'valid_slots':[int(x['slot']) for x in slots],
                       'J_by_slot':slot_iou,'J_oracle':float(slot_iou[best]),'J_selected':float(slot_iou[chosen]),
                       'action_regret':float(slot_iou[best]-slot_iou[chosen]),'oracle_slot':int(best),
                       'selected_slot':int(chosen),'scores':rscore,'modality_scores':modality_scores,'active_slot':0,
                       'candidate_set_hit':bool(chosen==best),'oracle_in_candidate_set':bool(best in cand_slots)})
        tr.state=anchor_state.tolist(); tr.prev_search=clone(prev_s); tr.prev_temporal_tokens=clone(prev_t); tr.frame_id=anchor_id; tr.z_dict1=slots[0]['template']
        info=seq.frame_info(t+1); info.update(sequence_name=seq.name,frame_num=t+1,gt_bbox=gt[t+1]); tr.track(frames[t+1],info)
        wf=t>0 and t%25==0
        if wf:
          wb=anchor_state
          tm,_rf=template(tr,frames[t],wb)
          if len(slots)<4: slots.append({'slot':len(slots),'bbox':box(wb).tolist(),'template':tm,'rf':float(_rf)})
          else: slots=[slots[0]]+slots[2:]+[{'slot':max(x['slot'] for x in slots[1:])+1,'bbox':box(wb).tolist(),'template':tm,'rf':float(_rf)}]
    finally: h.remove()
    return rows
def main():
    p=argparse.ArgumentParser(); p.add_argument('--method',choices=['r0','r1'],required=True); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--run-id',required=True); p.add_argument('--sequence-list',required=True); p.add_argument('--split',default='train'); p.add_argument('--max-sequences',type=int,default=0); p.add_argument('--output-root',default='output/experiments/apsr-r01'); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    out=Path(a.output_root)/a.run_id; out.mkdir(parents=True,exist_ok=a.resume); seqs=names(a)
    api=Tracker('tbsi_track',a.config,a.split,a.run_id); par=api.get_parameters(); par.checkpoint=os.path.abspath(a.checkpoint); par.debug=0; tr=api.create_tracker(par)
    rows=[]
    if a.resume:
      for f in out.glob('*.jsonl'): rows += [json.loads(x) for x in open(f) if x.strip()]
    for s in seqs:
      f=out/(s.name+'.jsonl')
      if a.resume and f.exists() and f.stat().st_size: print(json.dumps({'sequence':s.name,'status':'resume-skip'}),flush=True); continue
      st=time.time(); rs=run_seq(s,tr,a.method); rows+=rs
      with open(f,'w') as w:
        for r in rs: w.write(json.dumps(r,ensure_ascii=False)+'\n')
      print(json.dumps({'sequence':s.name,'frames':len(rs),'seconds':round(time.time()-st,2)}),flush=True)
    reg=np.array([r['action_regret'] for r in rows],float); seqnames=sorted(set(r['sequence'] for r in rows)); sm=np.array([np.mean([r['action_regret'] for r in rows if r['sequence']==s]) for s in seqnames])
    rng=np.random.default_rng(42); b=np.array([rng.choice(sm,len(sm),replace=True).mean() for _ in range(20000)]) if len(sm) else np.array([0.])
    summary={'status':'ok','method':a.method.upper(),'run_id':a.run_id,'config':a.config,'checkpoint':os.path.abspath(a.checkpoint),'sequence_list':os.path.abspath(a.sequence_list),'sequence_manifest_sha256':hashlib.sha256(Path(a.sequence_list).read_bytes()).hexdigest(),'sequences':seqnames,'frames':len(reg),'mean_action_regret':float(reg.mean()) if len(reg) else 0.,'median_action_regret':float(np.median(reg)) if len(reg) else 0.,'sequence_mean_action_regret':float(sm.mean()) if len(sm) else 0.,'sequence_bootstrap_ci95':[float(x) for x in np.quantile(b,[.025,.975])],'oracle_candidate_set_recall':float(np.mean([r['oracle_in_candidate_set'] for r in rows])) if rows else 0.,'selected_oracle_agreement':float(np.mean([r['candidate_set_hit'] for r in rows])) if rows else 0.}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)); print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
