#!/usr/bin/env python3
"""Fit a deployment-visible top-K readout ranker with an explicit no-op."""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box

NAME='TBSI_E_MHSA_FULL_4090D_seed42'
CONFIG=ROOT/'experiments/tbsi_track'/(NAME+'.yaml')
CHECKPOINT=ROOT/'output/experiments'/NAME/'checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG_SHA='fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
CHECKPOINT_SHA='1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'

def sha256(p):
 h=hashlib.sha256(); f=open(p,'rb')
 for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 f.close(); return h.hexdigest()
def iou(a,b):
 ix=max(0.,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0])); iy=max(0.,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1])); inter=ix*iy; un=a[2]*a[3]+b[2]*b[3]-inter
 return inter/un if un>0 else 0.
def decode(net,tr,response,size,offset,state,shape,resize):
 box=net.box_head.cal_bbox(response,size,offset).view(-1,4).mean(0)*(tr.params.search_size/resize); cx=state[0]+.5*state[2]; cy=state[1]+.5*state[3]; half=.5*tr.params.search_size/resize
 return clip_box([float(box[0]+cx-half-.5*box[2]),float(box[1]+cy-half-.5*box[3]),float(box[2]),float(box[3])],shape[0],shape[1],margin=10)
def fit_ridge(x,y,alpha):
 mu=x.mean(0); sd=x.std(0); sd[sd<1e-6]=1.; z=np.clip((x-mu)/sd,-8,8.); ym=float(y.mean()); w=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-ym))
 return {'mean':mu.tolist(),'scale':sd.tolist(),'bias':ym,'weight':w.tolist()}
def pred(m,x):
 z=np.clip((x-np.asarray(m['mean']))/np.asarray(m['scale']),-8,8); return float(m['bias'])+z@np.asarray(m['weight'])

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--trace-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--topk',type=int,default=8); ap.add_argument('--alpha',type=float,default=10.); args=ap.parse_args()
 out=Path(args.output_dir).resolve()
 if out.exists() and any(out.iterdir()): raise RuntimeError('refusing nonempty output')
 if sha256(CONFIG)!=CONFIG_SHA or sha256(CHECKPOINT)!=CHECKPOINT_SHA: raise RuntimeError('protected M1 hash mismatch')
 data=np.load(args.features,allow_pickle=False); req={'fused','sequence','frame','label'}
 if not req.issubset(set(data.files)): raise RuntimeError('missing feature fields')
 trace={}
 for p in Path(args.trace_dir).glob('*/frames.jsonl'):
  for line in p.read_text().splitlines():
   if line.strip(): r=json.loads(line); trace[(p.parent.name,int(r['frame_id']))]=r
 seqs=sorted(set(str(v) for v in data['sequence'])); ranked=sorted(seqs,key=lambda s:hashlib.sha256(('ranker20261013:'+s).encode()).hexdigest()); fit=set(ranked[:max(1,int(len(seqs)*2/3))]); held=set(ranked)-fit
 listpath=out.parent/'ranker_sequence_list.txt'; listpath.parent.mkdir(parents=True,exist_ok=True); listpath.write_text('\n'.join(seqs)+'\n'); seqmap={s.name:s for s in LasHeRDataset('trainingset',str(listpath)).get_sequence_list()}
 os.environ['TBSI_INFER_FP16']='0'; torch.set_num_threads(2); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
 params=parameters(NAME); params.debug=0; params.save_all_boxes=False; params.checkpoint=str(CHECKPOINT); tr=TBSITrack(params,'m1_topk_ranker'); net=tr.network.eval(); rows=[]; xrows=[]; yrows=[]
 with torch.no_grad():
  for i in range(len(data['sequence'])):
   seq=str(data['sequence'][i]); frame=int(data['frame'][i]); t=trace[(seq,frame)]; state=np.asarray(t['tracker_state_before'],float); gt=np.asarray(t['gt_box'],float)
   rgb=cv2.imread(seqmap[seq].frames[0][frame],cv2.IMREAD_COLOR); tir=cv2.imread(seqmap[seq].frames[1][frame],cv2.IMREAD_COLOR); image=np.concatenate((cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB),cv2.cvtColor(tir,cv2.COLOR_BGR2RGB)),2); _,resize,_=sample_target(image,state,params.search_factor,output_sz=params.search_size)
   fused=torch.as_tensor(data['fused'][i].transpose(2,0,1)[None],dtype=torch.float32,device='cuda'); score,_,size,offset=net.box_head(fused); response=tr.output_window*score; flat=response.flatten(1)[0]; order=torch.argsort(flat,descending=True).tolist()[:args.topk]; vals=flat.detach().cpu().numpy(); p=np.clip(vals,1e-8,None); p=p/p.sum(); ent=float(-(p*np.log(p)).sum()/np.log(len(p))); topgap=float(vals[order[0]]-vals[order[1]]) if len(order)>1 else 0.; base=iou(decode(net,tr,response,size,offset,state,image.shape,resize),gt)
   stateq=np.array([float(vals.max()),topgap,ent,float(vals.mean()),float(vals.std())],float)
   for rank,idx in enumerate(order,1):
    x,y=idx%16,idx//16; resp=response.clone(); resp[:,:,y,x]=flat.max()+1e-3; pred_iou=iou(decode(net,tr,resp,size,offset,state,image.shape,resize),gt); off=offset[0,:,y,x].detach().cpu().numpy(); sz=size[0,:,y,x].detach().cpu().numpy(); cand=np.r_[stateq,rank/args.topk,x/15.,y/15.,float(vals[idx]),float(vals[order[0]]-vals[idx]),off,sz]
    xrows.append(cand); yrows.append(pred_iou-base); rows.append({'sequence':seq,'frame':frame,'label':str(data['label'][i]),'rank':rank,'candidate_index':idx,'base_iou':base,'candidate_iou':pred_iou,'utility':pred_iou-base,'split':'fit' if seq in fit else 'heldout'})
   if (i+1)%16==0: print('states={}/{}'.format(i+1,len(data['sequence'])),flush=True)
 xarr=np.asarray(xrows); yarr=np.asarray(yrows); train=np.asarray([r['split']=='fit' for r in rows]); model=fit_ridge(xarr[train],yarr[train],args.alpha); held_rows=[]; grouped={}
 for i,r in enumerate(rows):
  if r['split']!='heldout': continue
  score=pred(model,xarr[i]); grouped.setdefault((r['sequence'],r['frame']),[]).append((score,float(r['utility']),r))
 for key,items in grouped.items():
  best=max(items,key=lambda z:z[0]); no_op=0.; actual=best[1] if best[0]>0 else no_op; rr=best[2]; held_rows.append({'sequence':key[0],'frame':key[1],'predicted_utility':best[0],'selected_utility':actual,'rank':rr['rank'],'abstain':int(best[0]<=0),'oracle_utility':max([0.]+[v[1] for v in items])})
 out.mkdir(parents=True,exist_ok=True)
 for name,content in [('per_candidate.csv',rows),('heldout_selection.csv',held_rows)]:
  with (out/name).open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(content[0])); w.writeheader(); w.writerows(content)
 seqnames=sorted(set(r['sequence'] for r in held_rows)); selected=np.array([r['selected_utility'] for r in held_rows]); oracle=np.array([r['oracle_utility'] for r in held_rows]);
 per={s:float(np.mean([r['selected_utility'] for r in held_rows if r['sequence']==s])) for s in seqnames}; op={s:float(np.mean([r['oracle_utility'] for r in held_rows if r['sequence']==s])) for s in seqnames}; delta=np.array([per[s] for s in seqnames]); od=np.array([op[s] for s in seqnames])
 summary={'heldout_states':len(held_rows),'heldout_sequences':len(seqnames),'fit_sequences':sorted(fit),'heldout_sequence_names':seqnames,'selected_mean_utility':float(selected.mean()),'selected_sequence_macro_utility':float(delta.mean()),'oracle_sequence_macro_utility':float(od.mean()),'oracle_fraction':float(delta.mean()/od.mean()) if od.mean()>0 else 0.,'harmful_intervention_rate':float(np.mean(selected< -1e-4)),'abstention_rate':float(np.mean([r['abstain'] for r in held_rows])),'sequence_positive':int((delta>0).sum()),'sequence_negative':int((delta<0).sum()),'model':model,'features':int(xarr.shape[1]),'alpha':args.alpha,'id':'M1-READOUT-TOPK-RANKER-20261013','status':'RANKER-FIT-COMPLETE','limitations':['fixed-state utility, not closed-loop tracking','24-sequence cohort and 8 heldout sequences','candidate actions use response peak override; no offset/size correction']}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
