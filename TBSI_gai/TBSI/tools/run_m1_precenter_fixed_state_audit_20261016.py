#!/usr/bin/env python3
"""Compare M1 and pre-CENTER residual on identical frozen states/crops."""
import csv, json, os, sys
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

EVENTS = ROOT / 'output/experiments/M1_CANDIDATE_RECALL_READOUT_20261010/analysis/per_event.csv'
TRACE = ROOT / 'output/experiments/M1_CANDIDATE_RECALL_READOUT_20261010/trace'
M1_CFG = 'TBSI_E_MHSA_FULL_4090D_seed42'
CAND_CFG = 'M1_PRECENTER_GATED_RESIDUAL_PROXY_20261016'
M1_CKPT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
CAND_CKPT = ROOT / 'output/experiments/M1_PRECENTER_GATED_RESIDUAL_PROXY_20261016/experiments/M1_PRECENTER_GATED_RESIDUAL_PROXY_20261016/checkpoints/TBSITrack_ep0003.pth.tar'
OUT = ROOT / 'output/experiments/M1_PRECENTER_GATED_RESIDUAL_PROXY_20261016/fixed_state_audit'
CANDIDATE_ROWS = []

def iou(a, b):
    ix=max(0.,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0])); iy=max(0.,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    inter=ix*iy; union=a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union>0 else 0.

def decode_candidate(tracker, network, response, size, offset, state, shape, resize):
    box=network.box_head.cal_bbox(response,size,offset).view(-1,4).mean(0)
    box=box*(tracker.params.search_size/resize)
    cx=state[0]+.5*state[2]; cy=state[1]+.5*state[3]; half=.5*tracker.params.search_size/resize
    return [float(box[0]+cx-half-.5*box[2]),float(box[1]+cy-half-.5*box[3]),float(box[2]),float(box[3])]

def load_events():
    with EVENTS.open() as f: rows=list(csv.DictReader(f))
    return rows

def trace_row(seq, frame):
    path=TRACE/seq/'frames.jsonl'
    for line in path.read_text().splitlines():
        r=json.loads(line)
        if int(r['frame_id']) == int(frame): return r
    raise RuntimeError('missing trace '+seq+' '+str(frame))

def run_arm(cfg_name, checkpoint, events, seqmap):
    os.environ['TBSI_CHECKPOINT_OVERRIDE']=str(checkpoint)
    os.environ['TBSI_INFER_FP16']='0'; os.environ['TBSI_FIXED_STATE_DIAG']='1'
    params=parameters(cfg_name); params.debug=0; params.save_all_boxes=False; params.checkpoint=str(checkpoint)
    tracker=TBSITrack(params, 'm1_precenter_fixed_state_audit')
    rows=[]; current_seq=None
    for e in events:
        seq=e['sequence']; frame=int(e['frame']); tr=trace_row(seq,frame); state=np.asarray(tr['tracker_state_before'],dtype=np.float64)
        gt=np.asarray(tr['gt_box'],dtype=np.float64); ds=seqmap[seq]
        if current_seq != seq:
            first=np.concatenate((cv2.cvtColor(cv2.imread(ds.frames[0][0]),cv2.COLOR_BGR2RGB),cv2.cvtColor(cv2.imread(ds.frames[1][0]),cv2.COLOR_BGR2RGB)),2)
            tracker.initialize(first, {'init_bbox':np.asarray(ds.ground_truth_rect[0],dtype=np.float64),'sequence_name':seq})
            current_seq=seq
        tracker.state=state.tolist()
        rgb=cv2.cvtColor(cv2.imread(ds.frames[0][frame]),cv2.COLOR_BGR2RGB); tir=cv2.cvtColor(cv2.imread(ds.frames[1][frame]),cv2.COLOR_BGR2RGB)
        image=np.concatenate((rgb,tir),2)
        out=tracker.track(image, {'gt_bbox':gt,'sequence_name':seq,'frame_num':frame})
        d=tracker._last_fixed_state_diag
        response_map=tracker.output_window.cuda()*d['score_map'].cuda()
        response=response_map[0,0]
        flat=response.flatten(); top=int(flat.argmax()); tx,ty=top%16,top//16
        st=state; resize=float(d['resize_factor']); half=.5*params.search_size/resize
        cx=st[0]+.5*st[2]; cy=st[1]+.5*st[3]; cropx=cx-half; cropy=cy-half
        gcx=(gt[0]+.5*gt[2]-cropx)*resize/16.; gcy=(gt[1]+.5*gt[3]-cropy)*resize/16.
        gx,gy=max(0,min(15,int(gcx))),max(0,min(15,int(gcy))); gtidx=gy*16+gx
        gt_score = response[gy, gx]
        rank=int((flat>gt_score).sum().item()+1)
        pred=np.asarray(out['target_bbox'],dtype=np.float64)
        topk=torch.argsort(flat,descending=True)[:8].tolist(); top2=float(flat[topk[1]])
        oracle=-1.; center_rule=-1.; center_idx=-1
        for idx in topk:
            x,y=idx%16,idx//16; rr=response_map.clone(); rr[0,0,y,x]=flat[topk[0]]+1e-4
            val=iou(decode_candidate(tracker,tracker.network,rr,d['size_map'].cuda(),d['offset_map'].cuda(),state,d['image_shape'],resize),gt)
            oracle=max(oracle,val)
            if center_idx<0 or (x-7.5)**2+(y-7.5)**2 < (center_idx%16-7.5)**2+(center_idx//16-7.5)**2: center_idx=idx; center_rule=val
            CANDIDATE_ROWS.append({'arm':cfg_name,'sequence':seq,'frame':frame,'state_label':e['state_label'],'rank':topk.index(idx)+1,'x':x,'y':y,'score':float(flat[idx]),'margin':margin if 'margin' in locals() else float(flat[topk[0]]-flat[topk[1]]),'center_dist':float(((x-7.5)**2+(y-7.5)**2)**0.5/10.6066),'size_w':float(d['size_map'][0,0,y,x]),'size_h':float(d['size_map'][0,1,y,x]),'offset_x':float(d['offset_map'][0,0,y,x]),'offset_y':float(d['offset_map'][0,1,y,x]),'utility':float(val-iou(pred,gt))})
        # Conservative no-op fallback: only apply center-prior rerank when peak margin is small.
        margin=float(flat[topk[0]]-top2)
        row={'sequence':seq,'frame':frame,'state_label':e['state_label'],'top1_x':tx,'top1_y':ty,'gt_x':gx,'gt_y':gy,'gt_rank':rank,'top1_is_gt_cell':int(top==gtidx),'fixed_state_iou':iou(pred,gt),'top1_score':float(flat[top]),'gt_cell_score':float(gt_score),'top8_oracle_iou':oracle,'top8_center_prior_iou':center_rule,'top1_top2_margin':margin}
        for threshold in (0.0,0.002,0.005): row['top8_center_prior_noop_{}'.format(str(threshold).replace('.','p'))]=center_rule if margin < threshold else row['fixed_state_iou']
        rows.append(row)
    return rows

def main():
    events=load_events(); seqs=sorted({e['sequence'] for e in events}); list_path=OUT/'sequence_list.txt'; OUT.mkdir(parents=True,exist_ok=True); list_path.write_text('\n'.join(seqs)+'\n')
    ds=LasHeRDataset('trainingset',str(list_path)).get_sequence_list(); seqmap={s.name:s for s in ds}
    allrows=[]
    for arm,cfg,ckpt in [('m1',M1_CFG,M1_CKPT),('candidate',CAND_CFG,CAND_CKPT)]:
        for r in run_arm(cfg,ckpt,events,seqmap): r['arm']=arm; allrows.append(r)
    with (OUT/'per_event.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
    summary=[]
    for arm in ('m1','candidate'):
        sub=[r for r in allrows if r['arm']==arm]
        for label in ('all','matched_good','pre_onset','severe_onset'):
            q=sub if label=='all' else [r for r in sub if r['state_label']==label]
            item={'arm':arm,'label':label,'n':len(q),'fixed_state_iou':float(np.mean([r['fixed_state_iou'] for r in q])),'gt_cell_rank':float(np.mean([r['gt_rank'] for r in q])),'gt_cell_recall':float(np.mean([r['top1_is_gt_cell'] for r in q])),'score_margin_gt_top1':float(np.mean([r['gt_cell_score']-r['top1_score'] for r in q])),'top8_oracle_iou':float(np.mean([r['top8_oracle_iou'] for r in q])),'top8_center_prior_iou':float(np.mean([r['top8_center_prior_iou'] for r in q]))}
            for threshold in ('0p0','0p002','0p005'):
                item['center_prior_noop_'+threshold]=float(np.mean([r['top8_center_prior_noop_'+threshold] for r in q]))
            summary.append(item)
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    with (OUT/'candidate_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(CANDIDATE_ROWS[0])); w.writeheader(); w.writerows(CANDIDATE_ROWS)
    print(json.dumps({'events':len(events),'sequences':len(seqs),'output':str(OUT)},indent=2))

if __name__=='__main__': main()
