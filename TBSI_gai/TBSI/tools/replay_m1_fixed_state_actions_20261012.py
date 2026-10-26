#!/usr/bin/env python3
"""Fixed-crop/state causal action replay for the protected M1 checkpoint.

This is deliberately offline: every arm sees the same saved search tokens,
template tokens, tracker state and ground-truth frame. It does not update the
tracker state and therefore cannot be interpreted as a closed-loop result.
"""
import argparse, csv, hashlib, json, math, os, random, sys
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
from lib.models.layers.single_signal_prefusion import SingleSignalPrefusion, image_features_from_patch
from lib.utils.box_ops import clip_box

NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
CONFIG = ROOT / 'experiments/tbsi_track' / (NAME + '.yaml')
CHECKPOINT = ROOT / 'output/experiments' / NAME / 'checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG_SHA = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
CHECKPOINT_SHA = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'
ARTIFACT = ROOT / 'output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/train72_screen_v1'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def iou(a, b):
    ix = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0],b[0]))
    iy = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1],b[1]))
    inter = ix * iy
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.

def box_from_output(network, tracker, out, state, shape):
    response = tracker.output_window * out['score_map']
    box = network.box_head.cal_bbox(response, out['size_map'], out['offset_map']).view(-1,4).mean(0)
    # The saved replay uses the original search crop geometry, supplied by caller.
    return box

def decode(network, tracker, out, state, image_shape, resize):
    box = box_from_output(network, tracker, out, state, image_shape)
    box = box * (tracker.params.search_size / resize)
    cx, cy = state[0] + .5*state[2], state[1] + .5*state[3]
    half = .5 * tracker.params.search_size / resize
    xywh = [float(box[0] + cx-half-.5*box[2]), float(box[1] + cy-half-.5*box[3]),
            float(box[2]), float(box[3])]
    return clip_box(xywh, image_shape[0], image_shape[1], margin=10)

def action(x, direction, alpha):
    c = x.shape[1] // 2
    y = x.clone()
    if direction > 0:
        y[:, :c] *= 1.0 + alpha; y[:, c:] *= 1.0 - alpha
    elif direction < 0:
        y[:, :c] *= 1.0 - alpha; y[:, c:] *= 1.0 + alpha
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--trace-dir', required=True, action='append',
                    help='repeat for each frozen trace batch represented in features')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--alpha', type=float, default=.15,
                    help='symmetric RGB/TIR action magnitude; fixed before evaluation')
    args = ap.parse_args()
    outdir = Path(args.output_dir).resolve()
    if outdir.exists() and any(outdir.iterdir()): raise RuntimeError('refusing nonempty output')
    if sha256(CONFIG) != CONFIG_SHA or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('protected M1 hash mismatch')
    data = np.load(args.features, allow_pickle=True)
    required = {'concat','template_rgb','template_tir','template_fg','sequence','frame','label','gt_xy'}
    if not required.issubset(set(data.files)): raise RuntimeError('feature asset lacks template/state fields')
    trace_roots = [Path(p) for p in args.trace_dir]
    trace = {}
    for trace_root in trace_roots:
        for p in trace_root.glob('*/frames.jsonl'):
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line); key=(p.parent.name, int(r['frame_id']))
                    if key in trace and trace[key] != r: raise RuntimeError('conflicting duplicate trace '+str(key))
                    trace[key] = r
    os.environ['TBSI_INFER_FP16'] = '0'
    torch.set_num_threads(2); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    params = parameters(NAME); params.debug = 0; params.save_all_boxes = False; params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'm1_fixed_state_action_replay'); network = tracker.network.eval()
    seq_names=sorted(set(str(v) for v in data['sequence']))
    list_path=outdir.parent / 'fixed_state_sequence_list.txt'
    list_path.parent.mkdir(parents=True,exist_ok=True)
    list_path.write_text('\n'.join(seq_names)+'\n')
    dataset = LasHeRDataset('trainingset', str(list_path)).get_sequence_list()
    seqmap = {s.name:s for s in dataset}
    artifact = str(ARTIFACT)
    if not (0.0 <= args.alpha <= 0.5):
        raise ValueError('--alpha must be in [0, .5]')
    gates = {m: SingleSignalPrefusion(m, artifact, alpha=args.alpha).cuda() for m in ('s1','s2','s3','s1s2','s1s2s3')}
    rng = random.Random(args.seed)
    rows=[]
    for i in range(len(data['sequence'])):
        seq, frame = str(data['sequence'][i]), int(data['frame'][i]); key=(seq,frame)
        if key not in trace or seq not in seqmap: raise RuntimeError('missing trace state '+str(key))
        tr=trace[key]; state=np.asarray(tr['tracker_state_before'],dtype=np.float64); gt=np.asarray(tr['gt_box'],dtype=np.float64)
        rgb=cv2.imread(seqmap[seq].frames[0][frame], cv2.IMREAD_COLOR)
        tir=cv2.imread(seqmap[seq].frames[1][frame], cv2.IMREAD_COLOR)
        if rgb is None or tir is None or rgb.shape[:2] != tir.shape[:2]: raise RuntimeError('missing/misaligned image '+str(key))
        image=np.concatenate((cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB),cv2.cvtColor(tir,cv2.COLOR_BGR2RGB)),2)
        crop, resize, _ = sample_target(image, state, params.search_factor, output_sz=params.search_size)
        x=torch.as_tensor(data['concat'][i].transpose(2,0,1)[None],dtype=torch.float32,device='cuda')
        with torch.no_grad():
            fused=network.tbsi_fuse_search(x); score,_bbox,size,offset=network.box_head(fused); base={'score_map':score,'size_map':size,'offset_map':offset}
        trgb=torch.as_tensor(data['template_rgb'][i][None],dtype=torch.float32,device='cuda')
        ttir=torch.as_tensor(data['template_tir'][i][None],dtype=torch.float32,device='cuda')
        cat=torch.cat((trgb,x[:,:768].flatten(2).transpose(1,2),ttir,x[:,768:].flatten(2).transpose(1,2)),1)
        for gate in gates.values():
            gate.template_fg=torch.as_tensor(data['template_fg'][i],dtype=torch.float32,device='cuda')
        gates['s3'].runtime_s3_features=image_features_from_patch(crop,state)
        for gate in gates.values():
            if gate.components is not None:
                for component in gate.components:
                    component.template_fg=gate.template_fg
                    component.runtime_s3_features=gates['s3'].runtime_s3_features
        predictions={}
        for name,gate in gates.items():
            rgb=cat[:,64:320,:]; tir=cat[:,384:640,:]
            pred=gate._predict(cat,rgb,tir) if gate.components is None else np.mean([c._predict(cat,rgb,tir) for c in gate.components])
            if gate.components is not None:
                ds=[1 if c._predict(cat,rgb,tir)>=0 else -1 for c in gate.components]; direction=1 if sum(ds)>0 else (-1 if sum(ds)<0 else 0)
            else: direction=1 if pred>0 else (-1 if pred<0 else 0)
            predictions[name]=(direction,float(pred))
        rand_dir=1 if rng.random() >= .5 else -1
        dirs={'no_op':0,'fixed_rgb':1,'fixed_tir':-1,'random':rand_dir}
        for name,(d,p) in predictions.items(): dirs[name]=d
        dirs['joint_lowconf_noop']=0 if abs(predictions['s1s2s3'][1]) < .005 else predictions['s1s2s3'][0]
        for arm,d in dirs.items():
            inp=action(x,d,args.alpha)
            with torch.no_grad():
                f=network.tbsi_fuse_search(inp); sc,_bbox,sz,of=network.box_head(f); o={'score_map':sc,'size_map':sz,'offset_map':of}
            pred_box=decode(network,tracker,o,state,image.shape,resize)
            hann=(sc*tracker.output_window)[0,0].float(); gtxy=data['gt_xy'][i]; gx,gy=int(gtxy[0]),int(gtxy[1]); flat=hann.flatten(); peak=int(flat.argmax().item()); px,py=peak%16,peak//16; rank=int((flat>hann[gy,gx]).sum().item()+1)
            gt_cx,gt_cy=gt[0]+.5*gt[2],gt[1]+.5*gt[3]
            pred_cx,pred_cy=pred_box[0]+.5*pred_box[2],pred_box[1]+.5*pred_box[3]
            rows.append({'sequence':seq,'frame':frame,'label':str(data['label'][i]),'arm':arm,'direction':d,'signal_prediction':predictions.get(arm,(np.nan,np.nan))[1] if arm in predictions else np.nan,'iou':iou(pred_box,gt),'peak_distance_tokens':float(math.hypot(px+.5-float(gtxy[0]),py+.5-float(gtxy[1]))),'decoded_center_error_px':float(math.hypot(pred_cx-gt_cx,pred_cy-gt_cy)),'decoded_size_rel_l1':float((abs(pred_box[2]-gt[2])+abs(pred_box[3]-gt[3]))/(gt[2]+gt[3]+1e-8)),'hann_peak_is_gt_cell':int(px==gx and py==gy),'gt_rank':rank})
    outdir.mkdir(parents=True); fields=list(rows[0]);
    with (outdir/'per_state_actions.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    arms=list(dict.fromkeys(r['arm'] for r in rows)); seqs=sorted(set(r['sequence'] for r in rows)); summary=[]
    per_sequence={arm:{s:float(np.mean([r['iou'] for r in rows if r['arm']==arm and r['sequence']==s])) for s in seqs} for arm in arms}
    rng_np=np.random.default_rng(args.seed); boot_n=20000
    for arm in arms:
        vals=[r['iou'] for r in rows if r['arm']==arm]
        diffs=np.asarray([per_sequence[arm][s]-per_sequence['no_op'][s] for s in seqs]) if arm!='no_op' else np.zeros(len(seqs))
        boot=np.asarray([rng_np.choice(diffs,size=len(diffs),replace=True).mean() for _ in range(boot_n)])
        summary.append({'arm':arm,'n_states':len(vals),'n_sequences':len(seqs),'mean_iou':float(np.mean(vals)),'median_iou':float(np.median(vals)),'mean_gt_rank':float(np.mean([r['gt_rank'] for r in rows if r['arm']==arm])),'delta_iou_vs_no_op_sequence_mean':float(diffs.mean()),'paired_sequence_bootstrap95_low':float(np.quantile(boot,.025)),'paired_sequence_bootstrap95_high':float(np.quantile(boot,.975)),'sequence_wins_vs_no_op':int((diffs>0).sum()),'sequence_losses_vs_no_op':int((diffs<0).sum()),'sequence_ties_vs_no_op':int((diffs==0).sum())})
    with (outdir/'summary.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    meta={'id':'M1-UTILITY-NOOP-RHO-SWEEP-20261013','parent':NAME,'features_sha256':sha256(Path(args.features)),'config_sha256':sha256(CONFIG),'checkpoint_sha256':sha256(CHECKPOINT),'trace_dirs':[str(p.resolve()) for p in trace_roots],'trace_manifest_sha256':[sha256(p/'trace_sequences.txt') for p in trace_roots],'seed':args.seed,'alpha':args.alpha,'n_states':len(data['sequence']),'n_sequences':len(seqs),'arms':arms,'low_confidence_threshold':.005,'interpretation':'same crop/state/head; offline decoded-IoU action utility, not closed-loop tracking','limitations':['working-tree replay, not archived frame parity','fixed states are selected from prior diagnostic cohort','S3 uses the saved crop image and is GT-state replay only','sequence-bootstrap applies to a selected diagnostic cohort, not full LasHeR']}
    (outdir/'summary.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
