#!/usr/bin/env python3
"""Paired fixed-state analysis for M1 and query-conditioned template MHSA."""
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

MANIFEST = ROOT / 'output/experiments/M1_PRECISION_SCREEN_MANIFEST_20261020/manifest.json'
CKPT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
OUT = ROOT / 'output/experiments/M1_QUERY_UNCERTAINTY_FIXED25_ANALYSIS_20261020'

def iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    inter = max(0., min(ax2, bx2) - max(a[0], b[0])) * max(0., min(ay2, by2) - max(a[1], b[1]))
    union = max(0., a[2]) * max(0., a[3]) + max(0., b[2]) * max(0., b[3]) - inter
    return inter / union if union > 0 else 0.

def mask_for(gt, crop, n=16):
    x, y, side, _ = crop
    gx, gy, gw, gh = gt
    x1, x2 = (gx-x)*n/side, (gx+gw-x)*n/side
    y1, y2 = (gy-y)*n/side, (gy+gh-y)*n/side
    yy, xx = np.mgrid[:n, :n]
    m = ((xx+.5 >= x1) & (xx+.5 <= x2) & (yy+.5 >= y1) & (yy+.5 <= y2))
    if not m.any(): m[int(np.clip((y1+y2)/2,0,n-1)), int(np.clip((x1+x2)/2,0,n-1))] = True
    return m

def feature_stats(feat, mask):
    z = feat.detach().float()[0].permute(1,2,0).cpu().numpy()
    t, b = z[mask].mean(0), z[~mask]
    bn = np.linalg.norm(b, axis=1); bp = b[int(np.argmax(bn))]
    return {'feature_target_norm': float(np.linalg.norm(t)),
            'feature_background_peak_norm': float(np.linalg.norm(bp)),
            'feature_target_bg_l2': float(np.linalg.norm(t-bp)),
            'feature_target_bg_cosine': float(np.dot(t,bp)/(np.linalg.norm(t)*np.linalg.norm(bp)+1e-8))}

def read_pair(seq, frame, seqmap):
    s = seqmap[seq]
    a = cv2.cvtColor(cv2.imread(s.frames[0][frame]), cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(cv2.imread(s.frames[1][frame]), cv2.COLOR_BGR2RGB)
    return np.concatenate((a,b), axis=2)

def run_arm(mode, events, seqmap):
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(CKPT)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_TEMPLATE_UNCERTAINTY_MODE'] = mode
    if mode == 'query_soft':
        os.environ['TBSI_TEMPLATE_UNCERTAINTY_THRESHOLD'] = '0.5'
        os.environ['TBSI_TEMPLATE_UNCERTAINTY_TEMPERATURE'] = '0.15'
    params = parameters('TBSI_E_MHSA_FULL_4090D_seed42')
    params.debug = 0; params.save_all_boxes = False; params.checkpoint = str(CKPT)
    tracker = TBSITrack(params, 'm1_query_uncertainty_fixed25')
    net = tracker.network; cap = {}
    def fpre(_, inp): cap['pre'] = inp[0].detach().float().cpu()
    def fpost(_, inp, out): cap['post'] = out.detach().float().cpu()
    hooks = [net.tbsi_fuse_search.register_forward_pre_hook(fpre), net.tbsi_fuse_search.register_forward_hook(fpost)]
    rows = []
    for idx, e in enumerate(events, 1):
        seq, frame = e['sequence'], int(e['frame']); trace = e
        first = read_pair(seq, 0, seqmap)
        tracker.initialize(first, {'init_bbox': np.asarray(seqmap[seq].ground_truth_rect[0]), 'sequence_name': seq})
        tracker.state = list(map(float, trace['tracker_state_before']))
        tracker.prev_search = None; tracker.prev_temporal_tokens = None; tracker.prev_head_feature = None
        cap.clear()
        pred = tracker.track(read_pair(seq, frame, seqmap), {'gt_bbox': trace['gt_box'], 'sequence_name': seq, 'frame_num': frame})
        diag = tracker._last_fixed_state_diag
        score = diag['score_map'][0,0].numpy() * tracker.output_window[0,0].detach().float().cpu().numpy()
        m = mask_for(trace['gt_box'], trace['crop_xywh']); target = float(score[m].max()); bg = float(score[~m].max())
        gt_rank = 1 + int(np.sum(score[~m] > target)); top = np.unravel_index(np.argmax(score), score.shape)
        row = {'mode': mode, 'sequence': seq, 'role': e.get('role',''), 'frame': frame,
               'fixed_state_iou': iou(pred['target_bbox'], trace['gt_box']), 'gt_rank': gt_rank,
               'target_peak': target, 'background_peak': bg, 'peak_margin': target-bg,
               'top1_is_gt': bool(m[top]), 'pred_x': pred['target_bbox'][0], 'pred_y': pred['target_bbox'][1]}
        if 'pre' in cap and 'post' in cap:
            row.update({'fusion_input_norm': float(cap['pre'].norm()), 'fusion_output_norm': float(cap['post'].norm())})
            row.update(feature_stats(cap['post'], m))
        rows.append(row)
        print('[{}] {}/{} {} IoU={:.3f} rank={} margin={:.4f}'.format(mode, idx, len(events), seq, row['fixed_state_iou'], gt_rank, target-bg), flush=True)
    for h in hooks: h.remove()
    return rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads(MANIFEST.read_text()); selected = payload['selection'] if isinstance(payload,dict) else payload
    names = sorted({e['sequence'] for e in selected})
    dataset = LasHeRDataset('testingset', str(OUT / 'sequence_list.txt')) if False else None
    (OUT / 'sequence_list.txt').write_text('\n'.join(names)+'\n')
    dataset = LasHeRDataset('testingset', str(OUT / 'sequence_list.txt')).get_sequence_list()
    seqmap = {s.name:s for s in dataset}; events=[]
    for e in selected:
        events.append({'sequence':e['sequence'], 'frame':int(e['frame']), 'role':e.get('role',''),
                       'tracker_state_before':e['tracker_state_before'], 'gt_box':e['gt_box'], 'crop_xywh':e['crop_xywh']})
    off = run_arm('off', events, seqmap); query = run_arm('query_soft', events, seqmap)
    rows=[]
    for a,b in zip(off,query):
        r=dict(b); r['off_iou']=a['fixed_state_iou']; r['query_iou_delta']=b['fixed_state_iou']-a['fixed_state_iou']
        r['off_gt_rank']=a['gt_rank']; r['gt_rank_delta']=b['gt_rank']-a['gt_rank']; r['off_margin']=a['peak_margin']; r['margin_delta']=b['peak_margin']-a['peak_margin']; r['top1_flip']=a['top1_is_gt'] != b['top1_is_gt']; rows.append(r)
        for key in ('feature_target_norm','feature_background_peak_norm','feature_target_bg_l2','feature_target_bg_cosine','fusion_input_norm','fusion_output_norm'):
            if key in a and key in b:
                r['off_'+key] = a[key]
                r[key+'_delta'] = b[key] - a[key]
    with (OUT/'per_event.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    def summary(q):
        ds=[r['query_iou_delta'] for r in q]; return {'n':len(q),'mean_iou_delta':float(np.mean(ds)),'wins':sum(x>1e-6 for x in ds),'losses':sum(x<-1e-6 for x in ds),'mean_margin_delta':float(np.mean([r['margin_delta'] for r in q])),'mean_rank_delta':float(np.mean([r['gt_rank_delta'] for r in q]))}
    out={'all':summary(rows),'failure':summary([r for r in rows if r['role']=='failure']),'control':summary([r for r in rows if r['role']=='control'])}
    (OUT/'summary.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
