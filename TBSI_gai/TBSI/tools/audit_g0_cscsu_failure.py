#!/usr/bin/env python3
"""Read-only frame audit for Cscsu failure attribution on one sequence."""
import argparse, json, os, sys
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.test.utils.load_text import load_text
from lib.train.data.processing_utils import sample_target

def read_frame(seq, n):
    rgb = cv2.cvtColor(cv2.imread(seq.frames[0][n]), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(seq.frames[1][n]), cv2.COLOR_BGR2RGB)
    return np.concatenate([rgb, tir], axis=2)

def crop_info(state, factor, size, gt):
    cx = float(state[0]) + float(state[2]) / 2.; cy = float(state[1]) + float(state[3]) / 2.
    side = float(size) / float(factor); x0 = cx - side / 2.; y0 = cy - side / 2.
    x, y, w, h = [float(v) for v in gt]
    px, py, pw, ph = (x-x0)*factor, (y-y0)*factor, w*factor, h*factor
    inside = bool(0 <= px+pw/2 <= size and 0 <= py+ph/2 <= size)
    return {'crop_x': x0, 'crop_y': y0, 'resize_factor': float(factor), 'search_size': int(size),
            'projected_gt_xywh': [px, py, pw, ph], 'target_center_in_crop': inside}

def box_iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix1, iy1 = max(ax,bx), max(ay,by); ix2, iy2 = min(ax+aw,bx+bw), min(ay+ah,by+bh)
    inter = max(0., ix2-ix1)*max(0.,iy2-iy1); union=aw*ah+bw*bh-inter
    return inter/union if union > 0 else 0.

def residual_ratio(adapter, features, key):
    device = next(adapter.parameters()).device
    early = torch.from_numpy(features[key]).float().to(device)
    late_key = 'late_input_rgb' if key == 'early_rgb' else 'late_input_tir'
    late = torch.from_numpy(features[late_key]).float().to(device)
    if early.ndim == 2:
        early = early.unsqueeze(0)
    if late.ndim == 2:
        late = late.unsqueeze(0)
    b, n, c = early.shape; side = int(n ** .5)
    fmap = early.transpose(1, 2).reshape(b, c, side, side)
    selected = adapter.scsu(fmap).flatten(2).transpose(1, 2)
    proj = adapter.rgb_proj if key == 'early_rgb' else adapter.tir_proj
    gamma = adapter.gamma_rgb if key == 'early_rgb' else adapter.gamma_tir
    residual = float(gamma.detach()) * torch.nn.functional.linear(selected, proj.weight.detach())
    return float(residual.norm() / late.norm().clamp_min(1e-8)), float(residual.norm()), float(late.norm())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--c0-pred',required=True)
    ap.add_argument('--cscsu-pred',required=True); ap.add_argument('--sequence',default='blkboywillstand'); ap.add_argument('--output',required=True)
    args=ap.parse_args(); os.environ['TBSI_INFER_FP16']='0'
    ds=LasHeRDataset(split='trainingset', sequence_list_file=str(ROOT/'experiments/proxy/lasher_rpp_v1/g0_cplain_20261004/cplain_eval_sequences.txt')).get_sequence_list()
    seq=next(s for s in ds if s.name==args.sequence)
    params=parameters('G0_CSCSU_DIAGNOSTIC_seed42'); params.debug=0; params.save_all_boxes=False; params.checkpoint=args.checkpoint
    tracker=TBSITrack(params,'lasher_train'); tracker.diagnostic=True; tracker.diagnostic_early_block=2; tracker.diagnostic_late_block=9
    adapter=tracker.network.backbone.early_search_residual
    c0=np.loadtxt(args.c0_pred,delimiter='\t',ndmin=2); cs=np.loadtxt(args.cscsu_pred,delimiter='\t',ndmin=2)
    tracker.initialize(read_frame(seq,0), {'init_bbox':seq.ground_truth_rect[0]})
    rows=[]
    for fn in range(1,len(seq.frames[0])):
        state_before=list(tracker.state); image=read_frame(seq,fn)
        resize=sample_target(image,state_before,params.search_factor,output_sz=params.search_size)[1]
        result=tracker.track(image, {'sequence_name':seq.name,'frame_num':fn,'gt_bbox':seq.ground_truth_rect[fn]})
        f=getattr(tracker,'_last_diagnostic_features',None)
        row={'frame':fn,'c0_box':c0[fn].tolist(),'cscsu_box':cs[fn].tolist(),
             'box_max_abs_diff':float(np.max(np.abs(c0[fn]-cs[fn]))),
             'c0_iou':box_iou(c0[fn],seq.ground_truth_rect[fn]),
             'cscsu_iou':box_iou(cs[fn],seq.ground_truth_rect[fn]),
             'iou_delta_cscsu_minus_c0':box_iou(cs[fn],seq.ground_truth_rect[fn])-box_iou(c0[fn],seq.ground_truth_rect[fn]),
             'state_before':state_before,'crop':crop_info(state_before,resize,params.search_size,seq.ground_truth_rect[fn])}
        if f is not None:
            arrays={k:f[k][0].float().cpu().numpy() for k in ('early_rgb','early_tir','late_input_rgb','late_input_tir')}
            row['rgb_residual_over_late'],row['rgb_residual_norm'],row['rgb_late_norm']=residual_ratio(adapter,arrays,'early_rgb')
            row['tir_residual_over_late'],row['tir_residual_norm'],row['tir_late_norm']=residual_ratio(adapter,arrays,'early_tir')
        rows.append(row)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/'frame_rows.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    def first(pred):
        return next((r for r in rows if pred(r)), None)
    summary={'sequence':args.sequence,'frames':len(rows),
             'first_box_diff_gt_1px':first(lambda r:r['box_max_abs_diff']>1.),
             'first_box_diff_gt_5px':first(lambda r:r['box_max_abs_diff']>5.),
             'first_iou_delta_le_minus_005':first(lambda r:r['iou_delta_cscsu_minus_c0']<=-.05),
             'first_iou_delta_le_minus_020':first(lambda r:r['iou_delta_cscsu_minus_c0']<=-.20),
             'target_center_outside_crop_frames':sum(not r['crop']['target_center_in_crop'] for r in rows),
             'max_rgb_residual_over_late':max(r.get('rgb_residual_over_late',0.) for r in rows),
             'max_tir_residual_over_late':max(r.get('tir_residual_over_late',0.) for r in rows),
             'mean_rgb_residual_over_late':float(np.mean([r.get('rgb_residual_over_late',0.) for r in rows])),
             'mean_tir_residual_over_late':float(np.mean([r.get('tir_residual_over_late',0.) for r in rows]))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
