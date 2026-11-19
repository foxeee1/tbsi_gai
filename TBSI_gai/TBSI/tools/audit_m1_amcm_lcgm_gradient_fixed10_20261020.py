#!/usr/bin/env python3
"""Exact actor-loss directional gradients for LCGM on fixed M1 frames."""
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target
from lib.train.actors.tbsi_track import TBSITrackActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
from lib.utils.focal_loss import FocalLoss

CFG = 'M1_AMCM_LCGM_QUICK15_3EP_20261019'
CKPT = ROOT / 'output/experiments/M1_AMCM_LCGM_QUICK15_3EP_20261019/checkpoints/TBSITrack_ep0003.pth.tar'
MANIFEST = ROOT / 'output/experiments/M1_AMCM_LCGM_FIXED10_DIAG_20261019/fixed_event_manifest.json'
OUT = ROOT / 'output/experiments/M1_AMCM_LCGM_GRADIENT_AUDIT_20261020'


def read_pair(seq, frame, seqmap):
    item = seqmap[seq]
    v = cv2.cvtColor(cv2.imread(item.frames[0][frame]), cv2.COLOR_BGR2RGB)
    i = cv2.cvtColor(cv2.imread(item.frames[1][frame]), cv2.COLOR_BGR2RGB)
    if v is None or i is None:
        raise RuntimeError('frame read failed: {} {}'.format(seq, frame))
    return np.concatenate((v, i), axis=2)


def make_actor(tracker):
    class Settings:
        batchsize = 1
    objective = {'giou': giou_loss, 'l1': torch.nn.L1Loss(), 'focal': FocalLoss(),
                 'cls': torch.nn.BCEWithLogitsLoss()}
    weights = {'giou': float(tracker.cfg.TRAIN.GIOU_WEIGHT),
               'l1': float(tracker.cfg.TRAIN.L1_WEIGHT), 'focal': 1.0, 'cls': 1.0}
    return TBSITrackActor(tracker.network, objective, weights, Settings(), tracker.cfg)


def event_loss_and_grad(tracker, actor, event, seqmap):
    seq, frame = event['sequence'], int(event['frame'])
    trace = event
    image = read_pair(seq, frame, seqmap)
    state = list(map(float, trace['m1_state_before']))
    tracker.state = state
    crop, resize_factor, att_mask = sample_target(
        image, state, tracker.params.search_factor, output_sz=tracker.params.search_size)
    search = tracker.preprocessor.process(crop, att_mask, half=False)
    search_v = search.tensors[:, :3].detach().clone()
    search_i = search.tensors[:, 3:].detach().clone()
    templ_v = tracker.z_dict1.tensors[:, :3].detach().clone()
    templ_i = tracker.z_dict1.tensors[:, 3:].detach().clone()

    crop_x, crop_y, crop_side, _ = map(float, trace['crop_xywh'])
    gx, gy, gw, gh = map(float, trace['gt_box'])
    box = torch.tensor([[(gx-crop_x)/crop_side, (gy-crop_y)/crop_side,
                         gw/crop_side, gh/crop_side]], device='cuda', dtype=torch.float32)
    # Match the crop transform used by the tracker: square resize and normalized box labels.
    actor.net.eval()
    actor.net.zero_grad(set_to_none=True)
    captured = {}
    module = tracker.network.amcm_reliability

    def save_outputs(_module, _inputs, output):
        captured['rgb'] = output[0]
        captured['tir'] = output[1]
        captured['drgb'] = output[0] - _inputs[0]
        captured['dtir'] = output[1] - _inputs[1]
        for key in ('rgb', 'tir', 'drgb', 'dtir'):
            captured[key].retain_grad()

    handle = module.register_forward_hook(save_outputs)
    def save_fusion(_module, _inputs, output):
        captured['fusion'] = output
        output.retain_grad()
    fusion_handle = tracker.network.tbsi_fuse_search.register_forward_hook(save_fusion)
    pred = tracker.network(template=[templ_v, templ_i], search=[search_v, search_i],
                           return_last_attn=False)
    handle.remove()
    fusion_handle.remove()
    gt_dict = {'search_anno': box.view(1, 1, 4)}
    loss, status = actor.compute_losses(pred, gt_dict, return_status=True)
    loss.backward()

    h, w = pred['score_map'].shape[-2:]
    yy, xx = np.mgrid[:h, :w]
    x1, x2 = (gx-crop_x)*w/crop_side, (gx+gw-crop_x)*w/crop_side
    y1, y2 = (gy-crop_y)*h/crop_side, (gy+gh-crop_y)*h/crop_side
    gt_mask = ((xx + .5 >= x1) & (xx + .5 <= x2) &
               (yy + .5 >= y1) & (yy + .5 <= y2))
    if not gt_mask.any():
        gt_mask[int(np.clip((y1+y2)/2, 0, h-1)), int(np.clip((x1+x2)/2, 0, w-1))] = True
    hann = tracker.output_window[0, 0].detach().float().cpu().numpy()
    score = (pred['score_map'][0, 0].detach().float().cpu().numpy() * hann)
    bg_score = np.where(~gt_mask, score, -np.inf)
    bg_y, bg_x = np.unravel_index(np.argmax(bg_score), bg_score.shape)
    bg_mask = np.zeros_like(gt_mask); bg_mask[bg_y, bg_x] = True
    row = {'sequence': seq, 'frame': frame, 'loss_total': status['Loss/total'],
           'loss_giou': status['Loss/giou'], 'loss_l1': status['Loss/l1'],
           'loss_center_focal': status['Loss/location'],
           'gt_cell_count': int(gt_mask.sum()), 'hard_bg_xy': '{},{}'.format(bg_x, bg_y),
           'crop_contains_gt': bool(x2 > 0 and y2 > 0 and x1 < w and y1 < h)}
    for mod in ('rgb', 'tir'):
        delta = captured['d' + mod].detach().float()
        grad = captured[mod].grad.detach().float()
        per_cell = (delta * grad).sum(dim=1)[0].cpu().numpy()
        row[mod + '_directional_dloss_gt_mean'] = float(per_cell[gt_mask].mean())
        row[mod + '_directional_dloss_hardbg'] = float(per_cell[bg_y, bg_x])
        row[mod + '_directional_dloss_gt_sum'] = float(per_cell[gt_mask].sum())
        row[mod + '_residual_rel_rms'] = float(delta.norm() / (captured[mod].detach().float().norm() + 1e-12))
        row[mod + '_residual_grad_cosine_gt'] = float(F.cosine_similarity(
            delta[:, :, gt_mask].reshape(1, -1), grad[:, :, gt_mask].reshape(1, -1)).item())
    row['tbsi_fuse_output_grad_norm'] = float(captured['fusion'].grad.norm())
    return row


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(CKPT)
    os.environ['TBSI_INFER_FP16'] = '0'
    manifest = json.loads(MANIFEST.read_text())
    names = sorted({r['sequence'] for r in manifest})
    OUT.mkdir(parents=True, exist_ok=True)
    list_path = OUT / 'sequence_list.txt'; list_path.write_text('\n'.join(names) + '\n')
    seqmap = {s.name: s for s in LasHeRDataset('testingset', str(list_path)).get_sequence_list()}
    params = parameters(CFG); params.debug = 0; params.save_all_boxes = False; params.checkpoint = str(CKPT)
    tracker = TBSITrack(params, 'lcgm_gradient_audit')
    actor = make_actor(tracker)
    tracker.network.eval()
    rows = []
    for idx, event in enumerate(manifest, 1):
        first = read_pair(event['sequence'], 0, seqmap)
        tracker.initialize(first, {'init_bbox': np.asarray(seqmap[event['sequence']].ground_truth_rect[0]),
                                   'sequence_name': event['sequence']})
        with torch.enable_grad():
            row = event_loss_and_grad(tracker, actor, event, seqmap)
        rows.append(row)
        print('[{}/{}] {} grad-dot-res RGB={:.4g} TIR={:.4g}'.format(
            idx, len(manifest), row['sequence'], row['rgb_directional_dloss_gt_mean'],
            row['tir_directional_dloss_gt_mean']), flush=True)
    with (OUT / 'per_sequence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {'events': len(rows), 'rgb_gt_loss_decrease': sum(r['rgb_directional_dloss_gt_mean'] < 0 for r in rows),
               'tir_gt_loss_decrease': sum(r['tir_directional_dloss_gt_mean'] < 0 for r in rows),
               'rgb_gt_direction_mean': float(np.mean([r['rgb_directional_dloss_gt_mean'] for r in rows])),
               'tir_gt_direction_mean': float(np.mean([r['tir_directional_dloss_gt_mean'] for r in rows])),
               'scope': 'LCGM checkpoint; exact project actor GIoU+L1+CENTER-focal objective; fixed M1 state/crop/template'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
