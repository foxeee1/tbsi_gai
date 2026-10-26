#!/usr/bin/env python3
"""Training-free end-to-end AMCM interface screen for recent non-tracking modules.

The candidates consume frozen M1 RGB/TIR search maps and template tokens and
emit a parameter-free spatial proxy. No candidate is trained or injected into
the tracker; this only tests whether the existing interface contains the
signal required by each paper family.
"""
import csv, json, os, sys
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

MANIFEST = ROOT/'output/experiments/M1_PRECISION_SCREEN_MANIFEST_20261020/manifest.json'
CKPT = ROOT/'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
CFG = 'TBSI_E_MHSA_FULL_4090D_seed42'
OUT = ROOT/'output/experiments/M1_NONTRACKING_NEW_MODULES_SCREEN_20261020'

NAMES = [
    'uncertainty_modulation', 'orthogonal_sequential',
    'morphology_region', 'cala_local', 'spectral_spatial_decoupled',
    'ngram_local_fusion', 'spatial_spectral_collab', 'caff_cross_modal',
    'grounding_adapter', 'dynamic_meta_fusion', 'tctfusion',
    'edge_guided_cross_modal', 'ctgan_cross_modal', 'text_guided_modulation',
]

def read_pair(seq, frame, seqmap):
    s = seqmap[seq]
    a = cv2.cvtColor(cv2.imread(s.frames[0][frame]), cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(cv2.imread(s.frames[1][frame]), cv2.COLOR_BGR2RGB)
    return np.concatenate((a, b), 2)

def auc(scores, mask):
    p = np.asarray(scores)[mask].ravel(); n = np.asarray(scores)[~mask].ravel()
    if not len(p) or not len(n): return float('nan')
    return float(((p[:, None] > n[None, :]).sum() + .5 *
                  (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))

def event_mask(event, size=16):
    x, y, w, h = map(float, event['crop_xywh']); gx, gy, gw, gh = map(float, event['gt_box'])
    x1, x2 = (gx-x)*size/w, (gx+gw-x)*size/w; y1, y2 = (gy-y)*size/h, (gy+gh-y)*size/h
    yy, xx = np.mgrid[:size, :size]
    m = ((xx+.5 >= x1) & (xx+.5 <= x2) & (yy+.5 >= y1) & (yy+.5 <= y2))
    if not m.any(): m[int(np.clip((y1+y2)/2, 0, size-1)), int(np.clip((x1+x2)/2, 0, size-1))] = True
    return m

def template_maps(rgb, tir, trgb, ttir):
    rn, tn = F.normalize(rgb, dim=1), F.normalize(tir, dim=1)
    kr, kt = F.normalize(trgb, dim=-1), F.normalize(ttir, dim=-1)
    sr = (rn * kr.mean(1)[:, :, None, None]).sum(1, keepdim=True)
    st = (tn * kt.mean(1)[:, :, None, None]).sum(1, keepdim=True)
    # Explicit token condition, unlike the old global-mean proxy.
    qr = rn.flatten(2).transpose(1, 2); qt = tn.flatten(2).transpose(1, 2)
    mr = torch.bmm(qr, kr.transpose(1, 2)).amax(-1).view(rgb.shape[0], 1, 16, 16)
    mt = torch.bmm(qt, kt.transpose(1, 2)).amax(-1).view(tir.shape[0], 1, 16, 16)
    mean = .5 * (sr + st); agreement = (rn * tn).sum(1, keepdim=True)
    edge = (F.avg_pool2d(mean, 3, 1, 1) - mean).abs()
    local3 = F.avg_pool2d(mean, 3, 1, 1); local5 = F.avg_pool2d(mean, 5, 1, 2)
    return sr, st, mr, mt, mean, agreement, edge, local3, local5

def candidate_maps(rgb, tir, trgb, ttir):
    sr, st, mr, mt, mean, agree, edge, l3, l5 = template_maps(rgb, tir, trgb, ttir)
    # All outputs are spatial [1,16,16] proxies that could be expanded to a
    # same-shape AMCM residual. They intentionally contain no trainable weight.
    noise = 0.03
    r_pert = F.normalize(rgb + noise * rgb.std() * torch.sign(rgb), dim=1)
    t_pert = F.normalize(tir + noise * tir.std() * torch.sign(tir), dim=1)
    sens = ((r_pert * F.normalize(trgb.mean(1), dim=1)[:, :, None, None]).sum(1, keepdim=True) - sr).abs()
    # Orthogonal incremental content, reliability weighted local fusion, and
    # region morphology are distinct from the previously screened modules.
    proj_r = (rgb * tir).sum(1, keepdim=True) / (tir.pow(2).sum(1, keepdim=True) + 1e-6) * tir
    orth = (rgb - proj_r).pow(2).mean(1, keepdim=True).sqrt()
    erosion = -F.max_pool2d(-mean, 3, 1, 1)
    morph = F.max_pool2d(erosion, 5, 1, 2)
    morph = mean + (mean - morph).relu()
    local_attn = (l3 * torch.sigmoid(4 * mean) + l5 * torch.sigmoid(4 * mean)) / 2
    ch_weight = torch.var(torch.stack((rgb.mean(1), tir.mean(1)), 0), 0, keepdim=True)
    spectral = mean * torch.sigmoid(4 * (agree - agree.mean())) + ch_weight
    ngram = .5 * (l3 + l5) + .25 * F.avg_pool2d(mean, 7, 1, 3)
    collab = .5 * (local_attn + spectral)
    diffusion = mean
    for _ in range(2): diffusion = .5 * diffusion + .5 * F.avg_pool2d(diffusion, 3, 1, 1)
    caff = diffusion * (0.5 + 0.5 * torch.sigmoid(agree)) + .5 * (mr + mt)
    dynamic = (mr * torch.sigmoid(sr) + mt * torch.sigmoid(st)) / (torch.sigmoid(sr) + torch.sigmoid(st) + 1e-6)
    triple = torch.maximum(torch.maximum(mr, mt), local_attn)
    edge_guided = mean * (1 + edge / (edge.mean() + 1e-6))
    ctgan = .5 * (mr + mt) * torch.sigmoid(agree)
    text_guided = torch.maximum(mr, mt) + .25 * (mr - mt).abs()
    return {
        'uncertainty_modulation': mean - 2 * sens,
        'orthogonal_sequential': mean - .25 * orth,
        'morphology_region': morph,
        'cala_local': local_attn,
        'spectral_spatial_decoupled': spectral,
        'ngram_local_fusion': ngram,
        'spatial_spectral_collab': collab,
        'caff_cross_modal': caff,
        'grounding_adapter': dynamic,
        'dynamic_meta_fusion': dynamic,
        'tctfusion': triple,
        'edge_guided_cross_modal': edge_guided,
        'ctgan_cross_modal': ctgan,
        'text_guided_modulation': text_guided,
    }

def summarize(name, score, mask):
    score = np.asarray(score, float); target, bg = score[mask], score[~mask]
    return {'candidate': name, 'auc': auc(score, mask),
            'target_bg_margin': float(target.mean()-bg.mean()),
            'target_bg_ratio': float(abs(target).mean()/(abs(bg).mean()+1e-8)),
            'target_max_bg_max_margin': float(target.max()-bg.max())}

def main():
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(CKPT); os.environ['TBSI_INFER_FP16'] = '0'
    manifest = json.loads(MANIFEST.read_text())['selection']; names = sorted({e['sequence'] for e in manifest})
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'sequence_list.txt').write_text('\n'.join(names)+'\n')
    seqmap = {s.name: s for s in LasHeRDataset('testingset', str(OUT/'sequence_list.txt')).get_sequence_list()}
    p = parameters(CFG); p.debug = 0; p.save_all_boxes = False; p.checkpoint = str(CKPT)
    tracker = TBSITrack(p, 'm1_new_module_screen'); captured = {}
    def bb(_m, _i, o):
        if isinstance(o, dict): o = o.get('feat', o.get('backbone_feat', next(iter(o.values()))))
        if isinstance(o, dict): o = o.get('feat', o.get('tokens', o.get('x', next(iter(o.values())))))
        if isinstance(o, (list, tuple)): o = o[0] if torch.is_tensor(o[0]) else o[-1]
        captured['backbone'] = o
    h = tracker.network.backbone.register_forward_hook(bb); rows = []
    for i, event in enumerate(manifest, 1):
        first = read_pair(event['sequence'], 0, seqmap)
        tracker.initialize(first, {'init_bbox': np.asarray(seqmap[event['sequence']].ground_truth_rect[0]), 'sequence_name': event['sequence']})
        tracker.state = list(map(float, event['tracker_state_before'])); captured.clear()
        tracker.track(read_pair(event['sequence'], int(event['frame']), seqmap), {'gt_bbox': event['gt_box'], 'sequence_name': event['sequence'], 'frame_num': event['frame']})
        cat = captured['backbone'].detach().float(); rgb = cat[:,64:320,:].transpose(1,2).reshape(1,-1,16,16); tir = cat[:,-256:,:].transpose(1,2).reshape(1,-1,16,16)
        trgb, ttir = cat[:,:64,:], cat[:,320:384,:]; mask = event_mask(event)
        for name, score in candidate_maps(rgb, tir, trgb, ttir).items():
            row = {'sequence': event['sequence'], 'frame': event['frame'], 'role': event['role'], 'm1_auc': event['m1_auc']}
            row.update(summarize(name, score[0,0].detach().cpu().numpy(), mask)); rows.append(row)
        if i % 5 == 0: print('[{}/{}] {}'.format(i, len(manifest), event['sequence']), flush=True)
    h.remove(); out_rows = OUT/'per_event_candidate.csv'
    with out_rows.open('w', newline='') as f: w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=[]
    for name in NAMES:
        rr=[r for r in rows if r['candidate']==name]
        for role in ('failure','control','all'):
            q=rr if role=='all' else [r for r in rr if r['role']==role]
            summary.append({'candidate':name,'role':role,'events':len(q),'mean_auc':float(np.mean([float(r['auc']) for r in q])),'mean_margin':float(np.mean([float(r['target_bg_margin']) for r in q])),'mean_target_bg_ratio':float(np.mean([float(r['target_bg_ratio']) for r in q])),'auc_above_chance':sum(float(r['auc'])>0.5 for r in q)})
    with (OUT/'summary.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    (OUT/'protocol.json').write_text(json.dumps({'scope':'M1 fixed 20 failures + 5 controls; frozen epoch15; no training/no tracker injection','candidates':NAMES,'interface':'[B,C,16,16] + [B,64,C] -> parameter-free spatial proxy [B,1,16,16]','warning':'mechanism proxy screen, not paper reproduction or tracking performance'},indent=2))
    print(json.dumps(summary, indent=2))
if __name__=='__main__': main()
