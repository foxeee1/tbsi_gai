#!/usr/bin/env python3
"""Token-level target/background margin audit for the protected M1 checkpoint.

This is an analysis-only script. It reuses the frozen 50-sequence manifest and
fixed events from the uncertainty-gate diagnostic, and does not alter model
weights, loss, data split, or tracker state outside each fixed event.
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack

CONFIG_NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
CONFIG = ROOT / 'experiments/tbsi_track' / (CONFIG_NAME + '.yaml')
CHECKPOINT = ROOT / 'output/experiments' / CONFIG_NAME / 'checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG_SHA256 = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e8bcebdc34f484ac8ec96316b4e1e'
CONFIG_SHA256 = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
CHECKPOINT_SHA256 = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'
MANIFEST = ROOT / 'experiments/manifests/m1_msha_uncertainty_gate_50seq_20261021.txt'
EVENTS = ROOT / 'output/experiments/M1_MSHA_UNCERTAINTY_GATE_50SEQ_20261021_RETRY/validation_events.json'
OUT = ROOT / 'output/experiments/M1_MSHA_TOKEN_MARGIN_AUDIT_50SEQ_20261021'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_pair(seq, frame):
    rgb = cv2.imread(seq.frames[0][frame], cv2.IMREAD_COLOR)
    tir = cv2.imread(seq.frames[1][frame], cv2.IMREAD_COLOR)
    if rgb is None or tir is None or rgb.shape[:2] != tir.shape[:2]:
        raise RuntimeError('missing or misaligned frame {} {}'.format(seq.name, frame))
    return np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)), axis=2)


def box_iou(a, b):
    ix = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


def grid_mask(gt, crop, n=16):
    x, y, side, _ = crop
    gx, gy, gw, gh = gt
    x1, x2 = (gx - x) * n / side, (gx + gw - x) * n / side
    y1, y2 = (gy - y) * n / side, (gy + gh - y) * n / side
    yy, xx = np.mgrid[:n, :n]
    mask = ((xx + .5 >= x1) & (xx + .5 <= x2) &
            (yy + .5 >= y1) & (yy + .5 <= y2))
    if not mask.any():
        mask[int(np.clip((y1 + y2) / 2, 0, n - 1)),
             int(np.clip((x1 + x2) / 2, 0, n - 1))] = True
    return mask.reshape(-1)


def search_crop(state, resize_factor, search_size):
    cx, cy = state[0] + state[2] / 2., state[1] + state[3] / 2.
    side = float(search_size) / float(resize_factor)
    return [cx - side / 2., cy - side / 2., side, side]


def normalize(x):
    return torch.nn.functional.normalize(x.float(), dim=-1, eps=1e-6)


def vector_margin(target, bg):
    """Signed separation: target-to-prototype similarity minus bg similarity."""
    t = normalize(target).mean(0, keepdim=True)
    b = normalize(bg)
    return float((normalize(target) @ t.t()).mean() - (b @ t.t()).mean())


def score_tokens(template, search, template_fg, search_fg):
    """Return target-conditioned template score and RGB/TIR agreement margins."""
    tn, sn = normalize(template), normalize(search)
    proto = tn[template_fg].mean(0, keepdim=True)
    score = (sn @ proto.t()).squeeze(-1)
    return score, float(score[search_fg].mean() - score[~search_fg].mean())


def attach_hooks(tracker, captured):
    net = tracker.network
    handles = []
    layers = getattr(net.backbone, 'tbsi_layers', [])
    configured_locs = [3, 6, 9]
    for index, layer in enumerate(layers):
        fusion = getattr(layer, 'v2_template_fusion', None)
        if fusion is None:
            continue
        slot = 'loc{}'.format(configured_locs[index] if index < len(configured_locs) else index)
        ctx = {}
        captured[slot] = ctx

        def outer_pre(module, inputs, ctx=ctx):
            xv, xi, lens = inputs[:3]
            ctx['rgb_search'] = xv[:, int(lens):, :].detach().float().cpu()
            ctx['tir_search'] = xi[:, int(lens):, :].detach().float().cpu()

        def fusion_pre(module, inputs, ctx=ctx, key=slot):
            ctx['rgb_template'] = inputs[0].detach().float().cpu()
            ctx['tir_template'] = inputs[1].detach().float().cpu()

        def fusion_post(module, inputs, output, ctx=ctx, key=slot):
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                ctx['rgb_template_post'] = output[0].detach().float().cpu()
                ctx['tir_template_post'] = output[1].detach().float().cpu()

        def layer_post(module, inputs, output, ctx=ctx):
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                lens = int(inputs[2])
                ctx['rgb_search_post'] = output[0][:, lens:, :].detach().float().cpu()
                ctx['tir_search_post'] = output[1][:, lens:, :].detach().float().cpu()

        handles.extend([layer.register_forward_pre_hook(outer_pre),
                        fusion.register_forward_pre_hook(fusion_pre),
                        fusion.register_forward_hook(fusion_post),
                        layer.register_forward_hook(layer_post)])

    def fuse_pre(module, inputs):
        captured['pre_fusion'] = inputs[0].detach().float().cpu()

    def fuse_post(module, inputs, output):
        captured['post_fusion'] = output.detach().float().cpu()

    handles.extend([net.tbsi_fuse_search.register_forward_pre_hook(fuse_pre),
                    net.tbsi_fuse_search.register_forward_hook(fuse_post)])
    return handles


def new_tracker():
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(CHECKPOINT)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_INTERNAL_DIAGNOSTIC'] = '0'
    os.environ['TBSI_TEMPLATE_UNCERTAINTY_MODE'] = 'off'
    p = parameters(CONFIG_NAME)
    p.debug = 0
    p.save_all_boxes = False
    p.checkpoint = str(CHECKPOINT)
    return TBSITrack(p, 'm1_token_margin_audit')


def run_event(event, seq):
    tracker = new_tracker()
    captured = {}
    handles = attach_hooks(tracker, captured)
    tracker.initialize(read_pair(seq, 0), {'init_bbox': np.asarray(seq.ground_truth_rect[0]),
                                           'sequence_name': seq.name})
    tracker.state = list(map(float, event['state_before']))
    tracker.prev_search = None
    tracker.prev_temporal_tokens = None
    tracker.prev_head_feature = None
    pred = tracker.track(read_pair(seq, int(event['frame'])), {
        'gt_bbox': event['gt_box'], 'sequence_name': seq.name,
        'frame_num': int(event['frame'])})
    diag = tracker._last_fixed_state_diag
    mask = torch.from_numpy(grid_mask(event['gt_box'], event['crop_xywh']))
    score = diag['score_map'][0, 0].numpy() * tracker.output_window[0, 0].detach().float().cpu().numpy()
    mask_grid = mask.numpy().reshape(16, 16)
    bg = np.where(~mask_grid, score, -np.inf)
    bg_yx = np.unravel_index(int(np.argmax(bg)), score.shape)
    row = {'sequence': seq.name, 'frame': int(event['frame']),
           'fixed_state_iou': box_iou(pred['target_bbox'], event['gt_box']),
           'gt_peak': float(score[mask_grid].max()), 'hard_bg_peak': float(score[bg_yx]),
           'center_margin': float(score[mask_grid].max() - score[bg_yx]),
           'gt_rank': int(1 + np.sum(score[~mask_grid] > score[mask_grid].max())),
           'hard_bg_y': int(bg_yx[0]), 'hard_bg_x': int(bg_yx[1])}
    bg_idx = int(bg_yx[0] * 16 + bg_yx[1])
    fg_idx = mask.bool()
    template_fg_idx = torch.from_numpy(grid_mask(event['gt_box'], event['crop_xywh'], n=8))
    bg_mask = ~fg_idx
    for key, ctx in captured.items():
        if not isinstance(ctx, dict):
            continue
        for mod in ('rgb', 'tir'):
            templ = ctx.get(mod + '_template')
            search = ctx.get(mod + '_search')
            post = ctx.get(mod + '_search_post')
            if templ is None or search is None:
                continue
            templ, search = templ[0], search[0]
            score_tok, tm = score_tokens(templ, search, template_fg_idx, fg_idx)
            agree = (normalize(ctx['rgb_search'][0]) * normalize(ctx['tir_search'][0])).sum(-1) if 'rgb_search' in ctx else None
            row['{}_{}_template_match_margin'.format(key, mod)] = tm
            row['{}_{}_template_match_fg'.format(key, mod)] = float(score_tok[fg_idx].mean())
            row['{}_{}_template_match_bg'.format(key, mod)] = float(score_tok[bg_mask].mean())
            if post is not None:
                post = post[0]
                post_score, post_tm = score_tokens(templ, post, template_fg_idx, fg_idx)
                row['{}_{}_post_template_match_margin'.format(key, mod)] = post_tm
            if mod == 'rgb' and agree is not None:
                row['{}_cross_modal_agreement_margin'.format(key)] = float(agree[fg_idx].mean() - agree[bg_mask].mean())
        if 'rgb_search' in ctx and 'tir_search' in ctx:
            rgb, tir = ctx['rgb_search'][0], ctx['tir_search'][0]
            row['{}_cross_modal_agreement_fg'.format(key)] = float((normalize(rgb) * normalize(tir)).sum(-1)[fg_idx].mean())
            row['{}_cross_modal_agreement_bg'.format(key)] = float((normalize(rgb) * normalize(tir)).sum(-1)[bg_mask].mean())
    pre = captured.get('pre_fusion')
    post = captured.get('post_fusion')
    if pre is not None and post is not None:
        pre, post = pre[0], post[0]
        c = pre.shape[-1] // 2
        for name, feat in (('pre_rgb', pre[:, :c]), ('pre_tir', pre[:, c:]), ('post', post)):
            # The search fusion implementation has changed layout across
            # historical branches. Only score a tensor when its token count
            # matches the fixed 16x16 search grid; preserve the observed shape.
            if feat.ndim == 3 and feat.shape[0] == 256:
                token_feat = feat
            elif feat.ndim == 3 and feat.shape[-1] == 256:
                token_feat = feat.transpose(0, 1)
            elif feat.ndim == 3 and feat.shape[1] * feat.shape[2] == 256:
                token_feat = feat.permute(1, 2, 0).reshape(256, -1)
            else:
                row['{}_shape'.format(name)] = str(tuple(feat.shape))
                continue
            target, background = token_feat[fg_idx], token_feat[bg_idx:bg_idx + 1]
            row['{}_target_bg_margin'.format(name)] = vector_margin(target, background)
    for h in handles:
        h.remove()
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(OUT))
    args = ap.parse_args()
    if sha256(CONFIG) != CONFIG_SHA256 or sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing to reuse non-empty output directory: {}'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in MANIFEST.read_text().splitlines() if x.strip()]
    events = json.loads(EVENTS.read_text())
    list_path = out / 'sequence_list.txt'; list_path.write_text('\n'.join(names) + '\n')
    seqs = LasHeRDataset('testingset', str(list_path)).get_sequence_list()
    seqmap = {s.name: s for s in seqs}
    rows = []
    for i, event in enumerate(events, 1):
        row = run_event(event, seqmap[event['sequence']])
        rows.append(row)
        print('[token-audit] {}/{} {} frame={} center_margin={:.5f}'.format(
            i, len(events), event['sequence'], event['frame'], row['center_margin']), flush=True)
    cols = sorted({k for r in rows for k in r})
    with (out / 'per_event.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    summary = {'id': 'M1-MSHA-TOKEN-MARGIN-AUDIT-50SEQ-20261021',
               'manifest': str(MANIFEST), 'manifest_sha256': sha256(MANIFEST),
               'requested_sequences': len(names), 'eligible_events': len(events),
               'sequences_without_event': len(names) - len(set(e['sequence'] for e in events)),
               'config_sha256': sha256(CONFIG), 'checkpoint_sha256': sha256(CHECKPOINT),
               'protocol': 'fixed state, off uncertainty mode; target mask from GT; hard background is off-arm top CENTER*Hann cell',
               'decision': 'TOKEN_LEVEL_AUDIT_COMPLETE'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    main()
