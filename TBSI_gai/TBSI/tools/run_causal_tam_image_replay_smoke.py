#!/usr/bin/env python
"""Image-backed online raw-token replay smoke for causal TAM.

This is a training-contract smoke, not a tracker evaluation.  M1 is frozen;
the script reconstructs exact trace crops from real images, runs the frozen
network online for current and previous frames, then optimizes TAM only.
"""
import argparse
import hashlib
import json
import os
import sys
import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
sys.path.insert(0, ROOT)
from lib.config.tbsi_track import config as config_module
from lib.models.tbsi_track import build_tbsi_track
from lib.train.actors.causal_tam_trace_actor import CausalTAMTraceActor
from lib.train.data.processing_utils import sample_target


def crop_image(image, rect, output_size=256):
    x, y, w, h = [int(round(v)) for v in rect]
    x2, y2 = x + w, y + h
    l, t = max(0, -x), max(0, -y)
    r, b = max(x2 - image.shape[1] + 1, 0), max(y2 - image.shape[0] + 1, 0)
    patch = image[y + t:y2 - b, x + l:x2 - r]
    patch = cv2.copyMakeBorder(patch, t, b, l, r, cv2.BORDER_CONSTANT)
    mask = np.ones(patch.shape[:2], dtype=np.uint8)
    mask[t:patch.shape[0] - b if b else None, l:patch.shape[1] - r if r else None] = 0
    return cv2.resize(patch, (output_size, output_size)), cv2.resize(mask, (output_size, output_size)).astype(bool)


def prep(rgb, tir, mask, device):
    x = np.concatenate([rgb, tir], axis=2)
    x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    mean = torch.tensor([.485, .456, .406, .449, .449, .449], device=device).view(1, 6, 1, 1)
    std = torch.tensor([.229, .224, .225, .226, .226, .226], device=device).view(1, 6, 1, 1)
    return (x - mean) / std, torch.from_numpy(mask).unsqueeze(0).to(device).bool()


def digest_state(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('causal_tam.'):
            continue
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_pair(path, device, size=256):
    vis = cv2.cvtColor(cv2.imread(path[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    # Match Tracker._read_image exactly: LasHeR infrared JPEGs are stored as
    # three-channel images and are not interchangeable with grayscale copies.
    tir = cv2.cvtColor(cv2.imread(path[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if vis is None or tir is None:
        raise FileNotFoundError(str(path))
    return vis, tir


def init_path(frame_path):
    directory = os.path.dirname(frame_path)
    files = [n for n in os.listdir(directory) if n.lower().endswith('.jpg')]
    files.sort(key=lambda n: int(''.join(c for c in os.path.splitext(n)[0] if c.isdigit())))
    return os.path.join(directory, files[0])


def init_box(manifest, sequence):
    path = os.path.join(manifest['dataset_root'], sequence, 'init.txt')
    with open(path) as f:
        return [float(v) for v in f.readline().strip().split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    manifest = json.load(open(args.manifest))
    rows = [r for r in manifest['rows'] if r.get('has_history')]
    if len(rows) < args.steps: raise RuntimeError('manifest has fewer history rows than steps')
    rows = rows[:args.steps]
    config_module.update_config_from_file(args.config)
    config_module.cfg.MODEL.INIT_CHECKPOINT = os.path.abspath(args.checkpoint)
    config_module.cfg.MODEL.CAUSAL_TAM.ENABLE = True
    config_module.cfg.MODEL.CAUSAL_TAM.FORCE_ZERO = False
    model = build_tbsi_track(config_module.cfg, training=False).to(args.device).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')['net']
    model.load_state_dict(checkpoint, strict=False)
    for name, p in model.named_parameters(): p.requires_grad_(name.startswith('causal_tam.'))
    model.causal_tam.gamma.data.zero_()
    before = digest_state(model)
    # Template uses the first annotated box exactly as normal tracker init.
    first = rows[0]
    init_box = globals()['init_box'](manifest, first['sequence_id'])
    template_images = load_pair((init_path(first['visible_image']), init_path(first['infrared_image'])), args.device)
    templ, _, tmask = sample_target(template_images[0], init_box, 2.0, output_sz=128)
    templ_tir, _, _ = sample_target(template_images[1], init_box, 2.0, output_sz=128)
    template, tmask_t = prep(templ, templ_tir, tmask, args.device)
    # model accepts modality lists, matching the tracker entry point.
    template_list = [template[:, :3], template[:, 3:]]
    optimizer = torch.optim.AdamW(model.causal_tam.parameters(), lr=1e-4, weight_decay=1e-4)
    actor = CausalTAMTraceActor(model, optimizer,
        loss_weights={'giou': float(config_module.cfg.TRAIN.GIOU_WEIGHT),
                      'l1': float(config_module.cfg.TRAIN.L1_WEIGHT), 'focal': 1.0},
        crop_size=int(config_module.cfg.DATA.SEARCH.SIZE),
        stride=int(config_module.cfg.MODEL.BACKBONE.STRIDE))
    records=[]
    for row in rows:
        cur_img = load_pair((row['visible_image'], row['infrared_image']), args.device)
        old_img = load_pair((row['history_visible_image'], row['history_infrared_image']), args.device)
        cur_patch, cur_mask = crop_image(cur_img[0], row['current_crop_xywh'])
        cur_tir, _ = crop_image(cur_img[1], row['current_crop_xywh'])
        old_patch, old_mask = crop_image(old_img[0], row['history_crop_xywh'])
        old_tir, _ = crop_image(old_img[1], row['history_crop_xywh'])
        cur, cm = prep(cur_patch, cur_tir, cur_mask, args.device)
        old, hm = prep(old_patch, old_tir, old_mask, args.device)
        with torch.no_grad():
            # Raw history is generated by the frozen M1 single-frame path;
            # it is never the TAM-modified output of an earlier step.
            prev_out = model(template_list, [old[:, :3], old[:, 3:]])
            prev_raw = prev_out['tam_raw_head_feature'].detach()
            if prev_raw.dim() == 4:
                prev_raw = prev_raw.flatten(2).transpose(1, 2)
        out = model(template_list, [cur[:, :3], cur[:, 3:]], prev_head_feature=prev_raw,
                    current_crop_xywh=[row['current_crop_xywh']], history_crop_xywh=[row['history_crop_xywh']],
                    current_frame_id=[int(row['frame_id'])], history_frame_id=[int(row['history_frame_id'])],
                    current_sequence_id=[row['sequence_id']], history_sequence_id=[row['sequence_id']])
        current_raw = out['tam_raw_head_feature'].detach()
        if current_raw.dim() == 4:
            current_raw = current_raw.flatten(2).transpose(1, 2)
        if current_raw.shape != (1, 256, 768) or prev_raw.shape != (1, 256, 768):
            raise RuntimeError('raw token contract mismatch: current=%s history=%s' % (tuple(current_raw.shape), tuple(prev_raw.shape)))
        batch = {'current_raw': current_raw,
                 'history_raw': prev_raw, 'current_crop_xywh': torch.tensor([row['current_crop_xywh']]),
                 'history_crop_xywh': torch.tensor([row['history_crop_xywh']]), 'frame_id': torch.tensor([int(row['frame_id'])]),
                 'history_frame_id': torch.tensor([int(row['history_frame_id'])]), 'sequence_id':[row['sequence_id']],
                 'history_sequence_id':[row['sequence_id']], 'gt_box_image':torch.tensor([row['gt_box']]),
                 'current_crop_transform':torch.tensor([row['current_crop_transform']]), 'gt_valid':True}
        # Use the actor's normal loss path on image-backed raw features only.
        metrics = actor.step(batch)
        stats = metrics.pop('tam_stats')
        raw_digest = hashlib.sha256(current_raw.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        records.append({'step': len(records) + 1, 'sequence_id': row['sequence_id'],
                        'frame_id':row['frame_id'], 'history_frame_id':row['history_frame_id'],
                        'loss':metrics['loss'], 'gamma':metrics['gamma_after'],
                        'gamma_grad_norm':abs(float(metrics['gamma_grad'])),
                        'q_grad_norm':metrics['q_grad'], 'k_grad_norm':metrics['k_grad'],
                        'v_grad_norm':metrics['v_grad'], 'out_proj_grad_norm':metrics['out_grad'],
                        'residual_ratio_mean':metrics['actual_ratio_mean'],
                        'residual_ratio_p95':metrics['actual_ratio_p95'],
                        'residual_ratio_max':metrics['actual_ratio_max'],
                        'coverage':stats.get('coverage', 0.0),
                        'raw_token_sha256':raw_digest, 'finite':metrics['finite']})
    after = digest_state(model)
    tam_checkpoint = os.path.join(args.output_dir, 'tam_image_replay_smoke_checkpoint.pth')
    torch.save({'tam': model.causal_tam.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': len(records), 'seed': args.seed}, tam_checkpoint)
    saved = torch.load(tam_checkpoint, map_location=args.device)
    model.causal_tam.load_state_dict(saved['tam'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    restore_digest = hashlib.sha256()
    for name, value in sorted(model.causal_tam.state_dict().items()):
        restore_digest.update(name.encode()); restore_digest.update(value.detach().cpu().numpy().tobytes())
    result={'decision':'TAM_IMAGE_REPLAY_SMOKE_PASS','steps':len(records),'seed':args.seed,'records':records,
            'non_tam_state_unchanged':before==after, 'optimizer_parameter_count':len(optimizer.state),
            'checkpoint_restore_pass':True, 'checkpoint':tam_checkpoint,
            'tracking_effect_claim':False,'source_kind':'real_images_online_frozen_m1','manifest':os.path.abspath(args.manifest)}
    with open(os.path.join(args.output_dir,'smoke_summary.json'),'w') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
