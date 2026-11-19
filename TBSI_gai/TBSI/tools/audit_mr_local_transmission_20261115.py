#!/usr/bin/env python3
"""Frozen learned-MR regional message audit, independent of old admission tools."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TBSI_INFER_FP16'] = '0'
import csv
import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from run_dual_message_p1 import setup, config, sha, environment, write_json, state_hash
from lib.test.evaluation import get_dataset
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import clip_box, giou_loss, box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap

RUN = ROOT/'output/experiments/MSHA-MR-LOCAL-TRANSMISSION-AUDIT-S42-20261115'
PARENT = ROOT/'output/experiments/MSHA-MR-CBMR-QUICK-S42-20261115'
PREVIOUS = ROOT/'output/experiments/MSHA-MR-CBMR-AUDIT-S42-20261115'
CHECKPOINT = PARENT/'control/checkpoint_u0468.pth.tar'
CFG = ROOT/'experiments/tbsi_track/MR-CBMR-QUICK-S42-20261115-candidate.yaml'
EXPECTED_SHA = 'a517e65a6e166a7e2b1f3520a7bcb37f8fff65dfe991ad077ea2e311662ffd6e'
REGIONS = ('inside', 'boundary', 'outside')
ACTIONS = ('drop_detail', 'shuffle_detail', 'drop_common', 'matched_common')


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest_tensor(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def coverage(box, side=8):
    # Independently written geometry, checked against a unit grid below.
    x, y, w, h = box.reshape(-1).tolist()
    yy, xx = np.mgrid[:side, :side]
    ox = np.maximum(0, np.minimum((xx+1)/side, min(x+w, 1))-np.maximum(xx/side, max(x, 0)))
    oy = np.maximum(0, np.minimum((yy+1)/side, min(y+h, 1))-np.maximum(yy/side, max(y, 0)))
    return torch.tensor((ox*oy*side*side).ravel(), device=box.device, dtype=torch.float32)


def masks(box):
    c = coverage(box)
    return dict(inside=c >= 1-1e-6, boundary=(c > 1e-6) & (c < 1-1e-6), outside=c <= 1e-6)


def components(message, mask):
    selected = message[:, :, mask, :]
    if selected.shape[-2] == 0:
        return torch.zeros_like(message), torch.zeros_like(message)
    mu = selected.mean(-2, keepdim=True)
    common = torch.zeros_like(message)
    detail = torch.zeros_like(message)
    common[:, :, mask, :] = mu.expand_as(selected)
    detail[:, :, mask, :] = selected-mu
    return common, detail


def action_delta(message, mask, action):
    common, detail = components(message, mask)
    if action == 'drop_detail':
        return -detail
    if action == 'drop_common':
        return -common
    if action == 'matched_common':
        return -common * (detail.norm()/common.norm().clamp_min(1e-20)).clamp(max=1)
    result = torch.zeros_like(message)
    result[:, :, mask, :] = message[:, :, mask, :].roll(1, dims=-2)-message[:, :, mask, :]
    return result


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(0, np.minimum(a[:2]+a[2:], b[:2]+b[2:])-np.maximum(a[:2], b[:2])).prod()
    return float(inter/max(float(a[2:].prod()+b[2:].prod()-inter), 1e-12))


def read_image(paths):
    images = []
    for path in paths:
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError('Cannot read '+str(path))
        images.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return np.concatenate(images, axis=2)


def build_manifest():
    prior = json.loads((PREVIOUS/'frame_manifest.json').read_text())
    seqs = {s.name:s for s in get_dataset('lasher_test')}
    rows = []
    groups = json.loads((PARENT/'contracts/eval_groups.json').read_text())
    for name in dict.fromkeys(r['sequence'] for r in prior):
        entries = [r for r in prior if r['sequence'] == name]
        seq = seqs[name]
        for idx in (1, 4):
            r = dict(entries[idx], split='test', template_frame=0,
                     gt=seq.ground_truth_rect[entries[idx]['frame']].tolist(),
                     init_box=seq.ground_truth_rect[0].tolist(),
                     template_paths=[seq.frames[k][0] for k in (0,1)],
                     search_paths=[seq.frames[k][entries[idx]['frame']] for k in (0,1)])
            rows.append(r)
    manifest = json.loads((PARENT/'contracts/data_manifest.json').read_text())
    buckets = defaultdict(list)
    for r in manifest['train_sequences']:
        buckets[r['stratum']].append(r)
    for members in buckets.values():
        members.sort(key=lambda x:hashlib.sha256(('MR-LOCAL-AUDIT-S42-20261115'+x['sequence']).encode()).hexdigest())
    chosen = []
    for level in range(1000):
        for key in sorted(buckets):
            if level < len(buckets[key]):
                chosen.append(buckets[key][level])
            if len(chosen) == 24:
                break
        if len(chosen) == 24:
            break
    trainroot = Path(manifest['dataset_root'])/'trainingset'
    for seq in chosen:
        directory = trainroot/seq['sequence']
        gt = np.loadtxt(directory/'visible.txt', delimiter=',', ndmin=2)
        listing = [sorted((directory/m).glob('*')) for m in ('visible', 'infrared')]
        frame = len(gt)//2
        assert len(listing[0]) == len(listing[1]) == len(gt)
        assert (gt[[0, frame-1, frame], 2:] > 0).all()
        rows.append(dict(sequence=seq['sequence'], split='train', frame=frame, template_frame=0,
            crop_state=gt[frame-1].tolist(), gt=gt[frame].tolist(), init_box=gt[0].tolist(),
            template_paths=[str(ls[0]) for ls in listing], search_paths=[str(ls[frame]) for ls in listing],
            stratum=seq['stratum'], attributes=seq['attributes']))
    assert len(rows) == 72
    write_json(RUN/'frame_manifest.json', rows)
    return rows, groups


class Audit:
    def __init__(self, net):
        self.net = net
        self.selection = None
        self.alpha = 1.
        self.gradient = False
        self.messages = {}
        self.values = {}
        self.stages = {}
        self.calls = defaultdict(int)
        self.handles = []
        self.hosts = [x.v2_template_fusion for x in net.backbone.tbsi_layers]
        for layer, host in enumerate(self.hosts):
            def forward(module, rt, tr, a, b, boxes, layer=layer):
                self.calls[layer] += 1
                output = []
                for direction, message in (('rgb', rt), ('tir', tr)):
                    self.messages[(layer, direction)] = message if self.gradient else message.detach().clone()
                    if self.selection is not None and self.selection[:2] == (layer, direction):
                        region, action = self.selection[2:]
                        message = message + self.alpha*action_delta(message, self.region_masks[direction][region], action)
                    output.append(message)
                return tuple(output)
            host.cbmr.forward = MethodType(forward, host.cbmr)
            def save_v(module, args, out, layer=layer):
                b, n, triple = out.shape
                v = out.reshape(b, n, 3, self.hosts[layer].num_heads, triple//3//self.hosts[layer].num_heads)
                self.values[layer] = v[:, :, 2].permute(0, 2, 1, 3).detach().clone()
            self.handles.append(host.qkv.register_forward_hook(save_v))
            def save_fusion(module, args, out, layer=layer):
                self.stages['fusion'+str(layer)] = torch.cat(out, dim=1).detach().clone()
            self.handles.append(host.register_forward_hook(save_fusion))
            def save_proj(module, args, out, layer=layer):
                self.stages['projection'+str(layer)] = out.detach().clone()
            self.handles.append(host.proj.register_forward_hook(save_proj))
        for layer in (4, 7, 10, 11):
            def save_block(module, args, out, layer=layer):
                key = 'vit'+str(layer)+'_search'
                value = out[:, 64:].detach().clone()
                self.stages[key] = torch.cat((self.stages[key], value), dim=1) if key in self.stages else value
            self.handles.append(net.backbone.blocks[layer].register_forward_hook(save_block))
        def head(module, args, out):
            self.head_calls += 1
        self.handles.append(net.box_head.register_forward_hook(head))

    def reset(self):
        self.messages = {}
        self.values = {}
        self.stages = {}
        self.calls = defaultdict(int)
        self.head_calls = 0

    def forward(self, tracker, search):
        self.reset()
        result = self.net.forward(template=[tracker.z_dict1.tensors[:, :3], tracker.z_dict1.tensors[:, 3:]],
            search=[search[:, :3], search[:, 3:]], ce_template_mask=tracker.box_mask_z,
            template_boxes=torch.stack((tracker.template_boxes, tracker.template_boxes), dim=1))
        assert all(self.calls[i] == 1 for i in range(3)), self.calls
        assert self.head_calls == 1, self.head_calls
        self.stages['final_search'] = torch.cat((result['backbone_feat'][:, 64:320], result['backbone_feat'][:, -256:]), dim=1).detach().clone()
        self.stages['score'] = result['score_map'].detach().clone()
        self.stages['hann'] = self.stages['score']*tracker.output_window
        return result


def target_context(entry, resize, size):
    state, gt = entry['crop_state'], entry['gt']
    side = size/resize
    x0 = round(state[0]+.5*state[2]-.5*side)
    y0 = round(state[1]+.5*state[3]-.5*side)
    exact = np.array([(gt[0]-x0)*resize/size, (gt[1]-y0)*resize/size, gt[2]*resize/size, gt[3]*resize/size])
    cx, cy = exact[:2]+exact[2:]/2
    inside = bool(0 <= cx < 1 and 0 <= cy < 1)
    yy, xx = np.mgrid[:16, :16]
    exclusion = (xx >= math.floor(exact[0]*16)-1) & (xx <= math.ceil((exact[0]+exact[2])*16)+1)
    exclusion &= (yy >= math.floor(exact[1]*16)-1) & (yy <= math.ceil((exact[1]+exact[3])*16)+1)
    supervision = transform_image_to_crop(torch.tensor(gt), torch.tensor(state), resize, torch.tensor([size, size]), normalize=True)
    target_index = min(max(int(round(float((supervision[0]+supervision[2]/2)*16))), 0), 15)
    target_index += 16*min(max(int(round(float((supervision[1]+supervision[3]/2)*16))), 0), 15)
    return dict(center_inside=inside, gt_index=target_index, exclusion=torch.tensor(exclusion.ravel(), device='cuda'),
                supervision=supervision.cuda().float().reshape(1,4), exact=exact.tolist())


def task_loss(out, context):
    target = context['supervision']
    predicted = box_cxcywh_to_xyxy(out['pred_boxes'].reshape(-1,4))
    truth = box_xywh_to_xyxy(target).clamp(0,1)
    heatmap = generate_heatmap(target.unsqueeze(0), 256, 16)[-1].unsqueeze(1)
    return FocalLoss()(out['score_map'], heatmap)+2*giou_loss(predicted, truth)[0]+5*F.l1_loss(predicted, truth)


def measures(tracker, out, entry, resize, context):
    score = out['score_map'].flatten()
    logits = torch.logit(score.clamp(1e-6, 1-1e-6))
    response = tracker.output_window*out['score_map']
    rawbox = tracker.network.box_head.cal_bbox(response, out['size_map'], out['offset_map']).view(-1,4).mean(0)
    tracker.state = list(entry['crop_state'])
    box = clip_box(tracker.map_box_back(rawbox*tracker.params.search_size/resize, resize), entry['image_height'], entry['image_width'], margin=10)
    valid = context['center_inside'] and context.get('negative_index') is not None
    margin = float(logits[context['gt_index']]-logits[context['negative_index']]) if valid else None
    return dict(iou=iou(box, entry['gt']), box=json.dumps(box), raw_peak=int(score.argmax()),
        hann_peak=int(response.argmax()), margin=margin,
        target_response=float(score[context['gt_index']]) if valid else None,
        negative_response=float(score[context['negative_index']]) if valid else None,
        loss=float(task_loss(out, context)) if context['center_inside'] else None)


def checks():
    c = coverage(torch.tensor([.25,.25,.5,.5], device='cuda'))
    assert int((c == 1).sum()) == 16 and int((c == 0).sum()) == 48
    m = torch.randn(1,2,64,4,device='cuda')
    mask = c > 0
    common, detail = components(m,mask)
    assert torch.allclose((common+detail)[:,:,mask],m[:,:,mask],atol=3e-7)
    shuffled = m+action_delta(m,mask,'shuffle_detail')
    err = float((shuffled[:,:,mask].norm()-m[:,:,mask].norm()).abs())
    assert err < 1e-5
    matched = action_delta(m,mask,'matched_common')
    if common.norm() > detail.norm():
        assert torch.allclose(matched.norm(),detail.norm(),atol=1e-5)
    return dict(geometry=True, decomposition=True, shuffle_energy_max_error=err)


def main():
    global RUN, CHECKPOINT, CFG, EXPECTED_SHA
    parser=argparse.ArgumentParser()
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--precise',action='store_true')
    args=parser.parse_args()
    if args.full:
        full=ROOT/'output/experiments/MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109'
        CHECKPOINT=full/'train/checkpoint_u7020.pth.tar'
        CFG=full/'contracts/mr_only_g1.yaml'
        EXPECTED_SHA='2da4a00a1cb8af18a6abba019af1eb0699c3d01c184f7b23b97dbb9a5d4762f5'
        RUN=RUN/'full_mr_g1'
    elif args.precise:
        RUN=RUN/'quick_highest_precision'
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'completed.json').exists():
        raise RuntimeError('Refusing completed-output overwrite')
    setup(42)
    if args.precise:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    started = time.time()
    assert sha(CHECKPOINT) == EXPECTED_SHA
    manifest, groups = build_manifest()
    cfg = config(CFG)
    if args.full:
        # Explicit translation of old flag semantics; must pass original-workspace parity.
        cfg.MODEL.BACKBONE.CROSS_MASS_PRESERVING_RELATION=False
        cfg.MODEL.BACKBONE.MASS_REDIST_ENABLE=True
        cfg.MODEL.BACKBONE.CBMR_ENABLE=True
        cfg.MODEL.BACKBONE.CBMR_TEMPLATE_BOXES=True
    tracker = TBSITrack(SimpleNamespace(cfg=cfg, checkpoint=str(CHECKPOINT), template_factor=cfg.TEST.TEMPLATE_FACTOR,
        template_size=cfg.TEST.TEMPLATE_SIZE, search_factor=cfg.TEST.SEARCH_FACTOR,
        search_size=cfg.TEST.SEARCH_SIZE, debug=0, save_all_boxes=False), 'lasher_test')
    for p in tracker.network.parameters():
        p.requires_grad_(False)
    initial_hash = state_hash(tracker.network.state_dict())
    audit = Audit(tracker.network)
    tool_checks = checks()
    gates={n:float(v) for n,v in tracker.network.state_dict().items() if n.endswith('early_gate_scale')}
    if args.full:
        assert len(gates)==3 and all(v==0 for v in gates.values()),gates
    workspace_reference=None
    if args.full:
        workspace_reference=json.loads((RUN.parent/'full_reference.json').read_text())['rows']
    parity=[]
    write_json(RUN/'started.json', dict(environment=environment(), checkpoint=str(CHECKPOINT), checkpoint_sha256=EXPECTED_SHA,
        config=str(CFG), config_sha256=sha(CFG), parent_config_sha256=sha(ROOT/'experiments/tbsi_track/MR-CBMR-QUICK-S42-20261115-control.yaml'),
        manifest_sha256=sha(RUN/'frame_manifest.json'), initial_state_sha256=initial_hash,
        gpu_processes=1, workers=0, optimizer_steps=0, parent_train_updates=7020 if args.full else 468,
        gate_scales=gates, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        config_translation=('old CROSS_MASS_PRESERVING_RELATION+Value gate => current MASS_REDIST_ENABLE;CBMR diagnostic identity' if args.full else 'none'),tool_checks=tool_checks))
    baseline_rows, interventions, representation, gradient_rows, finite_rows = [], [], [], [], []
    for sample_index, entry in enumerate(manifest):
        image = read_image(entry['search_paths'])
        entry['image_height'], entry['image_width'] = image.shape[:2]
        tracker.initialize(read_image(entry['template_paths']), dict(init_bbox=entry['init_box']))
        audit.region_masks = {key:masks(tracker.template_boxes[0]) for key in ('rgb','tir')}
        patch, resize, amask = sample_target(image, entry['crop_state'], cfg.TEST.SEARCH_FACTOR, output_sz=cfg.TEST.SEARCH_SIZE)
        search = tracker.preprocessor.process(patch,amask,half=False).tensors
        input_sha = digest_tensor(search)+':'+digest_tensor(tracker.z_dict1.tensors)
        context = target_context(entry,resize,cfg.TEST.SEARCH_SIZE)
        ident = dict(sample=sample_index, split=entry['split'], sequence=entry['sequence'], frame=entry['frame'],
                     center_inside=context['center_inside'])
        audit.selection = None
        audit.gradient = False
        with torch.no_grad():
            out = audit.forward(tracker, search)
            if workspace_reference is not None:
                actual={key:digest_tensor(out[key]) for key in ('score_map','size_map','offset_map','backbone_feat')}
                expected=workspace_reference[sample_index]
                assert input_sha==expected['input_sha256']
                assert actual==expected['outputs'],(sample_index,actual,expected)
                parity.append(dict(sample=sample_index,input_equal=True,all_four_outputs_bitwise_equal=True))
            baseline_messages = {k:v.clone() for k,v in audit.messages.items()}
            baseline_values = {k:v.clone() for k,v in audit.values.items()}
            baseline_stages = {k:v.clone() for k,v in audit.stages.items()}
            negatives = out['score_map'].flatten().masked_fill(context['exclusion'], -torch.inf)
            context['negative_index'] = int(negatives.argmax()) if torch.isfinite(negatives).any() else None
            base = measures(tracker,out,entry,resize,context)
            if sample_index == 0:
                for host in audit.hosts:
                    host.cbmr_enabled = False
                unwrapped = tracker.network.forward(template=[tracker.z_dict1.tensors[:,:3],tracker.z_dict1.tensors[:,3:]],
                    search=[search[:,:3],search[:,3:]], template_boxes=torch.stack((tracker.template_boxes,tracker.template_boxes),1))
                for key in ('score_map','size_map','offset_map','backbone_feat'):
                    assert torch.equal(out[key],unwrapped[key]), key
                for host in audit.hosts:
                    host.cbmr_enabled = True
                tracker.state=list(entry['crop_state']); tracker.frame_id=entry['frame']-1
                tracked=tracker.track(image)['target_bbox']
                assert np.max(np.abs(np.array(tracked)-json.loads(base['box']))) < 1e-6
                tool_checks.update(identity_vs_mr_disabled_bitwise=True, manual_vs_tracker_box=True, hooks_execute=True)
                write_json(RUN/'tool_checks.json',tool_checks)
        baseline_rows.append(dict(ident, input_sha256=input_sha, gt_index=context['gt_index'], negative_index=context['negative_index'], **base))
        choices = []
        for layer in range(3):
            for direction in ('rgb','tir'):
                message = baseline_messages[(layer,direction)]
                # Opposite modality is the source for each receiver direction.
                source = baseline_values[layer][:,:,64:] if direction == 'rgb' else baseline_values[layer][:,:,:64]
                for region in REGIONS:
                    mask = audit.region_masks[direction][region]
                    count = int(mask.sum())
                    if count == 0:
                        continue
                    common, detail = components(message,mask)
                    vcommon, vdetail = components(source,mask)
                    raw = message[:,:,mask,:].norm()
                    denom = raw.square().clamp_min(1e-20)
                    representation.append(dict(ident, layer=layer, receiver=direction, region=region,tokens=count,
                        message_norm=float(raw),common_energy_fraction=float(common.square().sum()/denom),
                        detail_energy_fraction=float(detail.square().sum()/denom),
                        source_v_detail_energy_fraction=float(vdetail.square().sum()/(vcommon.square().sum()+vdetail.square().sum()).clamp_min(1e-20))))
                    for action in ACTIONS:
                        if count < 2 and action != 'drop_common':
                            continue
                        choices.append((layer,direction,region,action))
        # Gradient of real task supervision at exactly the same baseline input.
        grads = None
        if context['center_inside']:
            audit.gradient = True
            audit.selection = None
            z = tracker.z_dict1.tensors.detach().requires_grad_(True)
            tracker.z_dict1.tensors = z
            with torch.enable_grad():
                gout = audit.forward(tracker,search)
                assert torch.equal(gout['score_map'].detach(),baseline_stages['score'])
                loss = task_loss(gout,context)
                keys = list(audit.messages)
                gradients = torch.autograd.grad(loss,[audit.messages[k] for k in keys])
                grads = {k:g.detach() for k,g in zip(keys,gradients)}
            tracker.z_dict1.tensors = z.detach()
            del gout,loss,gradients,z
            audit.gradient = False
            for selection in choices:
                layer,direction,region,action=selection
                m=baseline_messages[(layer,direction)]
                d=action_delta(m,audit.region_masks[direction][region],action)
                g=grads[(layer,direction)]
                gradient_rows.append(dict(ident,layer=layer,receiver=direction,region=region,action=action,
                    directional_derivative=float((g*d).sum()),gradient_norm=float(g.norm()),edit_norm=float(d.norm()),
                    gradient_edit_cosine=float((g*d).sum()/(g.norm()*d.norm()).clamp_min(1e-20))))
        with torch.no_grad():
            for selection in choices:
                layer,direction,region,action=selection
                audit.selection=selection;audit.alpha=1.
                out=audit.forward(tracker,search)
                measured=measures(tracker,out,entry,resize,context)
                d=action_delta(baseline_messages[(layer,direction)],audit.region_masks[direction][region],action)
                record=dict(ident,layer=layer,receiver=direction,region=region,action=action,
                    edit_norm=float(d.norm()),edit_relative_full_message=float(d.norm()/baseline_messages[(layer,direction)].norm().clamp_min(1e-20)),
                    iou_delta=measured['iou']-base['iou'],margin_delta=(measured['margin']-base['margin']) if base['margin'] is not None else None,
                    loss_delta=(measured['loss']-base['loss']) if base['loss'] is not None else None,
                    raw_peak_changed=measured['raw_peak']!=base['raw_peak'],hann_peak_changed=measured['hann_peak']!=base['hann_peak'],
                    box_max_pixel_delta=float(np.max(np.abs(np.array(json.loads(measured['box']))-json.loads(base['box'])))))
                for name in ('projection'+str(layer),'fusion'+str(layer),'vit'+str((4,7,10)[layer])+'_search','final_search','score','hann'):
                    reference=baseline_stages[name]
                    delta=(audit.stages[name]-reference).norm()
                    alias=name if not name.startswith(('projection','fusion','vit')) else ('projection' if name.startswith('projection') else 'fusion' if name.startswith('fusion') else 'next_search')
                    record[alias+'_delta_norm']=float(delta)
                    record[alias+'_relative_delta']=float(delta/reference.norm().clamp_min(1e-20))
                interventions.append(record)
            if entry['split']=='train' and grads is not None:
                region='inside' if int(audit.region_masks['rgb']['inside'].sum())>=2 else 'boundary'
                for action in ('drop_detail','matched_common'):
                    selection=(2,'rgb',region,action)
                    d=action_delta(baseline_messages[(2,'rgb')],audit.region_masks['rgb'][region],action)
                    expected=float((grads[(2,'rgb')]*d).sum())
                    for epsilon in (.05,.1,.5) if args.precise else (.05,.1):
                        vals=[]
                        for sign in (-1,1):
                            audit.selection=selection;audit.alpha=epsilon*sign
                            out=audit.forward(tracker,search)
                            vals.append(float(task_loss(out,context)))
                        numerical=(vals[1]-vals[0])/(2*epsilon)
                        finite_rows.append(dict(ident,layer=2,receiver='rgb',region=region,action=action,epsilon=epsilon,
                            analytic=expected,numerical=numerical,absolute_error=abs(numerical-expected),
                            symmetric_relative_error=abs(numerical-expected)/max(abs(numerical)+abs(expected),1e-7),
                            sign_matches=bool(numerical*expected>=0)))
        assert digest_tensor(search)+':'+digest_tensor(tracker.z_dict1.tensors)==input_sha
        audit.selection=None;audit.alpha=1.
        if (sample_index+1)%4==0:
            for name,rows in (('baseline',baseline_rows),('interventions',interventions),('representation',representation),('gradients',gradient_rows),('finite_difference',finite_rows)):
                csv_write(RUN/(name+'.csv'),rows)
            write_json(RUN/'progress.json',dict(completed_samples=sample_index+1,total_samples=len(manifest),interventions=len(interventions),wall_seconds=time.time()-started))
            print(json.dumps(dict(completed_samples=sample_index+1,total_samples=len(manifest),interventions=len(interventions),wall_seconds=time.time()-started)),flush=True)
    final_hash=state_hash(tracker.network.state_dict())
    assert final_hash==initial_hash
    if workspace_reference is not None:
        write_json(RUN/'original_workspace_parity.json',dict(samples=parity,all_passed=len(parity)==len(manifest)))
    write_json(RUN/'completed.json',dict(environment=environment(), samples=len(manifest),interventions=len(interventions),
        gradients=len(gradient_rows),finite_difference=len(finite_rows),wall_seconds=time.time()-started,
        gpu_wall_hours=(time.time()-started)/3600,state_unchanged=True,final_state_sha256=final_hash,
        tool_checks=tool_checks,optimizer_steps=0))


if __name__=='__main__':
    main()
