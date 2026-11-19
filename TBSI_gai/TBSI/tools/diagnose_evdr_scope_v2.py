#!/usr/bin/env python3
"""Retrospective temporal localization and learned, common-crop interventions.

No admission thresholds, training, oracle-centered crops, or legacy auditors.
Frame indices are zero based. Outcome-selected events are diagnostic only.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ['TBSI_INFER_FP16'] = '0'
import numpy as np

P1 = ROOT / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3'
OUT = ROOT / 'experiments/early_gate_closed_loop_20261026/evdr_scope_v2_20261026'
RESULT = ROOT / 'output/test/early_gate_p1_sot_v3'
TRAIN = ROOT / 'output/experiments'


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(np.minimum(a[..., :2] + a[..., 2:], b[..., :2] + b[..., 2:]) -
                       np.maximum(a[..., :2], b[..., :2]), 0).prod(-1)
    return inter / np.maximum(a[..., 2:].prod(-1) + b[..., 2:].prod(-1) - inter, 1.e-12)


def crop_visibility(states, gt, factor=4.):
    """Exact rounded square geometry used by sample_target; no image clipping.

    GT fraction in the sampled crop is distinct from GT visibility/occlusion.
    """
    side = np.ceil(np.sqrt(states[:, 2] * states[:, 3]) * factor)
    lo = np.round(states[:, :2] + states[:, 2:] / 2 - side[:, None] / 2)
    center = gt[:, :2] + gt[:, 2:] / 2
    inside = ((center >= lo) & (center < lo + side[:, None])).all(1)
    inter = np.maximum(np.minimum(gt[:, :2] + gt[:, 2:], lo + side[:, None]) -
                       np.maximum(gt[:, :2], lo), 0).prod(1)
    fraction = inter / np.maximum(gt[:, 2:].prod(1), 1.e-12)
    return inside, fraction


def first_run(mask, length):
    for i in range(1, len(mask) - length + 1):
        if mask[i:i + length].all():
            return i
    return None


def temporal():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((P1 / 'analysis_u0468_screen96_max2000/per_sequence.csv').open()))
    frames, stats = {}, []
    for row in rows:
        name = row['sequence']
        gt = np.loadtxt(ROOT / 'data/lasher/testingset' / name / 'init.txt', delimiter=',', ndmin=2)
        boxes = [np.loadtxt(RESULT / (arm + '_u0468_screen96_max2000_20261026') /
                           (name + '.txt'), ndmin=2) for arm in ['control', 'candidate']]
        overlaps = [iou(p, gt) for p in boxes]
        # Validate this independent localizer against the already validated P1 metric.
        for arm, overlap in zip(['control', 'candidate'], overlaps):
            auc = np.mean([np.mean(overlap > t) for t in np.arange(0, 1.00001, .05)]) * 100
            if abs(auc - float(row['auc_' + arm])) > 1.e-7:
                raise ValueError('Independent AUC cross-check failed: ' + name)
        delta = overlaps[1] - overlaps[0]
        onset = first_run(delta < -.2, 10)
        gain_onset = first_run(delta > .2, 10)
        masks = [crop_visibility(np.concatenate([gt[:1], p[:-1]]), gt) for p in boxes]
        for key, value in [('control_iou', overlaps[0]), ('candidate_iou', overlaps[1]),
                           ('control_crop_gt_fraction', masks[0][1]),
                           ('candidate_crop_gt_fraction', masks[1][1])]:
            frames[name + '/' + key] = value
        s = {'sequence': name, 'group': row['group'], 'frames': len(gt),
             'auc_delta': float(row['auc_delta']), 'loss_onset_10frames': onset,
             'gain_onset_10frames': gain_onset,
             'pre_onset_mean_iou_delta': float(delta[1:onset].mean()) if onset and onset > 1 else None,
             'post_onset_mean_iou_delta': float(delta[onset:].mean()) if onset else None,
             'candidate_crop_center_out_fraction': float((~masks[1][0]).mean()),
             'control_crop_center_out_fraction': float((~masks[0][0]).mean()),
             'candidate_gt_fraction_at_loss_onset': float(masks[1][1][onset]) if onset else None,
             'control_gt_fraction_at_loss_onset': float(masks[0][1][onset]) if onset else None,
             'new_failure_control_good': int(((overlaps[0] >= .5) & (overlaps[1] < .1)).sum())}
        stats.append(s)
    stats.sort(key=lambda r: r['auc_delta'])
    events = []
    selections = [('large_loss', r) for r in stats if r['auc_delta'] < -5]
    selections += [('gain', r) for r in sorted(stats, key=lambda r: -r['auc_delta'])[:3]]
    neutral = [r for r in stats if r['group'] == 'normal_protection' and abs(r['auc_delta']) < .2]
    selections += [('normal_stable', r) for r in sorted(neutral, key=lambda r: r['sequence'])[:3]]
    for label, row in selections:
        n = row['frames']
        pivot = (row['loss_onset_10frames'] if label == 'large_loss' else
                 row['gain_onset_10frames'] if label == 'gain' else n // 2)
        if pivot is None:
            pivot = n // 2
        chosen = sorted(set(max(1, min(n - 1, pivot + offset)) for offset in
                            [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]))
        events.append({'sequence': row['sequence'], 'selection': label, 'pivot': pivot,
                       'frames': chosen, 'group': row['group']})
    np.savez_compressed(OUT / 'temporal_arrays.npz', **frames)
    dump(OUT / 'temporal.json', {'definition': 'retrospective 96-sequence development cohort; indices zero based; onset=10 consecutive frames with candidate-control IoU<-0.2; threshold localizes events, never admits/rejects a module',
         'rows': stats, 'events': events,
         'prediction_identity': {arm: digest(RESULT / (arm + '_u0468_screen96_max2000_20261026') / 'identity.json') for arm in ['control', 'candidate']}})
    with (OUT / 'temporal.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)
    print(json.dumps({'sequences': len(stats), 'events': len(events), 'sampled_frames': sum(len(e['frames']) for e in events), 'worst9': stats[:9]}, indent=2))


def probe():
    import torch
    import torch.nn.functional as F
    from run_early_gate_p1_v2 import setup, environment
    from lib.test.evaluation.lasherdataset import LasHeRDataset
    from lib.test.evaluation.tracker import Tracker
    from lib.test.parameter.tbsi_track import parameters
    from lib.test.tracker.tbsi_track import TBSITrack
    setup(42)
    started = time.time()
    temporal_data = json.loads((OUT / 'temporal.json').read_text())
    names = [e['sequence'] for e in temporal_data['events']]
    list_path = OUT / 'retrospective_event_sequences.txt'
    list_path.write_text('\n'.join(names) + '\n')
    dataset = {s.name: s for s in LasHeRDataset('testingset', str(list_path)).get_sequence_list()}
    trackers, checkpoints = {}, {}
    for arm in ['control', 'candidate']:
        ckpt = TRAIN / ('EG_EVDR_P1_SOT_V3_' + arm.upper() + '_S42_20261026/checkpoint_u0468.pth.tar')
        os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(ckpt)
        p = parameters('EG_EVDR_P1_SOT_V3_' + arm.upper() + '_20261026')
        p.debug = 0
        trackers[arm] = TBSITrack(p, 'lasher_test')
        checkpoints[arm] = {'path': str(ckpt), 'sha256': digest(ckpt)}
    wrapper = Tracker('tbsi_track', 'EG_EVDR_P1_SOT_V3_CANDIDATE_20261026', 'lasher_test')
    operators = [l.v2_template_fusion for l in trackers['candidate'].network.backbone.tbsi_layers]
    cache, call_rows, per_token, context = {}, [], [], {}

    def fusion_pre(module, args):
        rgb, tir = args[:2]
        x = module.norm(torch.cat([rgb, tir], dim=1))
        b, n, c = x.shape
        qkv = module.qkv(x).reshape(b, n, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        a = ((qkv[0] @ qkv[1].transpose(-2, -1)) * module.scale).softmax(-1)
        n = rgb.shape[1]
        cache[id(module.value_direction)] = {'attn': [a[:, :, n:, :n], a[:, :, :n, n:]], 'call': 0}

    def operator_hook(module, args, output):
        v, src, rec, evidence = args[:4]
        state = cache[id(module)]; role = state['call']; state['call'] += 1
        a = state['attn'][role]
        src_code = F.gelu(module.source(module.norm(src)))
        rec_code = module.receiver(module.norm(rec))
        cond = module.condition(evidence.to(src_code.dtype))
        b, n, c = src.shape; h, r = module.num_heads, module.rank
        rec_head = rec_code.reshape(b, n, h, r).permute(0, 2, 1, 3)
        mass = a.sum(-2)
        # Cross-attention column normalization: actual receiver distribution
        # for each source token, using this head's existing QK weights.
        reverse = a.transpose(-2, -1) / mass.clamp_min(1.e-20).unsqueeze(-1)
        aligned = reverse @ rec_head
        normal = rec_head
        full_code = src_code * (1 + torch.tanh(rec_code + cond))
        base_raw = torch.einsum('bnhr,hrd->bhnd', src_code.reshape(b, n, h, r), module.up)
        raw = torch.einsum('bnhr,hrd->bhnd', full_code.reshape(b, n, h, r), module.up)
        unit = F.normalize(v.float(), dim=-1)
        tangent = raw.float() - (raw.float() * unit).sum(-1, keepdim=True) * unit
        base_tangent = base_raw.float() - (base_raw.float() * unit).sum(-1, keepdim=True) * unit
        rel = (output.float() - v.float()).norm(dim=-1) / v.float().norm(dim=-1).clamp_min(1.e-12)
        mask = trackers['candidate'].c1_template_mask.to(v.device)
        # Offline mask stratification only: not consumed by either operator.
        row = {**context, 'layer': operators.index(next(f for f in operators if f.value_direction is module)),
               'source': ['rgb', 'tir'][role], 'mean_relative_value_change': float(rel.mean()),
               'p95_relative_value_change': float(torch.quantile(rel.float(), .95)),
               'fg_relative_change': float(rel[..., mask].mean()) if mask.any() else None,
               'bg_relative_change': float(rel[..., ~mask].mean()) if (~mask).any() else None,
               'bg_share_squared_change': float((output[..., ~mask, :] - v[..., ~mask, :]).square().sum() /
                                                (output - v).square().sum().clamp_min(1.e-20)),
               'mean_source_tangent_norm': float(base_tangent.norm(dim=-1).mean()),
               'mean_conditional_tangent_norm': float((tangent-base_tangent).norm(dim=-1).mean()),
               'diagonal_receiver_mass': float((a.diagonal(dim1=-2, dim2=-1) / mass.clamp_min(1.e-20)).mean()),
               'aligned_vs_colocated_receiver_rel_l2': float((aligned-normal).norm() / normal.norm().clamp_min(1.e-20)),
               'aligned_vs_colocated_receiver_cosine': float(F.cosine_similarity(aligned, normal, dim=-1).mean()),
               'conditioner_change_if_aligned': float((torch.tanh(aligned + cond.reshape(b,n,h,r).permute(0,2,1,3)) -
                                                      torch.tanh(normal + cond.reshape(b,n,h,r).permute(0,2,1,3))).abs().mean()),
               'up_norm': float(module.up.norm())}
        call_rows.append(row)
        per_token.append(np.stack([rel[0].detach().cpu().numpy(),
                                   mass[0].detach().cpu().numpy(),
                                   (a.diagonal(dim1=-2,dim2=-1)/mass.clamp_min(1.e-20))[0].detach().cpu().numpy()], axis=0))

    hooks = []
    for fusion in operators:
        hooks += [fusion.register_forward_pre_hook(fusion_pre), fusion.value_direction.register_forward_hook(operator_hook)]
    comparisons = []
    with torch.inference_mode():
        for event in temporal_data['events']:
            name = event['sequence']; sequence = dataset[name]
            gt = sequence.ground_truth_rect
            first = np.concatenate([wrapper._read_image(sequence.frames[m][0]) for m in [0, 1]], axis=2)
            for tracker in trackers.values():
                tracker.initialize(first, {'init_bbox': gt[0].tolist()})
            saved = {arm: np.loadtxt(RESULT / (arm + '_u0468_screen96_max2000_20261026') / (name+'.txt'), ndmin=2) for arm in trackers}
            for frame in event['frames']:
                image = np.concatenate([wrapper._read_image(sequence.frames[m][frame]) for m in [0, 1]], axis=2)
                for state_source in ['control', 'candidate']:
                    state = saved[state_source][frame-1].tolist()
                    boxes = {}
                    for mode in ['candidate_on', 'candidate_off', 'control']:
                        context.update(sequence=name, frame=frame, selection=event['selection'], state_source=state_source, intervention=mode)
                        tracker = trackers['control' if mode == 'control' else 'candidate']
                        tracker.state = list(state); tracker.prev_search = None; tracker.prev_temporal_tokens = None
                        for fusion in operators:
                            fusion.value_direction_enabled = mode != 'candidate_off'
                        boxes[mode] = np.asarray(tracker.track(image)['target_bbox'])
                    replay = 'candidate_on' if state_source == 'candidate' else 'control'
                    error = float(np.max(np.abs(boxes[replay] - saved[state_source][frame])))
                    if error > 2.e-4:
                        raise ValueError('Native replay failed: %s/%d/%s %.9g' % (name, frame, state_source, error))
                    comparisons.append({'sequence': name, 'frame': frame, 'selection': event['selection'],
                         'state_source': state_source, 'state': state, 'gt': gt[frame].tolist(), 'replay_maxabs': error,
                         **{mode: {'box': box.tolist(), 'iou': float(iou(box, gt[frame]))} for mode, box in boxes.items()}})
            print(json.dumps({'sequence': name, 'events_done': len(comparisons), 'seconds': time.time()-started}), flush=True)
    for hook in hooks:
        hook.remove()
    dump(OUT / 'common_crop.json', {'environment': environment(), 'checkpoints': checkpoints,
         'scope': 'retrospective one-step interventions from actual preceding native tracker states; no rollout or retraining conclusion',
         'gpu_wall_hours': (time.time()-started)/3600., 'comparisons': comparisons})
    dump(OUT / 'operator_readouts.json', call_rows)
    np.savez_compressed(OUT / 'per_token_readouts.npz', measurements=np.asarray(per_token),
                        channels=np.array(['relative_value_change','cross_attention_source_mass','colocated_receiver_fraction']))
    print(json.dumps({'comparisons':len(comparisons),'operator_calls':len(call_rows),'gpu_wall_hours':(time.time()-started)/3600.}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['temporal', 'probe'])
    args = p.parse_args()
    temporal() if args.mode == 'temporal' else probe()
