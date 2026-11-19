#!/usr/bin/env python3
"""Fixed-state stage attribution for query-conditioned M1 template participation."""
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
CONFIG_SHA256 = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
CHECKPOINT_SHA256 = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'
VALIDATION_LIST = ROOT / 'experiments/diagnostics/m1_identity_calibration_v2/validation_sequences.txt'
TEST_EVENTS = ROOT / 'output/experiments/M1_PRECISION_SCREEN_MANIFEST_20261020/selected_events.csv'
OUT = ROOT / 'output/experiments/M1_TEMPLATE_QUERY_STAGE_DIAGNOSTIC_20261020'


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


def search_crop(state, resize_factor, search_size):
    cx, cy = state[0] + state[2] / 2., state[1] + state[3] / 2.
    side = float(search_size) / float(resize_factor)
    return [cx - side / 2., cy - side / 2., side, side]


def crop_contains(gt, crop):
    x, y, w, h = gt
    cx, cy, side, _ = crop
    return x >= cx and y >= cy and x + w <= cx + side and y + h <= cy + side


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
    return mask


def tensor_feature_stats(tensor, mask, bg_yx):
    z = tensor.detach().float()[0].permute(1, 2, 0).cpu().numpy()
    t = z[mask].mean(axis=0)
    b = z[bg_yx[0], bg_yx[1]]
    return {'target_bg_l2': float(np.linalg.norm(t - b)),
            'target_bg_cosine': float(np.dot(t, b) /
                                      (np.linalg.norm(t) * np.linalg.norm(b) + 1e-12)),
            'target_norm': float(np.linalg.norm(t)),
            'background_norm': float(np.linalg.norm(b))}


def query_stats(rgb_t, tir_t, rgb_s, tir_s, fg_mask):
    rn = torch.nn.functional.normalize(rgb_t.float(), dim=-1, eps=1e-6)
    tn = torch.nn.functional.normalize(tir_t.float(), dim=-1, eps=1e-6)
    rs = torch.nn.functional.normalize(rgb_s.float(), dim=-1, eps=1e-6)
    ts = torch.nn.functional.normalize(tir_s.float(), dim=-1, eps=1e-6)
    agreement = (rn * tn).sum(-1)[0]
    q = .5 * (rn @ rs.transpose(1, 2)).amax(-1)[0] + \
        .5 * (tn @ ts.transpose(1, 2)).amax(-1)[0]
    reliability = .5 * agreement + .5 * q
    p = torch.sigmoid((reliability - .5) / .15).detach().cpu()
    q = q.detach().cpu()
    reliability = reliability.detach().cpu()
    fg = fg_mask.cpu().bool()
    bg = ~fg
    return {'query_match_mean': float(q.mean()),
            'reliability_mean': float(reliability.mean()),
            'participation_mean': float(p.mean()),
            'participation_fg_mean': float(p[fg].mean()) if fg.any() else float('nan'),
            'participation_bg_mean': float(p[bg].mean()) if bg.any() else float('nan'),
            'participation_fg_bg_delta': float(p[fg].mean() - p[bg].mean())
            if fg.any() and bg.any() else float('nan'),
            'query_fg_mean': float(q[fg].mean()) if fg.any() else float('nan'),
            'query_bg_mean': float(q[bg].mean()) if bg.any() else float('nan')}


def new_tracker(mode, tracker_label):
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(CHECKPOINT)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_INTERNAL_DIAGNOSTIC'] = '0'
    os.environ['TBSI_TEMPLATE_UNCERTAINTY_MODE'] = mode
    if mode in ('query_soft', 'message_gate'):
        os.environ['TBSI_TEMPLATE_UNCERTAINTY_THRESHOLD'] = '0.5'
        os.environ['TBSI_TEMPLATE_UNCERTAINTY_TEMPERATURE'] = '0.15'
    p = parameters(CONFIG_NAME)
    p.debug = 0
    p.save_all_boxes = False
    p.checkpoint = str(CHECKPOINT)
    return TBSITrack(p, tracker_label)


def discover_validation_events(seqmap, names):
    """Outcome rule frozen before run: first crop-contained frame with IoU < 0.30."""
    tracker = new_tracker('off', 'm1_query_stage_event_discovery')
    events = []
    for si, name in enumerate(names, 1):
        seq = seqmap[name]
        tracker.initialize(read_pair(seq, 0), {
            'init_bbox': np.asarray(seq.ground_truth_rect[0]), 'sequence_name': name})
        found = None
        for frame in range(1, min(len(seq.frames[0]), 161)):
            state_before = [float(v) for v in tracker.state]
            gt = [float(v) for v in seq.ground_truth_rect[frame]]
            pred = tracker.track(read_pair(seq, frame), {
                'gt_bbox': gt, 'sequence_name': name, 'frame_num': frame})
            diag = tracker._last_fixed_state_diag
            crop = search_crop(state_before, diag['resize_factor'], tracker.params.search_size)
            iou = box_iou(pred['target_bbox'], gt)
            if crop_contains(gt, crop) and iou < .30:
                found = {'sequence': name, 'frame': frame, 'role': 'validation',
                         'event_rule': 'first IoU<0.30 with GT fully inside predicted-state crop',
                         'state_before': state_before, 'gt_box': gt, 'crop_xywh': crop,
                         'discovery_iou': iou}
                break
        if found is not None:
            events.append(found)
        print('[discover] {}/{} {} {}'.format(
            si, len(names), name, 'event@{}'.format(found['frame']) if found else 'no eligible event'),
            flush=True)
    return events


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
        context = {}

        def outer_pre(module, inputs, ctx=context, key=slot):
            xv, xi, lens = inputs[:3]
            ctx['rgb_search'] = xv[:, int(lens):, :].detach().float()
            ctx['tir_search'] = xi[:, int(lens):, :].detach().float()

        def fusion_pre(module, inputs, ctx=context, key=slot):
            ctx['rgb_template'] = inputs[0].detach().float()
            ctx['tir_template'] = inputs[1].detach().float()
            if 'rgb_search' in ctx:
                mask = tracker.c1_template_mask
                stats = query_stats(ctx['rgb_template'], ctx['tir_template'],
                                    ctx['rgb_search'], ctx['tir_search'], mask)
                captured.setdefault(key, {})['query'] = stats
                captured[key]['template_pre'] = (
                    ctx['rgb_template'].cpu(), ctx['tir_template'].cpu())

        def fusion_post(module, inputs, output, ctx=context, key=slot):
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                captured.setdefault(key, {})['template_post'] = (
                    output[0].detach().float().cpu(), output[1].detach().float().cpu())

        def layer_post(module, inputs, output, key=slot):
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                lens = int(inputs[2])
                captured.setdefault(key, {})['search_post'] = (
                    output[0][:, lens:, :].detach().float().cpu(),
                    output[1][:, lens:, :].detach().float().cpu())

        handles.extend([layer.register_forward_pre_hook(outer_pre),
                        fusion.register_forward_pre_hook(fusion_pre),
                        fusion.register_forward_hook(fusion_post),
                        layer.register_forward_hook(layer_post)])

    def fuse_pre(module, inputs):
        captured['search_fusion_input'] = inputs[0].detach().float().cpu()

    def fuse_post(module, inputs, output):
        captured['search_fusion_output'] = output.detach().float().cpu()

    handles.extend([net.tbsi_fuse_search.register_forward_pre_hook(fuse_pre),
                    net.tbsi_fuse_search.register_forward_hook(fuse_post)])
    return handles


def run_fixed_arm(event, seq, mode, bg_yx=None):
    tracker = new_tracker(mode, 'm1_query_stage_fixed_state')
    captured = {}
    handles = attach_hooks(tracker, captured)
    tracker.initialize(read_pair(seq, 0), {
        'init_bbox': np.asarray(seq.ground_truth_rect[0]), 'sequence_name': seq.name})
    tracker.state = list(map(float, event['state_before']))
    tracker.prev_search = None
    tracker.prev_temporal_tokens = None
    tracker.prev_head_feature = None
    pred = tracker.track(read_pair(seq, int(event['frame'])), {
        'gt_bbox': event['gt_box'], 'sequence_name': seq.name, 'frame_num': int(event['frame'])})
    diag = tracker._last_fixed_state_diag
    mask = grid_mask(event['gt_box'], event['crop_xywh'])
    score = diag['score_map'][0, 0].numpy()
    score = score * tracker.output_window[0, 0].detach().float().cpu().numpy()
    if bg_yx is None:
        bgflat = np.where(~mask, score, -np.inf)
        bg_yx = np.unravel_index(int(np.argmax(bgflat)), score.shape)
    target_peak = float(score[mask].max())
    bg_peak = float(score[bg_yx])
    gt_rank = 1 + int(np.sum(score[~mask] > target_peak))
    top = np.unravel_index(int(np.argmax(score)), score.shape)
    row = {'sequence': seq.name, 'frame': int(event['frame']), 'role': event['role'], 'mode': mode,
           'fixed_state_iou': box_iou(pred['target_bbox'], event['gt_box']),
           'gt_rank': gt_rank, 'top1_is_gt': bool(mask[top]),
           'gt_peak': target_peak, 'hard_bg_peak': bg_peak,
           'gt_hard_bg_margin': target_peak - bg_peak,
           'hard_bg_y': int(bg_yx[0]), 'hard_bg_x': int(bg_yx[1]),
           'state_before': json.dumps(event['state_before']),
           'gt_box': json.dumps(event['gt_box']), 'crop_xywh': json.dumps(event['crop_xywh'])}
    for key, tensor_key in (('search_fusion_input', 'pre_fusion'),
                            ('search_fusion_output', 'post_fusion')):
        tensor = captured.get(key)
        if tensor is not None:
            if key == 'search_fusion_input':
                c = tensor.shape[1] // 2
                for mod, part in (('rgb', tensor[:, :c]), ('tir', tensor[:, c:])):
                    for stat, value in tensor_feature_stats(part, mask, bg_yx).items():
                        row['{}_{}_{}'.format(tensor_key, mod, stat)] = value
            else:
                for stat, value in tensor_feature_stats(tensor, mask, bg_yx).items():
                    row['{}_{}'.format(tensor_key, stat)] = value
    row['center_gt_hard_bg_margin'] = target_peak - bg_peak
    for layer_name, data in captured.items():
        if not isinstance(data, dict):
            continue
        for stat, value in data.get('query', {}).items():
            row['{}_{}'.format(layer_name, stat)] = value
        post = data.get('template_post')
        if post:
            fg = tracker.c1_template_mask.cpu().bool()
            for mod, values in zip(('rgb', 'tir'), post):
                z = values[0].numpy()
                t, b = z[fg].mean(0), z[~fg].mean(0)
                row['{}_{}_template_fg_bg_l2'.format(layer_name, mod)] = float(np.linalg.norm(t-b))
                row['{}_{}_template_fg_bg_cosine'.format(layer_name, mod)] = float(
                    np.dot(t, b) / (np.linalg.norm(t) * np.linalg.norm(b) + 1e-12))
        search = data.get('search_post')
        if search:
            for mod, values in zip(('rgb', 'tir'), search):
                feat = values.transpose(1, 2).reshape(1, values.shape[-1], 16, 16)
                for stat, value in tensor_feature_stats(feat, mask, bg_yx).items():
                    row['{}_{}_search_{}'.format(layer_name, mod, stat)] = value
    for handle in handles:
        handle.remove()
    return row, tuple(map(int, bg_yx))


def test_case_events():
    events = []
    with TEST_EVENTS.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['sequence'] not in ('whitesuvturn', 'leftunderbasket'):
                continue
            events.append({'sequence': row['sequence'], 'frame': int(row['frame']),
                           'role': 'test_case_study',
                           'state_before': json.loads(row['tracker_state_before']),
                           'gt_box': json.loads(row['gt_box']),
                           'crop_xywh': json.loads(row['crop_xywh'])})
    return events


def main():
    if sha256(CONFIG) != CONFIG_SHA256 or sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-list', default=str(VALIDATION_LIST))
    parser.add_argument('--split', default='trainingset', choices=('trainingset', 'testingset'))
    parser.add_argument('--output', default=str(OUT))
    parser.add_argument('--uncertainty-mode', default='query_soft',
                        choices=('query_soft', 'message_gate'))
    parser.add_argument('--include-test-cases', action='store_true')
    args = parser.parse_args()
    sequence_list = Path(args.sequence_list).resolve()
    output_root = Path(args.output).resolve()
    if not sequence_list.is_file():
        raise FileNotFoundError(str(sequence_list))
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError('refusing to reuse non-empty output directory: {}'.format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    names = [x.strip() for x in sequence_list.read_text().splitlines() if x.strip()]
    list_path = output_root / 'sequence_list.txt'
    list_path.write_text('\n'.join(names) + '\n')
    validation = LasHeRDataset(args.split, str(list_path)).get_sequence_list()
    seqmap = {s.name: s for s in validation}
    event_path = output_root / 'validation_events.json'
    if event_path.exists():
        dev_events = json.loads(event_path.read_text())
    else:
        dev_events = discover_validation_events(seqmap, names)
        event_path.write_text(json.dumps(dev_events, indent=2) + '\n')

    test_events = test_case_events() if args.include_test_cases else []
    if test_events:
        test_names = sorted({e['sequence'] for e in test_events})
        test_list = output_root / 'test_case_sequence_list.txt'
        test_list.write_text('\n'.join(test_names) + '\n')
        testset = LasHeRDataset('testingset', str(test_list)).get_sequence_list()
        seqmap.update({s.name: s for s in testset})
    all_events = dev_events + test_events
    if not dev_events:
        raise RuntimeError('no eligible crop-contained validation events discovered')

    rows = []
    for idx, event in enumerate(all_events, 1):
        seq = seqmap[event['sequence']]
        off, bg = run_fixed_arm(event, seq, 'off')
        query, _ = run_fixed_arm(event, seq, args.uncertainty_mode, bg)
        off['pair_id'] = query['pair_id'] = '{}:{}'.format(event['sequence'], event['frame'])
        for key, value in off.items():
            if key not in ('pair_id', 'sequence', 'frame', 'role', 'mode', 'state_before', 'gt_box', 'crop_xywh'):
                query['off_' + key] = value
                if isinstance(value, (int, float)) and isinstance(query.get(key), (int, float)):
                    query[key + '_delta'] = query[key] - value
        rows.extend((off, query))
        print('[fixed] {}/{} {} frame={} off_margin={:.5f} query_margin={:.5f} rank={}->{}'.format(
            idx, len(all_events), event['sequence'], event['frame'],
            off['gt_hard_bg_margin'], query['gt_hard_bg_margin'], off['gt_rank'], query['gt_rank']),
            flush=True)

    columns = sorted({k for row in rows for k in row})
    with (output_root / 'per_event.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {'protocol': {'validation_split': 'LasHeR {}; supplied sequence manifest'.format(args.split),
                            'event_rule': 'first frame <=160 with predicted IoU <0.30 and GT fully inside crop',
                            'arms': ['off', '{} tau=0.5 T=0.15'.format(args.uncertainty_mode)],
                            'hard_background': 'highest off-arm CENTER*Hann cell outside GT, reused in query arm',
                            'test_case_studies_excluded_from_validation_summary': True},
               'validation_event_count': len(dev_events),
               'test_case_study_count': len(test_events),
               'validation_sequences_without_event': sorted(set(names) - {e['sequence'] for e in dev_events}),
               'config_sha256': sha256(CONFIG), 'checkpoint_sha256': sha256(CHECKPOINT),
               'git_head': os.popen('git -C "{}" rev-parse HEAD'.format(ROOT)).read().strip(),
               'tbsi_layer_sha256': sha256(ROOT / 'lib/models/layers/tbsi_layer.py'),
               'tracker_sha256': sha256(ROOT / 'lib/test/tracker/tbsi_track.py'),
               'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
               'torch': torch.__version__, 'cuda': torch.version.cuda}
    for role in ('validation', 'test_case_study'):
        selected = [r for r in rows if r['role'] == role and r['mode'] == args.uncertainty_mode]
        if selected:
            summary[role] = {
                'n': len(selected),
                'mean_margin_delta': float(np.mean([r['gt_hard_bg_margin_delta'] for r in selected])),
                'mean_rank_delta': float(np.mean([r['gt_rank_delta'] for r in selected])),
                'rank_improved': int(sum(r['gt_rank_delta'] < 0 for r in selected)),
                'top1_became_gt': int(sum(not r['off_top1_is_gt'] and r['top1_is_gt'] for r in selected)),
                'participation_fg_bg_delta_mean': float(np.mean([
                    r['loc9_participation_fg_bg_delta'] for r in selected
                    if 'loc9_participation_fg_bg_delta' in r])),
                'query_fg_bg_delta_mean': float(np.mean([
                    r['loc9_query_fg_mean'] - r['loc9_query_bg_mean'] for r in selected
                    if 'loc9_query_fg_mean' in r and 'loc9_query_bg_mean' in r]))}
    summary['protocol']['sequence_list'] = str(sequence_list)
    summary['protocol']['split'] = args.split
    summary['protocol']['uncertainty_mode'] = args.uncertainty_mode
    summary['protocol']['sequence_count_requested'] = len(names)
    (output_root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
