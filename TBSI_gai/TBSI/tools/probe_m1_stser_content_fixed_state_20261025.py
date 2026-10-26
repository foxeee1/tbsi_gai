"""Replay one sequence with identical external crop states for two probe modes."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.test.utils.load_text import load_text
from analyze_m1_stser_content_pair_20261025 import sha
from run_m1_stser_content_campaign_20261025 import contracts, process_gate
from lib.models.layers.search_conditioned_t2s_router import SearchConditionedT2SRouter


def iou(a, b):
    left = max(a[0], b[0]), max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right[0] - left[0]) * max(0, right[1] - left[1])
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return float(intersection / union) if union > 0 else 0.0


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('quick3', 'full15'), required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--mode', choices=('off', 'roll_mask', 'roll_competition',
                                           'scale_only'), required=True)
    parser.add_argument('--max-frames', type=int, default=0,
                        help='0 means all frames; positive values are exploratory subsets')
    args = parser.parse_args()
    process_gate()
    contracts(args.scope)
    campaign = ROOT / 'experiments/snapshots/m1_stser_content_campaign_{}_20261025'.format(args.scope)
    lock = json.loads((campaign / 'contract.json').read_text())
    for relative, expected in lock['source_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('source changed after paired preparation: ' + relative)
    output = campaign / 'fixed_state_{}_{}.csv'.format(args.sequence, args.mode)
    if output.exists():
        raise FileExistsError(str(output))
    if args.scope == 'quick3':
        manifest = Path(lock['test_manifest'])
        if sha(manifest) != lock['quick_test_manifest_sha256']:
            raise ValueError('test manifest changed')
        os.environ['TBSI_RPP_TEST_LIST'] = str(manifest)
    dataset = get_dataset('rpp_lasher_test' if args.scope == 'quick3' else 'lasher_test')
    matches = [sequence for sequence in dataset if sequence.name == args.sequence]
    if len(matches) != 1:
        raise ValueError('sequence is not in the locked evaluation scope')
    sequence = matches[0]
    name = 'M1_STSER_CONTENT_{}_CANDIDATE_20261025'.format(args.scope.upper())
    epoch = 3 if args.scope == 'quick3' else 15
    checkpoint = ROOT / 'output/experiments' / name / 'checkpoints/TBSITrack_ep{:04d}.pth.tar'.format(epoch)
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    def completed_status(arm):
        resume = campaign / ('resume-test-' + arm + '.json')
        primary = campaign / ('test-' + arm + '.json')
        return json.loads((resume if resume.is_file() else primary).read_text())

    candidate_status = completed_status('candidate')
    control_status = completed_status('control')
    if (candidate_status.get('status') != 'complete' or
            control_status.get('status') != 'complete' or
            candidate_status['checkpoint_sha256_after'] != sha(checkpoint)):
        raise RuntimeError('both tests must be complete and candidate checkpoint unchanged')
    # The external state comes from the matched control arm, not the candidate.
    control_name = name.replace('_CANDIDATE_', '_CONTROL_')
    reference = (ROOT / 'output/test/tracking_results/tbsi_track' / control_name /
                 (sequence.name + '.txt'))
    if not reference.is_file():
        raise FileNotFoundError(str(reference))
    boxes = np.asarray(load_text(str(reference), delimiter=('\t', ','), dtype=np.float64)).reshape(-1, 4)
    gt = np.asarray(sequence.ground_truth_rect, dtype=np.float64).reshape(-1, 4)
    if len(boxes) != len(gt) or len(sequence.frames[0]) != len(gt):
        raise ValueError('reference boxes, GT, and RGB frames differ in length')
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(checkpoint)
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_TEMPLATE_UPDATE_MODE'] = 'off'
    os.environ['TBSI_STSER_CONTENT_PROBE'] = 'correct'
    params = parameters(name)
    if (getattr(params.cfg.MODEL, 'TEMPORAL_TOKENS', False) or
            getattr(params.cfg.MODEL, 'CAUSAL_TAM', False)):
        raise RuntimeError('fixed-state replay requires no temporal model state')
    params.debug = 0
    tracker = TBSITrack(params, sequence.dataset)
    routers = [module for module in tracker.network.modules()
               if isinstance(module, SearchConditionedT2SRouter)]
    if len(routers) != 1:
        raise RuntimeError('expected one content STSER layer')
    init_image = np.concatenate([read_rgb(sequence.frames[0][0]),
                                 read_rgb(sequence.frames[1][0])], axis=2)
    info = sequence.init_info()
    info['sequence_name'] = sequence.name
    tracker.initialize(init_image, info)
    limit = len(gt) if args.max_frames <= 0 else min(len(gt), args.max_frames)
    if limit <= 1:
        raise ValueError('at least two frames are required')
    rows = []
    for frame in range(1, limit):
        source_box = boxes[frame - 1].tolist()
        if not np.isfinite(source_box).all() or source_box[2] <= 0 or source_box[3] <= 0:
            raise ValueError('invalid control state at frame {}'.format(frame - 1))
        image = np.concatenate([read_rgb(sequence.frames[0][frame]),
                                read_rgb(sequence.frames[1][frame])], axis=2)
        predictions = {}
        deltas = {}
        previous_frame_id = tracker.frame_id
        for mode in ('correct', args.mode):
            os.environ['TBSI_STSER_CONTENT_PROBE'] = mode
            tracker.state = list(source_box)
            tracker.frame_id = previous_frame_id
            tracker.prev_search = None
            tracker.prev_temporal_tokens = None
            tracker.prev_head_feature = None
            tracker.prev_crop_xywh = None
            out = tracker.track(image, {'sequence_name': sequence.name, 'frame_num': frame})
            predictions[mode] = [float(x) for x in out['target_bbox']]
            deltas[mode] = {key: routers[0].last_stats[key] for key in
                            ('rgb_bounded_delta_norm', 'tir_bounded_delta_norm')}
        a, b = predictions['correct'], predictions[args.mode]
        rows.append({'sequence': sequence.name, 'frame': frame,
                     'external_state_xywh': json.dumps(source_box),
                     'gt_xywh': json.dumps(gt[frame].tolist()),
                     'correct_xywh': json.dumps(a),
                     'probe_xywh': json.dumps(b),
                     'correct_iou': iou(a, gt[frame]),
                     'probe_iou': iou(b, gt[frame]),
                     'probe_minus_correct_iou': iou(b, gt[frame]) - iou(a, gt[frame]),
                     'correct_rgb_bounded_delta_norm': deltas['correct']['rgb_bounded_delta_norm'],
                     'probe_rgb_bounded_delta_norm': deltas[args.mode]['rgb_bounded_delta_norm'],
                     'correct_tir_bounded_delta_norm': deltas['correct']['tir_bounded_delta_norm'],
                     'probe_tir_bounded_delta_norm': deltas[args.mode]['tir_bounded_delta_norm'],
                     'center_distance_px': math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2,
                                                      a[1] + a[3] / 2 - b[1] - b[3] / 2)})
    with output.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'output': str(output), 'checkpoint_sha256': sha(checkpoint),
                      'control_reference_sha256': sha(reference), 'frames': len(rows),
                      'mean_probe_minus_correct_iou': float(np.mean(
                          [row['probe_minus_correct_iou'] for row in rows]))}, indent=2))


if __name__ == '__main__':
    main()
