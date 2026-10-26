#!/usr/bin/env python3
"""Audit saved full-test first updates against fixed-state model replays."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_m1_adaptive_first_update_causal_diagnostic_20261021 import (
    ROOT, build_tracker, iou, run_event,
)
from lib.test.evaluation.lasherdataset import LasHeRDataset


OUT = ROOT / 'output/experiments/M1_AMCM_ADAPTIVE_TEST_FIRST_UPDATE_PRECISION_REPLAY_ENTROPYFIX_20261021'
NAMES = ('bikeboyturntimes', 'blackboy', '10runone', 'runningcameragirl')
OFFICIAL = {
    'm1': ROOT / 'output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42',
    'adaptive': ROOT / 'output/test/tracking_results/tbsi_track/M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021_3',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def first_saved_box(path):
    with path.open() as stream:
        next(stream)
        return [float(value) for value in next(stream).split()]


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('output directory is not empty: {}'.format(OUT))
    OUT.mkdir(parents=True, exist_ok=True)
    sequence_file = OUT / 'sequence_list.txt'
    sequence_file.write_text('\n'.join(NAMES) + '\n')
    sequences = LasHeRDataset('testingset', str(sequence_file)).get_sequence_list()
    trackers = {'m1': build_tracker('m1'), 'm1_fp16': build_tracker('m1'),
                'adaptive': build_tracker('adaptive'),
                'adaptive_fp16': build_tracker('adaptive')}
    for arm in ('m1', 'adaptive'):
        half_tracker = trackers[arm + '_fp16']
        half_tracker.network.half()
        half_tracker.output_window = half_tracker.output_window.half()
        half_tracker.use_fp16 = True
        trackers[arm + '_fp16'] = half_tracker
    rows = []
    for sequence in sequences:
        init_box = [float(value) for value in sequence.ground_truth_rect[0]]
        gt_box = [float(value) for value in sequence.ground_truth_rect[1]]
        center_x = init_box[0] + init_box[2] / 2.0
        center_y = init_box[1] + init_box[3] / 2.0
        side = int(np.ceil(np.sqrt(init_box[2] * init_box[3]) * 4.0))
        event = {
            'sequence': sequence.name,
            'frame': 1,
            'state_before': init_box,
            'gt_box': gt_box,
            'crop_xywh': [float(int(round(center_x - side / 2.0))),
                          float(int(round(center_y - side / 2.0))),
                          float(side), float(side)],
            'm1_trace_iou': float('nan'),
        }
        for arm in ('m1', 'adaptive', 'zero', 'm1_fp16', 'adaptive_fp16'):
            tracker = trackers['m1' if arm == 'm1' else 'adaptive'
                               if arm == 'zero' else arm]
            replay = run_event('zero' if arm == 'zero' else 'adaptive'
                               if arm == 'adaptive_fp16' else 'm1'
                               if arm == 'm1_fp16' else arm, event, sequence, tracker)
            saved_path = OFFICIAL['m1' if arm.startswith('m1') else 'adaptive'] / (sequence.name + '.txt')
            saved_box = first_saved_box(saved_path)
            replay_box = json.loads(replay['predicted_bbox'])
            rows.append({
                'sequence': sequence.name, 'arm': arm,
                'saved_bbox': json.dumps(saved_box), 'replayed_bbox': json.dumps(replay_box),
                'saved_iou': iou(saved_box, gt_box),
                'replayed_iou': replay['fixed_state_iou'],
                'max_abs_bbox_delta': max(abs(a - b) for a, b in zip(saved_box, replay_box)),
                'gt_rank': replay['gt_rank'], 'gt_hard_bg_margin': replay['gt_hard_bg_margin'],
                'top1_x': replay['top1_x'], 'top1_y': replay['top1_y'],
                'decoded_width_px': replay['decoded_width_px'],
                'decoded_height_px': replay['decoded_height_px'],
                'residual_rgb_input_ratio': replay['amcm_residual_rgb_norm'] /
                    max(replay['amcm_input_rgb_norm'], 1e-12),
                'residual_tir_input_ratio': replay['amcm_residual_tir_norm'] /
                    max(replay['amcm_input_tir_norm'], 1e-12),
                'official_file': str(saved_path.relative_to(ROOT)),
                'official_sha256': sha256(saved_path),
            })
            print('{} {} saved_iou={:.4f} replay_iou={:.4f} bbox_delta={:.3f} rank={}'.format(
                sequence.name, arm, rows[-1]['saved_iou'], rows[-1]['replayed_iou'],
                rows[-1]['max_abs_bbox_delta'], rows[-1]['gt_rank']), flush=True)
    with (OUT / 'per_arm.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'summary.json').write_text(json.dumps({
        'id': 'M1-AMCM-ADAPTIVE-TEST-FIRST-UPDATE-REPLAY-20261021',
        'purpose': 'Provenance audit only; test cases are not used to select module parameters.',
        'sequences': list(NAMES), 'arms': ['m1', 'adaptive', 'zero', 'm1_fp16', 'adaptive_fp16'],
        'dataset': 'LasHeR testingset, frame 1 after initialization',
        'code': 'working tree; fixed-state diagnostic runner imported without modification',
        'rows': rows,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
