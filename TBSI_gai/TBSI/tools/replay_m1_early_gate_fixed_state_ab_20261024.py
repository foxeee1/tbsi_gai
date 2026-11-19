#!/usr/bin/env python3
"""Replay early gate on/off at identical saved candidate-control states."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker

EXP = ROOT / 'output/experiments/M1_EARLY_GATE_CANDIDATE_CHECKPOINT_AB_20261024'
FIXED_OUT = EXP / 'fixed_state_maps'
CFG = 'M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023_RETRY1'
CKPT = ROOT / 'output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023/checkpoints/TBSITrack_ep0015.pth.tar'
SEQUENCES = [
    'womanback2car', 'shinycarcoming', 'boydownplatform', '4thboywithwhite',
    'ab_bikeoccluded', '3rdfatboy', 'leftfarboycomingpicktheball',
]


def read_trace(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_events():
    events = []
    for name in SEQUENCES:
        rows = read_trace(EXP / 'trace/off' / name / 'frames.jsonl')
        n = len(rows)
        positions = sorted(set([0, round((n - 1) * 0.25), round((n - 1) * 0.50),
                                round((n - 1) * 0.75), n - 1]))
        for pos in positions:
            row = rows[pos]
            events.append({
                'sequence': name, 'frame_id': int(row['frame_id']),
                'state_before': row['tracker_state_before'],
                'gt_box': row['gt_box'],
                'control_crop_xywh': row['current_crop_xywh'],
            })
    return events


def run_arm(arm, strength, events, dataset):
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(CKPT)
    os.environ['TBSI_TEST_EARLY_GATE_STRENGTH'] = str(strength)
    os.environ['TBSI_INFER_FP16'] = '1'
    os.environ['TBSI_TEMPLATE_UPDATE_MODE'] = 'off'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_INTERNAL_DIAGNOSTIC'] = '1'
    os.environ['TBSI_ATTR_STATS'] = '1'
    tracker_info = Tracker('tbsi_track', CFG, 'lasher_test')
    params = tracker_info.get_parameters()
    params.debug = 0
    results = []
    head_maps = []
    sequence_map = {seq.name: seq for seq in dataset}
    for seq_name in SEQUENCES:
        seq = sequence_map[seq_name]
        tracker = tracker_info.create_tracker(params)
        init_info = seq.init_info()
        init_info['sequence_name'] = seq.name
        init_image = np.concatenate((tracker_info._read_image(seq.frames[0][0]),
                                     tracker_info._read_image(seq.frames[1][0])), axis=2)
        sequence_events = [event for event in events if event['sequence'] == seq_name]
        for event in sequence_events:
            frame_id = int(event['frame_id'])
            event_dir = FIXED_OUT / arm / seq.name / ('frame_{:06d}'.format(frame_id))
            os.environ['TBSI_M1_TRACE_DIR'] = str(event_dir)
            tracker.initialize(init_image, init_info)
            tracker.state = [float(v) for v in event['state_before']]
            tracker.frame_id = frame_id - 1
            image = np.concatenate((tracker_info._read_image(seq.frames[0][frame_id]),
                                    tracker_info._read_image(seq.frames[1][frame_id])), axis=2)
            frame_info = seq.frame_info(frame_id)
            frame_info.update({'sequence_name': seq.name, 'frame_num': frame_id,
                               'gt_bbox': np.asarray(event['gt_box'], dtype=np.float32)})
            tracker.track(image, frame_info)
            trace_file = event_dir / seq.name / 'frames.jsonl'
            row = read_trace(trace_file)[-1]
            results.append(row)
            diag = tracker._last_fixed_state_diag
            head_maps.append({
                'key': '{}:{}'.format(seq.name, frame_id),
                'score_map': diag['score_map'].numpy().astype(np.float32),
                'size_map': diag['size_map'].numpy().astype(np.float32),
                'offset_map': diag['offset_map'].numpy().astype(np.float32),
                'hann': tracker.output_window.detach().float().cpu().numpy().astype(np.float32),
            })
            print('[{}] {}/{} {} frame {}'.format(
                arm, len(results), len(events), seq.name, frame_id), flush=True)
    np.savez_compressed(
        FIXED_OUT / ('head_maps_{}.npz'.format(arm)),
        keys=np.asarray([row['key'] for row in head_maps]),
        score_map=np.stack([row['score_map'] for row in head_maps]),
        size_map=np.stack([row['size_map'] for row in head_maps]),
        offset_map=np.stack([row['offset_map'] for row in head_maps]),
        hann=np.stack([row['hann'] for row in head_maps]),
    )
    return results


def main():
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    events = select_events()
    FIXED_OUT.mkdir(parents=True, exist_ok=True)
    (FIXED_OUT / 'event_manifest.json').write_text(json.dumps(events, indent=2))
    dataset = get_dataset('lasher_test')
    off = run_arm('off', 0, events, dataset)
    on = run_arm('on', 1, events, dataset)
    payload = {'events': events, 'off': off, 'on': on}
    (FIXED_OUT / 'paired_trace.json').write_text(json.dumps(payload, indent=2))
    print('COMPLETE events={}'.format(len(events)))


if __name__ == '__main__':
    main()
