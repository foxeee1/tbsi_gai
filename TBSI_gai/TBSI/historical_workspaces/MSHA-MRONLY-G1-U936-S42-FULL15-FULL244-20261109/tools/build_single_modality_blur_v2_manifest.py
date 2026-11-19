#!/usr/bin/env python
"""Freeze a new, outcome-independent LasHeR train event manifest.

The 24 sequences are sampled from the alphabetically sorted train set after
excluding every sequence used by SJI/single-modality-degradation-v1. Three
frames per sequence are selected by a fixed uniform-in-time rule. No IoU or
tracker output is read when making this manifest.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from lib.test.evaluation.environment import env_settings
from synthetic_joint_interference_v1 import load_train_sequence


def sha256_file(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-manifest', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--n-sequences', type=int, default=24)
    ap.add_argument('--events-per-sequence', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20260926)
    ap.add_argument('--horizon', type=int, default=5)
    args = ap.parse_args()

    old = json.load(open(args.old_manifest))
    excluded = set(item['seq'] for item in old)
    settings = env_settings()
    train_root = os.path.join(settings.lasher_path, 'trainingset')
    available = sorted(name for name in os.listdir(train_root)
                       if os.path.isdir(os.path.join(train_root, name)) and name not in excluded)
    if len(available) < args.n_sequences:
        raise RuntimeError('only {} eligible sequences, need {}'.format(len(available), args.n_sequences))
    chooser = random.Random(args.seed)
    selected = sorted(chooser.sample(available, args.n_sequences))
    events = []
    for seq_index, name in enumerate(selected):
        sequence = load_train_sequence(settings.lasher_path, name)
        n = min(len(sequence.frames[0]), len(sequence.frames[1]), len(sequence.ground_truth_rect))
        max_frame = n - 1 - args.horizon
        if max_frame < args.events_per_sequence:
            raise RuntimeError('sequence {} has too few usable frames: {}'.format(name, n))
        # Three fixed positions, evenly spaced between after initialization and
        # the last frame with a complete H=5 future window.
        positions = np.linspace(1, max_frame, args.events_per_sequence)
        positions = [int(round(value)) for value in positions]
        if len(set(positions)) != args.events_per_sequence:
            raise RuntimeError('non-unique uniform positions for {}'.format(name))
        for local_index, frame in enumerate(positions):
            gt = [float(x) for x in sequence.ground_truth_rect[frame]]
            events.append({
                'event_id': 'SMB2-{:04d}'.format(len(events) + 1),
                'seq': name,
                'frame': frame,
                'gt': gt,
                'sequence_index': seq_index,
                'frame_index': local_index,
                'n_frames': n,
                'sampling_rule': 'round(linspace(1, n-1-horizon, 3))',
            })
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {
        'experiment': 'single_modality_blur_v2',
        'parent': 'single_modality_degradation_v1',
        'source_old_manifest': os.path.abspath(args.old_manifest),
        'source_old_manifest_sha256': sha256_file(args.old_manifest),
        'excluded_sequence_count': len(excluded),
        'eligible_sequence_count': len(available),
        'sequence_selection': {'seed': args.seed, 'n_sequences': args.n_sequences,
                               'rule': 'sorted eligible names; random.sample; final sort'},
        'frame_selection': {'events_per_sequence': args.events_per_sequence,
                           'horizon': args.horizon,
                           'rule': 'round(linspace(1, n_frames-1-horizon, events_per_sequence))'},
        'sequences': selected,
        'events': events,
        'note': 'Development train sequences only; not an unseen validation set for epoch-15 checkpoint.',
    }
    with open(os.path.join(args.output_dir, 'manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2)
    with open(os.path.join(args.output_dir, 'sequences.txt'), 'w') as handle:
        handle.write('\n'.join(selected) + '\n')
    with open(os.path.join(args.output_dir, 'events.csv'), 'w') as handle:
        handle.write('event_id,seq,frame\n')
        for event in events:
            handle.write('{},{},{}\n'.format(event['event_id'], event['seq'], event['frame']))
    print('FROZEN manifest={} sequences={} events={} excluded={}'.format(
        os.path.abspath(os.path.join(args.output_dir, 'manifest.json')), len(selected), len(events), len(excluded)))


if __name__ == '__main__':
    main()
