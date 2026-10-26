#!/usr/bin/env python3
"""Run one frozen-state arm for the M1 priority-2 reliability mechanism."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FLOW_PATH = ROOT / 'tools/run_m1_screening_fixed_state_flow_20261022.py'
SPEC = importlib.util.spec_from_file_location('m1_fixed_flow', str(FLOW_PATH))
flow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flow)
base = flow.base


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('parent', 'independent', 'shared', 'swapped'), required=True)
    parser.add_argument('--events', required=True)
    parser.add_argument('--sequence-list', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-states', type=int, default=0)
    args = parser.parse_args()
    event_path = Path(args.events).resolve()
    sequence_list = Path(args.sequence_list).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not event_path.is_file() or not sequence_list.is_file():
        raise FileNotFoundError('events or sequence list not found')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError('refusing to overwrite non-empty output directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    if sha256(base.CONFIG) != base.CONFIG_SHA256 or sha256(base.CHECKPOINT) != base.CHECKPOINT_SHA256:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')

    mode = {'parent': 'off', 'independent': 'p2_independent',
            'shared': 'p2_shared', 'swapped': 'p2_swapped'}[args.arm]
    os.environ['TBSI_NATIVE_MSHA_MESSAGE_MODE'] = mode
    os.environ['TBSI_NATIVE_MSHA_MESSAGE_ALPHA'] = '0.25'
    os.environ['TBSI_NATIVE_MSHA_MESSAGE_TEMPERATURE'] = '0.25'
    os.environ['TBSI_P2_SCORE_TEMPERATURE'] = '0.07'
    os.environ['TBSI_PRIORITY2_CAPTURE'] = '1'
    os.environ['TBSI_INFER_FP16'] = '0'
    os.environ['TBSI_FIXED_STATE_DIAG'] = '1'
    os.environ['TBSI_INTERNAL_DIAGNOSTIC'] = '0'
    os.environ['TBSI_TEMPLATE_UNCERTAINTY_MODE'] = 'off'
    os.environ['TBSI_SPARSE_CROSS_MODE'] = 'off'
    torch.set_num_threads(1)

    events = flow.read_events(event_path)
    names = [line.strip() for line in sequence_list.read_text().splitlines() if line.strip()]
    if set(events) - set(names):
        raise RuntimeError('event sequence outside the supplied cohort')
    tracker = base.new_tracker('off', 'm1_priority2_directional_reliability')
    sequence_map = {seq.name: seq for seq in base.LasHeRDataset(
        'trainingset', str(sequence_list)).get_sequence_list()}
    rows = []
    for name in sorted(events):
        sequence = sequence_map[name]
        tracker.initialize(base.read_pair(sequence, 0), {
            'init_bbox': sequence.ground_truth_rect[0], 'sequence_name': name})
        for event in events[name]:
            if args.max_states and len(rows) >= args.max_states:
                break
            row, _ = flow.flow.run_event(tracker, event, sequence)
            fusion_layers = [layer.v2_template_fusion for layer in
                             tracker.network.backbone.tbsi_layers]
            stats = [dict(getattr(layer, 'last_native_message_stats', {'mode': 'missing'}))
                     for layer in fusion_layers]
            if len(stats) != 3 or any(item.get('mode') != mode for item in stats):
                if args.arm != 'parent':
                    raise RuntimeError('priority-2 mode did not reach all three MSHA locations')
            row['native_msha_stats'] = json.dumps(stats, sort_keys=True)
            row['arm'] = args.arm
            rows.append(row)
            print('[{}] {}/{} {} {} frame={} replay_iou={:.4f} rank={} margin={:+.5f}'.format(
                args.arm, len(rows), sum(len(items) for items in events.values()), name,
                event['phase'], event['frame'], row['fixed_replay_iou'],
                row['center_gt_rank'], row['center_gt_hard_bg_margin']), flush=True)
        if args.max_states and len(rows) >= args.max_states:
            break

    if not rows:
        raise RuntimeError('no fixed states were evaluated')
    columns = sorted({key for row in rows for key in row})
    with (output_dir / 'per_event.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        'experiment': 'M1-PRIORITY2-DIRECTIONAL-RELIABILITY-FIXED46-20261023',
        'arm': args.arm, 'native_mode': mode,
        'parent': base.CONFIG_NAME,
        'config_sha256': sha256(base.CONFIG), 'checkpoint_sha256': sha256(base.CHECKPOINT),
        'runner_sha256': sha256(Path(__file__)), 'fixed_flow_sha256': sha256(FLOW_PATH),
        'events_sha256': sha256(event_path), 'sequence_list_sha256': sha256(sequence_list),
        'split': 'LasHeR official-train calibration cohort; previously exposed; mechanism diagnosis only',
        'state_count': len(rows), 'precision': 'FP32', 'torch_threads': 1,
        'alpha': 0.25, 'score_temperature': 0.07, 'reliability_temperature': 0.25,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'interpretation': 'fixed-state mechanism comparison; no training or tracking benchmark claim',
    }
    (output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'arm': args.arm, 'fixed_states': len(rows),
                      'output_dir': str(output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
