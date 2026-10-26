#!/usr/bin/env python3
"""Run full-test frozen-gradient diagnostics in resumable 30-sequence batches."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m1_fusion_flow_testset_diagnostic_20261009 import correlations

EXPERIMENT_ID = 'M1-FUSION-FLOW-FULL244-DIAG-20261009'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(str(tmp), str(path))


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch-size', type=int, default=30)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    trace_dir = root / 'closed_loop_trace' / 'trace'
    out = root / 'gradient_sampled_32'
    batch_root = out / 'batches'
    state_path = out / 'batch_state.json'
    names = [line.strip() for line in (trace_dir / 'trace_sequences.txt').read_text().splitlines()
             if line.strip()]
    if len(names) != 244:
        raise RuntimeError('gradient stage requires all 244 trace sequences')
    trace_manifest = json.loads((trace_dir / 'manifest.json').read_text())
    if trace_manifest.get('status') != 'TRACE-COMPLETE':
        raise RuntimeError('closed-loop trace is not marked TRACE-COMPLETE')
    if trace_manifest.get('completed_count') != 244:
        raise RuntimeError('closed-loop manifest does not confirm 244 completed sequences')
    out.mkdir(parents=True, exist_ok=True)
    batch_root.mkdir(exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        'experiment_id': EXPERIMENT_ID, 'trace_manifest_sha256': sha256(trace_dir / 'manifest.json'),
        'batch_size': args.batch_size, 'completed_batches': [], 'status': 'RUNNING'}
    if (state['experiment_id'] != EXPERIMENT_ID or
            state['trace_manifest_sha256'] != sha256(trace_dir / 'manifest.json') or
            state['batch_size'] != args.batch_size):
        raise RuntimeError('gradient batch state provenance mismatch')

    batches = [names[i:i + args.batch_size] for i in range(0, len(names), args.batch_size)]
    for batch_index, batch in enumerate(batches, 1):
        batch_name = 'batch_{:02d}'.format(batch_index)
        batch_out = batch_root / batch_name
        if batch_index in state['completed_batches']:
            summary_file = batch_out / 'summary.json'
            per_frame_file = batch_out / 'per_frame.csv'
            if summary_file.is_file() and per_frame_file.is_file():
                existing_summary = json.loads(summary_file.read_text())
                expected_rows = sum(existing_summary.get(
                    'sampled_trace_frames_per_sequence', {}).values())
                if (existing_summary.get('frame_count') == expected_rows and
                        existing_summary.get('parity_pass') and
                        len(read_csv(per_frame_file)) == expected_rows):
                    print('{} already complete; artifacts validated'.format(batch_name), flush=True)
                    continue
            if batch_index in state['completed_batches']:
                state['completed_batches'].remove(batch_index)
        batch_trace = batch_root / (batch_name + '_trace')
        if batch_trace.exists():
            shutil.rmtree(str(batch_trace))
        batch_trace.mkdir()
        (batch_trace / 'trace_sequences.txt').write_text('\n'.join(batch) + '\n')
        os.symlink(str(trace_dir / 'manifest.json'), str(batch_trace / 'manifest.json'))
        for name in batch:
            os.symlink(str(trace_dir / name), str(batch_trace / name))
        if batch_out.exists() and any(batch_out.iterdir()):
            shutil.rmtree(str(batch_out))
        command = [sys.executable, '-u', str(ROOT / 'tools/m1_fusion_flow_testset_diagnostic_20261009.py'),
                   '--trace-dir', str(batch_trace), '--checkpoint', args.checkpoint,
                   '--config', args.config, '--split', 'testingset',
                   '--frames-per-sequence', '32', '--output-dir', str(batch_out)]
        print('starting {} ({}/{}) sequences={}'.format(
            batch_name, batch_index, len(batches), len(batch)), flush=True)
        subprocess.run(command, check=True)
        state['completed_batches'].append(batch_index)
        state['completed_batches'] = sorted(set(state['completed_batches']))
        state['updated_unix_time'] = __import__('time').time()
        state['status'] = 'RUNNING'
        atomic_json(state_path, state)
        shutil.rmtree(str(batch_trace))

    all_rows = []
    batch_summaries = []
    per_sequence_sample_counts = {}
    for batch_index in range(1, len(batches) + 1):
        batch_out = batch_root / 'batch_{:02d}'.format(batch_index)
        summary = json.loads((batch_out / 'summary.json').read_text())
        sample_counts = summary.get('sampled_trace_frames_per_sequence', {})
        if set(sample_counts) != set(batches[batch_index - 1]):
            raise RuntimeError('{} sampled sequence/count manifest mismatch'.format(
                batch_out.name))
        per_sequence_sample_counts.update(sample_counts)
        batch_rows = read_csv(batch_out / 'per_frame.csv')
        observed_counts = {}
        for row in batch_rows:
            observed_counts[row['sequence']] = observed_counts.get(row['sequence'], 0) + 1
        if observed_counts != sample_counts:
            raise RuntimeError('{} rows do not match its valid-frame sample counts'.format(
                batch_out.name))
        all_rows.extend(batch_rows)
        batch_summaries.append(summary)
    expected_rows = sum(per_sequence_sample_counts.values())
    if len(per_sequence_sample_counts) != 244 or len(all_rows) != expected_rows:
        raise RuntimeError('expected {} valid-frame gradient rows over 244 sequences, got {}'
                           .format(expected_rows, len(all_rows)))
    fields = list(all_rows[0])
    with (out / 'per_frame.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with (out / 'metric_correlations.csv').open('w', newline='') as stream:
        rows = correlations(all_rows)
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    excluded = []
    for batch_index in range(1, len(batches) + 1):
        excluded.extend(read_csv(batch_root / 'batch_{:02d}'.format(batch_index) /
                                 'excluded_frames.csv'))
    with (out / 'excluded_frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sequence', 'frame', 'gt_box', 'reason'])
        writer.writeheader()
        writer.writerows(excluded)
    summary = {
        'experiment_id': EXPERIMENT_ID, 'status': 'GRADIENTS-COMPLETE',
        'sequence_count': 244, 'frames_per_sequence': 32, 'frame_count': len(all_rows),
        'per_sequence_sample_counts': per_sequence_sample_counts,
        'batch_count': len(batches), 'batch_size': args.batch_size,
        'trace_manifest_sha256': sha256(trace_dir / 'manifest.json'),
        'checkpoint_sha256': batch_summaries[0]['checkpoint_sha256'],
        'config_sha256': batch_summaries[0]['config_sha256'],
        'batch_summaries': [{'batch': i, 'frame_count': item['frame_count'],
                             'parity_pass': item['parity_pass'],
                             'max_abs_prediction_difference_vs_frozen_m1_trace':
                                 item['max_abs_prediction_difference_vs_frozen_m1_trace']}
                            for i, item in enumerate(batch_summaries, 1)],
        'all_batches_parity_pass': all(item['parity_pass'] for item in batch_summaries),
        'outputs': ['per_frame.csv', 'metric_correlations.csv', 'excluded_frames.csv',
                    'batches/'],
        'limitations': ['Frozen-checkpoint loss sensitivities; not optimizer gradients.',
                        'Current working-tree closed-loop replay is not directly comparable to archived M1.']}
    atomic_json(out / 'summary.json', summary)
    state['status'] = 'GRADIENTS-COMPLETE'
    state['updated_unix_time'] = __import__('time').time()
    atomic_json(state_path, state)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
