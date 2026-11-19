#!/usr/bin/env python3
"""Incremental registry updates for the C stable-utility quick pair."""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get('C_STABLE_UTILITY_RUN_ID', 'EG_C_STABLE_UTILITY_QUICK_S42_20261101')
RUN = ROOT / 'output/experiments' / RUN_ID
SOURCE = ROOT / 'experiments/early_gate_closed_loop_20261026/c_stable_utility_quick_20261101'
IDS = {
    'control': RUN_ID.replace('_S42_', '-CONTROL-S42-'),
    'candidate': RUN_ID.replace('_S42_', '-CANDIDATE-S42-'),
}


def register(arm, updates):
    asset_id = IDS[arm]
    table = ROOT / 'experiments/experiment_asset_registry.csv'
    rows = list(csv.DictReader(table.open(newline='')))
    fields = list(rows[0])
    matches = [i for i, row in enumerate(rows) if row['ID'] == asset_id]
    if len(matches) > 1:
        raise ValueError('Duplicate asset ID: ' + asset_id)
    row = dict(rows[matches[0]]) if matches else {key: 'NA' for key in fields}
    row.update({'ID': asset_id, **{key: str(value) for key, value in updates.items()}})
    if matches:
        rows[matches[0]] = row
        target = table.with_suffix('.csv.tmp')
        with target.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        target.replace(table)
    else:
        with table.open('a', newline='') as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)
    md = ROOT / 'experiments/实验资产数据库.md'
    start, end = '<!-- ' + asset_id + ' BEGIN -->', '<!-- ' + asset_id + ' END -->'
    block = start + '\n### ' + asset_id + '\n\n' + '\n'.join(
        '- **' + key + '**：' + row[key].replace('\n', ' ') for key in fields) + '\n' + end
    text = md.read_text()
    if start in text:
        left = text.index(start)
        right = text.index(end, left) + len(end)
        text = text[:left] + block + text[right:]
    else:
        text += '\n\n' + block + '\n'
    md.write_text(text)


def run_record(output, cfg, seed, arm, status, conclusion, checkpoint='NA', gpu_hours='NA'):
    short = 'control' if arm.lower() == 'control' else 'candidate'
    register(short, {
        'Date': '20261101',
        'Parent': 'C search-context architecture; same public-SOT fresh initialization pair; M1/Early gate are read-only references',
        'CommitOrSnapshot': str(SOURCE / 'source_at_launch') + '; git HEAD ' +
            (SOURCE / 'source_before/git_head.txt').read_text().strip(),
        'Config': str(Path(cfg).resolve()), 'Seed': seed,
        'Dataset': 'LasHeR train977; stratified causal sampling; 128 samples per optimizer update',
        'EvalScope': '468 successful updates; frozen screen96_max2000; FP32 tracker threads0; template update off',
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python 3.8.20/PyTorch 2.0.0+cu118/CUDA 11.8; AMP; batch16 accum8; workers4 prefetch2 persistentFalse; serial',
        'GPUHours': gpu_hours, 'Checkpoint': checkpoint,
        'AssumptionOrHypothesis': 'Ranking source-token gains only when local utility ordering agrees under center-peak and target-box peak definitions reduces noisy utility supervision and improves C.',
        'Conclusion': conclusion, 'Status': status, 'Evidence': str(RUN.resolve()),
        'Notes': 'Exploratory single-seed quick screen. Both arms compute identical utility taps; only loss coefficient differs. Public SOT initialization; no trained tracker continuation. Stable-pair coverage was about 99.5%, so the filter excluded very few pairs. Initial report generation hit a summary-key mismatch and was rerun after a script fix; no training or evaluation was repeated. No longer-budget promotion. Full evidence is stored at ' + str(RUN.resolve()) + '.'})


def finalize(args):
    summary = json.loads(Path(args.summary).read_text())
    results = summary['all_evaluated']['metrics']
    completed = {}
    for arm in ['control', 'candidate']:
        train_dir = RUN / arm / 'train'
        test_dir = RUN / arm / 'test'
        checkpoint_meta = json.loads((train_dir / 'checkpoint_u0468.json').read_text())
        done = json.loads((train_dir / 'completed.json').read_text())
        test_done = json.loads((test_dir / 'completed.json').read_text())
        completed[arm] = (checkpoint_meta, done, test_done)
    for arm in ['control', 'candidate']:
        checkpoint_meta, done, test_done = completed[arm]
        vals = {key: results[key]['control' if arm == 'control' else 'candidate']
                for key in ['auc', 'precision', 'norm_precision', 'op50', 'op75']}
        register(arm, {
            'AUC': vals['auc'], 'PR': vals['precision'], 'NPR': vals['norm_precision'],
            'OP50': vals['op50'], 'OP75': vals['op75'],
            'FPS': 'NA', 'GPUHours': done['gpu_wall_hours'] + test_done['seconds'] / 3600.0,
            'Checkpoint': checkpoint_meta['path'] + ' sha256=' + checkpoint_meta['sha256'],
            'Conclusion': '468 successful updates; screen96 AUC delta candidate-control=%+.4f pp; sequence 90%% CI [%.4f,%+.4f] pp; 96-sequence exploratory result.' %
                (results['auc']['delta'], *summary['all_evaluated']['sequence_bootstrap_90pct']),
            'Status': 'QUICK-PAIR-COMPLETE / SCREEN96-ANALYSIS-COMPLETE / SINGLE-SEED',
            'EvalScope': 'LasHeR screen96_max2000; %d frames; FP32; threads0; template update off; 468 successful updates' % test_done['frames'],
            'Evidence': str(RUN.resolve()),
            'Notes': 'Exploratory single-seed quick screen. Both arms compute identical utility taps; only loss coefficient differs. Public SOT initialization; no trained tracker continuation. Stable-pair coverage was about 99.5%, so the filter excluded very few pairs. Initial report generation hit a summary-key mismatch and was rerun after a script fix; no training or evaluation was repeated. No longer-budget promotion. Full evidence is stored at ' + str(RUN.resolve()) + '.'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize-summary')
    args = parser.parse_args()
    if args.finalize_summary:
        finalize(argparse.Namespace(summary=args.finalize_summary))


if __name__ == '__main__':
    main()
