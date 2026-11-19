#!/usr/bin/env python3
"""Keep the direct-margin quick pair registered in both asset tables."""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get('C_DIRECT_MARGIN_RUN_ID', 'EG_C_DIRECT_MARGIN_QUICK_RETRY1_S42_20261101')
RUN = ROOT / 'output/experiments' / RUN_ID
SOURCE = ROOT / 'experiments/early_gate_closed_loop_20261026/c_direct_margin_quick_20261101'
ASSET_IDS = {'control': 'EG-C-DIRECT-MARGIN-QUICK-CONTROL-S42-20261101',
             'candidate': 'EG-C-DIRECT-MARGIN-QUICK-CANDIDATE-S42-20261101'}


def register(arm, values):
    asset_id = ASSET_IDS[arm]
    table = ROOT / 'experiments/experiment_asset_registry.csv'
    with table.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if len(fields) != 23 or any(len(row) != len(fields) for row in rows):
        raise ValueError('Asset registry has an invalid row shape')
    hits = [index for index, row in enumerate(rows) if row['ID'] == asset_id]
    if len(hits) > 1:
        raise ValueError('Duplicate asset identity: ' + asset_id)
    row = dict(rows[hits[0]]) if hits else {key: 'NA' for key in fields}
    row.update({'ID': asset_id, **{key: str(value) for key, value in values.items()}})
    if hits:
        rows[hits[0]] = row
    else:
        rows.append(row)
    temp = table.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\r\n')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(table)

    md = ROOT / 'experiments/实验资产数据库.md'
    text = md.read_text(encoding='utf-8')
    start, end = '<!-- ' + asset_id + ' BEGIN -->', '<!-- ' + asset_id + ' END -->'
    block = start + '\n### ' + asset_id + '\n\n' + '\n'.join(
        '- **' + key + '**：' + row[key].replace('\n', ' ') for key in fields) + '\n' + end
    if text.count(start) > 1 or text.count(end) > 1:
        raise ValueError('Duplicate Markdown asset marker: ' + asset_id)
    if start in text:
        left, right = text.index(start), text.index(end, text.index(start)) + len(end)
        text = text[:left] + block + text[right:]
    else:
        text += '\n\n' + block + '\n'
    md.write_text(text, encoding='utf-8')


def run_record(output, config, seed, arm, status, conclusion, checkpoint='NA', gpu_hours='NA'):
    arm = 'control' if arm.lower() == 'control' else 'candidate'
    register(arm, {
        'Date': '20261101',
        'Parent': 'C search-conditioned source gain; public SOT fresh paired initialization; direct localization-margin auxiliary objective',
        'CommitOrSnapshot': str(SOURCE / 'source_at_launch') + '; git HEAD ' +
            (SOURCE / 'source_before/git_head.txt').read_text().strip(),
        'Config': str(Path(config).resolve()), 'Seed': seed,
        'Dataset': 'LasHeR train977; P1 SOT-v3 stratified causal sampler; 128 samples/update',
        'EvalScope': '468 successful updates; screen96_max2000; FP32; tracker threads0; template update off',
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python 3.8.20/PyTorch 2.0.0+cu118/CUDA 11.8; AMP; batch16 accum8; workers4 prefetch2 persistentFalse; serial',
        'GPUHours': gpu_hours, 'Checkpoint': checkpoint,
        'AssumptionOrHypothesis': 'A direct Hann-weighted CENTER target-versus-hard-negative margin objective improves the C search-conditioned source-gate tracker under a matched 468-update fresh-SOT protocol.',
        'Conclusion': conclusion, 'Status': status, 'Evidence': str(RUN.resolve()),
        'Notes': 'Single-seed exploratory quick pair. Margin loss uses center-neighborhood positives and GT-box-exterior negatives, with gradient ratio 0 vs 0.1 calibrated at score_map. Public SOT initialization in both arms; no trained tracker continuation. This loss can train the head/backbone directly, so total score change alone does not prove gate-path causal utility.'})


def finalize(path):
    summary = json.loads(Path(path).read_text())
    vals = summary['all_evaluated']['metrics']
    for arm in ['control', 'candidate']:
        training = RUN / arm / 'train'
        testing = RUN / arm / 'test'
        checkpoint = json.loads((training / 'checkpoint_u0468.json').read_text())
        done = json.loads((training / 'completed.json').read_text())
        tested = json.loads((testing / 'completed.json').read_text())
        metrics = {key: vals[key][arm] for key in vals}
        register(arm, {
            'AUC': metrics['auc'], 'PR': metrics['precision'], 'NPR': metrics['norm_precision'],
            'OP50': metrics['op50'], 'OP75': metrics['op75'], 'FPS': 'NA',
            'GPUHours': done['gpu_wall_hours'] + tested['seconds'] / 3600.0,
            'Checkpoint': checkpoint['path'] + ' sha256=' + checkpoint['sha256'],
            'Conclusion': '468 successful updates; screen96 AUC candidate-control=%+.4f pp; sequence bootstrap 95%% interval [%.4f,%+.4f] pp.' %
                (vals['auc']['delta'], *summary['all_evaluated']['sequence_bootstrap_95pct']),
            'Status': 'QUICK-PAIR-COMPLETE / SCREEN96-ANALYSIS-COMPLETE / SINGLE-SEED',
            'EvalScope': 'LasHeR screen96_max2000; %d frames; FP32; tracker threads0; template update off; 468 successful updates' % tested['frames'],
            'Evidence': str(RUN.resolve()) + '; ' + str(RUN / 'analysis/REPORT.md'),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize-summary')
    args = parser.parse_args()
    if args.finalize_summary:
        finalize(args.finalize_summary)


if __name__ == '__main__':
    main()
