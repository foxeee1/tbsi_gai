"""Synchronized incremental P1 asset registration; existing rows are retained."""
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def register(asset_id, updates):
    table = ROOT / 'experiments/experiment_asset_registry.csv'
    rows = list(csv.DictReader(table.open(newline='')))
    fields = list(rows[0])
    matches = [i for i, r in enumerate(rows) if r['ID'] == asset_id]
    if len(matches) > 1:
        raise ValueError('Duplicate asset identity: ' + asset_id)
    row = dict(rows[matches[0]]) if matches else {key: 'NA' for key in fields}
    row.update({'ID': asset_id, **{key: str(value) for key, value in updates.items()}})
    if set(row) != set(fields):
        raise ValueError('Unexpected registry fields')
    if matches:
        rows[matches[0]] = row
        temp = table.with_suffix('.csv.tmp')
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        temp.replace(table)
    else:
        with table.open('a', newline='') as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)
    md = ROOT / 'experiments/实验资产数据库.md'
    start, end = '<!-- ' + asset_id + ' BEGIN -->', '<!-- ' + asset_id + ' END -->'
    block = start + '\n### ' + asset_id + '\n\n' + '\n'.join(
        '- **' + key + '**：' + row[key].replace('\n', ' ') for key in fields) + '\n' + end
    text = md.read_text()
    if start in text:
        left = text.index(start); right = text.index(end, left) + len(end)
        text = text[:left] + block + text[right:]
    else:
        text += '\n\n' + block + '\n'
    md.write_text(text)


def run_record(output, cfg, seed, arm, status, conclusion, checkpoint='NA', gpu_hours='NA'):
    register('EG-EVDR-P1-SOT-V3-' + arm.upper() + '-S' + str(seed) + '-20261026', {
        'Date': '20261026', 'Parent': 'Early gate architecture control; BOTH single-stage from public SOT pretraining 1f6ebf59...; full15 is analysis reference only',
        'CommitOrSnapshot': 'HEAD 6d95ad0d09eea92e0f397fcc8256611e0e03998f + p1_sot_v3/source_at_launch/manifest.json',
        'Config': cfg, 'Seed': seed, 'Dataset': 'LasHeR train: 977 eligible sequences; 51 strata; 59904 pairs / 468 successful updates',
        'EvalScope': 'FP32 threads0; fixed broad paired cohort + full244 confirmation as preregistered; metrics pending',
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8; AMP train; B16 accum8; workers4 prefetch2 persistentFalse; single resource-intensive task',
        'GPUHours': gpu_hours, 'Checkpoint': checkpoint,
        'AssumptionOrHypothesis': 'Single-stage joint learning of cross-source Value direction improves low-AUC tracking beyond matched Early gate architecture trained from the same public SOT weights, without harming normal sequences.',
        'Conclusion': conclusion, 'Status': status, 'Evidence': str(output),
        'Notes': 'Single seed / exploratory quick-to-full transfer; 234 and 468 fixed successful-update checkpoints. Full training must start independently from the same public SOT weights and preserve the data/optimizer/schedule prefix. No trained-tracker continuation, multi-stage training or legacy admission rules.'})
