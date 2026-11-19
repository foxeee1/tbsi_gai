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
    prefix = 'EG-DUALMSG-V2-' if 'dual_message_p1_v2' in str(output) else 'EG-DUALMSG-P1-'
    parent = 'Early gate architecture; all arms single-stage from public SOT pretraining 1f6ebf59...; no trained-tracker continuation'
    hypothesis = 'HE/HD structured evidence and independent self/cross modulation under matched quick training.'
    dataset = 'LasHeR train: 977 eligible sequences; 51 strata; 59904 pairs / 468 successful updates'
    scope = 'FP32 threads0; fixed screen96; metrics pending'
    note = 'Single seed exploratory paired quick stage; quick direction does not guarantee full direction.'
    register(prefix + arm.upper() + '-S' + str(seed) + '-20261027', {
        'Date': '20261027', 'Parent': parent,
        'CommitOrSnapshot': '20261027 V2 source manifest in independent run directory; see run manifest',
        'Config': cfg, 'Seed': seed, 'Dataset': dataset,
        'EvalScope': scope,
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8; AMP train; B16 accum8; workers4 prefetch2 persistentFalse; single resource-intensive task',
        'GPUHours': gpu_hours, 'Checkpoint': checkpoint,
        'AssumptionOrHypothesis': hypothesis,
        'Conclusion': conclusion, 'Status': status, 'Evidence': str(output),
        'Notes': note})
