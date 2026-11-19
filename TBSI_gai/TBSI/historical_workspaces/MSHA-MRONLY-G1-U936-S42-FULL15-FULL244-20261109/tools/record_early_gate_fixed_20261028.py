"""Incremental registry writer for the 2026-10-28 fixed-gate comparison."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / 'experiments/experiment_asset_registry.csv'
MARKDOWN = ROOT / 'experiments/实验资产数据库.md'


def register(asset_id, updates):
    with TABLE.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames)
        rows = list(reader)
    matches = [i for i, row in enumerate(rows) if row['ID'] == asset_id]
    if len(matches) > 1:
        raise ValueError('Duplicate asset ID: ' + asset_id)
    row = dict(rows[matches[0]]) if matches else {field: 'NA' for field in fields}
    row.update({'ID': asset_id, **{key: str(value) for key, value in updates.items()}})
    if set(row) != set(fields):
        raise ValueError('Registry fields changed unexpectedly')
    if matches:
        rows[matches[0]] = row
        temp = TABLE.with_suffix('.csv.tmp')
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(TABLE)
    else:
        with TABLE.open('a', newline='') as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)
    start, end = '<!-- ' + asset_id + ' BEGIN -->', '<!-- ' + asset_id + ' END -->'
    block = start + '\n### ' + asset_id + '\n\n' + '\n'.join(
        '- **' + key + '**：' + row[key].replace('\n', ' ') for key in fields) + '\n' + end
    text = MARKDOWN.read_text()
    if start in text:
        left = text.index(start)
        right = text.index(end, left) + len(end)
        text = text[:left] + block + text[right:]
    else:
        text += '\n\n' + block + '\n'
    MARKDOWN.write_text(text)


def run_record(output, cfg, seed, arm, status, conclusion,
               checkpoint='NA', gpu_hours='NA'):
    arm_name = str(arm).upper().replace('_', '-')
    asset_id = ('EG-EARLY-GATE-FIXED-S' + str(seed) + '-20261028'
                if arm_name == 'FIXED' else
                'EG-EARLY-GATE-FIXED-' + arm_name + '-S' + str(seed) + '-20261028')
    register(asset_id, {
        'Date': '20261028',
        'Parent': 'EG-DUALMSG-V2-P-S42-20261027; same public-SOT pretraining lineage, no trained-tracker continuation',
        'CommitOrSnapshot': 'code_before.tar.gz plus current source hashes in early_gate_fixed_control_20261028/source_manifest.json',
        'Config': str(cfg),
        'Seed': seed,
        'Dataset': 'LasHeR train 977 eligible sequences / 59904 pairs / 468 updates; fixed screen96_max2000 evaluation',
        'EvalScope': 'FP32, tracker threads 0, template update off; fixed 96-sequence development cohort, 54666 frames',
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python 3.8.20 / PyTorch 2.0.0+cu118 / CUDA 11.8; AMP training; batch16/accum8; workers4/prefetch2/persistentFalse; serial task',
        'GPUHours': gpu_hours,
        'Checkpoint': checkpoint,
        'AssumptionOrHypothesis': 'Per-token early evidence conditioning improves cross-source Value amplitude control beyond a learned constant source gain per fusion layer.',
        'Conclusion': conclusion,
        'Status': status,
        'Evidence': str(output),
        'Notes': 'Single seed exploratory comparison. Fixed arm has lower active capacity and is not an equal-capacity control. Quick result cannot predict full-training direction.'})
