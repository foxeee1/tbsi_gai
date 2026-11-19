"""Registry update helper for the cross-mass-preserving relation run."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / 'experiments/experiment_asset_registry.csv'
MARKDOWN = ROOT / 'experiments/实验资产数据库.md'


def register(asset_id, updates):
    with TABLE.open(newline='') as stream:
        reader = csv.DictReader(stream); fields = list(reader.fieldnames); rows = list(reader)
    matches = [i for i, row in enumerate(rows) if row['ID'] == asset_id]
    if len(matches) > 1:
        raise ValueError('Duplicate asset ID: ' + asset_id)
    row = dict(rows[matches[0]]) if matches else {key: 'NA' for key in fields}
    row.update({'ID': asset_id, **{key: str(value) for key, value in updates.items()}})
    if matches: rows[matches[0]] = row
    else: rows.append(row)
    with TABLE.with_suffix('.csv.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    TABLE.with_suffix('.csv.tmp').replace(TABLE)
    start, end = '<!-- ' + asset_id + ' BEGIN -->', '<!-- ' + asset_id + ' END -->'
    block = start + '\n### ' + asset_id + '\n\n' + '\n'.join(
        '- **' + key + '**：' + row[key].replace('\n', ' ') for key in fields) + '\n' + end
    text = MARKDOWN.read_text()
    if start in text:
        left = text.index(start); right = text.index(end, left) + len(end); text = text[:left] + block + text[right:]
    else: text += '\n\n' + block + '\n'
    MARKDOWN.write_text(text)


def run_record(output, cfg, seed, arm, status, conclusion, checkpoint='NA', gpu_hours='NA'):
    upper_arm = str(arm).upper()
    if upper_arm.startswith('P0'):
        asset = 'EG-CROSS-RELATION-P0-S42-20261028'
    elif upper_arm.startswith('DIAG'):
        asset = 'EG-CROSS-RELATION-P-DIAG-S42-20261028'
    else:
        asset = 'EG-CROSS-RELATION-S42-20261028'
    register(asset, {
        'Date': '20261028',
        'Parent': 'Early gate P fixed-state diagnostic; public SOT single-stage paired contract',
        'CommitOrSnapshot': 'cross_mass_preserving_relation.py plus host feature-gate snapshot in run directory',
        'Config': cfg, 'Seed': seed,
        'Dataset': 'LasHeR 977 eligible sequences / 51 strata / 59904 pairs per 468 updates',
        'EvalScope': 'screen96_max2000, FP32, tracker threads=0, template update off',
        'AUC': 'NA', 'PR': 'NA', 'NPR': 'NA', 'OP50': 'NA', 'OP75': 'NA', 'FPS': 'NA',
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python/PyTorch/CUDA in started.json; AMP train B16 accum8 worker4 prefetch2 persistentFalse',
        'GPUHours': gpu_hours, 'Checkpoint': checkpoint,
        'AssumptionOrHypothesis': 'With Early evidence fixed as condition, receiver-source relation reallocation can improve cross-token selection without changing each query cross mass.',
        'Conclusion': conclusion, 'Status': status, 'Evidence': str(output),
        'Notes': 'Value multiplication is explicitly disabled; no auxiliary loss, self-path change, or template update. Quick result is exploratory.'})
