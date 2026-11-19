"""Incremental registry records for the 2026-10-28 FU experiment."""
import csv
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
    row = dict(rows[matches[0]]) if matches else {key: 'NA' for key in fields}
    row.update({'ID': asset_id, **{key: str(value) for key, value in updates.items()}})
    if set(row) != set(fields):
        raise ValueError('Registry schema changed unexpectedly')
    if matches:
        rows[matches[0]] = row
    else:
        rows.append(row)
    temp = TABLE.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(TABLE)

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
    output = Path(output)
    arm = str(arm).upper()
    retry = 'RETRY1' in arm or 'RETRY1' in str(output).upper()
    p0 = arm.startswith('P0')
    if retry:
        asset_id = ('EG-FU-RETRY1-P0-OPERATOR-CONTRACT-S42-20261028' if p0
                    else 'EG-FU-RETRY1-S42-20261028')
        snapshot = ROOT / 'experiments/early_gate_closed_loop_20261026/fine_grained_unified_20261028/retry1/source_before/source_before_retry1.tar.gz'
    else:
        asset_id = ('EG-FU-P0-OPERATOR-CONTRACT-S42-20261028' if p0
                    else 'EG-FU-S42-20261028')
        snapshot = ROOT / 'experiments/early_gate_closed_loop_20261026/fine_grained_unified_20261028/source_before/source_before.tar.gz'
    register(asset_id, {
        'Date': '20261028',
        'Parent': 'EG-DUALMSG-V2-P-S42-20261027 Early gate; public-SOT initialization and 468-update quick parent',
        'CommitOrSnapshot': str(snapshot),
        'Config': str(cfg),
        'Seed': seed,
        'Dataset': 'LasHeR stratified P1 SOT v3 manifest; exact samples/coverage recorded in run completed.json',
        'EvalScope': 'P0 has no tracking evaluation; P1 uses frozen screen96_max2000, FP32, threads=0, template update off',
        'AUC': 'NA', 'PR': 'NA', 'NPR': 'NA', 'OP50': 'NA', 'OP75': 'NA', 'FPS': 'NA',
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python/PyTorch/CUDA recorded in started.json; serial; AMP train B16/accum8/workers4/prefetch2/persistentFalse',
        'GPUHours': gpu_hours,
        'Checkpoint': checkpoint,
        'AssumptionOrHypothesis': 'Target/background-referenced fine-grained content and a receiver-source relation correction improve cross-modal allocation beyond the Early-gate parent.',
        'Conclusion': conclusion,
        'Status': status,
        'Evidence': str(output),
        'Notes': 'Single seed and reused development screen are exploratory. FU begins from public SOT pretrained weights in a fresh single-stage run; no trained tracker continuation.'})
