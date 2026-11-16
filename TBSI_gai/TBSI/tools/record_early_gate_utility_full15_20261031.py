"""Synchronized registry updates for the Early gate utility-loss full15 run."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / 'output/experiments/EG_EARLY_GATE_UTILITY_FULL15_S42_20261031'


def register(asset_id, values):
    table = ROOT / 'experiments/experiment_asset_registry.csv'
    with table.open(newline='') as stream:
        raw_rows = list(csv.reader(stream))
    fields = raw_rows[0]
    rows = [dict(zip(fields, raw[:len(fields)])) for raw in raw_rows[1:]]
    matches = [i for i, row in enumerate(rows) if row['ID'] == asset_id]
    if len(matches) > 1:
        raise ValueError('Duplicate asset ID: ' + asset_id)
    row = dict(rows[matches[0]]) if matches else {key: 'NA' for key in fields}
    row.update({'ID': asset_id, **{key: str(value) for key, value in values.items()}})
    if set(row) != set(fields):
        raise ValueError('Registry fields mismatch')
    if matches:
        raw_rows[matches[0] + 1][:len(fields)] = [row[key] for key in fields]
        temporary = table.with_suffix('.csv.tmp')
        with temporary.open('w', newline='') as stream:
            csv.writer(stream).writerows(raw_rows)
        temporary.replace(table)
    else:
        with table.open('a', newline='') as stream:
            csv.writer(stream).writerow([row[key] for key in fields])
    markdown = ROOT / 'experiments/实验资产数据库.md'
    start, end = '<!-- ' + asset_id + ' BEGIN -->', '<!-- ' + asset_id + ' END -->'
    block = start + '\n### ' + asset_id + '\n\n' + '\n'.join(
        '- **' + key + '**：' + row[key].replace('\n', ' ') for key in fields) + '\n' + end
    content = markdown.read_text()
    if start in content:
        left = content.index(start)
        right = content.index(end, left) + len(end)
        content = content[:left] + block + content[right:]
    else:
        content += '\n\n' + block + '\n'
    markdown.write_text(content)


def run_record(output, config, seed, arm, status, conclusion,
               checkpoint='NA', gpu_hours='NA'):
    arm_key = 'CONTROL' if arm.lower() == 'control' else 'UTILITY'
    register('EG-EARLY-GATE-UTILITY-FULL15-' + arm_key + '-S' + str(seed) + '-20261031', {
        'Date': '20261031',
        'Parent': 'Early gate P architecture; public SOT pretraining single-stage parent-control pair; M1 and Early gate full15 are read-only references',
        'CommitOrSnapshot': str(TRAIN_ROOT / 'source_at_launch/manifest.json'),
        'Config': str(Path(config).resolve()), 'Seed': seed,
        'Dataset': 'LasHeR train977; 59904 pairs/native epoch; fixed test244 list',
        'EvalScope': 'Full15=7020 successful updates; full244 FP32; tracker threads0; template update off; single seed',
        'AUC': 'NA', 'PR': 'NA', 'NPR': 'NA', 'OP50': 'NA', 'OP75': 'NA', 'FPS': 'NA',
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python 3.8.20; PyTorch 2.0.0+cu118; CUDA 11.8; AMP; batch16 accumulation8; workers4 prefetch2 persistent_workers=False; serial resource task',
        'GPUHours': gpu_hours, 'Checkpoint': checkpoint,
        'AssumptionOrHypothesis': 'A ranking loss from local Hann-windowed target-vs-strongest-competitor utility gradients can improve how original Early gate assigns source log-gain and improve closed-loop localization.',
        'Conclusion': conclusion, 'Status': status, 'Evidence': str(output),
        'Notes': 'Public SOT checkpoint SHA256 1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b; fresh optimizer; only utility loss coefficient differs (zero versus gradient-calibrated 0.1 task-gradient ratio). No checkpoint continuation. Full15/full244 single-seed result is exploratory.'})
