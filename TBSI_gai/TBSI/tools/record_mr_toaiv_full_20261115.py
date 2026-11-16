#!/usr/bin/env python3
"""Register the paired MR TO/AIV full run without deleting historical rows."""
import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'experiments/experiment_asset_registry.csv'
MARKDOWN = ROOT / 'experiments/实验资产数据库.md'
RUN = ROOT / 'output/experiments/MSHA-MR-TOAIV-CONTENT-PROJECTION-FULL15-FULL244-S42-20261115'
PUBLIC = ROOT / 'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def upsert(asset_id, updates):
    with REGISTRY.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames)
        rows = list(reader)
    matches = [idx for idx, row in enumerate(rows) if row['ID'] == asset_id]
    if len(matches) > 1:
        raise RuntimeError('duplicate asset id: ' + asset_id)
    row = {field: 'NA' for field in fields}
    if matches:
        row.update(rows[matches[0]])
    row.update({'ID': asset_id, **{key: str(value) for key, value in updates.items()}})
    if matches:
        rows[matches[0]] = row
    else:
        rows.append(row)
    temp = REGISTRY.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(REGISTRY)

    start = '<!-- ' + asset_id + ' BEGIN -->'
    end = '<!-- ' + asset_id + ' END -->'
    block = start + '\n### ' + asset_id + '\n\n' + '\n'.join(
        '- **' + field + '**：' + row[field].replace('\n', ' ')
        for field in fields) + '\n' + end
    text = MARKDOWN.read_text()
    if start in text:
        left = text.index(start)
        right = text.index(end, left) + len(end)
        text = text[:left] + block + text[right:]
    else:
        text += '\n\n' + block + '\n'
    MARKDOWN.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['control', 'candidate'], required=True)
    parser.add_argument('--status', required=True)
    parser.add_argument('--conclusion', required=True)
    parser.add_argument('--checkpoint', default='NA')
    parser.add_argument('--gpu-hours', default='NA')
    parser.add_argument('--auc', default='NA')
    parser.add_argument('--pr', default='NA')
    parser.add_argument('--npr', default='NA')
    parser.add_argument('--op50', default='NA')
    parser.add_argument('--op75', default='NA')
    parser.add_argument('--fps', default='NA')
    parser.add_argument('--evidence', default=str(RUN))
    args = parser.parse_args()
    arm = args.arm.upper()
    asset_id = ('MSHA-MR-TOAIV-CONTENT-PROJECTION-FULL15-S42-' + arm +
                '-20261115')
    config = RUN / ('workspace/experiments/tbsi_track/'
                    'MSHA_MR_TOAIV_CONTENT_PROJECTION_FULL15_S42_' + arm +
                    '_20261115.yaml')
    manifest = RUN / 'data_manifest.json'
    commit = 'independent workspace snapshot; protocol ' + str(RUN / 'PROTOCOL.md')
    updates = {
        'Date': '20261115',
        'Parent': 'MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109',
        'CommitOrSnapshot': commit,
        'Config': str(config),
        'Seed': '42',
        'Dataset': 'LasHeR train; p1_sot_v3 manifest; 977 sequences; 59904 samples/epoch x15',
        'EvalScope': 'Full15 training; full244 FP32 threads0 template update off (evaluation pending)',
        'AUC': args.auc, 'PR': args.pr, 'NPR': args.npr,
        'OP50': args.op50, 'OP75': args.op75, 'FPS': args.fps,
        'GPU': '1x NVIDIA GeForce RTX 4090 D 24 GiB',
        'HardwareConfig': 'Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8; AMP; B16 accum8; workers4 prefetch2 persistentFalse',
        'GPUHours': args.gpu_hours,
        'Checkpoint': args.checkpoint,
        'AssumptionOrHypothesis': ('Matched MR control; candidate tests frame-observable TO/AIV-conditioned Value content projection, '
                                   'alpha=0.25, without sequence-level labels.'),
        'Conclusion': args.conclusion,
        'Status': args.status,
        'Evidence': args.evidence,
        'Notes': ('Public SOT SHA256 ' + sha(PUBLIC) + '; config SHA256 ' + sha(config) +
                  '; manifest SHA256 ' + sha(manifest) + '; candidate-only variable is TBSI_MR_CONTENT_PROJECTION=to_aiv.')
    }
    upsert(asset_id, updates)
    print(asset_id)


if __name__ == '__main__':
    main()
