#!/usr/bin/env python3
"""Read-only resource, provenance, and dataset preflight for M1 full-244 diagnostics."""
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset

EXPECTED_CKPT = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'
EXPECTED_CONFIG = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
    config = ROOT / 'experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml'
    failures = []
    if not manifest_path.is_file():
        failures.append('manifest missing')
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text())
    if len(manifest.get('sequences', [])) != 244:
        failures.append('manifest does not contain 244 sequences')
    if not checkpoint.is_file() or sha256(checkpoint) != EXPECTED_CKPT:
        failures.append('protected M1 checkpoint missing or hash mismatch')
    if not config.is_file() or sha256(config) != EXPECTED_CONFIG:
        failures.append('protected M1 config missing or hash mismatch')
    if output_dir.exists():
        state = output_dir / 'run_state.json'
        unexpected = [p.name for p in output_dir.iterdir()
                      if p.name not in ('run_state.json', 'trace', 'trace_summary.json')]
        if unexpected:
            failures.append('unexpected entries in closed-loop output directory: {}'.format(unexpected))
        if (output_dir / 'trace').exists() and not state.is_file():
            failures.append('output path is non-empty but has no resumable run state')
        if state.is_file():
            previous = json.loads(state.read_text())
            if previous.get('manifest_sha256') != sha256(manifest_path):
                failures.append('existing output belongs to a different manifest')
    disk = shutil.disk_usage(output_dir if output_dir.exists() else output_dir.parent)
    available_gib = disk.free / float(1024 ** 3)
    if available_gib < 2.0:
        failures.append('less than 2 GiB disk space available (currently {:.2f} GiB)'.format(available_gib))
    sequences = LasHeRDataset(split='testingset').get_sequence_list()
    data_map = {sequence.name: sequence for sequence in sequences}
    names = manifest.get('sequences', [])
    if len(data_map) != 244 or set(data_map) != set(names):
        failures.append('dataset split does not match the 244-sequence manifest')
    missing_frames = []
    missing_files = []
    too_few_valid_gt = []
    total_frames = 0
    for name in names:
        sequence = data_map.get(name)
        if sequence is None:
            continue
        count = min(len(sequence.frames[0]), len(sequence.frames[1]),
                    len(sequence.ground_truth_rect))
        total_frames += count
        if (len(sequence.frames[0]) != len(sequence.frames[1]) or
                len(sequence.frames[0]) != len(sequence.ground_truth_rect) or count < 2):
            missing_frames.append(name)
        valid_gt = sum(1 for box in sequence.ground_truth_rect[:count]
                       if len(box) == 4 and all(math.isfinite(float(v)) for v in box) and
                       float(box[2]) > 0.0 and float(box[3]) > 0.0)
        if valid_gt < 32:
            too_few_valid_gt.append((name, valid_gt))
        for modality_paths in sequence.frames:
            missing_files.extend(path for path in modality_paths[:count]
                                 if not os.path.isfile(path))
            if missing_files:
                break
    if missing_frames:
        failures.append('invalid RGB/TIR/GT sequence lengths: {}'.format(missing_frames[:10]))
    if too_few_valid_gt:
        failures.append('fewer than 32 valid GT frames in sequences: {}'.format(too_few_valid_gt[:10]))
    if missing_files:
        failures.append('missing image files: {} (first: {})'.format(
            len(missing_files), missing_files[0]))
    archive_dir = ROOT / 'output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42'
    archive_missing = []
    archive_length_mismatch = []
    for name in names:
        sequence = data_map.get(name)
        prediction = archive_dir / (name + '.txt')
        if sequence is None or not prediction.is_file():
            archive_missing.append(name)
        elif sum(1 for _ in prediction.open()) != len(sequence.ground_truth_rect):
            archive_length_mismatch.append(name)
    if archive_missing:
        failures.append('missing archived M1 predictions: {}'.format(archive_missing[:10]))
    if archive_length_mismatch:
        failures.append('archived prediction/GT length mismatch: {}'.format(
            archive_length_mismatch[:10]))
    if platform.python_version() != '3.8.20':
        failures.append('expected Python 3.8.20, got {}'.format(platform.python_version()))
    try:
        import torch
        if torch.__version__ != '1.9.0+cu111' or torch.version.cuda != '11.1':
            failures.append('expected PyTorch 1.9.0+cu111 / CUDA 11.1, got {} / {}'.format(
                torch.__version__, torch.version.cuda))
        if not torch.cuda.is_available():
            failures.append('CUDA unavailable')
        elif '4090 D' not in torch.cuda.get_device_name(0):
            failures.append('expected RTX 4090 D, got {}'.format(torch.cuda.get_device_name(0)))
    except Exception as exc:
        failures.append('torch/CUDA environment check failed: {}'.format(exc))
    report = {
        'experiment_id': 'M1-FUSION-FLOW-FULL244-DIAG-20261009',
        'manifest': str(manifest_path), 'manifest_sha256': sha256(manifest_path),
        'sequence_count': len(names), 'dataset_sequence_count': len(data_map),
        'dataset_frames_including_initial': total_frames,
        'trace_frames_expected': total_frames - len(names),
        'gradient_frames_expected': len(names) * 32,
        'sequences_with_at_least_32_valid_gt': len(names) - len(too_few_valid_gt),
        'archived_prediction_count': len(names) - len(archive_missing),
        'checkpoint_sha256': sha256(checkpoint) if checkpoint.is_file() else 'MISSING',
        'config_sha256': sha256(config) if config.is_file() else 'MISSING',
        'available_disk_gib': available_gib,
        'python': platform.python_version(),
        'output_dir': str(output_dir), 'failures': failures,
        'status': 'PASS' if not failures else 'FAIL',
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
