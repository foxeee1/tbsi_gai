#!/usr/bin/env python3
"""Freeze fresh train-side sequences for the M1 CENTER readout diagnostic."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'data/lasher/trainingset'
PROXY = ROOT / 'experiments/proxy'
OUT = ROOT / 'experiments/diagnostics/m1_center_readout_fresh_v1'
COUNT = 24
PREFIX = 'm1-search-stage-20261009:'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def ranked_eligible():
    candidates = []
    for path in sorted(item for item in DATA.iterdir() if item.is_dir()):
        visible, infrared, gt = path / 'visible', path / 'infrared', path / 'init.txt'
        if not visible.is_dir() or not infrared.is_dir() or not gt.is_file():
            continue
        frames = sum(item.suffix.lower() == '.jpg' for item in visible.iterdir())
        if 150 <= frames <= 800:
            candidates.append((path.name, frames))
    return sorted(candidates, key=lambda item: hashlib.sha256(
        (PREFIX + item[0]).encode()).hexdigest())


def main():
    seen = {}
    exclusion_sources = {}
    for path in sorted(PROXY.rglob('*.txt')):
        if any(token in path.name.lower() for token in ('dev', 'eval', 'test', 'rescue')):
            names = {line.strip() for line in path.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#')}
            seen.update({name: 'proxy-list:' + str(path.relative_to(ROOT)) for name in names})
            exclusion_sources[str(path.relative_to(ROOT))] = sha256(path)

    output_root = ROOT / 'output/experiments'
    prior_trace_manifests = []
    for path in sorted(output_root.glob('M1_*/**/manifest.json')):
        try:
            manifest = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        split = str(manifest.get('split', '')).lower()
        is_train_trace = 'train' in split or 'SEARCH_STAGE' in str(path)
        if not is_train_trace or not isinstance(manifest.get('sequences'), list):
            continue
        names = []
        for item in manifest['sequences']:
            name = item.get('name') if isinstance(item, dict) else str(item)
            if name:
                names.append(name)
                seen.setdefault(name, 'prior-trace:' + str(path.relative_to(ROOT)))
        prior_trace_manifests.append({
            'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
            'sequence_count': len(names),
        })

    for path in sorted(output_root.glob('M1_*/**/selection_manifest.json')):
        try:
            manifest = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if 'STRIP_GT_CELL_COUNTERFACTUAL' not in str(path):
            continue
        for key in ('cohort_sequences', 'excluded_for_prior_strip_dev_overlap'):
            for name in manifest.get(key, []):
                seen.setdefault(name, 'prior-fixed-state:' + str(path.relative_to(ROOT)))
        exclusion_sources[str(path.relative_to(ROOT))] = sha256(path)

    event_csv = ROOT / 'experiments/diagnostics/frcb_admission_v1/events.csv'
    if event_csv.is_file():
        with event_csv.open(newline='') as stream:
            for row in csv.DictReader(stream):
                name = row.get('sequence') or row.get('sequence_name')
                if name:
                    seen.setdefault(name, 'prior-diagnostic-events:' +
                                     str(event_csv.relative_to(ROOT)))
        exclusion_sources[str(event_csv.relative_to(ROOT))] = sha256(event_csv)

    ranked = ranked_eligible()
    fresh = [(name, frames, index) for index, (name, frames) in enumerate(ranked)
             if index >= 120 and name not in seen][:COUNT]
    if len(fresh) != COUNT:
        raise RuntimeError('not enough fresh sequences after exclusion: {}'.format(len(fresh)))
    names = [name for name, _frames, _rank in fresh]
    overlaps = sorted(set(names) & set(seen))
    if overlaps:
        raise RuntimeError('sequence exclusion failed: ' + repr(overlaps))

    OUT.mkdir(parents=True, exist_ok=False)
    sequence_file = OUT / 'sequence_list.txt'
    sequence_file.write_text('\n'.join(names) + '\n')
    manifest = {
        'id': 'M1-CENTER-READOUT-FRESH-20261010',
        'parent': 'TBSI_E_MHSA_FULL_4090D_seed42 epoch15',
        'split': 'LasHeR trainingset',
        'data_path': str(DATA),
        'selection': 'SHA256 rank >=120, 150-800 frames, first 24 after removing prior dev/eval/test lists and prior M1 train-side diagnostic trace/event sequences; no metric used',
        'count': COUNT,
        'ranked_sequences': [
            {'name': name, 'frames': frames, 'sha256_rank': rank}
            for name, frames, rank in fresh
        ],
        'sequence_list_sha256': sha256(sequence_file),
        'excluded_unique_sequence_count': len(seen),
        'exclusion_source_hashes': exclusion_sources,
        'prior_trace_manifests': prior_trace_manifests,
        'protected_config_sha256': 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e',
        'protected_checkpoint_sha256': '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef',
        'event_rule': 'first frame of first run of 3 valid consecutive frames with crop coverage >=99.9% and M1 IoU <0.5; max one event per sequence; retain one preceding frame and nearest earlier matched-good frame where available',
        'comparison': 'same M1 template/checkpoint, same reconstructed crop/state, raw/Hann/top-k/GT-cell decoded with the same size/offset maps; RGB/TIR/concat/fused sequence-held-out probes use the same fold and target/background cells',
        'stop_condition': 'stop module search if no stable cross-sequence distinction between selection error, regression residual, and fusion-induced evidence loss; no training in this diagnostic',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
