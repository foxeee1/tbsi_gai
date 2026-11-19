#!/usr/bin/env python3
"""Build a reproducible M1 screening cohort and event-case manifest."""
import csv
import hashlib
import json
import math
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'data/lasher'
OUT = ROOT / 'experiments/protocols/m1_screening_v1_20261022'
TRAIN_FRACTION = 0.18
SEED = 20261022
SKIP_SEQUENCES = {'advancedredcup', 'cameraman_1202', 'mirroratleft'}
ATTRIBUTE_ZIP = DATA / 'Attributes&Order.zip'
SCENE_DIR = ROOT / 'output/experiments/M1_FULL244_SCENE_MINING_20261009'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def all_length_sequence_stats(path):
    """Compute scene proxies without excluding short or long official sequences."""
    boxes = np.loadtxt(str(path / 'init.txt'), delimiter=',', dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    rgb_files = sorted((path / 'visible').glob('*.jpg'))
    tir_files = sorted((path / 'infrared').glob('*.jpg'))
    n = min(len(boxes), len(rgb_files), len(tir_files))
    if n < 1:
        return None
    boxes = boxes[:n]
    valid = np.isfinite(boxes).all(1) & (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    valid_boxes = boxes[valid]
    if len(valid_boxes) < 1:
        return None
    centers = boxes[:, :2] + boxes[:, 2:4] * .5
    step = np.diff(centers, axis=0)
    denom = np.sqrt(np.maximum(1., boxes[1:, 2] * boxes[1:, 3]))
    valid_motion = valid[:-1] & valid[1:]
    motion = np.linalg.norm(step[valid_motion], axis=1) / denom[valid_motion]
    sample_indices = np.unique(np.linspace(0, n - 1, min(5, n)).round().astype(int))
    rgb_luma, tir_contrast = [], []
    for idx in sample_indices:
        box = boxes[idx]
        if not valid[idx]:
            continue
        rgb = cv2.imread(str(rgb_files[idx]), cv2.IMREAD_GRAYSCALE)
        tir = cv2.imread(str(tir_files[idx]), cv2.IMREAD_GRAYSCALE)
        if rgb is None or tir is None or rgb.shape != tir.shape:
            continue
        h, w = rgb.shape
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2 = min(w, int(math.ceil(box[0] + box[2])))
        y2 = min(h, int(math.ceil(box[1] + box[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        rgb_luma.append(float(rgb[y1:y2, x1:x2].mean()))
        pad_x, pad_y = max(3, int(box[2])), max(3, int(box[3]))
        rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        ring = tir[ry1:ry2, rx1:rx2].astype(np.float32).copy()
        ring[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = np.nan
        values = ring[np.isfinite(ring)]
        target_mean = float(tir[y1:y2, x1:x2].mean())
        bg_mean = float(np.nanmean(values)) if values.size else target_mean
        bg_std = float(np.nanstd(values)) if values.size else 0.0
        tir_contrast.append(abs(target_mean - bg_mean) / (bg_std + 1.0))
    if not rgb_luma or not tir_contrast:
        return None
    area = valid_boxes[:, 2] * valid_boxes[:, 3]
    return {
        'name': path.name, 'frames': int(n),
        'relative_area_median': float(np.median(area) / (640.0 * 512.0)),
        'relative_motion_p90': float(np.quantile(motion, .9)) if len(motion) else 0.0,
        'rgb_target_luma_median': float(np.median(rgb_luma)),
        'tir_target_ring_contrast_median': float(np.median(tir_contrast)),
    }


def load_attributes(split_names):
    with zipfile.ZipFile(str(ATTRIBUTE_ZIP)) as archive:
        order = [x.strip() for x in archive.read('Attributes_order.txt').decode().replace('\n', '').split(',')]
        rows = {}
        for member in archive.namelist():
            if not member.startswith('AttriSeqsTxt/') or not member.endswith('.txt'):
                continue
            name = Path(member).stem
            if name not in split_names:
                continue
            values = [int(x) for x in archive.read(member).decode().strip().split(',')]
            if len(values) == len(order):
                rows[name] = dict(zip(order, values))
    return order, rows


def quantile_edges(values, bins=4):
    ordered = sorted(values)
    return [ordered[min(len(ordered) - 1, int(round(i * (len(ordered) - 1) / bins)))]
            for i in range(bins + 1)]


def bin_value(value, edges):
    for index in range(1, len(edges) - 1):
        if value <= edges[index]:
            return index - 1
    return len(edges) - 2


def select_proportional(rows, attr_names, feature_names, count, seed=SEED):
    """Hash-seeded swap descent matching attribute and proxy marginals."""
    rng = random.Random(seed)
    n = len(rows)
    edges = {}
    for name in feature_names:
        edges[name] = quantile_edges([row[name] for row in rows])
    for row in rows:
        vector = [float(row['attributes'].get(name, 0)) for name in attr_names]
        for name in feature_names:
            qbin = bin_value(row[name], edges[name])
            vector.extend(1.0 if j == qbin else 0.0 for j in range(4))
        row['_balance_vector'] = vector
        row['_tie'] = hashlib.sha256(('M1-SCREEN-V1:{}:{}'.format(seed, row['name'])).encode()).hexdigest()

    totals = [sum(row['_balance_vector'][j] for row in rows) for j in range(len(rows[0]['_balance_vector']))]
    targets = [count * total / n for total in totals]
    weights = [1.0 / max(1.0, count * (total / n) * (1.0 - total / n)) for total in totals]

    selected = sorted(rows, key=lambda row: row['_tie'])[:count]
    remaining = [row for row in rows if row not in selected]
    sums = [sum(row['_balance_vector'][j] for row in selected) for j in range(len(targets))]
    current = sum(weights[j] * (sums[j] - targets[j]) ** 2 for j in range(len(targets)))
    for _ in range(30000):
        out_index = rng.randrange(count)
        in_index = rng.randrange(len(remaining))
        outgoing = selected[out_index]['_balance_vector']
        incoming = remaining[in_index]['_balance_vector']
        score = 0.0
        for j in range(len(targets)):
            candidate_sum = sums[j] - outgoing[j] + incoming[j]
            score += weights[j] * (candidate_sum - targets[j]) ** 2
        if score + 1e-12 < current:
            removed = selected[out_index]
            selected[out_index] = remaining[in_index]
            remaining[in_index] = removed
            current = score
            sums = [sums[j] - outgoing[j] + incoming[j] for j in range(len(targets))]
    selected.sort(key=lambda row: row['_tie'])
    for row in rows:
        row.pop('_balance_vector', None)
        row.pop('_tie', None)
    return selected, edges, current


def read_csv_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    generated_names = {'quick_train_sequences.txt', 'quick_train_manifest.json',
                       'full244_eval_sequences.txt', 'full244_eval_manifest.json',
                       'paper_case_events.csv', 'protocol_manifest.json', 'build_summary.json'}
    if OUT.exists() and any((OUT / name).exists() for name in generated_names):
        raise RuntimeError('refusing to overwrite generated protocol assets: {}'.format(OUT))
    train_rows = []
    train_path = DATA / 'trainingset'
    for path in sorted(train_path.iterdir()):
        if path.is_dir() and (path / 'visible').is_dir() and (path / 'infrared').is_dir() and (path / 'init.txt').is_file():
            row = all_length_sequence_stats(path)
            if row:
                train_rows.append(row)
    attr_names, attr_map = load_attributes({row['name'] for row in train_rows})
    train_rows = [row for row in train_rows if row['name'] in attr_map and row['name'] not in SKIP_SEQUENCES]
    for row in train_rows:
        row['attributes'] = attr_map.get(row['name'], {})
    train_count = int(round(len(train_rows) * TRAIN_FRACTION))
    feature_names = ('relative_area_median', 'relative_motion_p90',
                     'rgb_target_luma_median', 'tir_target_ring_contrast_median')
    selected, edges, objective = select_proportional(train_rows, attr_names, feature_names, train_count)
    OUT.mkdir(parents=True, exist_ok=True)

    selected_names = {row['name'] for row in selected}
    train_list = OUT / 'quick_train_sequences.txt'
    train_list.write_text('\n'.join(row['name'] for row in selected) + '\n')
    counts = {name: sum(row['attributes'].get(name, 0) for row in selected) for name in attr_names}
    population_counts = {name: sum(row['attributes'].get(name, 0) for row in train_rows) for name in attr_names}
    selected_rows = []
    for row in selected:
        clean = {k: v for k, v in row.items() if k not in ('attributes', '_balance_vector', '_tie')}
        clean['attributes'] = row['attributes']
        selected_rows.append(clean)
    write_json(OUT / 'quick_train_manifest.json', {
        'id': 'M1-QUICK-TRAIN-PROP18-V1-20261022',
        'split': 'LasHeR official trainingset', 'source_sequences': len(train_rows),
        'dataset_root': str(train_path), 'attribute_archive_sha256': sha256(ATTRIBUTE_ZIP),
        'selected_sequences': train_count, 'fraction': train_count / len(train_rows),
        'seed': SEED, 'selection_rule': 'Outcome-blind deterministic hash initialization followed by fixed-count swap descent minimizing weighted marginal deviation for all official overlapping attributes and quartile bins of target area, relative motion, RGB target luminance and TIR target-ring contrast.',
        'attribute_names': attr_names, 'population_attribute_counts': population_counts,
        'selected_attribute_counts': counts, 'proxy_quantile_edges': edges,
        'balance_objective': objective, 'sequence_list': str(train_list.relative_to(ROOT)),
        'sequence_list_sha256': sha256(train_list), 'sequences': selected_rows,
        'limitations': ['LasHeR annotations are sequence-level multi-label attributes, not frame-level event labels.',
                        'Image-derived scene measures are sampling proxies, not modality reliability ground truth.',
                        'M1 was trained on the complete official trainingset; this cohort is for candidate training, not an independent M1 test cohort.']
    })

    test_candidates = [p.name for p in (DATA / 'testingset').iterdir()
                       if p.is_dir() and p.name not in SKIP_SEQUENCES and (p / 'visible').is_dir() and
                       (p / 'infrared').is_dir() and (p / 'init.txt').is_file()]
    _, test_attributes = load_attributes(set(test_candidates))
    test_dirs = sorted(name for name in test_candidates if name in test_attributes)
    if len(train_rows) != 977 or len(test_dirs) != 244:
        raise RuntimeError('official LasHeR split audit mismatch: train={}, test={}'.format(
            len(train_rows), len(test_dirs)))
    test_list = OUT / 'full244_eval_sequences.txt'
    test_list.write_text('\n'.join(test_dirs) + '\n')
    write_json(OUT / 'full244_eval_manifest.json', {
        'id': 'M1-FULL244-EVAL-V1-20261022', 'split': 'LasHeR official testingset',
        'dataset_root': str(DATA / 'testingset'), 'attribute_archive_sha256': sha256(ATTRIBUTE_ZIP),
        'sequence_count': len(test_dirs), 'sequence_list': str(test_list.relative_to(ROOT)),
        'sequence_list_sha256': sha256(test_list), 'rule': 'Intersection of folders with RGB/TIR/GT and official Attributes&Order sequence labels, excluding the dataset loader SKIP_SEQUENCES; all official test sequences in lexical order; no sampling.' ,
        'qualification': 'Required screening evaluation. This split has extensive prior exposure in the research process; it is not a fresh untouched holdout.'
    })

    event_rows = read_csv_rows(SCENE_DIR / 'persistent_severe_events.csv')
    scene_rows = {row['sequence']: row for row in read_csv_rows(SCENE_DIR / 'per_sequence_scene.csv')}
    case_specs = [
        ('ab_pingpongball2', 10, 'tiny_target_high_relative_motion'),
        ('1strowleftboyturning', 33, 'similar_appearance_thermal_crossover_attributes'),
        ('leftunderbasket', 76, 'large_target_counterexample_with_scene_confounds'),
        ('basketball849', 72, 'low_light_background_thermal_crossover_attributes'),
    ]
    chosen = []
    for name, start, role in case_specs:
        match = next((row for row in event_rows if row['sequence'] == name and int(row['start_frame_0based']) == start), None)
        if match is None or name not in scene_rows:
            raise RuntimeError('missing archived event evidence: {}@{}'.format(name, start))
        chosen.append({'sequence': name, 'event_start_frame_0based': start,
                       'event_end_frame_0based_inclusive': int(match['end_frame_0based_inclusive']),
                       'event_length': int(match['length']), 'onset_iou': float(match['onset_iou']),
                       'onset_crop_coverage': float(match['onset_crop_coverage']),
                       'onset_center_error_gt_diag': float(match['onset_center_error_gt_diag']),
                       'first_full_crop_out_delay_frames': match['first_full_crop_out_delay_frames'],
                       'role': role, 'm1_auc': float(scene_rows[name]['m1_auc']),
                       'source': 'M1 archived full-244 scene mining; descriptive retrospective case only',
                       'exposure_status': 'TEST_SET_PREVIOUSLY_INSPECTED; never use for candidate selection or confirmation'})
    with (OUT / 'paper_case_events.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chosen[0].keys()))
        writer.writeheader()
        writer.writerows(chosen)

    write_json(OUT / 'protocol_manifest.json', {
        'id': 'M1-RESEARCH-SCREENING-PROTOCOL-V1-20261022',
        'parent': 'Protected M1 TBSI_E_MHSA_FULL_4090D_seed42 epoch15',
        'protected_parent_config_sha256': 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1',
        'protected_parent_checkpoint_sha256': '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef',
        'quick_train_manifest': 'quick_train_manifest.json',
        'eval_manifest': 'full244_eval_manifest.json',
        'paper_case_events': 'paper_case_events.csv',
        'historical_pressure_manifest': 'historical RPP-v1 proxy_test_manifest.json (60/244, overrepresents rare attributes; screening only)',
        'next_scientific_gate': 'Use train-side fixed-state events to classify A/B/C/D by earliest target-vs-hard-background evidence and CENTER readout. Do not train a new operator until the proposed intervention has a measurable path to CENTER rank under same crop/template/state.',
        'historical_case_limit': 'Four cases are retrospective examples selected from the already inspected official test split. They support problem illustration only, not method selection or unbiased performance claims.'
    })
    summary = {
        'quick_train_count': len(selected), 'quick_train_fraction': len(selected) / len(train_rows),
        'full244_eval_count': len(test_dirs), 'paper_cases': len(chosen),
        'manifest_sha256': {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        'train_sequence_names_overlap_existing_rpp_train': len(selected_names.intersection(
            set((ROOT / 'output/proxy_runs/rpp-v2/proxy_train_sequences.txt').read_text().splitlines())))
    }
    write_json(OUT / 'build_summary.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
