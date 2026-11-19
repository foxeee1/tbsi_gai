#!/usr/bin/env python3
"""Derive frozen search-feature energy and conflict signals for utility replay."""
import argparse
import csv
from pathlib import Path

import numpy as np


def local_energy(feature, x, y):
    value = feature.astype(np.float32)
    norm = np.linalg.norm(value, axis=-1)
    y0, y1 = max(0, y - 1), min(16, y + 2)
    x0, x1 = max(0, x - 1), min(16, x + 2)
    return float(norm[y0:y1, x0:x1].mean() / max(float(norm.mean()), 1e-8))


def local_conflict(rgb, tir, x, y):
    rgb = rgb.astype(np.float32)
    tir = tir.astype(np.float32)
    rgb /= np.maximum(np.linalg.norm(rgb, axis=-1, keepdims=True), 1e-8)
    tir /= np.maximum(np.linalg.norm(tir, axis=-1, keepdims=True), 1e-8)
    conflict = 1.0 - np.sum(rgb * tir, axis=-1)
    return float(conflict[max(0, y - 1):min(16, y + 2),
                          max(0, x - 1):min(16, x + 2)].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--center-csv', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = np.load(args.features)
    rgb_features = data['rgb']
    tir_features = data['tir']
    sequences = data['sequence']
    frames = data['frame']
    with open(args.center_csv, newline='') as stream:
        center_rows = [r for r in csv.DictReader(stream) if r['arm'] == 'baseline']
    center_map = {(r['sequence'], int(r['frame'])): r for r in center_rows}
    output = []
    for i, (seq, frame) in enumerate(zip(sequences, frames)):
        key = (str(seq), int(frame))
        if key not in center_map:
            raise RuntimeError('missing baseline CENTER row: {}'.format(key))
        center = center_map[key]
        x, y = int(float(center['window_peak_x'])), int(float(center['window_peak_y']))
        rgb_energy = local_energy(rgb_features[i], x, y)
        tir_energy = local_energy(tir_features[i], x, y)
        output.append({
            'sequence': key[0], 'frame': key[1],
            'local_feature_energy_ratio_gap_rgb_minus_tir': rgb_energy - tir_energy,
            'local_conflict_at_center_peak': local_conflict(rgb_features[i], tir_features[i], x, y),
            'template_support_gap_rgb_minus_tir': 'nan',
            'template_support_psr_gap_rgb_minus_tir': 'nan'})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0].keys()))
        writer.writeheader(); writer.writerows(output)
    print('wrote {} signal rows for {} states'.format(len(output), len(set(r['sequence'] for r in output))))


if __name__ == '__main__':
    main()
