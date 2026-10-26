#!/usr/bin/env python
"""Offline P0 audit for the TCSI shared/complementary interaction strategy.

The existing fusion snapshots contain attention mass summaries but not the
per-token Top-K indices required to measure exact shared/complementary sets.
This script therefore reports the observable fixed-budget evidence, runs a
deterministic set-decomposition sanity check, and refuses to claim training
readiness when raw support indices are absent.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict

import numpy as np


K_VALUES = (8, 16, 32)
REQUIRED_FIELDS = ('rt_topk_mass_8', 'rt_topk_mass_16', 'rt_topk_mass_32',
                   'tr_topk_mass_8', 'tr_topk_mass_16', 'tr_topk_mass_32',
                   'attention_entropy', 'cross_cosine_delta', 'layer')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with open(path, 'r') as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [key for key in REQUIRED_FIELDS if key not in row]
            if missing:
                raise ValueError('line {} missing {}'.format(line_no, ','.join(missing)))
            rows.append(row)
    if not rows:
        raise ValueError('snapshot is empty')
    return rows


def mean(rows, field):
    return float(np.mean([float(row[field]) for row in rows]))


def quantiles(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'q05': float(np.quantile(values, 0.05)), 'q95': float(np.quantile(values, 0.95))}


def decompose_support(rgb_indices, tir_indices, universe_size):
    """Exact set decomposition used by the eventual model implementation."""
    rgb = set(int(x) for x in rgb_indices)
    tir = set(int(x) for x in tir_indices)
    universe = set(range(int(universe_size)))
    shared = rgb & tir
    rgb_comp = rgb - tir
    tir_comp = tir - rgb
    invalid = universe - (shared | rgb_comp | tir_comp)
    return {'shared': sorted(shared), 'rgb_complementary': sorted(rgb_comp),
            'tir_complementary': sorted(tir_comp), 'invalid': sorted(invalid)}


def sanity_check():
    result = decompose_support([0, 1, 2, 7], [2, 3, 7, 9], 12)
    expected = {'shared': [2, 7], 'rgb_complementary': [0, 1],
                'tir_complementary': [3, 9], 'invalid': [4, 5, 6, 8, 10, 11]}
    if result != expected:
        raise AssertionError('support decomposition mismatch: {}'.format(result))
    all_sets = [set(result[key]) for key in ('shared', 'rgb_complementary', 'tir_complementary', 'invalid')]
    if sum(len(item) for item in all_sets) != 12:
        raise AssertionError('support sets do not partition universe')
    return {'passed': True, 'example': result}


def summarize(rows):
    summary = {'rows': len(rows), 'layers': sorted(set(int(row['layer']) for row in rows)),
               'sequences': len(set(row.get('sequence') for row in rows)),
               'budget': {}, 'global': {}, 'by_layer': {}, 'by_region': {}}
    for k in K_VALUES:
        rt = 'rt_topk_mass_{}'.format(k)
        tr = 'tr_topk_mass_{}'.format(k)
        summary['budget'][str(k)] = {
            'rgb_to_tir_mass': quantiles(rows, rt),
            'tir_to_rgb_mass': quantiles(rows, tr),
            'mean_retained_bidirectional_mass': float(np.mean([(float(r[rt]) + float(r[tr])) / 2.0 for r in rows])),
        }
    summary['global'] = {
        'attention_entropy': quantiles(rows, 'attention_entropy'),
        'cross_cosine_delta': quantiles(rows, 'cross_cosine_delta'),
        'rt_over_tt': quantiles(rows, 'rt_over_tt') if 'rt_over_tt' in rows[0] else None,
        'tr_over_rr': quantiles(rows, 'tr_over_rr') if 'tr_over_rr' in rows[0] else None,
    }
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row['layer'])].append(row)
    for layer, layer_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        summary['by_layer'][layer] = {
            'rows': len(layer_rows),
            'attention_entropy': quantiles(layer_rows, 'attention_entropy'),
            'cross_cosine_delta': quantiles(layer_rows, 'cross_cosine_delta'),
            'mass_16_mean': float(np.mean([(float(r['rt_topk_mass_16']) + float(r['tr_topk_mass_16'])) / 2.0 for r in layer_rows])),
        }
    for region in sorted(set(row.get('region', 'unknown') for row in rows)):
        region_rows = [row for row in rows if row.get('region', 'unknown') == region]
        summary['by_region'][region] = {
            'rows': len(region_rows),
            'attention_entropy': quantiles(region_rows, 'attention_entropy'),
            'cross_cosine_delta': quantiles(region_rows, 'cross_cosine_delta'),
            'mass_16_mean': float(np.mean([(float(r['rt_topk_mass_16']) + float(r['tr_topk_mass_16'])) / 2.0 for r in region_rows])),
        }
    return summary


def write_markdown(path, result):
    summary = result['observable_snapshot_summary']
    lines = [
        '# TCSI Strategy v1 P0 Audit', '',
        'Status: `DIAGNOSTIC`; no model parameters or canonical configuration changed.', '',
        '## Evidence', '',
        '- Snapshot rows: `{}`'.format(summary['rows']),
        '- Layers: `{}`'.format(', '.join(str(x) for x in summary['layers'])),
        '- Exact shared/complementary ratios: `UNAVAILABLE` because raw Top-K indices are absent.', '',
        '## Observable fixed-budget statistics', '',
        '| k | RGB->TIR mass mean | TIR->RGB mass mean | bidirectional retained mass |',
        '|---:|---:|---:|---:|',
    ]
    for k in K_VALUES:
        item = summary['budget'][str(k)]
        lines.append('| {} | {:.6f} | {:.6f} | {:.6f} |'.format(
            k, item['rgb_to_tir_mass']['mean'], item['tir_to_rgb_mass']['mean'],
            item['mean_retained_bidirectional_mass']))
    global_stats = summary['global']
    lines.extend(['', 'Global attention entropy mean: `{:.6f}`'.format(global_stats['attention_entropy']['mean']),
                  'Global cross-modal cosine delta mean: `{:.6f}`'.format(global_stats['cross_cosine_delta']['mean']), '',
                  '## Gate', '',
                  '- Set-decomposition numerical sanity: `{}`'.format(result['set_decomposition_sanity']['passed']),
                  '- Raw support-index contract: `FAIL` (not present in source snapshots).',
                  '- Training readiness: `STOP`; do not start TCSI training from these artifacts.', '',
                  'The existing snapshots show nonzero fixed-budget mass and small positive cross-modal changes, but they cannot establish shared versus complementary support. A real training decision requires a new probe that saves per-head/per-query `idx_rgb` and `idx_tir`, plus GT-aligned foreground/background masks and parameter-count accounting.'])
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def run(args):
    rows = read_jsonl(args.snapshot)
    result = {
        'experiment': 'tcsi_strategy_v1_p0_audit',
        'snapshot': os.path.abspath(args.snapshot),
        'snapshot_sha256': sha256_file(args.snapshot),
        'observable_snapshot_summary': summarize(rows),
        'set_decomposition_sanity': sanity_check(),
        'raw_support_indices_present': False,
        'training_ready': False,
        'decision': 'STOP_BEFORE_TRAINING',
        'reason': 'existing snapshot has top-k mass summaries but no per-query RGB/TIR support indices',
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'analysis.json'), 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    write_markdown(os.path.join(args.output_dir, 'analysis.md'), result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
