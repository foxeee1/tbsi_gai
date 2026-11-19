#!/usr/bin/env python3
"""Analyze frozen RGB/TIR correspondence on the fixed G0-Hard cohort."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.g0_post_tbsi_complement import average_precision, feature_metrics


def bootstrap(values, seed=42, draws=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    sampled = values[rng.randint(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()), 'lo95': float(np.quantile(sampled, 0.025)),
            'hi95': float(np.quantile(sampled, 0.975)), 'n_sequences': int(len(values))}


def dilate(mask, grid=16):
    mask = np.asarray(mask, dtype=bool).reshape(grid, grid)
    out = np.zeros_like(mask)
    for dh in (-1, 0, 1):
        for dw in (-1, 0, 1):
            src_h = slice(max(0, -dh), min(grid, grid - dh))
            src_w = slice(max(0, -dw), min(grid, grid - dw))
            dst_h = slice(max(0, dh), min(grid, grid + dh))
            dst_w = slice(max(0, dw), min(grid, grid + dw))
            out[dst_h, dst_w] |= mask[src_h, src_w]
    return out.reshape(-1)


def retrieval_metrics(rgb, tir, mask, xy_score):
    rgb = F.normalize(torch.from_numpy(rgb).float(), dim=-1)
    tir = F.normalize(torch.from_numpy(tir).float(), dim=-1)
    sim = (rgb @ tir.t()).numpy()
    mask = np.asarray(mask, dtype=bool)
    neigh = dilate(mask)
    bg = ~neigh
    def direction(matrix):
        target_queries = np.flatnonzero(mask)
        margins = []
        hits = []
        for q in target_queries:
            target_scores = matrix[q, neigh]
            bg_scores = matrix[q, bg]
            margins.append(float(target_scores.max() - bg_scores.max()) if bg.any() else 0.0)
            topk = np.argsort(-matrix[q])[:5]
            hits.append(float(neigh[topk].any()))
        return {'target_query_neighborhood_margin': float(np.mean(margins)),
                'target_query_neighborhood_top5_hit': float(np.mean(hits)),
                'target_queries': int(len(target_queries))}
    reverse = sim.T
    result = {'rgb_query_to_tir': direction(sim), 'tir_query_to_rgb': direction(reverse),
              'token_ap': average_precision(np.diagonal(sim), mask),
              'xy_ap': average_precision(xy_score, mask)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    groups = ('rescue_failure', 'offcenter', 'easy_centered')
    group_rows = {group: [row for row in rows if group in row['groups']] for group in groups}
    all_results = {}
    for group in groups:
        selected = group_rows[group]
        if not selected:
            all_results[group] = {'events': 0}
            continue
        records = []
        for idx, row in enumerate(selected):
            data = np.load(root / (row['event_id'] + '.npz'))
            mask = data['mask'].astype(bool)
            xy = -((np.indices((16, 16))[1].reshape(-1) - 8) ** 2 +
                   (np.indices((16, 16))[0].reshape(-1) - 8) ** 2).astype(float)
            record = {'event_id': row['event_id'], 'sequence': row['sequence'],
                      'm1_iou': row['m1_iou'], 'center_distance_tokens': row['center_distance_tokens']}
            for stage in ('post_tbsi', 'post_block'):
                record[stage] = feature_metrics(data[stage + '_rgb'], data[stage + '_tir'], mask, xy)
                record[stage + '_retrieval'] = retrieval_metrics(data[stage + '_rgb'],
                                                                  data[stage + '_tir'], mask, xy)
            records.append(record)
        # Cyclic mismatch within the same preselected condition; fixed one-token spatial shift.
        for idx, record in enumerate(records):
            current = np.load(root / (record['event_id'] + '.npz'))
            other = np.load(root / (records[(idx + 1) % len(records)]['event_id'] + '.npz'))
            mask = current['mask'].astype(bool)
            xy = -((np.indices((16, 16))[1].reshape(-1) - 8) ** 2 +
                   (np.indices((16, 16))[0].reshape(-1) - 8) ** 2).astype(float)
            for stage in ('post_tbsi', 'post_block'):
                shifted = np.roll(current[stage + '_tir'].reshape(16, 16, -1), 1, axis=1).reshape(256, -1)
                record[stage + '_event_shuffle'] = retrieval_metrics(current[stage + '_rgb'],
                                                                       other[stage + '_tir'], mask, xy)
                record[stage + '_spatial_shift'] = retrieval_metrics(current[stage + '_rgb'],
                                                                      shifted, mask, xy)

        def mean(path, records_in=records):
            values = []
            for record in records_in:
                value = record
                for key in path.split('.'):
                    value = value[key]
                values.append(value)
            return float(np.mean(values))

        seq_values = {}
        for seq in sorted(set(record['sequence'] for record in records)):
            seq_records = [record for record in records if record['sequence'] == seq]
            seq_values[seq] = {
                'correct_minus_event_ap': float(np.mean([record['post_tbsi']['token_ap'] - record['post_tbsi_event_shuffle']['token_ap'] for record in seq_records])),
                'correct_minus_spatial_ap': float(np.mean([record['post_tbsi']['token_ap'] - record['post_tbsi_spatial_shift']['token_ap'] for record in seq_records])),
                'correct_minus_event_retrieval_rgb': float(np.mean([
                    record['post_tbsi_retrieval']['rgb_query_to_tir']['target_query_neighborhood_margin'] -
                    record['post_tbsi_event_shuffle']['rgb_query_to_tir']['target_query_neighborhood_margin']
                    for record in seq_records])),
                'correct_minus_event_retrieval_tir': float(np.mean([
                    record['post_tbsi_retrieval']['tir_query_to_rgb']['target_query_neighborhood_margin'] -
                    record['post_tbsi_event_shuffle']['tir_query_to_rgb']['target_query_neighborhood_margin']
                    for record in seq_records])),
            }
        summary = {
            'events': len(records), 'sequences': len(seq_values),
            'm1_iou_mean': mean('m1_iou'),
            'center_distance_tokens_mean': mean('center_distance_tokens'),
            'post_tbsi': {
                'correct_token_ap': mean('post_tbsi.token_ap'),
                'event_shuffle_token_ap': mean('post_tbsi_event_shuffle.token_ap'),
                'spatial_shift_token_ap': mean('post_tbsi_spatial_shift.token_ap'),
                'xy_only_ap': mean('post_tbsi.xy_ap'),
                'correct_minus_event_ap': mean('post_tbsi.token_ap') - mean('post_tbsi_event_shuffle.token_ap'),
                'correct_minus_spatial_ap': mean('post_tbsi.token_ap') - mean('post_tbsi_spatial_shift.token_ap'),
                'correct_retrieval_rgb_margin': mean('post_tbsi_retrieval.rgb_query_to_tir.target_query_neighborhood_margin'),
                'event_retrieval_rgb_margin': mean('post_tbsi_event_shuffle.rgb_query_to_tir.target_query_neighborhood_margin'),
                'correct_retrieval_tir_margin': mean('post_tbsi_retrieval.tir_query_to_rgb.target_query_neighborhood_margin'),
                'event_retrieval_tir_margin': mean('post_tbsi_event_shuffle.tir_query_to_rgb.target_query_neighborhood_margin'),
                'correct_minus_event_retrieval_rgb': mean('post_tbsi_retrieval.rgb_query_to_tir.target_query_neighborhood_margin') - mean('post_tbsi_event_shuffle.rgb_query_to_tir.target_query_neighborhood_margin'),
                'correct_minus_event_retrieval_tir': mean('post_tbsi_retrieval.tir_query_to_rgb.target_query_neighborhood_margin') - mean('post_tbsi_event_shuffle.tir_query_to_rgb.target_query_neighborhood_margin'),
            },
            'sequence_bootstrap_95': {key: bootstrap([value[key] for value in seq_values.values()])
                                      for key in next(iter(seq_values.values()))},
            'sequence_rows': seq_values,
            'records': records,
        }
        all_results[group] = summary
    result = {'groups': all_results,
              'decision_rule': {
                  'requires_correct_minus_event_ap_gt': 0.03,
                  'requires_correct_minus_event_retrieval_rgb_gt': 0.0,
                  'requires_correct_minus_event_retrieval_tir_gt': 0.0,
                  'diagnostic_only': True,
              }}
    (root / 'hard_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({group: all_results[group].get('post_tbsi', {}) for group in groups}, indent=2))


if __name__ == '__main__':
    main()
