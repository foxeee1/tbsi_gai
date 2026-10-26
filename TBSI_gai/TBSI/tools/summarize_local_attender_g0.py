#!/usr/bin/env python3
"""Summarize completed G0 rows without rerunning the tracker."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True, type=Path)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    with args.rows.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in row:
            if key not in ('sequence', 'frame'):
                row[key] = float(row[key])
    usable = [row for row in rows if row['target_tokens'] > 0]
    mean = lambda key: float(np.mean([row[key] for row in usable]))
    result = {
        'events': len(rows),
        'usable_target_events': len(usable),
        'target_token_coverage': len(usable) / max(len(rows), 1),
        'sequences': len(set(row['sequence'] for row in rows)),
        'token_shape': [640, 768],
        'target_local_rescue_rgb_to_tir': mean('target_local_rescue_rgb_to_tir'),
        'background_local_rescue_rgb_to_tir': mean('background_local_rescue_rgb_to_tir'),
        'target_local_rescue_rate_rgb_to_tir': mean('target_local_rescue_rate_rgb_to_tir'),
        'target_local_rescue_tir_to_rgb': mean('target_local_rescue_tir_to_rgb'),
        'background_local_rescue_tir_to_rgb': mean('background_local_rescue_tir_to_rgb'),
        'target_similarity_margin_rgb': mean('target_template_similarity_rgb') - mean('background_template_similarity_rgb'),
        'target_similarity_margin_tir': mean('target_template_similarity_tir') - mean('background_template_similarity_tir'),
        'gate_definition': {
            'target_rescue_rate_gt': 0.50,
            'target_minus_background_rescue_gt': 0.01,
            'target_similarity_margin_gt': 0.0,
            'target_token_coverage_gt': 0.80,
            'diagnostic_only': True,
        },
    }
    result['status'] = 'PASS' if (
        result['target_token_coverage'] > 0.80 and
        result['target_local_rescue_rate_rgb_to_tir'] > 0.50 and
        result['target_local_rescue_rgb_to_tir'] - result['background_local_rescue_rgb_to_tir'] > 0.01 and
        result['target_similarity_margin_rgb'] > 0.0 and
        result['target_similarity_margin_tir'] > 0.0) else 'STOP'
    result['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
