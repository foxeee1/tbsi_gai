#!/usr/bin/env python3
"""Pre-review geometry screen for false-consensus candidate proposals.

Reads only the original candidate table. It never reads intervention outputs
or annotations and never changes the original CSV files.
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path


def box(value):
    return [float(x) for x in json.loads(value)]


def center(b):
    return b[0] + b[2] / 2., b[1] + b[3] / 2.


def point_in_box(p, b):
    return b[0] <= p[0] <= b[0] + b[2] and b[1] <= p[1] <= b[1] + b[3]


def expanded(b, ratio):
    dx, dy = b[2] * ratio, b[3] * ratio
    return [b[0] - dx, b[1] - dy, b[2] + 2 * dx, b[3] + 2 * dy]


def intersection_area(a, b):
    x = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return x * y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--image-root', required=True)
    ap.add_argument('--expanded-ratio', type=float, default=.10)
    ap.add_argument('--max-normalized-distance', type=float, default=.50)
    ap.add_argument('--random-rejected', type=int, default=40)
    ap.add_argument('--seed', type=int, default=3407)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    review_images = out / 'review_images'
    review_images.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(args.input, newline='')))
    derived = []
    passed = []
    rejected = []
    for row in rows:
        rgb = box(row['rgb_box']); tir = box(row['tir_box']); gt = box(row['gt_box'])
        gt_exp = expanded(gt, args.expanded_ratio)
        rgb_center, tir_center = center(rgb), center(tir)
        rgb_out = not point_in_box(rgb_center, gt_exp)
        tir_out = not point_in_box(tir_center, gt_exp)
        rgb_cov = intersection_area(rgb, gt) / max(rgb[2] * rgb[3], 1e-12)
        tir_cov = intersection_area(tir, gt) / max(tir[2] * tir[3], 1e-12)
        dist = float(row['center_distance'])
        diag = (gt[2] ** 2 + gt[3] ** 2) ** .5
        max_dist = min(64., .5 * diag)
        normalized = dist / max(diag, 1e-12)
        reasons = []
        if not rgb_out: reasons.append('rgb_center_inside_expanded_gt')
        if not tir_out: reasons.append('tir_center_inside_expanded_gt')
        if dist > max_dist: reasons.append('pair_distance_over_limit')
        item = dict(row)
        item.update({'gt_expanded_box': json.dumps(gt_exp), 'rgb_center_outside_gt': int(rgb_out),
                     'tir_center_outside_gt': int(tir_out), 'rgb_candidate_gt_coverage': rgb_cov,
                     'tir_candidate_gt_coverage': tir_cov, 'gt_diagonal': diag,
                     'pair_distance_limit': max_dist, 'pair_distance_normalized': normalized,
                     'screen_pass': int(not reasons), 'screen_reason': ';'.join(reasons)})
        derived.append(item)
        (passed if not reasons else rejected).append(item)
    fields = list(derived[0].keys())
    with open(out / 'candidate_v1.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(derived)
    rng = random.Random(args.seed)
    sample = rng.sample(rejected, min(args.random_rejected, len(rejected)))
    queue = []
    for item in passed:
        x = dict(item); x['review_priority'] = 'screen_pass'; queue.append(x)
    for item in sample:
        x = dict(item); x['review_priority'] = 'random_screen_rejected'; queue.append(x)
    queue_fields = fields + ['review_priority', 'event_type', 'same_physical_object', 'visible_in_both', 'alignment_ok', 'reviewer', 'comment']
    with open(out / 'review_queue_v1.1.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=queue_fields); w.writeheader()
        for item in queue:
            x = dict(item)
            x.update({'event_type': 'uncertain', 'same_physical_object': '', 'visible_in_both': '', 'alignment_ok': '', 'reviewer': '', 'comment': ''})
            w.writerow(x)
            src = Path(args.image_root) / (item['event_id'] + '.jpg')
            if src.is_file(): shutil.copy2(src, review_images / src.name)
    rules = {'version': 'v1.1_pre_review_screen', 'input_preserved': str(Path(args.input).resolve()),
             'expanded_gt_ratio': args.expanded_ratio, 'max_pair_distance_pixels': 64.0,
             'max_pair_distance_as_gt_diagonal_ratio': args.max_normalized_distance,
             'pass_rule': 'both candidate centers outside expanded GT and pair distance <= min(64px, 0.5*GT diagonal)',
             'random_rejected_count': len(sample), 'random_seed': args.seed,
             'intervention_outcomes_read': False, 'human_annotations_read': False,
             'counts': {'input': len(rows), 'screen_pass': len(passed), 'screen_rejected': len(rejected), 'review_queue': len(queue)}}
    (out / 'screening_rules.json').write_text(json.dumps(rules, indent=2, ensure_ascii=True))
    print(json.dumps(rules['counts'], indent=2), flush=True)


if __name__ == '__main__':
    main()
