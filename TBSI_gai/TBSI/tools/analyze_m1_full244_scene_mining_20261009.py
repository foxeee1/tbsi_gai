#!/usr/bin/env python3
"""Read-only scene and failure-onset audit of archived M1 test predictions."""
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'data/lasher/testingset'
PREDICTIONS = ROOT / 'output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42'
AUDIT = ROOT / 'output/experiments/M1_FULL244_SEQUENCE_AUDIT_20261009/per_sequence.csv'
CONFIG = ROOT / 'experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml'
CHECKPOINT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
OUTPUT = ROOT / 'output/experiments/M1_FULL244_SCENE_MINING_20261009'
EXPECTED_CONFIG = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
EXPECTED_CHECKPOINT = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def average(values):
    return float(np.mean(values)) if len(values) else None


def main():
    if OUTPUT.exists():
        raise RuntimeError('refusing existing output directory: {}'.format(OUTPUT))
    if sha256(CONFIG) != EXPECTED_CONFIG or sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError('protected M1 config/checkpoint hash mismatch')
    with AUDIT.open(newline='') as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) != 244 or len({row['sequence'] for row in source_rows}) != 244:
        raise RuntimeError('M1 full-test source audit must contain 244 unique sequences')

    per_sequence, events = [], []
    source_manifest = hashlib.sha256()
    annotation_manifest = hashlib.sha256()
    for source in source_rows:
        name = source['sequence']
        prediction_path = PREDICTIONS / (name + '.txt')
        gt_path = DATA / name / 'init.txt'
        visible_frames = sorted((DATA / name / 'visible').glob('*.jpg'))
        if not prediction_path.is_file() or not gt_path.is_file() or not visible_frames:
            raise RuntimeError('missing archive prediction, annotation, or image: ' + name)
        pred = np.loadtxt(str(prediction_path), ndmin=2, dtype=np.float64)
        gt = np.loadtxt(str(gt_path), delimiter=',', ndmin=2, dtype=np.float64)
        if pred.shape != gt.shape or pred.shape[1] != 4 or len(pred) < 2:
            raise RuntimeError('prediction/GT alignment failed: ' + name)
        image = cv2.imread(str(visible_frames[0]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('cannot read first visible frame: ' + name)
        height, width = image.shape[:2]
        source_manifest.update((name + ':' + sha256(prediction_path) + '\n').encode())
        annotation_manifest.update((name + ':' + sha256(gt_path) + '\n').encode())

        finite_gt = np.isfinite(gt).all(axis=1)
        valid = finite_gt & (gt[:, 2] > 0) & (gt[:, 3] > 0)
        valid[0] = False  # The first row initializes the tracker; it is not a tracking prediction.
        if not np.isfinite(pred).all() or (pred[:, 2:] <= 0).any():
            raise RuntimeError('nonfinite or nonpositive M1 prediction: ' + name)

        inter_w = np.maximum(0, np.minimum(gt[:, 0] + gt[:, 2], pred[:, 0] + pred[:, 2]) -
                             np.maximum(gt[:, 0], pred[:, 0]))
        inter_h = np.maximum(0, np.minimum(gt[:, 1] + gt[:, 3], pred[:, 1] + pred[:, 3]) -
                             np.maximum(gt[:, 1], pred[:, 1]))
        inter = inter_w * inter_h
        gt_area = gt[:, 2] * gt[:, 3]
        pred_area = pred[:, 2] * pred[:, 3]
        union = gt_area + pred_area - inter
        iou = np.full(len(gt), np.nan)
        iou[valid] = inter[valid] / union[valid]

        # The archive stores rounded boxes. This reconstructs the nominal next-frame crop;
        # it is not the original internal floating-point tracker state.
        prev = pred[:-1]
        side = np.ceil(np.sqrt(prev[:, 2] * prev[:, 3]) * 4.0)
        crop_x = np.rint(prev[:, 0] + prev[:, 2] / 2 - side / 2)
        crop_y = np.rint(prev[:, 1] + prev[:, 3] / 2 - side / 2)
        crop = np.column_stack((crop_x, crop_y, side, side))
        crop_inter_w = np.maximum(0, np.minimum(gt[1:, 0] + gt[1:, 2], crop[:, 0] + crop[:, 2]) -
                                  np.maximum(gt[1:, 0], crop[:, 0]))
        crop_inter_h = np.maximum(0, np.minimum(gt[1:, 1] + gt[1:, 3], crop[:, 1] + crop[:, 3]) -
                                  np.maximum(gt[1:, 1], crop[:, 1]))
        coverage = np.full(len(gt), np.nan)
        coverage[1:][valid[1:]] = ((crop_inter_w * crop_inter_h)[valid[1:]] /
                                   gt_area[1:][valid[1:]])
        margin = np.full(len(gt), np.nan)
        margin[1:] = np.minimum.reduce((gt[1:, 0] - crop[:, 0],
                                         gt[1:, 1] - crop[:, 1],
                                         crop[:, 0] + crop[:, 2] - gt[1:, 0] - gt[1:, 2],
                                         crop[:, 1] + crop[:, 3] - gt[1:, 1] - gt[1:, 3]))

        gt_center = gt[:, :2] + gt[:, 2:] / 2
        pred_center = pred[:, :2] + pred[:, 2:] / 2
        center_px = np.linalg.norm(pred_center - gt_center, axis=1)
        gt_diag = np.linalg.norm(gt[:, 2:], axis=1)
        center_norm = np.full(len(gt), np.nan)
        center_norm[valid] = center_px[valid] / gt_diag[valid]
        area_ratio = np.full(len(gt), np.nan)
        area_ratio[valid] = pred_area[valid] / gt_area[valid]

        severe = valid & (iou < 0.2)
        in_crop = valid & (coverage >= 0.999)
        severe_inside = severe & in_crop
        severe_partial = severe & (coverage > 0) & (coverage < 0.999)
        severe_outside = severe & (coverage <= 0)
        large_center = center_norm > 0.5
        large_area = (area_ratio < 0.5) | (area_ratio > 2.0)
        starts = np.flatnonzero(np.diff(np.r_[0, severe.astype(np.int8), 0]) == 1)
        ends = np.flatnonzero(np.diff(np.r_[0, severe.astype(np.int8), 0]) == -1)
        long_runs = []
        for start, end in zip(starts, ends):
            if end - start < 30:
                continue
            later_outside = np.flatnonzero(severe_outside[start:end])
            recovery = None
            good = valid & (iou >= 0.5)
            for index in range(end, len(gt) - 4):
                if good[index:index + 5].all():
                    recovery = index - start
                    break
            event = {
                'sequence': name, 'start_frame_0based': int(start),
                'end_frame_0based_inclusive': int(end - 1), 'length': int(end - start),
                'onset_iou': float(iou[start]), 'onset_crop_coverage': float(coverage[start]),
                'onset_crop_margin_px': float(margin[start]),
                'onset_center_error_gt_diag': float(center_norm[start]),
                'onset_area_ratio': float(area_ratio[start]),
                'first_full_crop_out_delay_frames': (int(later_outside[0])
                                                     if len(later_outside) else 'NA'),
                'first_5frame_recovery_delay_frames': recovery if recovery is not None else 'NA',
            }
            events.append(event)
            long_runs.append(event)

        consecutive_gt = valid[1:] & valid[:-1]
        motion = np.linalg.norm(np.diff(gt_center, axis=0), axis=1)
        motion_norm = motion[consecutive_gt] / gt_diag[:-1][consecutive_gt]
        row = {
            'sequence': name, 'attributes': source['attributes'],
            'm1_auc': float(source['m1_auc']),
            'frames': len(gt), 'valid_tracking_gt_frames': int(valid.sum()),
            'invalid_tracking_gt_frames': int(len(gt) - 1 - valid.sum()),
            'image_width': width, 'image_height': height,
            'median_gt_area_fraction': float(np.median(gt_area[valid] / (width * height))),
            'median_gt_short_side_px': float(np.median(np.minimum(gt[valid, 2], gt[valid, 3]))),
            'median_abs_log_gt_aspect': float(np.median(np.abs(np.log(gt[valid, 2] / gt[valid, 3])))),
            'p90_gt_motion_per_diag': float(np.quantile(motion_norm, 0.9))
            if len(motion_norm) else 'NA',
            'severe_valid_frames': int(severe.sum()),
            'severe_inside_frames': int(severe_inside.sum()),
            'severe_partial_frames': int(severe_partial.sum()),
            'severe_outside_frames': int(severe_outside.sum()),
            'severe_inside_center_large_frames': int((severe_inside & large_center).sum()),
            'severe_inside_area_large_frames': int((severe_inside & large_area).sum()),
            'severe_inside_center_only_frames': int((severe_inside & large_center & ~large_area).sum()),
            'severe_inside_area_only_frames': int((severe_inside & ~large_center & large_area).sum()),
            'severe_run_ge30_count': len(long_runs),
            'severe_run_ge30_frames': sum(item['length'] for item in long_runs),
            'first_severe_run_start_fraction': (long_runs[0]['start_frame_0based'] / len(gt)
                                                if long_runs else 'NA'),
            'first_severe_run_onset_inside': (int(long_runs[0]['onset_crop_coverage'] >= 0.999)
                                                 if long_runs else 'NA'),
            'longest_severe_run': max((item['length'] for item in long_runs), default=0),
            'archived_persistent_lt05_events': int(source['persistent_event_count']),
            'archived_persistent_to_end': int(source['persistent_to_end']),
        }
        per_sequence.append(row)

    low = [row for row in per_sequence if row['m1_auc'] < 40]
    very_low = [row for row in per_sequence if row['m1_auc'] < 20]
    labels = sorted({label for row in per_sequence for label in row['attributes'].split(';')})
    attributes = []
    for label in labels:
        members = [row for row in per_sequence if label in row['attributes'].split(';')]
        nonmembers = [row for row in per_sequence if label not in row['attributes'].split(';')]
        count_low = sum(row['m1_auc'] < 40 for row in members)
        other_low = sum(row['m1_auc'] < 40 for row in nonmembers)
        member_rate = count_low / len(members)
        other_rate = other_low / len(nonmembers)
        attributes.append({
            'attribute': label, 'member_sequences': len(members),
            'member_mean_auc': average([row['m1_auc'] for row in members]),
            'member_auc_lt40_count': count_low, 'member_auc_lt40_fraction': member_rate,
            'nonmember_auc_lt40_fraction': other_rate,
            'low_auc_risk_ratio': member_rate / other_rate if other_rate else 'NA',
            'member_severe_inside_frames': sum(row['severe_inside_frames'] for row in members),
            'member_severe_run_ge30_sequences': sum(row['severe_run_ge30_count'] > 0
                                                   for row in members),
        })

    severe_count = sum(row['severe_valid_frames'] for row in per_sequence)
    inside_count = sum(row['severe_inside_frames'] for row in per_sequence)
    starts_inside = sum(item['onset_crop_coverage'] >= 0.999 for item in events)
    starts_margin10 = sum(item['onset_crop_margin_px'] >= 10 for item in events)
    later_outside = [item for item in events if item['first_full_crop_out_delay_frames'] != 'NA']
    area_quantiles = np.quantile([row['median_gt_area_fraction'] for row in per_sequence],
                                 [0, 0.25, 0.5, 0.75, 1])
    motion_quantiles = np.quantile([row['p90_gt_motion_per_diag'] for row in per_sequence],
                                   [0, 0.25, 0.5, 0.75, 1])

    def quartile_summary(key, edges):
        result = []
        for index in range(4):
            members = [row for row in per_sequence
                       if (edges[index] <= row[key] < edges[index + 1] or
                           (index == 3 and row[key] == edges[index + 1]))]
            result.append({'quartile': index + 1, 'lower': float(edges[index]),
                           'upper': float(edges[index + 1]), 'sequences': len(members),
                           'mean_auc': average([row['m1_auc'] for row in members]),
                           'auc_lt40_count': sum(row['m1_auc'] < 40 for row in members),
                           'severe_run_ge30_sequences': sum(row['severe_run_ge30_count'] > 0
                                                            for row in members)})
        return result

    payload = {
        'id': 'M1-FULL244-SCENE-MINING-20261009',
        'parent': 'TBSI_E_MHSA_FULL_4090D_seed42',
        'scope': 'Read-only archived M1 predictions; LasHeR testingset all 244 sequences',
        'source_audit_sha256': sha256(AUDIT),
        'prediction_manifest_sha256': source_manifest.hexdigest(),
        'annotation_manifest_sha256': annotation_manifest.hexdigest(),
        'config_sha256': EXPECTED_CONFIG, 'checkpoint_sha256': EXPECTED_CHECKPOINT,
        'script_sha256': sha256(Path(__file__)),
        'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
        'command': 'python tools/analyze_m1_full244_scene_mining_20261009.py',
        'official_sequence_equal_auc': average([row['m1_auc'] for row in per_sequence]),
        'sequence_count': len(per_sequence),
        'tracking_frame_count': sum(row['frames'] - 1 for row in per_sequence),
        'valid_tracking_gt_frames': sum(row['valid_tracking_gt_frames'] for row in per_sequence),
        'invalid_tracking_gt_frames': sum(row['invalid_tracking_gt_frames'] for row in per_sequence),
        'auc_lt20_sequences': len(very_low), 'auc_lt40_sequences': len(low),
        'auc_lt40_severe_frame_fraction': (sum(row['severe_valid_frames'] for row in low) /
                                           severe_count),
        'auc_lt40_severe_inside_frame_fraction': (sum(row['severe_inside_frames'] for row in low) /
                                                  inside_count),
        'severe_valid_frames': severe_count,
        'severe_inside_frames': inside_count,
        'severe_partial_frames': sum(row['severe_partial_frames'] for row in per_sequence),
        'severe_outside_frames': sum(row['severe_outside_frames'] for row in per_sequence),
        'severe_inside_center_large_frames': sum(row['severe_inside_center_large_frames']
                                                 for row in per_sequence),
        'severe_inside_area_large_frames': sum(row['severe_inside_area_large_frames']
                                               for row in per_sequence),
        'severe_inside_center_only_frames': sum(row['severe_inside_center_only_frames']
                                                for row in per_sequence),
        'severe_inside_area_only_frames': sum(row['severe_inside_area_only_frames']
                                              for row in per_sequence),
        'severe_run_ge30_count': len(events),
        'severe_run_ge30_sequences': sum(row['severe_run_ge30_count'] > 0
                                        for row in per_sequence),
        'severe_run_onset_inside': starts_inside,
        'severe_run_onset_margin_ge10px': starts_margin10,
        'severe_runs_later_outside': len(later_outside),
        'later_outside_median_delay_frames': float(np.median([
            item['first_full_crop_out_delay_frames'] for item in later_outside])),
        'gt_area_quartiles': quartile_summary('median_gt_area_fraction', area_quantiles),
        'gt_motion_quartiles': quartile_summary('p90_gt_motion_per_diag', motion_quantiles),
        'limits': [
            'Crop reconstructed from rounded archived boxes, not internal tracker state.',
            'Official AUC includes invalid-GT frames in its denominator; morphology excludes them.',
            'Official attributes overlap and are sequence-level tags, not local modality-quality labels.',
            'AUC thresholds and geometry buckets describe failures; no module causal claim or test-set selection.',
            'No tracker forward, feature extraction, perturbation, training, or GPU computation.',
        ],
    }
    if len(per_sequence) != 244 or payload['tracking_frame_count'] != 219048:
        raise RuntimeError('unexpected full-test membership or frame count')
    if abs(payload['official_sequence_equal_auc'] - 56.20771663529531) > 1e-6:
        raise RuntimeError('official M1 AUC reference mismatch')

    OUTPUT.mkdir(parents=True)
    write_csv(OUTPUT / 'per_sequence_scene.csv', per_sequence)
    write_csv(OUTPUT / 'persistent_severe_events.csv', events)
    write_csv(OUTPUT / 'attribute_scene.csv', attributes)
    (OUTPUT / 'summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
