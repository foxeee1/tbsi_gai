"""Geometry audit of saved predictions only; no tracker/model execution."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NAME = 'M1_STSER_SOT_FULL15_COMPETITION_20261025'
RUN = ROOT / 'output/experiments' / NAME
OUT = RUN / 'attribution_assets_only_20261025'
PRED = ROOT / 'output/test/tracking_results/tbsi_track' / NAME
DATA = Path('/root/autodl-tmp/TBSI_gai/data/lasher/testingset')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(threshold=0.2, min_length=5):
    with (RUN / 'analysis_lasher244/per_sequence_metrics.csv').open() as handle:
        seq_metrics = list(csv.DictReader(handle))
    records, episodes, inputs = [], [], []
    totals = dict(valid_noninit_frames=0, bad_frames=0,
                  bad_with_center_inside=0, bad_with_90pct_box_inside=0,
                  bad_with_center_outside=0)
    for metadata in seq_metrics:
        name = metadata['sequence']
        pred_path = PRED / (name + '.txt')
        gt_path = DATA / name / 'init.txt'
        pred = np.loadtxt(str(pred_path), delimiter='\t', ndmin=2)
        gt = np.loadtxt(str(gt_path), delimiter=',', ndmin=2)
        if pred.shape != gt.shape or pred.shape[1] != 4:
            raise ValueError('unaligned saved boxes: ' + name)
        inputs.append(dict(sequence=name, frames=len(gt),
                           prediction_sha256=sha(pred_path),
                           annotation_sha256=sha(gt_path)))
        valid = np.isfinite(gt).all(1) & (gt[:, 2:] > 0).all(1)
        if not np.isfinite(pred).all() or not (pred[:, 2:] > 0).all():
            raise ValueError('invalid predicted boxes: ' + name)
        previous = np.concatenate((gt[:1], pred[:-1]), axis=0)
        # Same nominal square window as sample_target. Saved coordinates may
        # be rounded, so this is a geometric reconstruction, not a pixel trace.
        side = np.ceil(np.sqrt(previous[:, 2] * previous[:, 3]) * 4.0)
        crop_lo = np.rint(previous[:, :2] + 0.5 * previous[:, 2:] - side[:, None] / 2)
        crop_hi = crop_lo + side[:, None]
        gt_hi = gt[:, :2] + gt[:, 2:]
        inter_crop = np.maximum(0, np.minimum(gt_hi, crop_hi) - np.maximum(gt[:, :2], crop_lo))
        coverage = inter_crop.prod(1) / np.maximum(gt[:, 2:].prod(1), 1e-12)
        center = gt[:, :2] + gt[:, 2:] / 2
        center_inside = ((center >= crop_lo) & (center <= crop_hi)).all(1)
        inter = np.maximum(0, np.minimum(pred[:, :2] + pred[:, 2:], gt_hi)
                           - np.maximum(pred[:, :2], gt[:, :2])).prod(1)
        iou = inter / np.maximum(pred[:, 2:].prod(1) + gt[:, 2:].prod(1) - inter, 1e-12)
        valid[0] = False
        bad = valid & (iou < threshold)
        row = dict(sequence=name, frames=len(gt), AUC=float(metadata['AUC']),
                   valid_noninit_frames=int(valid.sum()), bad_frames=int(bad.sum()),
                   bad_with_center_inside=int((bad & center_inside).sum()),
                   bad_with_90pct_box_inside=int((bad & (coverage >= .9)).sum()),
                   bad_with_center_outside=int((bad & ~center_inside).sum()),
                   persistent_episodes=0)
        edge = np.diff(np.concatenate(([0], bad.astype(np.int8), [0])))
        for start, stop in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
            if stop - start < min_length:
                continue
            row['persistent_episodes'] += 1
            # Temporal order is descriptive; this does not establish distractor
            # selection or the causal role of any fusion operator.
            outside = np.flatnonzero(~center_inside[start:stop])
            near_good = bool(np.any(iou[max(1, start - 5):start] >= .5))
            pred_center = pred[start, :2] + pred[start, 2:] / 2
            normalized_error = np.abs(pred_center - center[start]) / np.maximum(gt[start, 2:], 1e-12)
            episodes.append(dict(sequence=name, onset_frame_1based=int(start + 1),
                                 length=int(stop - start), onset_iou=float(iou[start]),
                                 onset_center_inside=bool(center_inside[start]),
                                 onset_gt_box_coverage=float(coverage[start]),
                                 onset_center_error_gt_units_linf=float(normalized_error.max()),
                                 good_within_previous5=near_good,
                                 first_center_outside_offset=(int(outside[0]) if len(outside) else None)))
        records.append(row)
        for key in totals:
            totals[key] += row[key]
    totals['bad_center_inside_pct'] = 100 * totals['bad_with_center_inside'] / totals['bad_frames']
    totals['bad_90pct_box_inside_pct'] = 100 * totals['bad_with_90pct_box_inside'] / totals['bad_frames']
    onset_inside = [e for e in episodes if e['onset_center_inside']]
    near_good = [e for e in episodes if e['good_within_previous5']]
    summary = dict(
        execution='existing prediction and annotation arrays only; no model, tracker, training or forward',
        scope='candidate-only LasHeR full244; no M1 pairing',
        definitions=dict(bad_iou_less_than=threshold, persistent_min_frames=min_length,
                         crop='sample_target nominal square from previous saved prediction; factor4; ceil/round',
                         visibility='geometric inclusion only; no occlusion/visibility labels'),
        totals=totals, sequence_count=len(records), frames_including_initialization=sum(r['frames'] for r in records),
        persistent_episodes=len(episodes),
        persistent_onsets_center_inside=len(onset_inside),
        persistent_onsets_90pct_box_inside=sum(e['onset_gt_box_coverage'] >= .9 for e in episodes),
        inside_onsets_then_center_exits_during_same_bad_episode=sum(
            e['first_center_outside_offset'] is not None for e in onset_inside),
        inside_onsets_median_exit_delay_frames=(float(np.median([
            e['first_center_outside_offset'] for e in onset_inside
            if e['first_center_outside_offset'] is not None])) if any(
                e['first_center_outside_offset'] is not None for e in onset_inside) else None),
        onsets_with_good_within_previous5=len(near_good),
        near_good_onsets_center_inside=sum(e['onset_center_inside'] for e in near_good),
        near_good_onsets_90pct_box_inside=sum(e['onset_gt_box_coverage'] >= .9 for e in near_good),
        onset_center_error_gt_units_linf_ge1=sum(e['onset_center_error_gt_units_linf'] >= 1 for e in episodes),
        limits=['No matched M1 trajectory: cannot identify added failures or gains.',
                'Geometric containment does not establish target visibility or feature recoverability.',
                'Temporal order does not identify the causal fusion component.',
                'Serialized prediction precision can move a reconstructed crop by a pixel.'])
    return summary, records, episodes, inputs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, records, episodes, inputs = audit()
    sensitivity = []
    for threshold, length in [(.1, 5), (.2, 10), (.3, 5)]:
        other, _, _, _ = audit(threshold, length)
        sensitivity.append(dict(threshold=threshold, min_length=length,
                                persistent_episodes=other['persistent_episodes'],
                                onsets_center_inside=other['persistent_onsets_center_inside'],
                                bad_center_inside_pct=other['totals']['bad_center_inside_pct']))
    summary['threshold_sensitivity'] = sensitivity
    summary['script_sha256'] = sha(Path(__file__))
    for name, rows in [('trajectory_per_sequence.csv', records), ('trajectory_failure_episodes.csv', episodes),
                       ('trajectory_input_manifest.csv', inputs)]:
        with (OUT / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / 'trajectory_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
