#!/usr/bin/env python3
"""Read-only sequence and attribute audit of archived M1 full-test results."""

import csv
import hashlib
import json
import statistics
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'output/experiments/m1_full_test_failure_map_20261005'
OFFICIAL = SOURCE / 'official_metric_and_persistence.json'
MORPHOLOGY = SOURCE / 'per_sequence.csv'
ATTRIBUTES = ROOT.parent / 'data/lasher/Attributes&Order.zip'
PREDICTIONS = ROOT / 'output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42'
OUTPUT = ROOT / 'output/experiments/M1_FULL244_SEQUENCE_AUDIT_20261009'
LABELS = ('NO', 'PO', 'TO', 'HO', 'MB', 'LI', 'HI', 'AIV', 'LR', 'DEF',
          'BC', 'SA', 'CM', 'TC', 'FL', 'OV', 'FM', 'SV', 'ARV')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    return sum(values) / len(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def main():
    source = json.loads(OFFICIAL.read_text())
    if source['missing'] or len(source['per_sequence']) != 244:
        raise RuntimeError('Archived official audit is not a complete 244-sequence result')
    if abs(source['aggregate']['M1']['official_auc_percent'] - 56.20771663529531) > 1e-9:
        raise RuntimeError('M1 official AUC differs from the accepted audit')
    if abs(source['aggregate']['M1']['official_op50_percent'] - 68.3339) > 0.001:
        raise RuntimeError('M1 OP50 differs from the protected reference')
    if abs(source['aggregate']['M1']['official_op75_percent'] - 47.1349) > 0.001:
        raise RuntimeError('M1 OP75 differs from the protected reference')

    with MORPHOLOGY.open(newline='') as stream:
        morphology = {row['sequence']: row for row in csv.DictReader(stream)}
    official = {row['sequence']: row for row in source['per_sequence']}
    if len(official) != 244 or set(morphology) != set(official):
        raise RuntimeError('Official and box-error sequence members do not match')

    rows = []
    prediction_digest = hashlib.sha256()
    timing_frames = 0
    timing_seconds = 0.0
    with zipfile.ZipFile(ATTRIBUTES) as archive:
        archive_members = set(archive.namelist())
        for name in sorted(official):
            member = 'AttriSeqsTxt/{}.txt'.format(name)
            prediction = PREDICTIONS / (name + '.txt')
            timing = PREDICTIONS / (name + '_time.txt')
            if member not in archive_members or not prediction.is_file() or not timing.is_file():
                raise RuntimeError('Missing attribute, prediction, or timing for ' + name)
            flags = [int(value) for value in archive.read(member).decode().strip().split(',')]
            if len(flags) != len(LABELS) or any(value not in (0, 1) for value in flags):
                raise RuntimeError('Unexpected LasHeR attribute row for ' + name)
            attributes = [label for label, flag in zip(LABELS, flags) if flag]
            item = official[name]
            m1 = item['M1']
            b0 = item['B0']
            box = morphology[name]
            events = m1['persistent_failure_segments']
            persistent_frames = sum(event['length_at_least_persistent_len'] for event in events)
            first_start = min((event['start_frame_1based'] for event in events), default=None)
            if m1['raw_length_relation'] != 'equal' or int(box['frames_pred']) != item['gt_frames']:
                raise RuntimeError('Prediction/GT length mismatch for ' + name)
            frame_times = [float(value) for value in timing.read_text().splitlines()]
            if len(frame_times) != item['gt_frames']:
                raise RuntimeError('Timing/GT length mismatch for ' + name)
            valid_times = [value for value in frame_times if value > 0]
            timing_frames += len(valid_times)
            timing_seconds += sum(valid_times)
            prediction_digest.update(name.encode() + b'\0' + sha256(prediction).encode() + b'\n')
            rows.append({
                'sequence': name,
                'frames': item['gt_frames'],
                'valid_gt_frames': m1['official_valid_gt_frames'],
                'attributes': ';'.join(attributes),
                'm1_auc': 100 * m1['official_sequence_curve_auc_sampled'],
                'b0_auc': 100 * b0['official_sequence_curve_auc_sampled'],
                'm1_minus_b0_auc': 100 * (m1['official_sequence_curve_auc_sampled'] -
                                            b0['official_sequence_curve_auc_sampled']),
                'm1_iou_lt_05_fraction': m1['frame_weighted_failure_rate_lt_0.5'],
                'm1_iou_lt_02_fraction': m1['frame_weighted_failure_rate_lt_0.2'],
                'persistent_event_count': len(events),
                'persistent_frames': persistent_frames,
                'longest_persistent_event': max(
                    (event['length_at_least_persistent_len'] for event in events), default=0),
                'first_persistent_frame': first_start if first_start is not None else 'NA',
                'first_persistent_fraction': first_start / item['gt_frames']
                if first_start is not None else 'NA',
                'persistent_to_end': int(any(event['recovery_frame_1based'] is None
                                             for event in events)),
                'center_error_gt_diag_mean': float(box['center_error_normalized_gt_diag_mean']),
                'area_log_error_mean': float(box['area_log_error_mean']),
                'aspect_log_error_mean': float(box['aspect_log_error_mean']),
                'positive_timing_frames': len(valid_times),
                'measured_fps': len(valid_times) / sum(valid_times) if valid_times else 'NA',
            })

    auc = mean([row['m1_auc'] for row in rows])
    if abs(auc - source['aggregate']['M1']['official_auc_percent']) > 1e-9:
        raise RuntimeError('Sequence-equal AUC failed to reproduce the official result')
    output_fields = list(rows[0])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'per_sequence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    attributes = []
    for label in LABELS:
        members = [row for row in rows if label in row['attributes'].split(';')]
        nonmembers = [row for row in rows if label not in row['attributes'].split(';')]
        if not members:
            raise RuntimeError('No full-test members carry attribute ' + label)
        attributes.append({
            'attribute': label,
            'sequences': len(members),
            'm1_mean_auc': mean([row['m1_auc'] for row in members]),
            'm1_median_auc': median([row['m1_auc'] for row in members]),
            'nonmember_mean_auc': mean([row['m1_auc'] for row in nonmembers]),
            'member_minus_nonmember_auc': mean([row['m1_auc'] for row in members]) -
                                          mean([row['m1_auc'] for row in nonmembers]),
            'persistent_sequences': sum(row['persistent_event_count'] > 0 for row in members),
            'persistent_sequence_fraction': mean(
                [int(row['persistent_event_count'] > 0) for row in members]),
            'persistent_frames': sum(row['persistent_frames'] for row in members),
            'm1_minus_b0_mean_auc': mean([row['m1_minus_b0_auc'] for row in members]),
        })
    with (OUTPUT / 'attribute_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(attributes[0]))
        writer.writeheader()
        writer.writerows(attributes)

    affected = [row for row in rows if row['persistent_event_count']]
    all_events = [event for item in source['per_sequence']
                  for event in item['M1']['persistent_failure_segments']]
    summary = {
        'identity': {
            'parent': 'TBSI-E-MHSA-FULL-20260927',
            'config': 'TBSI_E_MHSA_FULL_4090D_seed42', 'seed': 42,
            'checkpoint_epoch': 15,
            'checkpoint_sha256_from_protected_registry':
                '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef',
            'source_official_json_sha256': sha256(OFFICIAL),
            'source_box_error_csv_sha256': sha256(MORPHOLOGY),
            'attributes_zip_sha256': sha256(ATTRIBUTES),
            'prediction_manifest_sha256': prediction_digest.hexdigest(),
            'analysis_script_sha256': sha256(Path(__file__)),
            'python': sys.version.split()[0],
            'analysis_command': 'python tools/analyze_m1_full244_sequences_20261009.py',
        },
        'scope': 'LasHeR testingset; all 244 valid sequences; archived M1 epoch-15 predictions',
        'official': source['aggregate']['M1'],
        'descriptive': {
            'sequence_count': len(rows),
            'prediction_frames': sum(row['frames'] for row in rows),
            'sequences_with_persistent_failure': len(affected),
            'persistent_events': len(all_events),
            'persistent_frames': sum(row['persistent_frames'] for row in rows),
            'persistent_share_of_iou_lt_05_frames':
                sum(row['persistent_frames'] for row in rows) /
                sum(item['M1']['iou_lt_0.5_frames'] for item in source['per_sequence']),
            'affected_sequence_median_first_persistent_fraction': median(
                [row['first_persistent_fraction'] for row in affected]),
            'all_event_median_start_frame': median(
                [event['start_frame_1based'] for event in all_events]),
            'persistent_to_end_sequences': sum(row['persistent_to_end'] for row in rows),
            'total_positive_timing_frames': timing_frames,
            'timing_derived_fps': timing_frames / timing_seconds,
        },
        'lowest_auc_sequences': sorted(rows, key=lambda row: row['m1_auc'])[:12],
        'highest_persistent_burden': sorted(
            rows, key=lambda row: row['persistent_frames'], reverse=True)[:12],
        'largest_m1_regressions_vs_b0': sorted(
            rows, key=lambda row: row['m1_minus_b0_auc'])[:12],
        'attribute_summary': attributes,
        'limitations': [
            'All attributes overlap and represent sequence-level labels; association is not causation.',
            'The full-244 test set has been inspected before and must not select a module or insertion point.',
            'Only fused M1 predictions are archived; no M1 RGB-only or TIR-only prediction series exist.',
            'Archived full-test predictions lack framewise crop transforms and internal fusion states.',
            'Historical B0 runs differ in training contract; M1-B0 sequence differences are descriptive.',
            'Box-error summaries are not the official success-curve AUC.',
        ],
    }
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({
        'official_auc': auc,
        'descriptive': summary['descriptive'],
        'lowest_attributes': sorted(attributes, key=lambda item: item['m1_mean_auc'])[:8],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
