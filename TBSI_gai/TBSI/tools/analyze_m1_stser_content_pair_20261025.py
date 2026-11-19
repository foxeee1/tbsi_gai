"""Offline paired evaluation of saved STSER content predictions."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation import get_dataset
from lib.test.utils.load_text import load_text


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(values, repeats=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(42)
    sampled = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))]


def failure_episodes(iou, valid, window=3, threshold=.2, recovery=.5):
    bad = (iou < threshold) & valid
    episodes = []
    index = 1
    while index < len(bad):
        if not bad[index]:
            index += 1
            continue
        start = index
        while index < len(bad) and bad[index]:
            index += 1
        if index - start >= window:
            tail = np.flatnonzero(valid[index:] & (iou[index:] >= recovery))
            episodes.append({'start': start, 'end_exclusive': index,
                             'length': index - start,
                             'recovered_after_frame': int(index + tail[0]) if len(tail) else None})
    return episodes


def read_sequence(sequence, directory):
    path = directory / (sequence.name + '.txt')
    timing = directory / (sequence.name + '_time.txt')
    if not path.is_file() or not timing.is_file():
        raise FileNotFoundError('prediction/time missing: ' + str(path))
    boxes = np.asarray(load_text(str(path), delimiter=('\t', ','), dtype=np.float64)).reshape(-1, 4)
    gt = np.asarray(sequence.ground_truth_rect, dtype=np.float64).reshape(-1, 4)
    if len(boxes) != len(gt):
        raise ValueError('{} has {} boxes for {} ground-truth frames'.format(
            sequence.name, len(boxes), len(gt)))
    times = np.asarray(load_text(str(timing), delimiter=('\t', ','), dtype=np.float64)).reshape(-1)
    if len(times) != len(gt) or not np.isfinite(times).all() or (times <= 0).any():
        raise ValueError('timing count or values invalid: ' + str(timing))
    visible = (torch.as_tensor(sequence.target_visible, dtype=torch.uint8)
               if sequence.target_visible is not None else None)
    overlap, center, norm, valid = calc_seq_err_robust(
        torch.as_tensor(boxes, dtype=torch.float64), torch.as_tensor(gt, dtype=torch.float64),
        sequence.dataset, visible)
    overlap, center, norm, valid = [x.cpu().numpy() for x in (overlap, center, norm, valid)]
    thresholds = np.arange(21, dtype=np.float64) * .05
    success = (overlap[:, None] > thresholds[None, :]).mean(axis=0) * 100
    episodes = failure_episodes(overlap, valid)
    metrics = {'auc': float(success.mean()), 'op50': float(success[10]),
               'op75': float(success[15]),
               'precision': float((center <= 20).mean() * 100),
               'norm_precision': float((norm <= .2).mean() * 100),
               'fps': float(len(times) / times.sum()), 'frames': len(gt),
               'first_failure_frame': episodes[0]['start'] if episodes else None,
               'failure_episode_count': len(episodes),
               'failure_episode_frames': sum(item['length'] for item in episodes),
               'recovered_episode_count': sum(item['recovered_after_frame'] is not None
                                              for item in episodes),
               'bad_iou_fraction': float(((overlap < .2) & valid).mean()),
               'mean_valid_iou': float(overlap[valid].mean()) if valid.any() else None}
    return metrics, overlap, center, boxes, gt, float(times.sum())


def diagnostic_summary(run):
    logs = list(run.rglob('stser_content_flow.csv'))
    grad_logs = list(run.rglob('stser_content_preclip_grad.csv'))
    result = {'flow_files': [str(p) for p in logs], 'grad_files': [str(p) for p in grad_logs]}
    for label, paths, fields in (
            ('flow', logs, ('lambda', 'source_shift_l1', 'foreground_mass_base',
                            'foreground_mass_changed', 'valid_reference_fraction',
                            'content_message_norm', 'content_message_cosine_base')),
            ('preclip', grad_logs, ('global_grad_norm', 'content_lambda_grad_norm',
                                     'content_lambda_has_grad'))):
        rows = []
        for path in paths:
            with path.open(newline='') as handle:
                rows.extend(csv.DictReader(handle))
        result[label + '_row_count'] = len(rows)
        result[label + '_mean'] = {
            field: float(np.mean([float(row[field]) for row in rows if row.get(field) not in ('', None)]))
            for field in fields if any(row.get(field) not in ('', None) for row in rows)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('quick3', 'full15'), required=True)
    args = parser.parse_args()
    scope = args.scope
    campaign = ROOT / 'experiments/snapshots/m1_stser_content_campaign_{}_20261025'.format(scope)
    lock = json.loads((campaign / 'contract.json').read_text())
    names = {arm: 'M1_STSER_CONTENT_{}_{}_20261025'.format(scope.upper(), arm.upper())
             for arm in ('control', 'candidate')}
    for arm in names:
        resume = campaign / ('resume-test-' + arm + '.json')
        status = json.loads((resume if resume.is_file() else
                             campaign / ('test-' + arm + '.json')).read_text())
        if status['status'] != 'complete':
            raise RuntimeError(arm + ' test phase incomplete')
    if scope == 'quick3':
        manifest = Path(lock['test_manifest'])
        if sha(manifest) != lock['quick_test_manifest_sha256']:
            raise ValueError('quick test manifest changed')
        os.environ['TBSI_RPP_TEST_LIST'] = str(manifest)
    dataset = get_dataset('rpp_lasher_test' if scope == 'quick3' else 'lasher_test')
    if scope == 'quick3':
        expected = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
        if [s.name for s in dataset] != expected:
            raise ValueError('quick dataset order differs from locked manifest')
    else:
        expected = [line.strip() for line in Path(
            lock['full_list_sources']['evaluation']['resolved_path']).read_text().splitlines()
            if line.strip()]
        if [s.name for s in dataset] != expected:
            raise ValueError('full evaluation members differ from frozen list')
    directories = {arm: ROOT / 'output/test/tracking_results/tbsi_track' / name
                   for arm, name in names.items()}
    runs = {arm: ROOT / 'output/experiments' / name for arm, name in names.items()}
    epoch = 3 if scope == 'quick3' else 15
    keys = ('auc', 'precision', 'norm_precision', 'op50', 'op75',
            'bad_iou_fraction', 'mean_valid_iou', 'fps',
            'failure_episode_count', 'failure_episode_frames', 'recovered_episode_count')
    rows, failure_rows, episode_rows = [], [], []
    frame_count = {'control': 0, 'candidate': 0}
    seconds = {'control': 0.0, 'candidate': 0.0}
    for sequence in dataset:
        arm_data = {}
        for arm in names:
            metrics, iou, center, boxes, gt, timing = read_sequence(sequence, directories[arm])
            arm_data[arm] = (metrics, iou, center, boxes, gt)
            frame_count[arm] += metrics['frames']
            seconds[arm] += timing
        control, candidate = arm_data['control'][0], arm_data['candidate'][0]
        row = {'sequence': sequence.name, 'frames': control['frames'],
               'control_first_failure_frame': control['first_failure_frame'],
               'candidate_first_failure_frame': candidate['first_failure_frame']}
        for key in keys:
            a, b = control[key], candidate[key]
            row['control_' + key] = a
            row['candidate_' + key] = b
            row['delta_' + key] = b - a if a is not None and b is not None else None
        rows.append(row)
        for arm in names:
            metrics, iou, center, boxes, gt = arm_data[arm]
            valid = np.isfinite(center)
            for event in failure_episodes(iou, valid):
                episode_rows.append({'sequence': sequence.name, 'arm': arm,
                                     'start': event['start'],
                                     'end_exclusive': event['end_exclusive'],
                                     'length': event['length'],
                                     'recovered_after_frame': event['recovered_after_frame']})
            onset = metrics['first_failure_frame']
            if onset is None:
                continue
            for frame in range(max(1, onset - 2), min(len(gt), onset + 6)):
                failure_rows.append({'sequence': sequence.name, 'arm': arm,
                                     'onset': onset, 'frame': frame,
                                     'iou': float(iou[frame]),
                                     'center_error_px': float(center[frame]),
                                     'pred_xywh': json.dumps(boxes[frame].tolist()),
                                     'gt_xywh': json.dumps(gt[frame].tolist())})
    if not rows:
        raise ValueError('no evaluated sequences')
    result = {'scope': scope, 'classification': 'proxy screen' if scope == 'quick3' else 'full LasHeR test',
              'comparison': 'candidate minus matched structural control',
              'not_a_multiseed_claim': True, 'sequence_count': len(rows),
              'checkpoint_sha256': {arm: sha(runs[arm] / 'checkpoints/TBSITrack_ep{:04d}.pth.tar'.format(epoch))
                                    for arm in names},
              'config_sha256': lock['config_sha256'],
              'shared_initial_sha256': lock['shared_initial_sha256'],
              'checkpoint_progress': {
                  arm: json.loads((campaign / ('train-' + arm + '_checkpoint.json')).read_text())
                  for arm in names},
              'test_manifest_sha256': sha(manifest) if scope == 'quick3' else None,
              'fps_total': {arm: frame_count[arm] / seconds[arm] for arm in names},
              'failure_episode_totals': {
                  arm: {'count': sum(row['arm'] == arm for row in episode_rows),
                        'frames': sum(row['length'] for row in episode_rows if row['arm'] == arm),
                        'recovered': sum(row['arm'] == arm and
                                         row['recovered_after_frame'] is not None
                                         for row in episode_rows)} for arm in names},
              'macro_sequence_mean': {}, 'paired_delta': {},
              'diagnostics': {arm: diagnostic_summary(runs[arm]) for arm in names},
              'm1_reference_aggregate_only': {'auc': 56.21, 'precision': 70.21,
                                                'norm_precision': 66.20,
                                                'op50': 68.33, 'op75': 47.13}
              if scope == 'full15' else None}
    for key in keys:
        matched = [row for row in rows if row['delta_' + key] is not None]
        delta = np.array([row['delta_' + key] for row in matched])
        result['macro_sequence_mean'][key] = {
            arm: float(np.mean([row[arm + '_' + key] for row in matched])) for arm in names}
        result['paired_delta'][key] = {'mean': float(delta.mean()),
                                       'median': float(np.median(delta)),
                                       'ci95_sequence_bootstrap': bootstrap(delta),
                                       'wins': int((delta > 1e-12).sum()),
                                       'losses': int((delta < -1e-12).sum()),
                                       'ties': int((np.abs(delta) <= 1e-12).sum()),
                                       'at_least_2_points_worse': int((delta <= -2).sum())
                                       if key in ('auc', 'precision', 'norm_precision', 'op50', 'op75') else None}
    out = campaign / 'analysis'
    if out.exists():
        raise FileExistsError('analysis already exists: ' + str(out))
    out.mkdir()
    with (out / 'per_sequence.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if failure_rows:
        with (out / 'failure_onset_windows.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(failure_rows[0]))
            writer.writeheader()
            writer.writerows(failure_rows)
    if episode_rows:
        with (out / 'failure_episodes.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
            writer.writeheader()
            writer.writerows(episode_rows)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
