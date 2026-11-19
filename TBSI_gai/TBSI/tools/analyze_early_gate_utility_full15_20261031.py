#!/usr/bin/env python3
"""Paired full244 reporting for the Early gate source-utility loss experiment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lib.test.analysis.extract_results import calc_seq_err_robust

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/early_gate_closed_loop_20261026'
METRICS = ('auc', 'precision', 'norm_precision', 'op50', 'op75')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def score(prediction, ground_truth):
    if prediction.shape != ground_truth.shape or not np.isfinite(prediction).all():
        raise ValueError('Prediction shape or finite-value check failed')
    iou, center, norm, valid = calc_seq_err_robust(
        torch.tensor(prediction, dtype=torch.float64),
        torch.tensor(ground_truth, dtype=torch.float64), 'lasher')
    iou, center, norm, valid = iou.numpy(), center.numpy(), norm.numpy(), valid.numpy()
    thresholds = np.arange(0., 1.00001, .05)
    success = np.array([(iou > threshold).mean() for threshold in thresholds])
    result = {'auc': float(100 * success.mean()),
              'precision': float(100 * (center <= 20).mean()),
              'norm_precision': float(100 * (norm <= .20).mean()),
              'op50': float(100 * (iou > .5).mean()),
              'op75': float(100 * (iou > .75).mean()),
              'failure_fraction': float((iou < .1).mean())}
    return result, iou


def load(folder, name, frames):
    pred = np.loadtxt(str(Path(folder) / (name + '.txt')), ndmin=2)
    gt = np.loadtxt(str(ROOT / 'data/lasher/testingset' / name / 'init.txt'),
                    delimiter=',', ndmin=2)
    if len(gt) != frames or pred.shape != gt.shape:
        raise ValueError('Frame count mismatch for ' + name)
    return score(pred, gt)


def summarize(rows):
    values = {metric: np.array([row[metric + '_delta'] for row in rows])
              for metric in METRICS}
    rng = np.random.RandomState(20261031)
    samples = rng.randint(0, len(rows), size=(20000, len(rows)))
    auc_bootstrap = values['auc'][samples].mean(axis=1)
    auc = values['auc']
    positives = auc[auc > 0]
    return {
        'sequences': len(rows), 'frames': int(sum(row['frames'] for row in rows)),
        'metrics': {metric: {'control_mean': float(np.mean([r[metric + '_control'] for r in rows])),
                             'utility_mean': float(np.mean([r[metric + '_utility'] for r in rows])),
                             'delta_mean': float(values[metric].mean()),
                             'delta_median': float(np.median(values[metric]))}
                    for metric in METRICS},
        'auc_delta_90pct_sequence_bootstrap': np.quantile(auc_bootstrap, [.05, .95]).tolist(),
        'auc_delta_95pct_sequence_bootstrap': np.quantile(auc_bootstrap, [.025, .975]).tolist(),
        'auc_wins_ties_losses': {'wins': int((auc > 1e-6).sum()),
                                 'ties': int((np.abs(auc) <= 1e-6).sum()),
                                 'losses': int((auc < -1e-6).sum())},
        'auc_losses_over_5pp': int((auc < -5).sum()),
        'auc_losses_over_10pp': int((auc < -10).sum()),
        'worst_decile_auc_delta_mean': float(np.sort(auc)[:max(1, int(np.ceil(.1 * len(auc))))].mean()),
        'mean_auc_delta_without_top3_gains': float(np.sort(auc)[:-3].mean()),
        'top3_share_positive_auc_gain': float(np.sort(positives)[-3:].sum() / positives.sum()) if len(positives) else None,
        'sequence_bootstrap_scope': 'sequence uncertainty for these fixed checkpoints; does not estimate seed uncertainty',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', required=True)
    parser.add_argument('--utility', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    control, utility, output = map(Path, (args.control, args.utility, args.output))
    manifest = json.loads((BASE / 'p1_sot_v3/data_manifest.json').read_text())
    names = [row['sequence'] for row in manifest['test_sequences']]
    if len(names) != 244 or len(set(names)) != 244:
        raise ValueError('Frozen test manifest is not exactly 244 unique sequences')
    rows = []
    for item in manifest['test_sequences']:
        name, frames = item['sequence'], int(item['frames'])
        c, ciou = load(control, name, frames)
        u, uiou = load(utility, name, frames)
        row = {'sequence': name, 'frames': frames}
        for metric in METRICS:
            row[metric + '_control'] = c[metric]
            row[metric + '_utility'] = u[metric]
            row[metric + '_delta'] = u[metric] - c[metric]
        row['failure_fraction_control'] = c['failure_fraction']
        row['failure_fraction_utility'] = u['failure_fraction']
        row['failure_fraction_delta'] = u['failure_fraction'] - c['failure_fraction']
        row['recovered_failure_frames'] = int(((ciou < .1) & (uiou >= .5)).sum())
        row['new_failure_frames'] = int(((ciou >= .5) & (uiou < .1)).sum())
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per_sequence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary.update({'protocol': 'Full15 public-SOT single-stage pair; fixed LasHeR test244; FP32; tracker threads0; template update off',
                    'control_identity_sha256': sha(control / 'identity.json'),
                    'utility_identity_sha256': sha(utility / 'identity.json'),
                    'manifest_sha256': sha(BASE / 'p1_sot_v3/data_manifest.json'),
                    'test_sequence_list_sha256': sha(BASE / 'p1_sot_v3/test_sequences.txt'),
                    'scope_is_full244': True, 'single_seed': True})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    lines = ['# Early gate utility loss: full244 paired results', '',
             '| Metric | P control | P + utility rank | Delta (pp) |', '|---|---:|---:|---:|']
    for metric, label in zip(METRICS, ('AUC', 'Precision', 'Norm Precision', 'OP50', 'OP75')):
        values = summary['metrics'][metric]
        lines.append('| %s | %.4f | %.4f | %+.4f |' %
                     (label, values['control_mean'], values['utility_mean'], values['delta_mean']))
    lines += ['', '244 sequences, %d frames. AUC delta 90%% sequence bootstrap: [%.4f, %.4f] pp.' %
              (summary['frames'], *summary['auc_delta_90pct_sequence_bootstrap']),
              'Single-seed sequence resampling does not estimate training-seed variance.',
              'The local utility derivative is a training-time proxy; full244 is closed-loop task evidence, not proof that the proxy is a valid long-term source label.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
