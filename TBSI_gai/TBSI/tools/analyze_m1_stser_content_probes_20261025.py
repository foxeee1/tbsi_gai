"""Compare fixed-checkpoint inference interventions with the candidate run."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation import get_dataset
from analyze_m1_stser_content_pair_20261025 import bootstrap, read_sequence, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('quick3', 'full15'), required=True)
    scope = parser.parse_args().scope
    campaign = ROOT / 'experiments/snapshots/m1_stser_content_campaign_{}_20261025'.format(scope)
    lock = json.loads((campaign / 'contract.json').read_text())
    candidate = 'M1_STSER_CONTENT_{}_CANDIDATE_20261025'.format(scope.upper())
    if scope == 'quick3':
        manifest = Path(lock['test_manifest'])
        if sha(manifest) != lock['quick_test_manifest_sha256']:
            raise ValueError('test manifest changed')
        os.environ['TBSI_RPP_TEST_LIST'] = str(manifest)
    dataset = get_dataset('rpp_lasher_test' if scope == 'quick3' else 'lasher_test')
    modes = ('off', 'roll_mask', 'roll_competition', 'scale_only')
    for mode in modes:
        phase = 'probe-' + mode.replace('_', '-')
        status = json.loads((campaign / (phase + '.json')).read_text())
        if status['status'] != 'complete':
            raise RuntimeError(phase + ' incomplete')
        if status['checkpoint_sha256_before'] != status['checkpoint_sha256_after']:
            raise RuntimeError(phase + ' changed checkpoint')
    root = ROOT / 'output/test/tracking_results/tbsi_track'
    baseline_dir = root / candidate
    rows = []
    for sequence in dataset:
        baseline, _, _, _, _, _ = read_sequence(sequence, baseline_dir)
        for mode in modes:
            probe_dir = root / (candidate + '_content_' + mode)
            probe, _, _, _, _, _ = read_sequence(sequence, probe_dir)
            row = {'sequence': sequence.name, 'probe': mode,
                   'frames': baseline['frames']}
            for key in ('auc', 'precision', 'norm_precision', 'op50', 'op75'):
                row['candidate_' + key] = baseline[key]
                row['probe_' + key] = probe[key]
                row['probe_minus_candidate_' + key] = probe[key] - baseline[key]
            rows.append(row)
    summary = {'scope': scope, 'comparison': 'same candidate checkpoint, inference-only intervention',
               'claim_limit': 'post-training sensitivity, not a retraining ablation',
               'sequence_count': len(dataset), 'probes': {}}
    for mode in modes:
        selected = [row for row in rows if row['probe'] == mode]
        summary['probes'][mode] = {}
        for key in ('auc', 'precision', 'norm_precision', 'op50', 'op75'):
            delta = np.array([row['probe_minus_candidate_' + key] for row in selected])
            summary['probes'][mode][key] = {
                'mean': float(delta.mean()), 'ci95_sequence_bootstrap': bootstrap(delta),
                'wins': int((delta > 1e-12).sum()), 'losses': int((delta < -1e-12).sum())}
    output = campaign / 'probe_analysis'
    if output.exists():
        raise FileExistsError('probe analysis already exists')
    output.mkdir()
    with (output / 'per_sequence.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
