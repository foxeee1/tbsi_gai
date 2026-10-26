#!/usr/bin/env python3
"""Run NLformer candidate on the frozen prior 50-sequence E2 cohort."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--sequence-list', type=Path,
                    default=ROOT / 'experiments/proxy/lasher_e2_50seq_20261017.txt')
    ap.add_argument('--out-dir', type=Path,
                    default=ROOT / 'output/test/tracking_results/tbsi_track/M1_AMCM_NLFORMER_QUICK15_3EP_20261020_50SEQ')
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    # Keep the tracker in FP32 for the numerical audit and avoid AMP changing
    # the published Woodbury branch semantics.
    os.environ['TBSI_INFER_FP16'] = '0'
    names = [x.strip() for x in args.sequence_list.read_text().splitlines() if x.strip()]
    all_sequences = {s.name: s for s in get_dataset('lasher_test')}
    missing = [n for n in names if n not in all_sequences]
    if missing:
        raise RuntimeError('sequences missing from lasher_test: ' + ','.join(missing))
    dataset = [all_sequences[n] for n in names]
    tracker = Tracker('tbsi_track', args.config, 'lasher_test')
    tracker.results_dir = str(args.out_dir)
    print('Running {} frozen sequences serially'.format(len(dataset)), flush=True)
    run_dataset(dataset, [tracker], debug=0, threads=args.threads, num_gpus=1)


if __name__ == '__main__':
    main()
