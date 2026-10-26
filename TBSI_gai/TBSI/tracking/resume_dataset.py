"""Resume a dataset evaluation from a fixed sequence index."""
import argparse
import os
import sys

PRJ_PATH = os.path.join(os.path.dirname(__file__), '..')
if PRJ_PATH not in sys.path:
    sys.path.append(PRJ_PATH)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-index', type=int, required=True)
    parser.add_argument('--tracker-name', default='tbsi_track')
    parser.add_argument('--tracker-param', required=True)
    parser.add_argument('--dataset-name', default='lasher_test')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--num-gpus', type=int, default=1)
    args = parser.parse_args()

    dataset = get_dataset(args.dataset_name)[args.start_index:]
    tracker = Tracker(args.tracker_name, args.tracker_param, args.dataset_name, None)
    print('Resuming from index {}: {} sequences'.format(args.start_index, len(dataset)), flush=True)
    run_dataset(dataset, [tracker], debug=False, threads=args.threads, num_gpus=args.num_gpus)


if __name__ == '__main__':
    main()
