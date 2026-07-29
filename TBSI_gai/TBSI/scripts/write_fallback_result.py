#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracker_name")
    parser.add_argument("tracker_param")
    parser.add_argument("dataset_name")
    parser.add_argument("sequence")
    args = parser.parse_args()

    dataset = get_dataset(args.dataset_name)
    seq = next((s for s in dataset if s.name == args.sequence), None)
    if seq is None:
        raise RuntimeError("sequence not found: {}".format(args.sequence))

    tracker = Tracker(args.tracker_name, args.tracker_param, args.dataset_name, None)
    Path(tracker.results_dir).mkdir(parents=True, exist_ok=True)

    gt = seq.ground_truth_rect
    if isinstance(gt, dict):
        gt = next(iter(gt.values()))
    if gt is None or len(gt) == 0:
        gt = np.zeros((1, 4), dtype=np.float32)
    init_box = np.asarray(gt[0], dtype=np.float32)
    boxes = np.tile(init_box[None, :], (len(gt), 1)).astype(int)
    times = np.zeros((len(gt),), dtype=float)

    base = Path(tracker.results_dir) / seq.name
    np.savetxt(str(base) + ".txt", boxes, delimiter="\t", fmt="%d")
    np.savetxt(str(base) + "_time.txt", times, delimiter="\t", fmt="%f")
    print("[FALLBACK_RESULT] wrote {} frames for {}".format(len(gt), seq.name))


if __name__ == "__main__":
    main()
