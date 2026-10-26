#!/usr/bin/env python3
"""Build blind-review panels from raw RGB/TIR frames and reviewed candidate ROIs."""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset


def parse_box(value):
    box = [float(x) for x in json.loads(value)]
    if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
        raise ValueError(f"invalid box: {value}")
    return box


def panel(image_path, box, color):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    x, y, w, h = map(int, np.rint(box))
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--sequence-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"Output directory is non-empty; refuse overwrite: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    sequences = {sequence.name: sequence for sequence in
                 LasHeRDataset(split="trainingset", sequence_list_file=str(args.sequence_list)).get_sequence_list()}
    with args.queue.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        sequence = sequences.get(row["seq"])
        if sequence is None:
            raise KeyError(f"sequence not found in frozen list: {row['seq']}")
        index = int(row["frame"])
        if index < 0 or index >= len(sequence.frames[0]):
            raise IndexError(f"frame out of range: {row['event_id']}")
        rgb = panel(sequence.frames[0][index], parse_box(row["rgb_box"]), (255, 255, 255))
        tir = panel(sequence.frames[1][index], parse_box(row["tir_box"]), (255, 255, 255))
        height = max(rgb.shape[0], tir.shape[0])
        if rgb.shape[0] != height:
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * height / rgb.shape[0]), height))
        if tir.shape[0] != height:
            tir = cv2.resize(tir, (int(tir.shape[1] * height / tir.shape[0]), height))
        sheet = np.concatenate([rgb, tir], axis=1)
        output = args.out / f"{row['event_id']}.jpg"
        if output.exists():
            raise SystemExit(f"refuse overwrite: {output}")
        if not cv2.imwrite(str(output), sheet):
            raise RuntimeError(f"failed to write {output}")
    print(f"built {len(rows)} clean review images in {args.out}; only candidate ROI outlines are shown")


if __name__ == "__main__":
    main()
