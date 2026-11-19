#!/usr/bin/env python3
"""Render GT/pred overlays for manual review of selected train-side events."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'data/lasher/trainingset'


def draw_box(image, box, color, thickness=2):
    x, y, w, h = [int(round(float(value))) for value in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-arm', required=True)
    parser.add_argument('--trace-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing nonempty review output')
    output.mkdir(parents=True)
    with open(args.per_arm, newline='') as stream:
        events = [row for row in csv.DictReader(stream)
                  if row['arm'] == 'baseline' and row['label'] == 'severe_onset']
    index_rows = []
    for row in events:
        name, frame = row['sequence'], int(row['frame'])
        trace_path = Path(args.trace_dir) / name / 'frames.jsonl'
        with trace_path.open() as stream:
            item = next((json.loads(line) for line in stream
                         if int(json.loads(line)['frame_id']) == frame), None)
        if item is None:
            raise RuntimeError('trace frame missing: {} {}'.format(name, frame))
        panels = []
        for modality in ('visible', 'infrared'):
            images = sorted((DATA / name / modality).glob('*.jpg'),
                            key=lambda path: int(path.stem[1:]))
            if frame >= len(images):
                raise RuntimeError('frame outside image list: {} {} {}'.format(
                    name, frame, modality))
            image = cv2.imread(str(images[frame]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError('image missing: {} {} {}'.format(name, frame, modality))
            draw_box(image, item['gt_box'], (0, 220, 0))
            draw_box(image, item['pred_box'], (0, 0, 255))
            scale = min(640 / image.shape[1], 400 / image.shape[0])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.putText(image, '{} {} GT=green pred=red'.format(name, modality), (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(image)
        height = max(panel.shape[0] for panel in panels)
        width = sum(panel.shape[1] for panel in panels)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        x = 0
        for panel in panels:
            canvas[:panel.shape[0], x:x + panel.shape[1]] = panel
            x += panel.shape[1]
        filename = '{}_frame_{:05d}.jpg'.format(name, frame)
        cv2.imwrite(str(output / filename), canvas)
        index_rows.append({'sequence': name, 'frame': frame,
                           'file': filename, 'trace_iou': row['trace_iou']})
    with (output / 'index.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    print('rendered {} event pairs'.format(len(index_rows)))


if __name__ == '__main__':
    main()
