#!/usr/bin/env python
"""Offline rendering of the three pre-registered SJI extreme events."""
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from synthetic_joint_interference_v1 import image_loader, load_train_sequence, patch_image
from lib.test.evaluation.environment import env_settings

EVENTS = ('SJI-00005', 'SJI-00014', 'SJI-00030')


def rect(canvas, box, color, label, scale=1.0):
    x, y, w, h = [int(round(float(v) * scale)) for v in box]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, max(1, int(round(2 * scale))))
    cv2.putText(canvas, label, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, 0.5 * scale), color, max(1, int(round(scale))))


def resize(image, width=480):
    scale = width / float(image.shape[1])
    return cv2.resize(image, (width, int(round(image.shape[0] * scale)))), scale


def main():
    out_dir = os.path.join(ROOT, 'output/experiments/synthetic_joint_interference_v1/batch/extreme_event_inspection')
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(ROOT, 'output/experiments/synthetic_joint_interference_v1/batch')
    selected = {x['event_id']: x for x in json.load(open(os.path.join(base, 'selected_events.json')))}
    rows = list(csv.DictReader(open(os.path.join(base, 'event_results.csv'))))
    by_event = {}
    for row in rows:
        if row['event_id'] in EVENTS:
            by_event.setdefault(row['event_id'], {})[row['condition']] = [float(row[k]) for k in ('pred_x','pred_y','pred_w','pred_h')]
    settings = env_settings()
    for event_id in EVENTS:
        event = selected[event_id]
        seq = load_train_sequence(settings.lasher_path, event['seq'])
        frame = int(event['frame'])
        rgb = image_loader(seq.frames[0][frame])
        tir = image_loader(seq.frames[1][frame])
        gt = event['gt']; dest_rgb = event['dest_rgb']; dest_tir = event['dest_tir']
        rgb_paste, _ = patch_image(rgb, gt, dest_rgb)
        tir_paste, _ = patch_image(tir, gt, dest_tir)
        panels = []
        for image, title, dest, preds in ((rgb, 'RGB original', dest_rgb, False),
                                           (rgb_paste, 'RGB 11_same', dest_rgb, True),
                                           (tir, 'TIR original', dest_tir, False),
                                           (tir_paste, 'TIR 11_same', dest_tir, True)):
            panel, scale = resize(image)
            if title.startswith('TIR'):
                panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            else:
                panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            rect(panel, gt, (0, 255, 0), 'GT', scale)
            rect(panel, dest, (255, 0, 255), 'paste', scale)
            if preds:
                colors = {'00': (255, 255, 0), '10': (0, 165, 255), '01': (255, 128, 0), '11': (0, 0, 255)}
                for condition in ('00', '10', '01', '11'):
                    rect(panel, by_event[event_id][condition], colors[condition], condition, scale)
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            panels.append(panel)
        height = max(x.shape[0] for x in panels)
        panels = [cv2.copyMakeBorder(x, 0, height - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in panels]
        sheet = np.hstack(panels)
        cv2.imwrite(os.path.join(out_dir, event_id + '_audit.jpg'), sheet)
        print(event_id, event['seq'], event['frame'])


if __name__ == '__main__':
    main()
