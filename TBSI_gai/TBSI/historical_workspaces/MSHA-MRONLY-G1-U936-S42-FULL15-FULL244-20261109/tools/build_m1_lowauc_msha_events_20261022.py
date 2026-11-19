#!/usr/bin/env python3
"""Build fixed event rows at first sustained low-IoU onset for low-AUC M1 sequences."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ROOT / 'experiments/manifests/m1_lowauc20_msha_20261022.txt'
TRACE = ROOT / 'output/experiments/M1_FUSION_FLOW_FULL244_DIAG_20261009/closed_loop_trace/trace'
OUT = ROOT / 'output/experiments/M1_LOWAUC20_MSHA_CAUSAL_20261022/manifest'


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


def crop_contains(box, crop):
    return box[0] >= crop[0] and box[1] >= crop[1] and box[0] + box[2] <= crop[0] + crop[2] and box[1] + box[3] <= crop[1] + crop[3]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in SEQUENCES.read_text().splitlines() if x.strip()]
    events = []
    for name in names:
        rows = [json.loads(x) for x in (TRACE / name / 'frames.jsonl').read_text().splitlines() if x.strip()]
        values = [(r, iou(r['pred_box'], r['gt_box'])) for r in rows if r['frame_id'] > 1 and r['gt_valid']]
        selected = None
        for idx, (row, value) in enumerate(values):
            window = values[idx:idx + 10]
            if value < .5 and len(window) == 10 and all(v < .5 for _, v in window) and crop_contains(row['gt_box'], row['current_crop_xywh']):
                selected = row
                break
        if selected is None:
            selected = min(values, key=lambda item: item[1])[0]
        events.append({'event_id': 'L{:03d}'.format(len(events) + 1), 'seq': name,
                       'frame': int(selected['frame_id']), 'group': 'lowauc',
                       'failure_label': 'first_sustained_iou_lt_0.5',
                       'trace_iou': iou(selected['pred_box'], selected['gt_box']),
                       'crop_xywh': json.dumps(selected['current_crop_xywh']),
                       'state_before': json.dumps(selected['tracker_state_before'])})
    fields = list(events[0])
    for filename, records in (('regression_events.csv', events), ('gain_events.csv', [])):
        with (OUT / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(records)
    (OUT / 'event_manifest.json').write_text(json.dumps({'selection': 'lowest official M1 AUC 20, first 10-frame sustained IoU<0.5 crop-contained onset; sequence-minimum fallback', 'events': events}, indent=2) + '\n')
    print('events', len(events), 'max_frame', max(row['frame'] for row in events))


if __name__ == '__main__':
    main()
