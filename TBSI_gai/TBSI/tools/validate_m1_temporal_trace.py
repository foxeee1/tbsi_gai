#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
import hashlib
import numpy as np


def finite_box(value):
    return isinstance(value, list) and len(value) == 4 and all(math.isfinite(float(x)) for x in value)


def validate(root):
    errors = []
    summary = {'sequences': 0, 'rows': 0, 'history_rows': 0, 'frame_gaps': [],
               'cross_sequence_errors': 0, 'causal_errors': 0, 'crop_errors': 0,
               'duplicate_frames': 0, 'raw_shape_errors': 0,
               'raw_file_errors': 0, 'raw_checksum_errors': 0}
    for seq in sorted(os.listdir(root)):
        path = os.path.join(root, seq, 'frames.jsonl')
        if not os.path.isfile(path):
            continue
        rows = []
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                try:
                    row = json.loads(line)
                    rows.append(row)
                except Exception as exc:
                    errors.append('%s:%d invalid json: %s' % (path, line_no, exc))
        summary['sequences'] += 1
        summary['rows'] += len(rows)
        frames = [int(r.get('frame_id', -1)) for r in rows]
        if frames != sorted(frames):
            errors.append('%s frames not sorted' % path)
        if len(frames) != len(set(frames)):
            summary['duplicate_frames'] += 1
            errors.append('%s duplicate frame ids' % path)
        for i, row in enumerate(rows):
            if row.get('sequence_id') != seq:
                summary['cross_sequence_errors'] += 1
            cur = row.get('current_crop_xywh')
            if not finite_box(cur) or float(cur[2]) <= 0 or float(cur[3]) <= 0:
                summary['crop_errors'] += 1
            if row.get('raw_head_shape') != [1, 256, 768]:
                summary['raw_shape_errors'] += 1
            raw_file = row.get('raw_head_file')
            if raw_file:
                candidate = os.path.join(root, seq, raw_file)
                try:
                    raw = np.load(candidate, allow_pickle=False)
                    if list(raw.shape) != [1, 256, 768] or not np.isfinite(raw).all():
                        summary['raw_file_errors'] += 1
                    digest = hashlib.sha256(np.ascontiguousarray(raw, dtype=np.float32).tobytes()).hexdigest()
                    if digest != row.get('raw_head_sha256'):
                        summary['raw_checksum_errors'] += 1
                except Exception:
                    summary['raw_file_errors'] += 1
            has = bool(row.get('has_history'))
            if i == 0:
                if has or row.get('history_frame_id') is not None or row.get('history_crop_xywh') is not None:
                    errors.append('%s first row has history' % path)
                continue
            summary['history_rows'] += 1
            prev = rows[i - 1]
            hid = row.get('history_frame_id')
            if not has or hid != prev.get('frame_id'):
                summary['causal_errors'] += 1
            if row.get('history_sequence_id') != seq:
                summary['cross_sequence_errors'] += 1
            if hid is None or int(hid) >= int(row['frame_id']):
                summary['causal_errors'] += 1
            gap = int(row['frame_id']) - int(prev['frame_id'])
            if gap != 1:
                summary['frame_gaps'].append(gap)
            if row.get('history_crop_xywh') != prev.get('current_crop_xywh'):
                summary['crop_errors'] += 1
            if row.get('history_crop_transform') != prev.get('current_crop_transform'):
                summary['crop_errors'] += 1
        if not rows:
            errors.append('%s empty' % path)
    summary['pass'] = (not errors and summary['sequences'] > 0 and summary['rows'] > 0
                       and summary['raw_file_errors'] == 0
                       and summary['raw_checksum_errors'] == 0)
    return summary, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace-dir', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    summary, errors = validate(args.trace_dir)
    result = {'summary': summary, 'errors': errors,
              'decision': 'TRACE_READY_FOR_G0' if summary['pass'] else 'STOP_TRACE_INVALID'}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f: json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if summary['pass'] else 2


if __name__ == '__main__':
    sys.exit(main())
