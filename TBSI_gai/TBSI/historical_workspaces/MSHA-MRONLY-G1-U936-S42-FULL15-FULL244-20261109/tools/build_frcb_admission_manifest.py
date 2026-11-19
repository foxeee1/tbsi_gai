#!/usr/bin/env python3
"""Build deterministic diagnostic events from the frozen RPP-v1 train list."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequence-list', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--events-per-sequence', type=int, choices=(2, 3), default=2)
    args = ap.parse_args()
    source = Path(args.sequence_list).resolve()
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing non-empty output {}'.format(out))
    names = [x.strip() for x in source.read_text().splitlines() if x.strip()]
    if not names:
        raise RuntimeError('empty sequence list')
    dataset = LasHeRDataset(split='trainingset', sequence_list_file=str(source)).get_sequence_list()
    rows = []
    for seq in dataset:
        count = min(len(seq.frames[0]), len(seq.frames[1]), len(seq.ground_truth_rect))
        if count < 8:
            raise RuntimeError('sequence too short for future horizon: {} ({})'.format(seq.name, count))
        candidates = [1, max(1, count // 2)]
        if args.events_per_sequence == 3:
            candidates.append(max(1, (2 * count) // 3))
        used = []
        for frame in candidates:
            frame = min(frame, count - 6)
            if frame not in used:
                used.append(frame)
        for index, frame in enumerate(used):
            rows.append({'event_id': 'FRCB-{:04d}'.format(len(rows)),
                         'sequence': seq.name, 'frame': frame,
                         'event_index': index, 'future_horizon': 5})
    out.mkdir(parents=True)
    (out / 'sequence_list.txt').write_text('\n'.join(names) + '\n')
    with (out / 'events.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        'protocol': 'FRCB-ADMISSION-V1',
        'sequence_source': str(source),
        'sequence_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'sequence_count': len(names), 'event_count': len(rows),
        'events_per_sequence': args.events_per_sequence,
        'future_horizon': 5, 'selection': 'deterministic positions 1, floor(N/2)[, floor(2N/3)]',
        'training': False, 'test_split': False,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
