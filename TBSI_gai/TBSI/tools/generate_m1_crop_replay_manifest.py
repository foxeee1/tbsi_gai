#!/usr/bin/env python
"""Join the frozen M1 temporal trace to its real LasHeR image files.

The trace remains the authority for tracker state and crop geometry.  This
manifest only adds image paths and provenance; it never recomputes crops from
GT boxes.
"""
import argparse
import hashlib
import json
import os


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def image_path(root, sequence, frame_id, modality):
    directory = os.path.join(root, sequence, modality)
    files = [name for name in os.listdir(directory) if name.lower().endswith('.jpg')]
    files.sort(key=lambda name: int(''.join(ch for ch in os.path.splitext(name)[0] if ch.isdigit())))
    # Tracker trace frame_id starts at 1 for the first post-initialization
    # track() call; dataset frame index 0 is the initialization frame.
    index = int(frame_id)
    if index < 0 or index >= len(files):
        raise FileNotFoundError('%s frame %s (found %d files)' % (directory, frame_id, len(files)))
    return os.path.abspath(os.path.join(directory, files[index]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace-dir', required=True)
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    trace_dir = os.path.abspath(args.trace_dir)
    dataset_root = os.path.abspath(args.dataset_root)
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256(checkpoint)
    rows = []
    sequences = []
    for sequence in sorted(os.listdir(trace_dir)):
        path = os.path.join(trace_dir, sequence, 'frames.jsonl')
        if not os.path.isfile(path):
            continue
        sequences.append(sequence)
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                frame_id = int(row['frame_id'])
                item = dict(row)
                item['visible_image'] = image_path(dataset_root, sequence, frame_id, 'visible')
                item['infrared_image'] = image_path(dataset_root, sequence, frame_id, 'infrared')
                if row.get('has_history'):
                    hf = int(row['history_frame_id'])
                    item['history_visible_image'] = image_path(dataset_root, sequence, hf, 'visible')
                    item['history_infrared_image'] = image_path(dataset_root, sequence, hf, 'infrared')
                else:
                    item['history_visible_image'] = None
                    item['history_infrared_image'] = None
                item['source_trace_dir'] = trace_dir
                item['source_checkpoint_sha256'] = checkpoint_hash
                item['source_kind'] = 'canonical_m1_replay'
                rows.append(item)
    if not rows:
        raise RuntimeError('no trace rows found')
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    manifest = {
        'schema_version': 1,
        'kind': 'm1_crop_replay_manifest',
        'source_kind': 'canonical_m1_replay',
        'trace_dir': trace_dir,
        'dataset_root': dataset_root,
        'checkpoint': checkpoint,
        'checkpoint_sha256': checkpoint_hash,
        'sequences': sequences,
        'rows': rows,
        'gt_usage': 'target/posthoc_only; never used for crop selection',
        'crop_source': 'frozen_trace_actual_crop_xywh',
        'image_indexing': 'trace frame_id maps directly to zero-based sorted numeric dataset index; frame 0 is initialization',
    }
    with open(output, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(json.dumps({
        'status': 'MANIFEST_PASS', 'rows': len(rows),
        'history_rows': sum(bool(r.get('has_history')) for r in rows),
        'sequences': sequences, 'output': output,
        'checkpoint_sha256': checkpoint_hash,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
