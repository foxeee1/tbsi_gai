#!/usr/bin/env python3
"""Create post-forward target/background token labels from crop geometry."""
import argparse
import json
from pathlib import Path

import numpy as np


def token_mask(row, grid=16):
    tr = row['search_crop_transform']
    x0, y0 = float(tr['crop_x']), float(tr['crop_y'])
    scale = float(tr['resize_factor'])
    size = float(tr['search_size'])
    gx, gy, gw, gh = [float(v) for v in row['gt_box_original']]
    gx, gy, gw, gh = (gx - x0) * scale, (gy - y0) * scale, gw * scale, gh * scale
    step = size / grid
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
    px0, py0 = xx * step, yy * step
    mask = (px0 + step > gx) & (py0 + step > gy) & (px0 < gx + gw) & (py0 < gy + gh)
    center_inside = (gx + gw / 2.0 >= 0 and gx + gw / 2.0 < size and
                     gy + gh / 2.0 >= 0 and gy + gh / 2.0 < size)
    return mask.reshape(-1), bool(center_inside), (gx, gy, gw, gh)


def dilate(mask, grid=16):
    m = mask.reshape(grid, grid)
    out = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys = slice(max(0, dy), min(grid, grid + dy))
            xs = slice(max(0, dx), min(grid, grid + dx))
            yd = slice(max(0, -dy), min(grid, grid - dy))
            xd = slice(max(0, -dx), min(grid, grid - dx))
            out[yd, xd] |= m[ys, xs]
    return out.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export-dir', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    export_dir = Path(args.export_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing non-empty output {}'.format(output))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    usable = 0
    with (export_dir / 'events.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            mask, center_inside, projected = token_mask(row)
            neighbor = dilate(mask)
            bg = neighbor & ~mask
            # If the one-token ring is empty, use all crop-background tokens; this is rare
            # and remains explicit in the metadata rather than silently dropping the event.
            if not bg.any():
                bg = ~mask
            row.update({'target_mask': mask.astype(np.uint8).tolist(),
                        'background_mask': bg.astype(np.uint8).tolist(),
                        'target_in_crop': bool(center_inside),
                        'target_token_count': int(mask.sum()),
                        'background_token_count': int(bg.sum()),
                        'target_smaller_than_one_token': bool(mask.sum() <= 1),
                        'projected_gt_xywh': [float(x) for x in projected],
                        'rescue_eligible': bool(center_inside and mask.any() and bg.any())})
            rows.append(row)
            usable += int(row['rescue_eligible'])
    with (output / 'token_labels.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    summary = {'events': len(rows), 'rescue_eligible_events': usable,
               'target_in_crop_events': sum(r['target_in_crop'] for r in rows),
               'target_outside_or_empty_events': sum(not r['rescue_eligible'] for r in rows),
               'small_target_events': sum(r['target_smaller_than_one_token'] for r in rows),
               'label_policy': 'GT projected after forward; one-token adjacent ring background; no GT in crop/model'}
    (output / 'label_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
