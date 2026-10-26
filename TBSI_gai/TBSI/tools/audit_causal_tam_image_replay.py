#!/usr/bin/env python
"""Compare online real-image M1 raw tokens with the accepted trace."""
import argparse, hashlib, json, os, sys
import cv2, numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lib.config.tbsi_track import config as config_module
from lib.models.tbsi_track import build_tbsi_track
from lib.train.data.processing_utils import sample_target
from run_causal_tam_image_replay_smoke import load_pair, crop_image, prep


def init_path(frame_path):
    directory = os.path.dirname(frame_path)
    files = [n for n in os.listdir(directory) if n.lower().endswith('.jpg')]
    files.sort(key=lambda n: int(''.join(c for c in os.path.splitext(n)[0] if c.isdigit())))
    return os.path.join(directory, files[0])


def init_box(manifest, sequence):
    path = os.path.join(manifest['dataset_root'], sequence, 'init.txt')
    with open(path) as f:
        return [float(v) for v in f.readline().strip().split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--trace-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--per-sequence', type=int, default=5)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    manifest = json.load(open(args.manifest))
    rows_by_seq = {}
    for row in manifest['rows']:
        rows_by_seq.setdefault(row['sequence_id'], []).append(row)
    selected = []
    for seq, rows in sorted(rows_by_seq.items()):
        rows = [r for r in rows if r.get('has_history')]
        if not rows: continue
        idx = np.linspace(0, len(rows) - 1, min(args.per_sequence, len(rows))).round().astype(int)
        selected.extend(rows[int(i)] for i in sorted(set(idx)))
    config_module.update_config_from_file(args.config)
    config_module.cfg.MODEL.INIT_CHECKPOINT = os.path.abspath(args.checkpoint)
    config_module.cfg.MODEL.CAUSAL_TAM.ENABLE = True
    config_module.cfg.MODEL.CAUSAL_TAM.FORCE_ZERO = True
    model = build_tbsi_track(config_module.cfg, training=False).to(args.device).eval()
    state = torch.load(args.checkpoint, map_location='cpu')['net']
    model.load_state_dict(state, strict=False)
    results = []
    for row in selected:
        first = next(r for r in manifest['rows'] if r['sequence_id'] == row['sequence_id'] and int(r['frame_id']) == 1)
        init_img = load_pair((init_path(first['visible_image']), init_path(first['infrared_image'])), args.device)
        box = init_box(manifest, row['sequence_id'])
        z_rgb, _, zmask = sample_target(init_img[0], box, 2.0, output_sz=128)
        z_tir, _, _ = sample_target(init_img[1], box, 2.0, output_sz=128)
        template, _ = prep(z_rgb, z_tir, zmask, args.device)
        cur_img = load_pair((row['visible_image'], row['infrared_image']), args.device)
        cur_rgb, cmask = crop_image(cur_img[0], row['current_crop_xywh'])
        cur_tir, _ = crop_image(cur_img[1], row['current_crop_xywh'])
        search, _ = prep(cur_rgb, cur_tir, cmask, args.device)
        with torch.no_grad():
            out = model([template[:, :3], template[:, 3:]], [search[:, :3], search[:, 3:]])
        raw = out['tam_raw_head_feature'].float().cpu().contiguous()
        trace_file = os.path.join(args.trace_dir, row['sequence_id'], row['raw_head_file'])
        expected = torch.from_numpy(np.load(trace_file, allow_pickle=False)).float()
        delta = (raw - expected).abs()
        results.append({'sequence_id':row['sequence_id'], 'frame_id':row['frame_id'],
                        'crop_xywh_equal': row['current_crop_xywh'] == next(r for r in manifest['rows'] if r['sequence_id']==row['sequence_id'] and int(r['frame_id'])==int(row['frame_id']))['current_crop_xywh'],
                        'shape': list(raw.shape), 'trace_shape': list(expected.shape),
                        'online_sha256': hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
                        'trace_sha256': hashlib.sha256(expected.numpy().tobytes()).hexdigest(),
                        'checksum_equal': hashlib.sha256(raw.numpy().tobytes()).hexdigest() == hashlib.sha256(expected.numpy().tobytes()).hexdigest(),
                        'max_abs_diff': float(delta.max()), 'mean_abs_diff': float(delta.mean()),
                        'finite': bool(torch.isfinite(raw).all())})
    result = {'decision':'TAM_IMAGE_REPLAY_PARITY_AUDIT', 'selected_rows':len(results),
              'per_sequence':args.per_sequence, 'results':results,
              'all_shapes_equal':all(r['shape']==r['trace_shape'] for r in results),
              'all_finite':all(r['finite'] for r in results),
              'checksum_equal_count':sum(r['checksum_equal'] for r in results),
              'max_abs_diff':max(r['max_abs_diff'] for r in results),
              'mean_abs_diff_mean':float(np.mean([r['mean_abs_diff'] for r in results])),
              'claim':'replay parity only; no tracking-performance claim'}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump(result, open(args.output,'w'), indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == '__main__': main()
