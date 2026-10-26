#!/usr/bin/env python3
"""Fit/evaluate bounded offset and size predictors with sequence-held-out no-op gating."""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.parameter.tbsi_track import parameters
from lib.test.tracker.tbsi_track import TBSITrack
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box

NAME = 'TBSI_E_MHSA_FULL_4090D_seed42'
CONFIG = ROOT / 'experiments/tbsi_track' / (NAME + '.yaml')
CHECKPOINT = ROOT / 'output/experiments' / NAME / 'checkpoints/TBSITrack_ep0015.pth.tar'
CONFIG_SHA = 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e'
CHECKPOINT_SHA = '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def iou(a, b):
    ix = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


def decode(network, tracker, response, size, offset, state, shape, resize):
    box = network.box_head.cal_bbox(response, size, offset).view(-1, 4).mean(0)
    box = box * (tracker.params.search_size / resize)
    cx, cy = state[0] + .5 * state[2], state[1] + .5 * state[3]
    half = .5 * tracker.params.search_size / resize
    xywh = [float(box[0] + cx - half - .5 * box[2]),
            float(box[1] + cy - half - .5 * box[3]), float(box[2]), float(box[3])]
    return clip_box(xywh, shape[0], shape[1], margin=10)


def fit(x, y, alpha=10.0):
    mu = x.mean(0); sd = x.std(0); sd[sd < 1e-6] = 1.
    z = np.clip((x - mu) / sd, -8., 8.)
    w = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y.mean(0)))
    return mu, sd, y.mean(0), w


def pred(model, x):
    mu, sd, b, w = model
    return b + np.clip((x - mu) / sd, -8., 8.) @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True); ap.add_argument('--signals', required=True)
    ap.add_argument('--trace-dir', required=True); ap.add_argument('--output-dir', required=True)
    ap.add_argument('--alpha', type=float, default=10.0); ap.add_argument('--rho', type=float, default=.5)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refusing nonempty output')
    out.mkdir(parents=True, exist_ok=True)
    if sha256(CONFIG) != CONFIG_SHA or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('protected M1 hash mismatch')
    data = np.load(args.features, allow_pickle=False)
    signals = { (r['sequence'], r['frame']): r for r in csv.DictReader(open(args.signals)) }
    trace = {}
    for p in Path(args.trace_dir).glob('*/frames.jsonl'):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line); trace[(p.parent.name, int(r['frame_id']))] = r
    seqs = sorted(set(str(x) for x in data['sequence']))
    list_path = out.parent / 'predictor_sequence_list.txt'; list_path.write_text('\n'.join(seqs) + '\n')
    seqmap = {s.name: s for s in LasHeRDataset('trainingset', str(list_path)).get_sequence_list()}
    os.environ['TBSI_INFER_FP16'] = '0'; torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    params = parameters(NAME); params.debug = 0; params.save_all_boxes = False; params.checkpoint = str(CHECKPOINT)
    tracker = TBSITrack(params, 'm1_bounded_readout_predictor'); network = tracker.network.eval()
    rows = []
    with torch.no_grad():
        for i in range(len(data['sequence'])):
            seq = str(data['sequence'][i]); frame = int(data['frame'][i]); tr = trace[(seq, frame)]
            state = np.asarray(tr['tracker_state_before'], dtype=np.float64); gt = np.asarray(tr['gt_box'], dtype=np.float64)
            rgb = cv2.imread(seqmap[seq].frames[0][frame], cv2.IMREAD_COLOR); tir = cv2.imread(seqmap[seq].frames[1][frame], cv2.IMREAD_COLOR)
            image = np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), cv2.cvtColor(tir, cv2.COLOR_BGR2RGB)), 2)
            _, resize, _ = sample_target(image, state, params.search_factor, output_sz=params.search_size)
            fused = torch.as_tensor(data['fused'][i].transpose(2, 0, 1)[None], dtype=torch.float32, device='cuda')
            score, bbox, size, offset = network.box_head(fused)
            response = tracker.output_window * score
            flat = response.flatten(1)[0]
            order = torch.argsort(flat, descending=True)
            peak = int(order[0].item()); px, py = peak % 16, peak // 16
            peak_margin = float((flat[order[0]] - flat[order[1]]).item())
            base_box = decode(network, tracker, response, size, offset, state, image.shape, resize); base_iou = iou(base_box, gt)
            gt_cx, gt_cy = gt[0] + .5 * gt[2], gt[1] + .5 * gt[3]
            scx, scy = state[0] + .5 * state[2], state[1] + .5 * state[3]
            target_offset = np.asarray([(gt_cx - scx) * resize / params.search_size + .5, (gt_cy - scy) * resize / params.search_size + .5]) * 16. - np.asarray([px, py])
            target_size = np.asarray([gt[2] * resize / params.search_size, gt[3] * resize / params.search_size])
            cur_offset = offset[0, :, py, px].detach().cpu().numpy(); cur_size = size[0, :, py, px].detach().cpu().numpy()
            do = target_offset - cur_offset; ds = target_size - cur_size
            oo = offset.clone(); oo[:, :, py, px] = offset[:, :, py, px] + torch.as_tensor(do * args.rho, device='cuda', dtype=offset.dtype)
            iz = iou(decode(network, tracker, response, size, oo, state, image.shape, resize), gt) - base_iou
            zz = size.clone(); zz[:, :, py, px] = torch.clamp(size[:, :, py, px] + torch.as_tensor(ds * args.rho, device='cuda', dtype=size.dtype), min=.01)
            io = iou(decode(network, tracker, response, zz, offset, state, image.shape, resize), gt) - base_iou
            rows.append({'sequence': seq, 'frame': frame, 'base_iou': base_iou, 'peak_x': px, 'peak_y': py,
                         'peak_margin': peak_margin,
                         'offset_dx': float(do[0]), 'offset_dy': float(do[1]), 'size_dw': float(ds[0]), 'size_dh': float(ds[1]),
                         'offset_utility': iz, 'size_utility': io, 'gt_cell_x': int(data['gt_xy'][i, 0]), 'gt_cell_y': int(data['gt_xy'][i, 1])})
            if (i + 1) % 16 == 0: print('states={}/{}'.format(i + 1, len(data['sequence'])), flush=True)
    # Only deployment-visible evidence is allowed; GT rank is diagnostic-only.
    fields = ['peak_margin']
    for r in rows:
        if (r['sequence'], str(r['frame'])) not in signals:
            raise RuntimeError('signal key audit failed')
    x = np.asarray([[r[f] for f in fields] for r in rows], dtype=float)
    yoff = np.asarray([[r['offset_dx'], r['offset_dy']] for r in rows]); ysize = np.asarray([[r['size_dw'], r['size_dh']] for r in rows])
    uoff = np.asarray([r['offset_utility'] for r in rows]); usize = np.asarray([r['size_utility'] for r in rows])
    seq_arr = np.asarray([r['sequence'] for r in rows]); records=[]; summary={}
    for threshold in [0., .04, .08, .12, .16]:
        selected=[]; harms=[]; covered=[]
        for held in seqs:
            te = seq_arr == held; tr = ~te
            mo = fit(x[tr], uoff[tr], args.alpha); mz = fit(x[tr], usize[tr], args.alpha)
            po = pred(mo, x[te]); pz = pred(mz, x[te]); idx = np.where(te)[0]
            for j, k in enumerate(idx):
                choices=[0., float(po[j]), float(pz[j])]; a=int(np.argmax(choices)); use=choices[a] > threshold
                val=([0., uoff[k], usize[k]][a] if use else 0.); selected.append(val); covered.append(use); harms.append(use and val < -1e-6)
                records.append({'sequence': seq_arr[k], 'frame': int(rows[k]['frame']), 'threshold': threshold, 'action': ['noop','offset','size'][a] if use else 'noop', 'predicted_utility': choices[a], 'selected_utility': val, 'offset_utility': uoff[k], 'size_utility': usize[k]})
        summary[str(threshold)]={'threshold':threshold,'states':len(selected),'coverage':float(np.mean(covered)),'selected_mean_utility':float(np.mean(selected)),'harmful_rate_all':float(np.mean(harms)),'harmful_rate_covered':float(np.sum(harms)/max(np.sum(covered),1))}
    with (out/'per_state_selection.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    with (out/'capture_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'id':'M1-BOUNDED-READOUT-PREDICTOR-20261014','parent':NAME,'rho':args.rho,'features':fields,'sequence_holdout':True,'models':summary,'limitations':['fixed-state decoded replay','selector labels use peak-cell decoded utility','no closed-loop tracking']},indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
