"""
Exp4 direct evaluation on 50 baseline sequences.
Avoids tracker cache issue by creating fresh model per sequence.
"""
import sys, os, math, time, cv2, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import _init_paths

from lib.test.evaluation import get_dataset
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target
from lib.test.evaluation.environment import env_settings
from lib.utils.box_ops import clip_box
from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.models.tbsi_track import build_tbsi_track

PRJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
yaml_file = os.path.join(PRJ, 'experiments/tbsi_track/vitb_256_tbsi_exp4_degstate.yaml')
update_config_from_file(yaml_file)

save_dir = env_settings().save_dir
ckpt_file = os.path.join(save_dir, "checkpoints/train/tbsi_track/vitb_256_tbsi_exp4_degstate/TBSITrack_ep0003.pth.tar")
out_dir = os.path.join(PRJ, "output/test/tracking_results/tbsi_track/vitb_256_tbsi_exp4_degstate")
os.makedirs(out_dir, exist_ok=True)

# 50 baseline sequences
with open('/tmp/baseline_50_seqs.txt') as f:
    seq_names = [l.strip() for l in f if l.strip()]

dataset = get_dataset('lasher_test')
seq_map = {seq.name: seq for seq in dataset}
total, success_count = len(seq_names), 0

for seq_name in seq_names:
    if seq_name not in seq_map:
        print(f"SKIP {seq_name}: not found")
        continue

    result_file = os.path.join(out_dir, f"{seq_name}.txt")
    if os.path.exists(result_file):
        print(f"SKIP {seq_name}: already done")
        success_count += 1
        continue

    seq = seq_map[seq_name]
    net = build_tbsi_track(cfg, training=False)
    ckpt = torch.load(ckpt_file, map_location='cpu')['net']
    net.load_state_dict(ckpt, strict=True)
    net.cuda().eval()
    preproc = Preprocessor()

    # Initialize
    init_info = seq.init_info()
    img0 = cv2.imread(seq.frames[0][0])[:,:,::-1]
    img1 = cv2.imread(seq.frames[1][0])[:,:,::-1]
    image = np.concatenate([img0, img1], 2) if img0.ndim == 3 else img0

    z_patch, zf, _ = sample_target(image, init_info['init_bbox'], cfg.TEST.TEMPLATE_FACTOR, output_sz=cfg.TEST.TEMPLATE_SIZE)
    template = preproc.process(z_patch, None)

    state = init_info['init_bbox']
    pred_boxes = [state]
    H, W, _ = image.shape

    for f_idx in range(len(seq.frames[0])):
        if f_idx > 0:
            fp0 = seq.frames[0][f_idx]
            fp1 = seq.frames[1][f_idx]
            img_v = cv2.imread(fp0)[:, :, ::-1] if isinstance(fp0, str) else fp0
            img_i = cv2.imread(fp1)[:, :, ::-1] if isinstance(fp1, str) else fp1
            image = np.concatenate([img_v, img_i], 2)

        x_patch, xf, _ = sample_target(image, state, cfg.TEST.SEARCH_FACTOR, output_sz=cfg.TEST.SEARCH_SIZE)
        search = preproc.process(x_patch, None)

        with torch.no_grad():
            out = net(template=[template.tensors[:,:3], template.tensors[:,3:]],
                      search=[search.tensors[:,:3], search.tensors[:,3:]],
                      return_last_attn=False)

        # Post-process
        feat_sz = int(cfg.TEST.SEARCH_SIZE // 16)
        pred_sm = out['score_map']
        from lib.test.utils.hann import hann2d
        window = hann2d(torch.tensor([feat_sz, feat_sz]).long(), centered=True).cuda()
        response = window * pred_sm
        pred_box = net.box_head.cal_bbox(response, out['size_map'], out['offset_map'])
        pred_box = pred_box.view(-1, 4)
        pred_box_mean = (pred_box.mean(dim=0) * cfg.TEST.SEARCH_SIZE / xf).tolist()

        cx, cy, w, h = state
        hs = 0.5 * cfg.TEST.SEARCH_SIZE / xf
        cx_r = pred_box_mean[0] + (cx + w/2 - hs)
        cy_r = pred_box_mean[1] + (cy + h/2 - hs)
        state = clip_box([cx_r - pred_box_mean[2]/2, cy_r - pred_box_mean[3]/2, pred_box_mean[2], pred_box_mean[3]], H, W, 10)
        pred_boxes.append(state)

    # Save results
    np.savetxt(result_file, np.array(pred_boxes), delimiter='\t', fmt='%.6f')
    del net
    torch.cuda.empty_cache()
    success_count += 1
    print(f"[{success_count}/{total}] {seq_name} OK ({len(pred_boxes)} frames)")

print(f"\nDone! {success_count}/{total} sequences completed.")
