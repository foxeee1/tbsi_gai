#!/usr/bin/env python3
"""Fixed-crop MR/B0 template K/V path decomposition.

This is an inference-only diagnostic.  It compares the two registered full
checkpoints on the same template, search crop, and tracker state.  The MR
counterfactual arms replace only the first 64 template-token K and/or V rows
in the standard ViT attention blocks after the first TBSI insertion.
"""
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output/experiments/MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109/workspace"
sys.path.insert(0, str(SRC))

from lib.test.evaluation.lasherdataset import LasHeRDataset
from lib.test.utils import TrackerParams
from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.test.tracker.tbsi_track import TBSITrack


MR_CKPT = ROOT / "output/experiments/MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109/train/checkpoint_u7020.pth.tar"
B0_CKPT = ROOT / "output/experiments/MSHA-B0-U936-S42-FULL15-FULL244-20261109/train/checkpoint_u7020.pth.tar"
B0_CONFIG = ROOT / "output/experiments/MSHA-B0-U936-S42-FULL15-FULL244-20261109/contracts/b0.yaml"
MR_CONFIG = ROOT / "output/experiments/MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109/contracts/mr_only_g1.yaml"
PAIR_CSV = ROOT / "output/experiments/MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109/analysis/per_sequence_vs_b0.csv"
OUT = ROOT / "output/experiments/MR_KV_PATH_DIAGNOSTIC_FIXEDCROP_20261114_RETRY2"
EVENTS = 8
TEMPLATE_TOKENS = 64
FIRST_POST_TBSI_BLOCK = 4


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_frame(seq, index):
    rgb = cv2.cvtColor(cv2.imread(seq.frames[0][index]), cv2.COLOR_BGR2RGB)
    tir = cv2.cvtColor(cv2.imread(seq.frames[1][index]), cv2.COLOR_BGR2RGB)
    return np.concatenate((rgb, tir), axis=2)


def box_iou(a, b):
    ax1, ay1, aw, ah = [float(x) for x in a]
    bx1, by1, bw, bh = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def select_events():
    rows = list(csv.DictReader(PAIR_CSV.open()))
    rows.sort(key=lambda r: float(r["delta_auc"]))
    chosen = []
    # Fixed before reading any diagnostic output: three losses, two ties, three gains.
    chosen.extend(rows[:3])
    ties = sorted(rows, key=lambda r: abs(float(r["delta_auc"])))
    for row in ties:
        if row["sequence"] not in {x["sequence"] for x in chosen}:
            chosen.append(row)
        if len(chosen) >= 5:
            break
    for row in reversed(rows):
        if row["sequence"] not in {x["sequence"] for x in chosen}:
            chosen.append(row)
        if len(chosen) >= EVENTS:
            break
    return chosen[:EVENTS]


class QKVCapture:
    def __init__(self, tracker, mode="capture", reference=None):
        self.tracker = tracker
        self.mode = mode
        self.reference = reference or {}
        self.values = {}
        self.calls = {}
        self.handles = []
        for index, block in enumerate(tracker.network.backbone.blocks):
            handle = block.attn.qkv.register_forward_hook(self._hook(index))
            self.handles.append(handle)

    def _hook(self, index):
        def hook(module, inputs, output):
            call = self.calls.get(index, 0)
            self.calls[index] = call + 1
            key = (index, call)
            out = output
            if self.mode == "override" and index >= FIRST_POST_TBSI_BLOCK:
                ref = self.reference.get(key)
                if ref is None:
                    raise RuntimeError("missing B0 qkv reference for {}".format(key))
                ref = ref.to(device=out.device, dtype=out.dtype)
                c = out.shape[-1] // 3
                out = out.clone()
                if self.mode_name in ("k", "kv"):
                    out[:, :TEMPLATE_TOKENS, c:2 * c] = ref[:, :TEMPLATE_TOKENS, c:2 * c]
                if self.mode_name in ("v", "kv"):
                    out[:, :TEMPLATE_TOKENS, 2 * c:3 * c] = ref[:, :TEMPLATE_TOKENS, 2 * c:3 * c]
            self.values[key] = out.detach().float().cpu()
            return out
        return hook

    def set_mode(self, name):
        self.mode = "capture" if name == "capture" else "override"
        self.mode_name = name
        self.values.clear()
        self.calls.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def make_tracker(config_name, checkpoint):
    os.environ["TBSI_CHECKPOINT_OVERRIDE"] = str(checkpoint)
    os.environ["TBSI_INFER_FP16"] = "0"
    params = TrackerParams()
    config_path = B0_CONFIG if config_name == "b0" else MR_CONFIG
    update_config_from_file(str(config_path))
    params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.debug = 0
    params.save_all_boxes = False
    params.checkpoint = str(checkpoint)
    return TBSITrack(params, "lasher_test")


def parse_qkv(tensor, heads):
    b, n, three_c = tensor.shape
    c = three_c // 3
    qkv = tensor.reshape(b, n, 3, heads, c // heads).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def attention(q, k):
    scale = q.shape[-1] ** -0.5
    return torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)


def block_metrics(b0_t, mr_t, heads):
    q0, k0, v0 = parse_qkv(b0_t, heads)
    qm, km, vm = parse_qkv(mr_t, heads)
    bm = attention(qm, km)
    b0 = attention(q0, k0)

    def y(q, k, v):
        return attention(q, k)[:, :, TEMPLATE_TOKENS:, :] @ v

    # Template-only K/V replacement in the MR query and search-side K/V.
    k_template_b0 = km.clone()
    k_template_b0[:, :, :TEMPLATE_TOKENS] = k0[:, :, :TEMPLATE_TOKENS]
    v_template_b0 = vm.clone()
    v_template_b0[:, :, :TEMPLATE_TOKENS] = v0[:, :, :TEMPLATE_TOKENS]
    y_mr = bm[:, :, TEMPLATE_TOKENS:, :] @ vm
    y_k = y(qm, k_template_b0, vm)
    y_v = bm[:, :, TEMPLATE_TOKENS:, :] @ v_template_b0
    y_kv = y(qm, k_template_b0, v_template_b0)
    y_b0 = b0[:, :, TEMPLATE_TOKENS:, :] @ v0

    def mean_norm(x):
        return float(x.float().reshape(x.shape[0], -1).norm(dim=-1).mean())

    def mean_abs(x):
        return float(x.float().abs().mean())

    mr_cross = bm[:, :, TEMPLATE_TOKENS:, :TEMPLATE_TOKENS].sum(-1).mean()
    b0_cross = b0[:, :, TEMPLATE_TOKENS:, :TEMPLATE_TOKENS].sum(-1).mean()
    mr_self = bm[:, :, TEMPLATE_TOKENS:, TEMPLATE_TOKENS:].sum(-1).mean()
    b0_self = b0[:, :, TEMPLATE_TOKENS:, TEMPLATE_TOKENS:].sum(-1).mean()
    return {
        "mr_b0_output_norm": mean_norm(y_mr - y_b0),
        "template_v_effect_norm": mean_norm(y_mr - y_v),
        "template_k_effect_norm": mean_norm(y_mr - y_k),
        "template_kv_effect_norm": mean_norm(y_mr - y_kv),
        "search_self_effect_norm": mean_norm(
            bm[:, :, TEMPLATE_TOKENS:, TEMPLATE_TOKENS:] @ vm[:, :, TEMPLATE_TOKENS:] -
            b0[:, :, TEMPLATE_TOKENS:, TEMPLATE_TOKENS:] @ vm[:, :, TEMPLATE_TOKENS:]),
        "cross_attention_mass_mr": float(mr_cross),
        "cross_attention_mass_b0": float(b0_cross),
        "self_attention_mass_mr": float(mr_self),
        "self_attention_mass_b0": float(b0_self),
        "template_k_route_delta": mean_abs(bm[:, :, TEMPLATE_TOKENS:, :TEMPLATE_TOKENS] -
                                            attention(qm, k_template_b0)[:, :, TEMPLATE_TOKENS:, :TEMPLATE_TOKENS]),
        "template_v_delta_norm": mean_norm(vm[:, :, :TEMPLATE_TOKENS] - v0[:, :, :TEMPLATE_TOKENS]),
    }


def main():
    if not MR_CKPT.is_file() or not B0_CKPT.is_file():
        raise FileNotFoundError("registered MR/B0 checkpoints are required")
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("refusing to overwrite {}".format(OUT))
    OUT.mkdir(parents=True)
    selected = select_events()
    seq_list = OUT / "event_sequences.txt"
    seq_list.write_text("\n".join(row["sequence"] for row in selected) + "\n")
    dataset = LasHeRDataset("testingset", str(seq_list)).get_sequence_list()
    seqmap = {s.name: s for s in dataset}
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    rows = []
    for selected_row in selected:
        name = selected_row["sequence"]
        seq = seqmap[name]
        frame = min(10, len(seq.ground_truth_rect) - 1)
        init_info = {"init_bbox": np.asarray(seq.ground_truth_rect[0], dtype=np.float32)}
        state = np.asarray(seq.ground_truth_rect[max(0, frame - 1)], dtype=np.float32)
        image0, image = read_frame(seq, 0), read_frame(seq, frame)
        info = {"previous_output": {}, "sequence_name": name, "frame_num": frame,
                "gt_bbox": np.asarray(seq.ground_truth_rect[frame], dtype=np.float32)}

        b0 = make_tracker("b0", B0_CKPT)
        b0.initialize(image0, init_info)
        b0.state = state.copy()
        cap0 = QKVCapture(b0, "capture")
        out_b0 = b0.track(image, info)
        b0_values = dict(cap0.values)
        cap0.close()
        b0_iou = box_iou(out_b0["target_bbox"], seq.ground_truth_rect[frame])
        del b0

        mr = make_tracker("mr_only_g1", MR_CKPT)
        mr.initialize(image0, init_info)
        mr.state = state.copy()
        cap = QKVCapture(mr, "capture", b0_values)
        arm_outputs = {}
        arm_iou = {}
        arm_metrics = {}
        native_values = None
        for arm in ("native", "k", "v", "kv"):
            mr.initialize(image0, init_info)
            mr.state = state.copy()
            cap.set_mode(arm)
            out = mr.track(image, info)
            arm_outputs[arm] = [float(x) for x in out["target_bbox"]]
            arm_iou[arm] = box_iou(out["target_bbox"], seq.ground_truth_rect[frame])
            if arm == "native":
                native_values = dict(cap.values)
        for arm in ("native", "k", "v", "kv"):
            arm_metrics[arm] = []
            for key, mr_qkv in sorted(native_values.items()):
                block, call = key
                if block < FIRST_POST_TBSI_BLOCK or key not in b0_values:
                    continue
                arm_metrics[arm].append({"block": block, "modality_call": call,
                                         **block_metrics(b0_values[key], mr_qkv,
                                                         mr.network.backbone.blocks[block].attn.num_heads)})
        cap.close()
        del mr

        for arm in ("native", "k", "v", "kv"):
            metrics = arm_metrics[arm]
            for item in metrics:
                rows.append({"sequence": name, "frame": frame,
                             "delta_auc_pp": float(selected_row["delta_auc"]),
                             "arm": arm, "b0_iou": b0_iou, "mr_iou": arm_iou[arm],
                             "box_iou_delta_vs_native": arm_iou[arm] - arm_iou["native"],
                             **item})
        print("[event] {} frame={} delta_auc={:+.3f} native_iou={:.4f} k={:.4f} v={:.4f} kv={:.4f}".format(
            name, frame, float(selected_row["delta_auc"]), arm_iou["native"],
            arm_iou["k"], arm_iou["v"], arm_iou["kv"]), flush=True)

    with (OUT / "per_block.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "FIXED-CROP-KV-DIAGNOSTIC-COMPLETE",
        "protocol": "8 testingset events selected before reading diagnostic output; GT template and GT previous-frame state; FP32; no training; template K/V replacement in standard ViT blocks >=4",
        "selection": [{"sequence": r["sequence"], "delta_auc_pp": float(r["delta_auc"])} for r in selected],
        "mr_checkpoint_sha256": sha256(MR_CKPT),
        "b0_checkpoint_sha256": sha256(B0_CKPT),
        "pair_csv_sha256": sha256(PAIR_CSV),
        "git_head": os.popen("git -C '{}' rev-parse HEAD".format(ROOT)).read().strip(),
        "rows": len(rows),
        "arms": ["native", "k", "v", "kv"],
        "limitations": ["fixed-state evidence only", "B0 K/V come from its own checkpoint input trajectory", "does not establish closed-loop tracking benefit"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "provenance.json").write_text(json.dumps({
        "source_workspace": str(SRC), "source_tbsi_layer_sha256": sha256(SRC / "lib/models/layers/tbsi_layer.py"),
        "source_vit_sha256": sha256(SRC / "lib/models/tbsi_track/vit_tbsi_care.py"),
        "script_sha256": sha256(Path(__file__)), "mr_checkpoint": str(MR_CKPT), "b0_checkpoint": str(B0_CKPT),
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
