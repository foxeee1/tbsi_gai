#!/usr/bin/env python3
"""Attribute-wise LasHeR evaluation for two tracker result folders.

The aggregation follows lib.test.analysis.extract_results: sequence-level
curves are averaged within each attribute group, then AUC/OP/Precision are
read from the averaged curves.
"""
import argparse
import csv
import json
import os
import re
import zipfile

import numpy as np


ATTR_ZIP_DEFAULT = "/root/autodl-tmp/TBSI_gai/data/lasher/Attributes&Order.zip"
DATASET_DEFAULT = "/root/autodl-tmp/TBSI_gai/data/lasher/testingset"
RESULTS_DEFAULT = "/root/autodl-tmp/TBSI_gai/TBSI/output/test/tracking_results/tbsi_track"


def load_boxes(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(x) for x in re.split(r"[\t, ]+", line)])
    return np.asarray(rows, dtype=np.float64)


def overlap(pred, gt):
    tl = np.maximum(pred[:, :2], gt[:, :2])
    br = np.minimum(pred[:, :2] + pred[:, 2:] - 1.0,
                    gt[:, :2] + gt[:, 2:] - 1.0)
    wh = np.maximum(br - tl + 1.0, 0.0)
    inter = wh[:, 0] * wh[:, 1]
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - inter
    return inter / np.maximum(union, 1e-12)


def center_error(pred, gt, normalized=False):
    pc = pred[:, :2] + 0.5 * (pred[:, 2:] - 1.0)
    gc = gt[:, :2] + 0.5 * (gt[:, 2:] - 1.0)
    if normalized:
        pc = pc / np.maximum(gt[:, 2:], 1e-12)
        gc = gc / np.maximum(gt[:, 2:], 1e-12)
    return np.sqrt(((pc - gc) ** 2).sum(axis=1))


def sequence_curves(pred_path, gt_path):
    gt = load_boxes(gt_path).reshape(-1, 4)
    pred = load_boxes(pred_path).reshape(-1, 4)
    if pred.shape[0] > gt.shape[0]:
        pred = pred[:gt.shape[0]]
    elif pred.shape[0] < gt.shape[0]:
        pred = np.concatenate([pred, np.zeros((gt.shape[0] - pred.shape[0], 4))])
    # Match the project evaluator: initialization is always correct.
    pred[0] = gt[0]
    valid = (gt[:, 2:] > 0).all(axis=1)
    ov = overlap(pred, gt)
    ce = center_error(pred, gt)
    cne = center_error(pred, gt, normalized=True)
    ov[~valid] = -1.0
    ce[~valid] = np.inf
    cne[~valid] = np.inf
    ov_th = np.arange(0.0, 1.0 + 0.05, 0.05)
    ce_th = np.arange(0.0, 51.0)
    cne_th = np.arange(0.0, 51.0) / 100.0
    denom = float(len(gt))
    return {
        "overlap": (ov[:, None] > ov_th[None, :]).sum(axis=0) / denom,
        "center": (ce[:, None] <= ce_th[None, :]).sum(axis=0) / denom,
        "center_norm": (cne[:, None] <= cne_th[None, :]).sum(axis=0) / denom,
        "ao": float(ov[valid].mean()) if valid.any() else float("nan"),
    }


def summarize(items):
    overlap = np.stack([x["overlap"] for x in items]).mean(axis=0)
    center = np.stack([x["center"] for x in items]).mean(axis=0)
    center_norm = np.stack([x["center_norm"] for x in items]).mean(axis=0)
    return {
        "AUC": float(overlap.mean() * 100.0),
        "OP50": float(overlap[10] * 100.0),
        "OP75": float(overlap[15] * 100.0),
        "Precision": float(center[20] * 100.0),
        "NormPrecision": float(center_norm[20] * 100.0),
        "AO": float(np.mean([x["ao"] for x in items]) * 100.0),
        "N": len(items),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--results-root", default=RESULTS_DEFAULT)
    ap.add_argument("--dataset-root", default=DATASET_DEFAULT)
    ap.add_argument("--attr-zip", default=ATTR_ZIP_DEFAULT)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with zipfile.ZipFile(args.attr_zip) as zf:
        order = zf.read("Attributes_order.txt").decode().strip().split(", ")
        attrs = {}
        for name in os.listdir(args.dataset_root):
            attr_file = "AttriSeqsTxt/{}.txt".format(name)
            try:
                values = [int(x) for x in zf.read(attr_file).decode().strip().split(",")]
            except KeyError:
                continue
            if len(values) == len(order):
                attrs[name] = dict(zip(order, values))

    base_dir = os.path.join(args.results_root, args.baseline)
    meth_dir = os.path.join(args.results_root, args.method)
    per_seq = {}
    missing = []
    for name in sorted(attrs):
        gt_path = os.path.join(args.dataset_root, name, "init.txt")
        bp = os.path.join(base_dir, name + ".txt")
        mp = os.path.join(meth_dir, name + ".txt")
        if not (os.path.isfile(gt_path) and os.path.isfile(bp) and os.path.isfile(mp)):
            missing.append(name)
            continue
        per_seq[name] = {
            "attributes": attrs[name],
            "baseline": sequence_curves(bp, gt_path),
            "method": sequence_curves(mp, gt_path),
        }

    rows = []
    overall_b = summarize([x["baseline"] for x in per_seq.values()])
    overall_m = summarize([x["method"] for x in per_seq.values()])
    for attr in order:
        selected = [x for x in per_seq.values() if x["attributes"].get(attr, 0) == 1]
        if not selected:
            continue
        b = summarize([x["baseline"] for x in selected])
        m = summarize([x["method"] for x in selected])
        row = {"attribute": attr, "N": b["N"]}
        for key in ("AUC", "OP50", "OP75", "Precision", "NormPrecision", "AO"):
            row["baseline_" + key] = b[key]
            row["method_" + key] = m[key]
            row["delta_" + key] = m[key] - b[key]
        rows.append(row)
    rows.sort(key=lambda x: x["delta_AUC"], reverse=True)

    csv_path = os.path.join(args.out_dir, "attribute_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = os.path.join(args.out_dir, "attribute_metrics.json")
    with open(json_path, "w") as f:
        json.dump({"order": order, "overall_baseline": overall_b,
                   "overall_method": overall_m, "rows": rows,
                   "missing": missing}, f, indent=2)

    print("Evaluated sequences: {} / {}".format(len(per_seq), len(attrs)))
    print("Missing results: {}".format(len(missing)))
    print("Overall:", json.dumps({"baseline": overall_b, "method": overall_m}, indent=2))
    print("\nAttribute ranking by ΔAUC:")
    print("attribute,N,baseline_AUC,method_AUC,delta_AUC,delta_OP50,delta_OP75,delta_Precision,delta_NormPrecision")
    for row in rows:
        print("{attribute},{N},{baseline_AUC:.3f},{method_AUC:.3f},{delta_AUC:+.3f},{delta_OP50:+.3f},{delta_OP75:+.3f},{delta_Precision:+.3f},{delta_NormPrecision:+.3f}".format(**row))


if __name__ == "__main__":
    main()
