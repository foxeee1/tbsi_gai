#!/usr/bin/env python3
"""Sequence-held-out test of whether S1 state harm is predictable.

This is an offline diagnostic only. It never changes tracker code or loads a
new checkpoint. The label is future_h{10,30} > 0, where positive means that
the S1-state rollout has lower future IoU than the M1-state rollout.
"""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output/experiments/M1_S1_CLOSED_LOOP_RISK_DIAGNOSTIC_20261015/paired_rollout.csv"
DEFAULT_OUT = ROOT / "output/experiments/M1_S1_FUTURE_HARM_PREDICTOR_20261016"


def rank_auc(y, score):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    npos = int(y.sum())
    return float((ranks[y].sum() - npos * (npos + 1) / 2) /
                 (npos * (len(y) - npos)))


def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(x, y, steps=1200, lr=0.08, l2=1.0):
    """Small deterministic ridge logistic fit, avoiding an extra dependency."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    z = (x - mu) / sd
    w = np.zeros(z.shape[1], dtype=float)
    b = 0.0
    n = max(len(y), 1)
    for _ in range(steps):
        p = sigmoid(z @ w + b)
        g = p - y
        w -= lr * ((z.T @ g) / n + l2 * w / n)
        b -= lr * float(g.mean())
    return mu, sd, w, b


def predict(x, params):
    mu, sd, w, b = params
    return sigmoid(((np.asarray(x, dtype=float) - mu) / sd) @ w + b)


def summarize(y, score, coverage=0.25):
    y, score = np.asarray(y, bool), np.asarray(score, float)
    k = max(1, int(np.ceil(len(score) * coverage)))
    top = np.argsort(-score, kind="mergesort")[:k]
    return {
        "frames": int(len(y)),
        "harm_fraction": float(y.mean()) if len(y) else None,
        "auroc": rank_auc(y, score),
        "top_25pct_harm_rate": float(y[top].mean()) if len(top) else None,
        "top_25pct_enrichment": float(y[top].mean() / y.mean()) if len(top) and y.mean() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.out)
    if out.exists():
        raise RuntimeError("refusing to overwrite existing output: {}".format(out))
    df = pd.read_csv(inp)
    required = {"sequence", "instant_harm", "score_center", "score_size", "score_joint",
                "state_gap_x", "state_gap_y", "state_gap_logw", "state_gap_logh",
                "future_h10", "future_h30"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError("missing columns: {}".format(missing))
    out.mkdir(parents=True)
    feature_sets = {
        "gap": ["state_gap_x", "state_gap_y", "state_gap_logw", "state_gap_logh"],
        "action_scores": ["score_center", "score_size", "score_joint"],
        "instant": ["instant_harm"],
        "all": ["state_gap_x", "state_gap_y", "state_gap_logw", "state_gap_logh",
                "score_center", "score_size", "score_joint", "instant_harm"],
    }
    results = {"input": str(inp), "input_sha256": hashlib.sha256(inp.read_bytes()).hexdigest(),
               "label_definition": "future_h10/h30 > 0 means S1-state future IoU is lower than M1-state",
               "sequence_count": int(df.sequence.nunique()), "rows": int(len(df)), "horizons": {}}
    for horizon in (10, 30):
        d = df[df["future_h{}".format(horizon)].notna()].copy()
        y = (d["future_h{}".format(horizon)].to_numpy(float) > 0.0).astype(int)
        horizon_result = {"positive_harm_fraction": float(y.mean()), "models": {}}
        sequences = sorted(d.sequence.unique())
        for name, cols in feature_sets.items():
            oof = np.full(len(d), np.nan, dtype=float)
            per_seq = []
            for seq in sequences:
                tr = d.sequence.to_numpy() != seq
                te = ~tr
                if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
                    oof[te] = float(y[tr].mean())
                else:
                    params = fit_logistic(d.loc[tr, cols].to_numpy(float), y[tr])
                    oof[te] = predict(d.loc[te, cols].to_numpy(float), params)
                per_seq.append({"sequence": seq, **summarize(y[te], oof[te])})
            horizon_result["models"][name] = {
                "features": cols,
                "pooled": summarize(y, oof),
                "sequence_mean_auroc": float(np.nanmean([x["auroc"] for x in per_seq if x["auroc"] is not None]))
                    if any(x["auroc"] is not None for x in per_seq) else None,
                "per_sequence": per_seq,
            }
        # Baselines use the same held-out rows and do not fit parameters.
        for name, col in (("baseline_score_center", "score_center"),
                          ("baseline_score_size", "score_size"),
                          ("baseline_score_joint", "score_joint"),
                          ("baseline_instant_harm", "instant_harm")):
            s = d[col].to_numpy(float)
            horizon_result["models"][name] = {"features": [col], "pooled": summarize(y, s)}
        results["horizons"][str(horizon)] = horizon_result
    (out / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    lines = ["# Future harm predictor diagnostic", "", "Input: `{}`".format(inp),
             "Label: future IoU(M1-state) - IoU(S1-state) > 0", "", "| horizon | model | AUROC | top25 harm | enrichment |", "|---:|---|---:|---:|---:|"]
    for h, hr in results["horizons"].items():
        for name, mr in hr["models"].items():
            p = mr["pooled"]
            lines.append("| {} | {} | {} | {} | {} |".format(h, name,
                "NA" if p["auroc"] is None else "{:.4f}".format(p["auroc"]),
                "NA" if p["top_25pct_harm_rate"] is None else "{:.4f}".format(p["top_25pct_harm_rate"]),
                "NA" if p["top_25pct_enrichment"] is None else "{:.4f}".format(p["top_25pct_enrichment"])))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
