#!/usr/bin/env python3
"""Compute the RPP proxy/full ranking gate from durable analysis logs."""
import argparse
import json
import math
import re
from pathlib import Path


def auc_from_log(path):
    text = Path(path).read_text(errors="replace")
    matches = re.findall(r"\|\s*AUC\s*\|.*?\n([^\n]*?\|\s*([0-9]+(?:\.[0-9]+)?))", text)
    if not matches:
        raise RuntimeError("AUC not found in " + str(path))
    return float(matches[-1][1])


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    a = [v - mx for v in x]
    b = [v - my for v in y]
    den = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(i * j for i, j in zip(a, b)) / den if den else 0.0


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", metavar="NAME=LOG=FULL_AUC")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    records = []
    for item in args.records:
        name, log, full = item.split("=", 2)
        records.append({"name": name, "proxy_auc": auc_from_log(log), "full_auc": float(full), "log": log})
    records.sort(key=lambda r: r["name"])
    proxy = [r["proxy_auc"] for r in records]
    full = [r["full_auc"] for r in records]
    base_proxy = next(r["proxy_auc"] for r in records if "baseline" in r["name"])
    base_full = next(r["full_auc"] for r in records if "baseline" in r["name"])
    candidates = [r for r in records if "baseline" not in r["name"]]
    direction = sum((r["proxy_auc"] - base_proxy >= 0) == (r["full_auc"] - base_full >= 0) for r in candidates) / len(candidates)
    sp = pearson(ranks(proxy), ranks(full))
    pe = pearson(proxy, full)
    sms = next(r for r in records if "smsa" in r["name"])
    result = {
        "records": records,
        "pearson_auc": pe,
        "spearman_auc": sp,
        "direction_agreement": direction,
        "smsa_proxy_above_baseline": sms["proxy_auc"] > base_proxy,
        "smsa_full_above_baseline": sms["full_auc"] > base_full,
        "thresholds": {"direction_min": 0.75, "spearman_min": 0.6},
    }
    result["status"] = "pass" if direction >= 0.75 and sp >= 0.6 and result["smsa_proxy_above_baseline"] else "fail"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "pass" else 20)


if __name__ == "__main__":
    main()
