#!/usr/bin/env python3
"""Freeze deterministic sequence-level LasHeR RPP train/test proxies."""
import argparse
import csv
import hashlib
import json
import random
import zipfile
from pathlib import Path

SKIP = {"advancedredcup", "cameraman_1202", "mirroratleft"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_sequences(path):
    with open(path, newline="") as f:
        return [row[0].strip() for row in csv.reader(f) if row and row[0].strip() not in SKIP]


def read_attributes(zip_path, sequences):
    with zipfile.ZipFile(zip_path) as zf:
        names = [x.strip() for x in zf.read("Attributes_order.txt").decode().split(",")]
        attrs = {}
        for seq in sequences:
            member = "AttriSeqsTxt/{}.txt".format(seq)
            try:
                vals = [int(x) for x in zf.read(member).decode().strip().split(",")]
            except KeyError:
                vals = [0] * len(names)
            if len(vals) != len(names):
                raise ValueError("attribute width mismatch for {}".format(seq))
            attrs[seq] = vals
    return names, attrs


def stratified_select(sequences, attrs, fraction_or_count, seed):
    n = int(round(len(sequences) * fraction_or_count)) if fraction_or_count < 1 else int(fraction_or_count)
    n = max(1, min(len(sequences), n))
    width = len(next(iter(attrs.values())))
    population = [sum(attrs[s][j] for s in sequences) for j in range(width)]
    target = [p * n / float(len(sequences)) for p in population]
    rng = random.Random(seed)
    remaining = list(sequences)
    selected = []
    counts = [0] * width
    while len(selected) < n:
        # Deficit-driven greedy sampling preserves rare and difficult labels;
        # seeded jitter makes ties deterministic without random subset sampling.
        best = None
        best_score = None
        for seq in remaining:
            vec = attrs[seq]
            gain = sum(max(target[j] - counts[j], 0.0) / max(target[j], 1.0) for j in range(width) if vec[j])
            rarity = sum(1.0 / max(population[j], 1) for j in range(width) if vec[j])
            score = gain + 0.05 * rarity + rng.random() * 1e-6
            if best_score is None or score > best_score:
                best, best_score = seq, score
        selected.append(best)
        remaining.remove(best)
        counts = [counts[j] + attrs[best][j] for j in range(width)]
    return selected, population, counts


def write_proxy(out, split, source_list, attrs, names, fraction_or_count, seed):
    selected, population, counts = stratified_select(source_list, attrs, fraction_or_count, seed)
    list_name = "proxy_{}_sequences.txt".format(split)
    manifest_name = "proxy_{}_manifest.json".format(split)
    list_path = out / list_name
    list_path.write_text("\n".join(selected) + "\n")
    manifest = {
        "protocol": "RPP-v1",
        "split": split,
        "seed": seed,
        "source_sequence_count": len(source_list),
        "selected_sequence_count": len(selected),
        "selected_fraction": len(selected) / float(len(source_list)),
        "attribute_names": names,
        "population_attribute_counts": population,
        "selected_attribute_counts": counts,
        "sequence_list_sha256": sha256_file(list_path),
        "sequences": selected,
    }
    (out / manifest_name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data/lasher")
    ap.add_argument("--out", default="output/proxy_runs/rpp-v1")
    args = ap.parse_args()
    root = Path(args.data_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    zip_path = root / "Attributes&Order.zip"
    train = read_sequences(root / "trainingsetList.txt")
    test = read_sequences(root / "testingsetList.txt")
    names, train_attrs = read_attributes(zip_path, train)
    _, test_attrs = read_attributes(zip_path, test)
    train_manifest = write_proxy(out, "train", train, train_attrs, names, 0.18, 42)
    test_manifest = write_proxy(out, "test", test, test_attrs, names, 60, 3407)
    summary = {
        "protocol": "RPP-v1",
        "attribute_order": names,
        "source_zip_sha256": sha256_file(zip_path),
        "train": train_manifest,
        "test": test_manifest,
    }
    (out / "proxy_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "train": len(train_manifest["sequences"]), "test": len(test_manifest["sequences"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
