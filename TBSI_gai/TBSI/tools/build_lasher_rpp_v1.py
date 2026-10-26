#!/usr/bin/env python3
"""Build the single frozen LasHeR RPP-v1 train/test sequence manifests.

The official LasHeR attribute archive is the source of truth. Selection is
sequence-level and deterministic; the greedy objective preserves the
population prevalence of every multi-label attribute while retaining ordinary
and difficult sequences.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import zipfile
from collections import Counter


SKIP_SEQUENCES = {"advancedredcup", "cameraman_1202", "mirroratleft"}
DEFAULT_ATTR_ZIP = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/lasher/Attributes&Order.zip")
)
DEFAULT_DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/lasher"))
ATTR_PREFIX = "AttriSeqsTxt/"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path):
    with open(path, newline="") as handle:
        names = [row[0].strip() for row in csv.reader(handle) if row and row[0].strip()]
    return [name for name in names if name not in SKIP_SEQUENCES]


def read_attributes(attr_zip):
    with zipfile.ZipFile(attr_zip) as archive:
        order = archive.read("Attributes_order.txt").decode("utf-8-sig").strip()
        names = [item.strip() for item in order.split(",") if item.strip()]
        attributes = {}
        for member in archive.namelist():
            if not member.startswith(ATTR_PREFIX) or not member.endswith(".txt"):
                continue
            sequence = os.path.basename(member)[:-4]
            values = [int(value) for value in archive.read(member).decode("utf-8").strip().split(",")]
            if len(values) != len(names):
                raise ValueError("attribute width mismatch for %s" % sequence)
            attributes[sequence] = {name: value for name, value in zip(names, values)}
    return names, attributes


def stable_key(sequence, seed):
    return hashlib.sha256((str(seed) + ":" + sequence).encode("utf-8")).hexdigest()


def select_sequences(sequences, attributes, attribute_names, count, seed):
    if count <= 0 or count > len(sequences):
        raise ValueError("selection count must be in [1, source count]")

    ordered = sorted(sequences, key=lambda item: stable_key(item, seed))
    population = Counter(name for seq in sequences for name in attribute_names if attributes[seq].get(name, 0))
    fraction = float(count) / float(len(sequences))
    target = {name: min(population[name], int(round(population[name] * fraction))) for name in attribute_names}
    selected = []
    remaining = set(ordered)
    counts = Counter()

    # Fill the hardest-to-cover attribute deficits first, with a deterministic
    # tie-breaker. This is a multi-label coverage objective, not random sampling.
    while len(selected) < count:
        best = None
        best_score = None
        for sequence in remaining:
            score = 0.0
            for name in attribute_names:
                if not attributes[sequence].get(name, 0):
                    continue
                deficit = max(target[name] - counts[name], 0)
                if deficit:
                    score += float(deficit) / max(target[name], 1)
            # Keep ordinary sequences eligible after attribute deficits are met.
            score += 1e-9 * (1.0 / (1 + ordered.index(sequence)))
            tie = stable_key(sequence, seed)
            if best_score is None or (score, tie) > best_score:
                best = sequence
                best_score = (score, tie)
        selected.append(best)
        remaining.remove(best)
        for name in attribute_names:
            if attributes[best].get(name, 0):
                counts[name] += 1

    return selected, population, counts


def write_manifest(output_dir, split, source_list, selected, population, selected_counts,
                   attribute_names, attr_zip, seed, target_fraction):
    os.makedirs(output_dir, exist_ok=True)
    list_name = "proxy_%s_sequences.txt" % split
    manifest_name = "proxy_%s_manifest.json" % split
    list_bytes = ("\n".join(selected) + "\n").encode("utf-8")
    list_path = os.path.join(output_dir, list_name)
    with open(list_path, "wb") as handle:
        handle.write(list_bytes)
    manifest = {
        "protocol": "LasHeR-RPP-v1",
        "split": split,
        "selection": "sequence-level stratified multi-label greedy coverage",
        "selection_seed": seed,
        "target_fraction": target_fraction,
        "source_sequence_count": len(source_list),
        "selected_sequence_count": len(selected),
        "selected_fraction": float(len(selected)) / float(len(source_list)),
        "attribute_source": os.path.abspath(attr_zip),
        "attribute_source_sha256": sha256_file(attr_zip),
        "attribute_names": attribute_names,
        "population_attribute_counts": [population[name] for name in attribute_names],
        "selected_attribute_counts": [selected_counts[name] for name in attribute_names],
        "sequence_list_sha256": sha256_bytes(list_bytes),
        "sequences": selected,
    }
    with open(os.path.join(output_dir, manifest_name), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return list_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--attributes-zip", default=DEFAULT_ATTR_ZIP)
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "../experiments/proxy/lasher_rpp_v1"))
    parser.add_argument("--train-fraction", type=float, default=0.18)
    parser.add_argument("--test-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if not 0.15 <= args.train_fraction <= 0.20:
        raise ValueError("train fraction must be between 0.15 and 0.20")
    if not os.path.isfile(args.attributes_zip):
        raise FileNotFoundError(args.attributes_zip)

    attribute_names, attributes = read_attributes(args.attributes_zip)
    def list_file(split):
        candidates = [
            os.path.join(args.data_root, "..", "%singsetList.txt" % split),
            os.path.join(args.data_root, "..", "..", "data", "lasher", "%singsetList.txt" % split),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        raise FileNotFoundError("LasHeR %s list not found: %s" % (split, candidates))

    split_specs = {
        "train": (list_file("train"), None, args.train_fraction),
        "test": (list_file("test"), args.test_count, None),
    }
    for split, (list_file, count, fraction) in split_specs.items():
        source = read_split(os.path.abspath(list_file))
        if split == "train":
            count = max(1, int(round(len(source) * args.train_fraction)))
        if count > len(source):
            raise ValueError("requested %s sequences, only %s available" % (count, len(source)))
        selected, population, selected_counts = select_sequences(
            source, attributes, attribute_names, count, args.seed + (0 if split == "train" else 1))
        path = write_manifest(args.output_dir, split, source, selected, population, selected_counts,
                              attribute_names, args.attributes_zip, args.seed, fraction)
        print("%s: %d/%d -> %s" % (split, len(selected), len(source), path))


if __name__ == "__main__":
    main()
