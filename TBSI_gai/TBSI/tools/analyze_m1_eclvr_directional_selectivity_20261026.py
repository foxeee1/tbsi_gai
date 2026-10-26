#!/usr/bin/env python3
"""Offline, sequence-clustered direction analysis for the ECLVR CALIB48 export."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_mean(values, weights):
    denom = float(weights.sum())
    if denom <= 1e-8:
        return float("nan")
    return float((values * weights[None, :, None]).sum(axis=1).mean() / denom)


def direction_stats(delta, fg):
    """Return mask-conditioned direction effects; delta is [head, token, dim]."""
    norms = np.linalg.norm(delta, axis=-1)
    unit = delta / np.maximum(norms[..., None], 1e-8)
    fg = np.clip(fg.astype(np.float64), 0.0, 1.0)
    bg = 1.0 - fg
    fg_den = max(float(fg.sum()), 1e-8)
    bg_den = max(float(bg.sum()), 1e-8)
    fg_centroid = (unit * fg[None, :, None]).sum(axis=1) / fg_den
    bg_centroid = (unit * bg[None, :, None]).sum(axis=1) / bg_den
    fg_coherence = np.linalg.norm(fg_centroid, axis=-1)
    bg_coherence = np.linalg.norm(bg_centroid, axis=-1)
    gap = np.linalg.norm(fg_centroid - bg_centroid, axis=-1)
    return float(gap.mean()), float(fg_coherence.mean()), float(bg_coherence.mean())


def event_metrics(arrays, location, modality):
    delta = arrays["l{}_delta_{}".format(location, modality)].astype(np.float64)
    current = arrays["l{}_current_{}_value".format(location, modality)].astype(np.float64)
    fg_index = 0 if modality == "rgb" else 1
    fg = arrays["l{}_fg_masks".format(location)][fg_index].astype(np.float64)
    bg = 1.0 - np.clip(fg, 0.0, 1.0)

    dnorm = np.linalg.norm(delta, axis=-1)
    unit = delta / np.maximum(dnorm[..., None], 1e-8)
    cnorm = np.linalg.norm(current, axis=-1)
    cos = (delta * current).sum(axis=-1) / np.maximum(dnorm * cnorm, 1e-8)
    abs_radial = np.abs(cos)
    orth = np.sqrt(np.maximum(1.0 - np.square(np.clip(cos, -1.0, 1.0)), 0.0))

    def group_mean(values, weights):
        return float((values * weights[None, :]).sum() / (values.shape[0] * max(float(weights.sum()), 1e-8)))

    direction_gap, fg_coherence, bg_coherence = direction_stats(delta, fg)
    fg_centroid = (unit * fg[None, :, None]).sum(axis=1) / max(float(fg.sum()), 1e-8)
    fg_centroid = fg_centroid / np.maximum(np.linalg.norm(fg_centroid, axis=-1, keepdims=True), 1e-8)
    target_alignment = (unit * fg_centroid[:, None, :]).sum(axis=-1).mean(axis=0)
    radial_mean = cos.mean(axis=0)
    orth_mean = orth.mean(axis=0)
    delta_norm_mean = dnorm.mean(axis=0)
    evidence = arrays["l{}_evidence".format(location)].astype(np.float64)

    def corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.square(a).sum() * np.square(b).sum())
        return float((a * b).sum() / denom) if denom > 1e-10 else float("nan")

    evidence_correlations = {}
    direction_targets = {
        "target_alignment": target_alignment,
        "signed_radial": radial_mean,
        "orthogonal_share": orth_mean,
        "delta_norm": delta_norm_mean,
    }
    for evidence_index in range(evidence.shape[1]):
        for target_name, target_values in direction_targets.items():
            evidence_correlations["evidence{}_corr_{}".format(evidence_index, target_name)] = corr(
                evidence[:, evidence_index], target_values)
    shifted_gaps = []
    grid = fg.reshape(8, 8)
    for dy in range(8):
        for dx in range(8):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(grid, shift=(dy, dx), axis=(0, 1)).reshape(64)
            shifted_gaps.append(direction_stats(delta, shifted)[0])

    result = {
        "radial_cos_fg": group_mean(cos, fg),
        "radial_cos_bg": group_mean(cos, bg),
        "abs_radial_fg": group_mean(abs_radial, fg),
        "abs_radial_bg": group_mean(abs_radial, bg),
        "orthogonal_share_fg": group_mean(orth, fg),
        "orthogonal_share_bg": group_mean(orth, bg),
        "direction_gap": direction_gap,
        "fg_direction_coherence": fg_coherence,
        "bg_direction_coherence": bg_coherence,
        "shifted_mask_direction_gap_median": float(np.median(shifted_gaps)),
        "direction_gap_minus_shifted_median": direction_gap - float(np.median(shifted_gaps)),
        "direction_gap_shift_percentile": float(np.mean(np.asarray(shifted_gaps) <= direction_gap)),
    }
    result.update(evidence_correlations)
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cluster_bootstrap(values, rng, iterations):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    draws = rng.randint(0, n, size=(iterations, n))
    means = values[draws].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--events", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    events_path = Path(args.events).resolve() if args.events else source / "events.jsonl"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    event_by_id = {row["event_id"]: row for row in events}
    npz_paths = sorted(source.glob("ECLVR-*.npz"))
    if len(npz_paths) != len(events):
        raise RuntimeError("event/NPZ count mismatch: {} vs {}".format(len(events), len(npz_paths)))

    rows = []
    for path in npz_paths:
        event_id = path.stem
        event = event_by_id[event_id]
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for location in (3, 6, 9):
            for modality in ("rgb", "tir"):
                row = {
                    "event_id": event_id,
                    "sequence": event["sequence"],
                    "frame": event["frame"],
                    "location": location,
                    "modality": modality,
                }
                row.update(event_metrics(arrays, location, modality))
                rows.append(row)

    metrics = [
        "radial_cos_fg", "radial_cos_bg", "abs_radial_fg", "abs_radial_bg",
        "orthogonal_share_fg", "orthogonal_share_bg", "direction_gap",
        "fg_direction_coherence", "bg_direction_coherence",
        "shifted_mask_direction_gap_median", "direction_gap_minus_shifted_median",
        "direction_gap_shift_percentile",
    ]
    evidence_metrics = [
        "evidence{}_corr_{}".format(i, target)
        for i in range(3)
        for target in ("target_alignment", "signed_radial", "orthogonal_share", "delta_norm")
    ]
    metrics.extend(evidence_metrics)
    per_sequence = []
    for location in (3, 6, 9):
        for modality in ("rgb", "tir"):
            subset = [r for r in rows if r["location"] == location and r["modality"] == modality]
            sequences = sorted({r["sequence"] for r in subset})
            for sequence in sequences:
                event_rows = [r for r in subset if r["sequence"] == sequence]
                out = {"sequence": sequence, "location": location, "modality": modality,
                       "event_count": len(event_rows)}
                for metric in metrics:
                    out[metric] = float(np.mean([r[metric] for r in event_rows]))
                per_sequence.append(out)

    rng = np.random.RandomState(args.seed)
    summaries = []
    for location in (3, 6, 9):
        for modality in ("rgb", "tir"):
            seq_rows = [r for r in per_sequence if r["location"] == location and r["modality"] == modality]
            summary = {"location": location, "modality": modality, "sequences": len(seq_rows)}
            for metric in metrics:
                values = [r[metric] for r in seq_rows]
                summary[metric + "_mean"] = float(np.mean(values))
                summary[metric + "_sequence_bootstrap_95ci"] = cluster_bootstrap(values, rng, args.bootstrap)
            gap_values = [r["direction_gap_minus_shifted_median"] for r in seq_rows]
            summary["direction_gap_above_shifted_median_sequences"] = int(sum(x > 0 for x in gap_values))
            summary["direction_gap_above_shifted_median_rate"] = float(np.mean(np.asarray(gap_values) > 0))
            summaries.append(summary)

    write_csv(output / "per_event_direction_metrics.csv", rows)
    write_csv(output / "per_sequence_direction_metrics.csv", per_sequence)
    write_csv(output / "summary.csv", summaries)
    provenance = {
        "input_dir": str(source),
        "events_jsonl": str(events_path),
        "events_sha256": sha256(events_path),
        "npz_count": len(npz_paths),
        "sequence_count": len({row["sequence"] for row in events}),
        "event_count": len(events),
        "bootstrap_unit": "sequence; events averaged within sequence before resampling",
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_seed": args.seed,
        "analysis": "offline CPU only; no model forward, training, or tracking inference",
        "direction_metric": "per-head norm of foreground minus background mean unit-DeltaV vectors, averaged over heads",
        "mask_null": "63 nonzero toroidal shifts of the 8x8 foreground mask; descriptive spatial control, not a formal independent permutation test",
    }
    (output / "analysis_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    for row in summaries:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
