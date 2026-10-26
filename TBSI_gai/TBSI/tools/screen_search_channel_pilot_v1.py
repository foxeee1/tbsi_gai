#!/usr/bin/env python3
"""Evidence gates for the search-stage cross-channel pilot.

This tool evaluates recorded P0 evidence and paired per-sequence proxy results.
It never launches training or tracking and never modifies experiment configs.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


REQUIRED_ARMS = ("C0", "C1", "C2", "C2_GAMMA_OFF")
REQUIRED_SEEDS = (42, 3407)
METRICS = ("auc", "pr", "npr", "op50", "op75")
PROXY_DIR = Path(__file__).resolve().parents[1] / "experiments" / "proxy" / "lasher_rpp_v1"


def finite_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be numeric".format(label))
    if not math.isfinite(number):
        raise ValueError("{} must be finite".format(label))
    return number


def read_json(path):
    with Path(path).open("r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def evaluate_p0(path, tolerance):
    data = read_json(path)
    failures = []
    checks = {
        "parent checkpoint fully loaded": data.get("parent_checkpoint_loaded") is True,
        "actual head input verified": data.get("actual_head_input_verified") is True,
        "adapter called in train and serial inference":
            data.get("called_in_train") is True and data.get("called_in_serial_inference") is True,
    }
    for label, passed in checks.items():
        if not passed:
            failures.append(label)

    for key in ("missing_keys", "unexpected_keys"):
        values = data.get(key)
        if not isinstance(values, list):
            failures.append("{} must be recorded as a list".format(key))
        elif values:
            if key == "missing_keys" and all(str(item).startswith("search_channel.") for item in values):
                pass
            else:
                failures.append("{} contains unexpected entries: {}".format(key, values))

    parity_fields = ("gamma_zero_head_max_abs_diff", "gamma_zero_box_max_abs_diff",
                     "template_max_abs_diff")
    parity = {}
    for key in parity_fields:
        try:
            value = finite_number(data.get(key), key)
            parity[key] = value
            if value > tolerance:
                failures.append("{}={} exceeds tolerance {}".format(key, value, tolerance))
        except ValueError as error:
            failures.append(str(error))

    gradients = data.get("step2_gradient_norms")
    required_gradients = ("qkv", "depthwise", "projection", "gamma_rgb", "gamma_tir")
    if not isinstance(gradients, dict):
        failures.append("step2_gradient_norms must be an object")
    else:
        for name in required_gradients:
            try:
                value = finite_number(gradients.get(name), "step2_gradient_norms.{}".format(name))
                if value <= 0:
                    failures.append("{} gradient norm must be positive".format(name))
            except ValueError as error:
                failures.append(str(error))

    try:
        effect = finite_number(data.get("enabled_output_max_abs_diff"), "enabled_output_max_abs_diff")
        if effect <= tolerance:
            failures.append("enabled adapter has no measurable output effect")
    except ValueError as error:
        failures.append(str(error))

    for key in ("extra_parameters", "peak_memory_mb", "latency_ms", "fps"):
        try:
            if finite_number(data.get(key), key) < 0:
                failures.append("{} cannot be negative".format(key))
        except ValueError as error:
            failures.append(str(error))

    return {
        "gate": "P0_PASS" if not failures else "P0_FAIL",
        "input": str(Path(path)),
        "tolerance": tolerance,
        "parity": parity,
        "failures": failures,
        "scope": "implementation validity only; no training or tracking claim",
    }


def trimmed_mean(values, trim_fraction=0.10):
    ordered = sorted(values)
    cut = int(len(ordered) * trim_fraction)
    core = ordered[cut:len(ordered) - cut] if cut else ordered
    return statistics.mean(core)


def summarize_delta(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "trimmed_mean_10pct": trimmed_mean(values),
        "win_rate": sum(value > 0 for value in values) / float(len(values)),
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
        "catastrophic_loss_le_minus_5": sum(value <= -5.0 for value in values),
        "large_gain_ge_plus_5": sum(value >= 5.0 for value in values),
    }


def load_rows(path):
    rows = {}
    required = {"seed", "sequence", "arm"}.union(METRICS)
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("CSV missing columns: {}".format(", ".join(sorted(missing))))
        for line_number, row in enumerate(reader, 2):
            seed = int(row["seed"])
            arm = row["arm"].strip().upper()
            sequence = row["sequence"].strip()
            if seed not in REQUIRED_SEEDS:
                raise ValueError("line {} has unregistered seed {}".format(line_number, seed))
            if arm not in REQUIRED_ARMS:
                raise ValueError("line {} has unknown arm {}".format(line_number, arm))
            if not sequence:
                raise ValueError("line {} has an empty sequence".format(line_number))
            key = (seed, sequence, arm)
            if key in rows:
                raise ValueError("duplicate row for {}".format(key))
            rows[key] = {metric: finite_number(row[metric], "line {} {}".format(line_number, metric))
                         for metric in METRICS}
    return rows


def evaluate_results(csv_path, rpp_admission_path, output_path, sequence_list_path):
    admission = read_json(rpp_admission_path)
    def admission_number(name):
        try:
            return finite_number(admission.get(name), name)
        except ValueError:
            return None

    anchor_count = admission_number("historical_anchor_count")
    direction_agreement = admission_number("direction_agreement")
    pearson = admission_number("pearson")
    spearman = admission_number("spearman")
    admission_checks = {
        "historical_anchor_count_at_least_4": anchor_count is not None and anchor_count >= 4,
        "direction_agreement_at_least_75pct": direction_agreement is not None and direction_agreement >= 0.75,
        "spearman_at_least_0_60": spearman is not None and spearman >= 0.60,
        "proxy_full_pearson_recorded": pearson is not None,
        "rank_consistency_passed": admission.get("rank_consistency_passed") is True,
        "smsa_above_baseline": admission.get("smsa_above_baseline") is True,
        "old_ctvm_negative_or_later": admission.get("old_ctvm_negative_or_later") is True,
        "serial_parallel_consistency_passed": admission.get("serial_parallel_consistency_passed") is True,
    }
    rows = load_rows(csv_path)
    sequence_path = Path(sequence_list_path)
    frozen_sequences = [line.strip() for line in sequence_path.read_text().splitlines() if line.strip()]
    if len(frozen_sequences) != len(set(frozen_sequences)):
        raise ValueError("frozen sequence list contains duplicates")
    manifest = read_json(PROXY_DIR / "proxy_test_manifest.json")
    sequence_sha256 = hashlib.sha256(sequence_path.read_bytes()).hexdigest()
    if (manifest.get("protocol") != "LasHeR-RPP-v1" or
            manifest.get("sequence_list_sha256") != sequence_sha256 or
            manifest.get("sequences") != frozen_sequences):
        raise ValueError("sequence list does not match the frozen RPP-v1 test manifest")
    frozen_set = set(frozen_sequences)
    observed_seeds = sorted(set(key[0] for key in rows))
    if observed_seeds != list(REQUIRED_SEEDS):
        raise ValueError("results must include exactly the fixed seed pair {}".format(REQUIRED_SEEDS))

    seed_reports = {}
    all_gates_pass = all(admission_checks.values())
    for seed in REQUIRED_SEEDS:
        seq_by_arm = {}
        for arm in REQUIRED_ARMS:
            seq_by_arm[arm] = {sequence: value for (row_seed, sequence, row_arm), value in rows.items()
                               if row_seed == seed and row_arm == arm}
        reference = set(seq_by_arm["C0"])
        if reference != frozen_set:
            raise ValueError("seed {} must contain exactly all {} frozen RPP-v1 test sequences"
                             .format(seed, len(frozen_set)))
        for arm in REQUIRED_ARMS[1:]:
            if set(seq_by_arm[arm]) != reference:
                raise ValueError("seed {} sequence set differs for {}".format(seed, arm))

        comparisons = {}
        comparisons_pass = True
        for arm in ("C0", "C1", "C2_GAMMA_OFF"):
            deltas = [seq_by_arm["C2"][sequence]["auc"] - seq_by_arm[arm][sequence]["auc"]
                      for sequence in sorted(reference)]
            summary = summarize_delta(deltas)
            passed = (summary["mean"] > 0 and summary["median"] > 0 and
                      summary["trimmed_mean_10pct"] > 0 and summary["win_rate"] > 0.5)
            summary["gate"] = "PASS" if passed else "FAIL"
            comparisons["C2_minus_{}".format(arm)] = summary
            comparisons_pass = comparisons_pass and passed

        metrics = {}
        for metric in METRICS:
            metrics[metric] = {}
            for arm in REQUIRED_ARMS:
                metrics[metric][arm] = statistics.mean(
                    seq_by_arm[arm][sequence][metric] for sequence in sorted(reference))
        seed_pass = comparisons_pass and all(admission_checks.values())
        seed_reports[str(seed)] = {
            "sequences": len(reference),
            "mean_metrics": metrics,
            "paired_auc": comparisons,
            "gate": "ELIGIBLE_FOR_FULL_CONFIRMATION" if seed_pass else "STOP_OR_REPAIR",
        }
        all_gates_pass = all_gates_pass and comparisons_pass

    result = {
        "gate": "ELIGIBLE_FOR_FULL_CONFIRMATION" if all_gates_pass else "STOP_OR_REPAIR",
        "scope": "RPP-v1 screening only; not an independent confirmation or paper result",
        "input_csv": str(Path(csv_path)),
        "rpp_admission": str(Path(rpp_admission_path)),
        "frozen_sequence_list": str(sequence_path),
        "frozen_sequence_list_sha256": sequence_sha256,
        "rpp_admission_checks": {key: "PASS" if value else "FAIL" for key, value in admission_checks.items()},
        "rpp_admission_values": {
            "historical_anchor_count": anchor_count,
            "direction_agreement": direction_agreement,
            "pearson": pearson,
            "spearman": spearman,
            "rank_consistency_passed": admission.get("rank_consistency_passed"),
            "smsa_above_baseline": admission.get("smsa_above_baseline"),
            "old_ctvm_negative_or_later": admission.get("old_ctvm_negative_or_later"),
            "serial_parallel_consistency_passed": admission.get("serial_parallel_consistency_passed"),
        },
        "seeds": seed_reports,
        "protocol_note": "All C0/C1/C2 arms must use the same frozen RPP-v1 train/test manifests and full 15-epoch proxy schedule. A 2-3 epoch continuation is not admitted by the current general RPP protocol.",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return all_gates_pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    p0 = subparsers.add_parser("p0", help="evaluate recorded whole-tracker P0 evidence")
    p0.add_argument("--input", required=True, help="P0 evidence JSON")
    p0.add_argument("--output", required=True, help="decision JSON output")
    p0.add_argument("--parity-tolerance", type=float, default=1e-6)
    screen = subparsers.add_parser("results", help="evaluate paired frozen-proxy sequence metrics")
    screen.add_argument("--csv", required=True, help="per-sequence metrics CSV")
    screen.add_argument("--rpp-admission", required=True, help="validated RPP history/consistency admission JSON")
    screen.add_argument("--sequence-list", default=str(PROXY_DIR / "proxy_test_sequences.txt"),
                        help="must match the frozen RPP-v1 list and manifest")
    screen.add_argument("--output", required=True, help="decision JSON output")
    args = parser.parse_args()
    if args.command == "p0":
        result = evaluate_p0(args.input, args.parity_tolerance)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["gate"] == "P0_PASS" else 2
    if args.command == "results":
        return 0 if evaluate_results(args.csv, args.rpp_admission, args.output, args.sequence_list) else 3
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
