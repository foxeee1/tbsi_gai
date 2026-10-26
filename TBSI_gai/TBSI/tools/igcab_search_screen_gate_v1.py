#!/usr/bin/env python3
"""Evaluate recorded evidence for the staged IG-CAB search-only pilot.

This gate evaluator does not edit model code, launch jobs, or create metrics.
All thresholds are supplied before a run and written into its evidence file.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROXY_DIR = ROOT / "experiments" / "proxy" / "lasher_rpp_v1"
SEEDS = (42, 3407)


def read_json(path):
    with Path(path).open("r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be numeric".format(name))
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def write_result(path, result):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def gate_p0(args):
    data = read_json(args.evidence)
    failures = []
    for key in ("parent_checkpoint_loaded", "actual_location9_boundary_verified",
                "template_unchanged", "cpu_fp32_stable", "cuda_fp32_warmed_stable"):
        if data.get(key) is not True:
            failures.append("{} must be true".format(key))
    missing = data.get("missing_keys")
    unexpected = data.get("unexpected_keys")
    if not isinstance(missing, list) or any(not str(key).endswith("igcab_search_adapter") and
                                              ".igcab_search_adapter." not in str(key)
                                              for key in missing):
        failures.append("checkpoint missing keys must be empty or only IG-CAB adapter keys")
    if not isinstance(unexpected, list) or unexpected:
        failures.append("unexpected checkpoint keys must be an empty list")
    calls = data.get("adapter_calls", {})
    if calls.get("train") != 1 or calls.get("serial_inference") != 1:
        failures.append("adapter must be called exactly once in train and serial inference")
    contract = data.get("tensor_contract", {})
    expected_contract = {"lens_z": 64, "search_tokens": 256, "channels": 768,
                         "spatial_height": 16, "spatial_width": 16,
                         "insertion_layer": 9}
    if any(contract.get(key) != expected for key, expected in expected_contract.items()):
        failures.append("tensor_contract does not match the predeclared location-9 contract")
    for key in ("gamma_zero_tracker_max_abs_diff", "gamma_zero_box_max_abs_diff",
                "template_max_abs_diff"):
        try:
            if number(data.get(key), key) > args.parity_tolerance:
                failures.append("{} exceeds parity tolerance".format(key))
        except ValueError as error:
            failures.append(str(error))
    for key in ("baseline_peak_memory_mb", "candidate_peak_memory_mb",
                "baseline_latency_ms", "candidate_latency_ms"):
        try:
            if number(data.get(key), key) <= 0:
                failures.append("{} must be a measured positive value".format(key))
        except ValueError as error:
            failures.append(str(error))
    try:
        extra_mem = (number(data["candidate_peak_memory_mb"], "candidate_peak_memory_mb") -
                     number(data["baseline_peak_memory_mb"], "baseline_peak_memory_mb"))
        latency_pct = 100.0 * (number(data["candidate_latency_ms"], "candidate_latency_ms") /
                               number(data["baseline_latency_ms"], "baseline_latency_ms") - 1.0)
        if extra_mem > args.max_extra_memory_mb:
            failures.append("peak memory increase exceeds declared budget")
        if latency_pct > args.max_latency_overhead_pct:
            failures.append("latency overhead exceeds declared budget")
    except (ValueError, KeyError, ZeroDivisionError) as error:
        failures.append("cannot evaluate resource budget: {}".format(error))
        extra_mem, latency_pct = None, None
    result = {
        "gate": "P0_PASS" if not failures else "P0_FAIL",
        "evidence": str(Path(args.evidence)),
        "declared_limits": {"max_extra_memory_mb": args.max_extra_memory_mb,
                            "max_latency_overhead_pct": args.max_latency_overhead_pct,
                            "parity_tolerance": args.parity_tolerance},
        "observed": {"extra_memory_mb": extra_mem, "latency_overhead_pct": latency_pct},
        "failures": failures,
        "scope": "real-interface and zero-gamma audit only",
    }
    write_result(args.output, result)
    return not failures


def gate_p1(args):
    p0 = read_json(args.p0_gate)
    data = read_json(args.evidence)
    failures = []
    if p0.get("gate") != "P0_PASS":
        failures.append("P0 gate has not passed")
    if data.get("optimizer_steps") != 3:
        failures.append("P1 evidence must contain exactly three actual-loss optimizer steps")
    if abs(number(data.get("gamma_init"), "gamma_init") - 1e-3) > 1e-12:
        failures.append("gamma_init must equal the predeclared 1e-3")
    steps = data.get("steps")
    if not isinstance(steps, list) or len(steps) != 3:
        failures.append("steps must contain three records")
        steps = []
    last_ratio = None
    required_gradients = ("down_rgb", "down_tir", "from_tir", "from_rgb", "up_rgb", "up_tir")
    for index, step in enumerate(steps):
        if step.get("finite_loss") is not True:
            failures.append("step {} loss is not finite".format(index + 1))
        gradients = step.get("gradient_norms", {})
        if index == len(steps) - 1:
            for name in required_gradients:
                try:
                    if number(gradients.get(name), name) <= 0:
                        failures.append("step 3 {} gradient must be nonzero".format(name))
                except ValueError as error:
                    failures.append(str(error))
        for name in ("gamma_rgb", "gamma_tir"):
            try:
                number(step.get("gamma", {}).get(name), "step{}.{}".format(index + 1, name))
            except ValueError as error:
                failures.append(str(error))
        try:
            last_ratio = number(step.get("write_ratio"), "step{}.write_ratio".format(index + 1))
            if not args.min_write_ratio <= last_ratio <= args.max_write_ratio:
                failures.append("step {} write ratio is outside the predeclared interval".format(index + 1))
        except ValueError as error:
            failures.append(str(error))
    result = {
        "gate": "P1_PASS" if not failures else "P1_FAIL",
        "evidence": str(Path(args.evidence)),
        "declared_limits": {"min_write_ratio": args.min_write_ratio,
                            "max_write_ratio": args.max_write_ratio},
        "last_write_ratio": last_ratio,
        "failures": failures,
        "scope": "three-step optimization validity only",
    }
    write_result(args.output, result)
    return not failures


def load_auc_csv(path, arms):
    rows = {}
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"seed", "sequence", "arm", "auc"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("CSV missing columns: {}".format(", ".join(sorted(missing))))
        for line, row in enumerate(reader, 2):
            seed = int(row["seed"])
            arm = row["arm"].strip()
            seq = row["sequence"].strip()
            if seed not in SEEDS or arm not in arms or not seq:
                raise ValueError("invalid seed/arm/sequence at CSV line {}".format(line))
            key = (seed, seq, arm)
            if key in rows:
                raise ValueError("duplicate paired row {}".format(key))
            rows[key] = number(row["auc"], "line {} auc".format(line))
    return rows


def paired_summary(deltas):
    ordered = sorted(deltas)
    trim = int(len(ordered) * 0.10)
    core = ordered[trim:len(ordered) - trim] if trim else ordered
    return {
        "n": len(ordered),
        "mean": statistics.mean(ordered),
        "trimmed_mean_10pct": statistics.mean(core),
        "median": statistics.median(ordered),
        "wins": sum(value > 0 for value in ordered),
        "ties": sum(value == 0 for value in ordered),
        "losses": sum(value < 0 for value in ordered),
        "catastrophic_losses_le_minus_5": sum(value <= -5 for value in ordered),
        "large_gains_ge_plus_5": sum(value >= 5 for value in ordered),
    }


def check_rpp_admission(path):
    data = read_json(path)
    try:
        valid = (int(data.get("historical_anchor_count", 0)) >= 4 and
                 number(data.get("direction_agreement"), "direction_agreement") >= 0.75 and
                 number(data.get("spearman"), "spearman") >= 0.60 and
                 math.isfinite(number(data.get("pearson"), "pearson")) and
                 math.isfinite(number(data.get("catastrophic_regression_detection_rate"),
                                      "catastrophic_regression_detection_rate")) and
                 data.get("rank_consistency_passed") is True and
                 data.get("serial_parallel_consistency_passed") is True and
                 data.get("smsa_above_baseline") is True and
                 data.get("old_ctvm_negative_or_later") is True)
    except (ValueError, TypeError):
        valid = False
    return data, valid


def gate_s1(args):
    p0, p1 = read_json(args.p0_gate), read_json(args.p1_gate)
    admission, admission_ok = check_rpp_admission(args.rpp_admission)
    failures = []
    if p0.get("gate") != "P0_PASS" or p1.get("gate") != "P1_PASS":
        failures.append("P0 and P1 must both pass")
    if not admission_ok:
        failures.append("RPP-v1 historical admission/consistency is absent or failed")
    data = read_json(args.evidence)
    if data.get("epochs") not in (2, 3):
        failures.append("S1 must be the predeclared 2- or 3-epoch diagnostic continuation")
    for key in ("same_m1_epoch15_parent", "same_initialization_seed", "same_optimizer_and_schedule",
                "same_data_order", "all_original_trainable_parameters_enabled"):
        if data.get(key) is not True:
            failures.append("{} must be true".format(key))
    try:
        plain_count = int(data.get("plain_parameter_count"))
        igcab_count = int(data.get("igcab_parameter_count"))
        mismatch = abs(igcab_count - plain_count) / float(max(plain_count, igcab_count))
        if mismatch > 0.05:
            failures.append("Cplain/Cig parameter mismatch exceeds 5%")
    except (TypeError, ValueError, ZeroDivisionError):
        mismatch = None
        failures.append("Cplain and Cig parameter counts are missing or invalid")
    rows = load_auc_csv(args.csv, {"C0", "Cplain", "Cig"})
    frozen = [line.strip() for line in (PROXY_DIR / "proxy_test_sequences.txt").read_text().splitlines()
              if line.strip()]
    manifest = read_json(PROXY_DIR / "proxy_test_manifest.json")
    frozen_path = PROXY_DIR / "proxy_test_sequences.txt"
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    if (manifest.get("protocol") != "LasHeR-RPP-v1" or
            manifest.get("sequences") != frozen or
            manifest.get("sequence_list_sha256") != frozen_hash):
        raise ValueError("frozen proxy sequence file differs from its manifest")
    by_seed = {}
    all_metrics_pass = True
    for seed in SEEDS:
        arm_rows = {arm: {seq: auc for (s, seq, a), auc in rows.items() if s == seed and a == arm}
                    for arm in ("C0", "Cplain", "Cig")}
        if any(set(values) != set(frozen) for values in arm_rows.values()):
            raise ValueError("seed {} must include exactly the frozen RPP-v1 test sequences in all arms"
                             .format(seed))
        comparisons = {}
        for parent, threshold in (("C0", 0.7), ("Cplain", 0.3)):
            summary = paired_summary([arm_rows["Cig"][seq] - arm_rows[parent][seq] for seq in frozen])
            passed = summary["mean"] >= threshold and summary["trimmed_mean_10pct"] > 0
            summary["required_mean"] = threshold
            summary["gate"] = "PASS" if passed else "FAIL"
            comparisons["Cig_minus_{}".format(parent)] = summary
            all_metrics_pass = all_metrics_pass and passed
        catastrophic = max(item["catastrophic_losses_le_minus_5"] for item in comparisons.values())
        if catastrophic > args.max_catastrophic_losses:
            all_metrics_pass = False
            failures.append("seed {} exceeds declared catastrophic-loss count".format(seed))
        by_seed[str(seed)] = {"sequences": len(frozen), "comparisons": comparisons,
                              "catastrophic_loss_limit": args.max_catastrophic_losses}
    if not all_metrics_pass:
        failures.append("one or more per-seed S1 performance thresholds failed")
    result = {
        "gate": "S1_PASS" if not failures else "S1_FAIL",
        "evidence": str(Path(args.evidence)),
        "rpp_admission": str(Path(args.rpp_admission)),
        "rpp_admission_passed": admission_ok,
        "parameter_mismatch_fraction": mismatch,
        "frozen_sequence_list_sha256": frozen_hash,
        "declared_limits": {"Cig_minus_C0_mean_auc_min": 0.7,
                            "Cig_minus_Cplain_mean_auc_min": 0.3,
                            "trimmed_mean_min_exclusive": 0.0,
                            "catastrophic_losses_max": args.max_catastrophic_losses},
        "seeds": by_seed,
        "failures": failures,
        "scope": "developmental RPP-v1 screening only; not independent confirmation",
    }
    write_result(args.output, result)
    return not failures


def gate_s2(args):
    s1 = read_json(args.s1_gate)
    if s1.get("gate") != "S1_PASS":
        raise ValueError("S2 requires a passing S1 gate")
    rows = load_auc_csv(args.csv, {"Cig", "Cig_GAMMA_OFF", "Cig_UNIFORM"})
    frozen_path = PROXY_DIR / "proxy_test_sequences.txt"
    frozen = [line.strip() for line in frozen_path.read_text().splitlines()
              if line.strip()]
    manifest = read_json(PROXY_DIR / "proxy_test_manifest.json")
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    if (manifest.get("protocol") != "LasHeR-RPP-v1" or
            manifest.get("sequences") != frozen or
            manifest.get("sequence_list_sha256") != frozen_hash):
        raise ValueError("frozen proxy sequence file differs from its manifest")
    results = {}
    failures = []
    for seed in SEEDS:
        arm_rows = {arm: {seq: auc for (s, seq, a), auc in rows.items() if s == seed and a == arm}
                    for arm in ("Cig", "Cig_GAMMA_OFF", "Cig_UNIFORM")}
        if any(set(values) != set(frozen) for values in arm_rows.values()):
            raise ValueError("seed {} must include exactly the frozen RPP-v1 test sequences in all arms"
                             .format(seed))
        comparisons = {}
        for perturbation in ("Cig_GAMMA_OFF", "Cig_UNIFORM"):
            summary = paired_summary([arm_rows["Cig"][seq] - arm_rows[perturbation][seq]
                                      for seq in frozen])
            passed = summary["mean"] >= args.min_mean_drop and \
                summary["trimmed_mean_10pct"] >= args.min_trimmed_drop
            summary["gate"] = "PASS" if passed else "FAIL"
            comparisons["Cig_minus_{}".format(perturbation)] = summary
            if not passed:
                failures.append("seed {} {} did not meet declared effect thresholds"
                                .format(seed, perturbation))
        results[str(seed)] = comparisons
    result = {
        "gate": "S2_PASS" if not failures else "S2_FAIL",
        "s1_gate": str(Path(args.s1_gate)),
        "frozen_sequence_list_sha256": frozen_hash,
        "declared_limits": {"mean_drop_min": args.min_mean_drop,
                            "trimmed_drop_min": args.min_trimmed_drop},
        "seeds": results,
        "failures": failures,
        "scope": "mechanism diagnostic; perturbation sensitivity is not causal proof",
    }
    write_result(args.output, result)
    return not failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage")
    p0 = commands.add_parser("p0")
    p0.add_argument("--evidence", required=True)
    p0.add_argument("--output", required=True)
    p0.add_argument("--max-extra-memory-mb", type=float, required=True)
    p0.add_argument("--max-latency-overhead-pct", type=float, required=True)
    p0.add_argument("--parity-tolerance", type=float, default=1e-6)
    p1 = commands.add_parser("p1")
    p1.add_argument("--evidence", required=True)
    p1.add_argument("--p0-gate", required=True)
    p1.add_argument("--output", required=True)
    p1.add_argument("--min-write-ratio", type=float, required=True)
    p1.add_argument("--max-write-ratio", type=float, required=True)
    s1 = commands.add_parser("s1")
    s1.add_argument("--evidence", required=True)
    s1.add_argument("--csv", required=True)
    s1.add_argument("--p0-gate", required=True)
    s1.add_argument("--p1-gate", required=True)
    s1.add_argument("--rpp-admission", required=True)
    s1.add_argument("--output", required=True)
    s1.add_argument("--max-catastrophic-losses", type=int, required=True)
    s2 = commands.add_parser("s2")
    s2.add_argument("--csv", required=True)
    s2.add_argument("--s1-gate", required=True)
    s2.add_argument("--output", required=True)
    s2.add_argument("--min-mean-drop", type=float, required=True)
    s2.add_argument("--min-trimmed-drop", type=float, required=True)
    args = parser.parse_args()
    if args.stage == "p0":
        passed = gate_p0(args)
    elif args.stage == "p1":
        passed = gate_p1(args)
    elif args.stage == "s1":
        passed = gate_s1(args)
    elif args.stage == "s2":
        passed = gate_s2(args)
    else:
        parser.print_help()
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
