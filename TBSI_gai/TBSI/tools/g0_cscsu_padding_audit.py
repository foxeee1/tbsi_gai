#!/usr/bin/env python3
"""Record receptive-field padding coverage for the fixed 16x16 SCSU grid."""
import json
from pathlib import Path


def valid_counts(size, kernel):
    pad = kernel // 2
    return [sum(0 <= i < size for i in range(pos - pad, pos + pad + 1))
            for pos in range(size)]


def main():
    size = 16
    rows = {}
    for kernel in (3, 5, 17):
        counts = valid_counts(size, kernel)
        rows[str(kernel)] = {
            "kernel": kernel,
            "valid_counts_per_position": counts,
            "min_valid": min(counts),
            "max_valid": max(counts),
            "mean_valid": sum(counts) / len(counts),
            "mean_valid_fraction": sum(counts) / (len(counts) * kernel),
        }
    result = {
        "grid": [size, size],
        "scsu_k1": 3,
        "scsu_k2": 5,
        "strip_k_formula": "3*k+2",
        "horizontal_and_vertical_strip_kernel": 17,
        "audit": rows,
        "interpretation": (
            "For the fixed 16x16 grid, the k2=5 strip kernel has length 17. "
            "Every output position uses zero padding for at least one kernel slot; "
            "edge positions use 9/17 valid slots and the best interior positions "
            "use 16/17. This is a structural mismatch hypothesis, not a causal proof."
        ),
    }
    out = Path("output/experiments/g0_cscsu_20261005/failure_audit/scscu_padding_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
