#!/usr/bin/env python3
"""Create isolated RPP configs from existing full-run anchors."""
import argparse
from pathlib import Path
import re


def replace_one(text, old, new, label):
    if old not in text:
        raise RuntimeError("missing {} in source config".format(label))
    return text.replace(old, new, 1)


def make(src, dst, epochs):
    text = Path(src).read_text()
    text = text.replace("LasHeR_train", "RPP_LasHeR_train")
    text = text.replace("LasHeR_test", "rpp_lasher_test")
    text = re.sub(r"(SAMPLE_PER_EPOCH:\s*)\d+", r"\g<1>10783", text, count=1)
    text = re.sub(r"(^\s+EPOCH:\s*)\d+", r"\g<1>{}".format(epochs), text, count=1, flags=re.MULTILINE)
    text = re.sub(r"(^TEST:\n(?:.*\n)*?^\s+EPOCH:\s*)\d+", r"\g<1>{}".format(epochs), text, count=1, flags=re.MULTILINE)
    text = "# GENERATED RPP-v1 CONFIG; source={}\n".format(src) + text
    Path(dst).write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="experiments/tbsi_track")
    args = ap.parse_args()
    root = Path(args.config_dir)
    anchors = {
        "v1.0.0-rpp-anchor-baseline-15ep": "v1.0.0-单卡3090-48G-等效全局128基线.yaml",
        "v1.0.7-rpp-anchor-smsa-15ep": "v1.0.7-smsa-only-full-15ep.yaml",
        "v1.0.9-rpp-anchor-ctvm-15ep": "v1.0.9-smsa-ctvm-full-15ep.yaml",
        "v1.1.3-rpp-anchor-sall-15ep": "v1.1.3-sall-full-15ep.yaml",
    }
    for name, source in anchors.items():
        make(root / source, root / (name + ".yaml"), 15)
    # Explicit CTVM pre-screen names are created from the existing SMSA+CTVM
    # implementation. P0 will reject unsupported detach/group variants rather
    # than silently pretending that a config-only change implemented them.
    for name, epochs in [
        ("v1.2.0-ctvm-c0-rpp-3ep", 3),
        ("v1.2.1-ctvm-c1-rpp-3ep", 3),
        ("v1.2.2-ctvm-c2-rpp-3ep", 3),
        ("v1.2.0-ctvm-c0-rpp-15ep", 15),
        ("v1.2.1-ctvm-c1-rpp-15ep", 15),
        ("v1.2.2-ctvm-c2-rpp-15ep", 15),
    ]:
        make(root / "v1.0.9-smsa-ctvm-full-15ep.yaml", root / (name + ".yaml"), epochs)
    print("prepared", len(anchors) + 6, "isolated RPP configs")


if __name__ == "__main__":
    main()
