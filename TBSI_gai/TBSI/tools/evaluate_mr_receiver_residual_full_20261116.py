#!/usr/bin/env python3
"""Evaluate one full MR residual arm on all LasHeR test sequences."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from run_dual_message_p1 import evaluate
from run_mr_receiver_residual_full_20261116 import CANDIDATE_CONFIG, CONTROL_CONFIG, MANIFEST


def main():
    p = argparse.ArgumentParser(); p.add_argument('arm', choices=['candidate', 'control']); p.add_argument('--output', required=True); p.add_argument('--checkpoint', required=True)
    a = p.parse_args()
    a.config = str(CANDIDATE_CONFIG if a.arm == 'candidate' else CONTROL_CONFIG)
    a.seed = 42; a.arm = a.arm; a.manifest = str(MANIFEST); a.list = None; a.prefetch_workers = 0
    evaluate(a)


if __name__ == '__main__': main()
