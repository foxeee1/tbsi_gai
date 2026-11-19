#!/usr/bin/env python3
"""Run the fixed-state residual/fusion structure audit for N-gram."""
import run_m1_amcm_test_failure_structure_diagnostic_20261019 as audit
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
audit.E6_CFG = 'M1_AMCM_NGRAM_QUICK15_3EP_20261020'
audit.E6_CKPT = ROOT / 'output/experiments/M1_AMCM_NGRAM_QUICK15_3EP_20261020/checkpoints/TBSITrack_ep0003.pth.tar'
audit.OUT = ROOT / 'output/experiments/M1_AMCM_NGRAM_STRUCTURE_AUDIT_20261020'

if __name__ == '__main__':
    audit.main()
