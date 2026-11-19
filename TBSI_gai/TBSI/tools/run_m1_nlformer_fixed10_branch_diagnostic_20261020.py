#!/usr/bin/env python3
"""Run the existing fixed-state diagnostic with M1 vs NLformer arms."""
import run_m1_amcm_test_failure_structure_diagnostic_20261019 as diag
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
diag.E6_CFG = 'M1_AMCM_NLFORMER_QUICK15_3EP_20261020'
diag.E6_CKPT = ROOT / 'output/experiments/M1_AMCM_NLFORMER_QUICK15_3EP_20261020/checkpoints/TBSITrack_ep0003.pth.tar'
diag.OUT = ROOT / 'output/experiments/M1_AMCM_NLFORMER_FIXED10_BRANCH_DIAG_20261020'

if __name__ == '__main__':
    diag.main()
