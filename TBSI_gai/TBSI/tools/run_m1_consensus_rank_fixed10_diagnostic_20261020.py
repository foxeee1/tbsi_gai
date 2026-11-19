#!/usr/bin/env python3
"""Fixed-state diagnostic for the AMCM residual-ranking candidate."""
import run_m1_amcm_test_failure_structure_diagnostic_20261019 as diag
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
diag.E6_CFG = 'M1_AMCM_CONSENSUS_RESIDUAL_RANK_QUICK15_3EP_20261020'
diag.E6_CKPT = ROOT / (
    'output/experiments/M1_AMCM_CONSENSUS_RESIDUAL_RANK_QUICK15_3EP_20261020/'
    'checkpoints/TBSITrack_ep0003.pth.tar')
diag.OUT = ROOT / (
    'output/experiments/M1_AMCM_CONSENSUS_RESIDUAL_RANK_FIXED10_DIAG_20261020')

if __name__ == '__main__':
    diag.main()
