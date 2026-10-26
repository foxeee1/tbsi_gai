#!/usr/bin/env python3
"""Exact actor-loss gradient audit for N-gram plus representation margin."""
import audit_m1_amcm_lcgm_gradient_fixed10_20261020 as audit
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
audit.CFG = 'M1_AMCM_NGRAM_MARGIN_QUICK15_3EP_20261020'
audit.CKPT = ROOT / 'output/experiments/M1_AMCM_NGRAM_MARGIN_QUICK15_3EP_20261020/checkpoints/TBSITrack_ep0003.pth.tar'
audit.OUT = ROOT / 'output/experiments/M1_AMCM_NGRAM_MARGIN_GRADIENT_AUDIT_20261020'

if __name__ == '__main__':
    audit.main()
