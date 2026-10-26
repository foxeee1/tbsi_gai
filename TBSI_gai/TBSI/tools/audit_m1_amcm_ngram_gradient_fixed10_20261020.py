#!/usr/bin/env python3
"""Run the existing exact actor-loss gradient audit for the N-gram branch."""
from audit_m1_amcm_lcgm_gradient_fixed10_20261020 import (  # noqa: E402
    main, CFG, CKPT, MANIFEST, OUT,
)
import audit_m1_amcm_lcgm_gradient_fixed10_20261020 as audit
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
audit.CFG = 'M1_AMCM_NGRAM_QUICK15_3EP_20261020'
audit.CKPT = ROOT / 'output/experiments/M1_AMCM_NGRAM_QUICK15_3EP_20261020/checkpoints/TBSITrack_ep0003.pth.tar'
audit.OUT = ROOT / 'output/experiments/M1_AMCM_NGRAM_GRADIENT_AUDIT_20261020'

if __name__ == '__main__':
    audit.main()
