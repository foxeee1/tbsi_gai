#!/usr/bin/env python3
"""Run the orthogonal-uncertainty AMCM candidate on the fixed 10-event cohort."""
import run_m1_candidate_closed_loop_20261020 as runner
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runner.MANIFEST = ROOT / 'output/experiments/M1_AMCM_TMATCH_FIXED10_DIAG_20261019/fixed_event_manifest.json'

if __name__ == '__main__':
    runner.main()
