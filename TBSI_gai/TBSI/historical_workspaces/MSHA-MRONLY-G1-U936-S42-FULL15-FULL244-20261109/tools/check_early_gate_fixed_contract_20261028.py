#!/usr/bin/env python3
"""Run one real LasHeR update to verify fixed-gate gradient transmission."""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from record_early_gate_fixed_20261028 import run_record
from run_dual_message_p1 import train


if __name__ == '__main__':
    base = ROOT / 'experiments/early_gate_closed_loop_20261026/early_gate_fixed_control_20261028'
    args = SimpleNamespace(
        config=str(ROOT / 'experiments/tbsi_track/EG_EARLY_GATE_FIXED_SOT_QUICK_S42_20261028.yaml'),
        output='/autodl-fs/data/early_gate_fixed_control_20261028/runs/P0',
        seed=42,
        arm='FIXED_P0',
        manifest='/autodl-fs/data/early_gate_closed_loop/dual_message_p1_v2_20261027/data_manifest.json',
        updates=1,
        milestones='1',
        diagnostic_only=True,
        shared_init_snapshot='/autodl-fs/data/early_gate_closed_loop/dual_message_p1_v2_20261027/shared_public_sot_init.pt',
        export_shared_init=None,
    )
    train(args, recorder=run_record)
