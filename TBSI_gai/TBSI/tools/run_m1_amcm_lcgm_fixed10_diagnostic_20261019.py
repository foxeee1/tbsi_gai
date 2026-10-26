#!/usr/bin/env python3
"""Paired fixed-state LCGM attribution on the ten M1 failure frames."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / 'run_m1_amcm_test_failure_structure_diagnostic_20261019.py').read_text()
source = source.replace("E6_CFG = 'M1_AMCM_DR_LOSS_E6_REPR_QUICK15_3EP_20261019'",
                        "E6_CFG = 'M1_AMCM_LCGM_QUICK15_3EP_20261019'")
source = source.replace("E6_CKPT = ROOT / 'output/experiments/M1_AMCM_DR_LOSS_E6_REPR_QUICK15_3EP_20261019/checkpoints/TBSITrack_ep0003.pth.tar'",
                        "E6_CKPT = ROOT / 'output/experiments/M1_AMCM_LCGM_QUICK15_3EP_20261019/checkpoints/TBSITrack_ep0003.pth.tar'")
source = source.replace("OUT = ROOT / 'output/experiments/M1_AMCM_TEST_FAILURE_STRUCTURE_DIAG_V2_20261019'",
                        "OUT = ROOT / 'output/experiments/M1_AMCM_LCGM_FIXED10_DIAG_20261019'")
source = source.replace("rows += run_arm('e6_amcm', E6_CFG, E6_CKPT, seqmap, events)",
                        "rows += run_arm('lcgm_amcm', E6_CFG, E6_CKPT, seqmap, events)")
source = source.replace("r['arm'] == 'e6_amcm'", "r['arm'] == 'lcgm_amcm'")
source = source.replace("'e6_amcm_iou': b['fixed_state_iou']", "'lcgm_amcm_iou': b['fixed_state_iou']")
source = source.replace("'e6_amcm_gt_rank': m1_rank", "'lcgm_amcm_gt_rank': m1_rank")
source = source.replace("'e6_amcm_gt_rank': e6_rank", "'lcgm_amcm_gt_rank': e6_rank")
source = source.replace("'e6_amcm_margin': e6_gt - e6_bg", "'lcgm_amcm_margin': e6_gt - e6_bg")
source = source.replace("'e6_gt_peak_score': e6_gt, 'e6_bg_peak_score': e6_bg", "'lcgm_gt_peak_score': e6_gt, 'lcgm_bg_peak_score': e6_bg")
source = source.replace("'mean_e6_amcm_fixed_iou'", "'mean_lcgm_fixed_iou'")
source = source.replace("'mean_e6_amcm_fixed_iou': float(np.mean([p['e6_amcm_iou'] for p in pairs]))",
                        "'mean_lcgm_fixed_iou': float(np.mean([p['lcgm_amcm_iou'] for p in pairs]))")
source = source.replace("'e6_amcm_fixed_iou'", "'lcgm_amcm_fixed_iou'")
source = source.replace("'e6_amcm'", "'lcgm_amcm'")
source = source.replace("e6_amcm_iou", "lcgm_amcm_iou")
source = source.replace("e6_amcm_gt_rank", "lcgm_amcm_gt_rank")
source = source.replace("e6_amcm_margin", "lcgm_amcm_margin")

# The forward diagnostic captures exact AMCM and fusion tensors. Gradient
# direction is computed in a separate training-contract probe because tracker
# inference intentionally runs without autograd.
exec(compile(source, str(HERE / 'run_m1_amcm_test_failure_structure_diagnostic_20261019.py'), 'exec'))
