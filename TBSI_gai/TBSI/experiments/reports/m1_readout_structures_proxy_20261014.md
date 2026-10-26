# M1 Readout Structures Proxy (2026-10-14)

## Protocol

- Parent checkpoint: protected M1 epoch 15, SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Dataset: the same fixed LasHeR union82 manifest used by the prior closed-loop proxy: low-AUC 62 + normal sentinel 20. Manifest SHA256 is recorded in `output/experiments/M1_READOUT_STRUCTURES_PROXY_20261014/contract.json`.
- Inference: FP32, serial `rpp_lasher_test`, same tracker/config/checkpoint for candidate runs 1021-1023. A fresh current-code M1 parent was run as 1024 after no-op auditing exposed a mismatch against historical run 9904.
- Three candidate structures: S1 independent offset/size utility admission; S2 S1 plus peak-margin and normalized-response-entropy reliability admission; S3 S2 plus two-frame confirmation and release.
- No training was performed. Existing train-side fitted bounded predictor parameters were reused.

## Paired Results

All deltas below are sequence-macro metric differences against the fresh current-code M1 parent run 1024. The prior run 9904 and existing bounded predictor run 9915 are retained in the matched summary and per-sequence CSV for reference.

| Structure | Cohort | AUC | PR | NPR | OP50 | OP75 | AUC wins/losses/ties |
|---|---|---:|---:|---:|---:|---:|---:|
| S1 independent actions | Low-AUC 62 | +3.73 | +5.11 | +4.70 | +5.08 | +3.06 | 38/23/1 |
| S1 independent actions | Normal 20 | +0.03 | +0.02 | +0.08 | +0.01 | -0.02 | 13/7/0 |
| S1 independent actions | Union 82 | +2.83 | +3.87 | +3.57 | +3.84 | +2.31 | 51/30/1 |
| S2 reliability gate | Low-AUC 62 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0/0/62 |
| S2 reliability gate | Normal 20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0/0/20 |
| S2 reliability gate | Union 82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0/0/82 |
| S3 hysteresis | Low-AUC 62 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0/0/62 |
| S3 hysteresis | Normal 20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0/0/20 |
| S3 hysteresis | Union 82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0/0/82 |

S1 also modestly exceeds the prior bounded predictor on this same cohort: union82 sequence-macro deltas are +0.08 AUC, +0.20 PR, +0.27 NPR, +0.34 OP50, and +0.06 OP75. Paired AUC improved on 41 sequences, declined on 38, and tied on 3.

## Action Audit and Attribution

- S1 processed 68,298 frames: 37,648 offset actions and 30,650 no-ops. It triggered on 37,156/62,434 low-AUC frames (59.5%) and 492/5,864 normal frames (8.4%). Size and joint actions were both zero. The nominal offset/size decoupling therefore exercised only the offset branch.
- S2 and S3 recorded 68,298 no-op frames each. Every one of their 82 prediction files is byte-identical to current-code M1 run 1024. Their zero deltas are verified no-op parity, not evidence that reliability gating or temporal hysteresis is ineffective.
- The normalized response entropy observed in this run was generally above 0.999, while S2/S3 required `<= 0.92`. This gate rejected every action. The threshold was not calibrated on the test cohort, and must not be tuned on these test results.
- S3 had no admitted action to confirm or release. Temporal hysteresis remains untested as a behavior, and the run does not test recovery from already accumulated tracker-state drift.
- The historical M1 run 9904 differs from the fresh current-code parent run 1024: current run 1024 is +0.52 union AUC relative to 9904. S2/S3 exact no-op parity with 1024 establishes the current-code causal comparison; causes of the historical run difference remain unresolved. Both comparison surfaces are preserved.

## Sequence-Level Changes

S1 AUC changes versus current-code M1 on low-AUC 62 were 38 wins, 23 losses, and 1 tie. On normal 20 they were 13 wins and 7 losses. The largest low-AUC improvements were `ab_bikeoccluded` (+57.71), `3bike1` (+56.28), `leftfarboycomingpicktheball` (+49.55), `right2ndflagformath` (+27.83), and `runningcameragirl` (+22.96). The largest declines were `shinycarcoming` (-14.54), `redmidboy` (-12.12), `rightbottlecomes` (-10.16), `boytakingbasketballfollowing` (-9.34), and `rightcomingstrongboy` (-3.48).

The complete 82-sequence comparison, including AUC/PR/NPR/OP50/OP75 deltas for M1 run 1024, historical M1 run 9904, old predictor run 9915, and S1-S3, is in `output/experiments/M1_READOUT_STRUCTURES_PROXY_20261014/per_sequence_matched_current_parent.csv`. The earlier historical-parent comparison is retained in `per_sequence_all_arms.csv`.

## Decision

`PROXY-COMPLETE / S1-PROMISING-OFFSET-ONLY / S2-ALL-NOOP / S3-NOT-EXERCISED / NO-FULL-TRAIN-YET`.

The current evidence supports another train-side calibration and independent size-action diagnostic before any full training. Preserve S1 as a promising single-seed proxy result; do not describe it as a verified full-dataset gain. Rebuild reliability calibration from train-side response statistics, define entropy using a scale that separates useful and unhelpful states, and run a small action-coverage check before another union82 evaluation. Keep S3 out of the evidence for or against hysteresis until its gate admits actions.

## Evidence

- Contract, logs, per-frame actions, summaries, and outputs: `output/experiments/M1_READOUT_STRUCTURES_PROXY_20261014/`.
- Candidate result directories: `output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_{1021,1022,1023}/`.
- Fresh parent run 1024 and paired reanalysis: `output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_1024/` and `output/experiments/M1_READOUT_STRUCTURES_PROXY_20261014/matched_current_parent_summary.json`.
- Tracker code snapshot before this experiment: `experiments/snapshots/tbsi_track_before_readout_structures_20261014.py`.

Hardware was 1x NVIDIA GeForce RTX 4090 D; environment was Python 3.8.20, PyTorch 2.0.0+cu118, CUDA 11.8. Candidate arm wall times were 34.1, 33.7, and 32.3 minutes. This is an 82-sequence proxy with one seed, not a full-244 result.
