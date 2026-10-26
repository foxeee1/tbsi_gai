# CFVI parent hot-start P2 analysis

Date: 2026-10-01  
Scope: RPP-v1 frozen proxy-60, serial FP32 tracker test, seed 42  
Parent: `TBSI-E-MHSA-FULL-20260927`, epoch-15 checkpoint  
Config: `CFVI_P2_RAW_VALUE_HOTSTART_seed42`  

## Results

| Condition | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| P0 parent | 51.52 | 62.71 | 40.79 | 65.82 | 60.88 |
| P1 fixed Haar-CFVI | 51.64 | 62.88 | 40.98 | 66.15 | 61.22 |
| P2 raw Value adapter | 51.29 | 62.45 | 40.70 | 65.57 | 60.64 |
| P2 - P0 | -0.23 | -0.26 | -0.09 | -0.25 | -0.24 |
| P2 - P1 | -0.35 | -0.43 | -0.28 | -0.58 | -0.58 |

P2 completed 3 training epochs and 60/60 proxy sequences. The epoch-3 checkpoint was moved to `output/.trash_cfvi_completed/` after evaluation to release space. No M2 or full-244 run was performed.

## Paired sequence analysis

For sequence-level AUC, P2 minus P0 had mean `-0.002272`, median `+0.000190`, win rate `53.3%`, loss rate `46.7%`, and bootstrap 95% CI `[-0.011260, 0.006096]`. The trimmed mean after removing the three largest gains and losses was `+0.000668`. The largest P2 gains were concentrated in `midrunboywithwhite` (`+0.1225`) and `boyaroundtrees` (`+0.0470`), while the largest losses included `ab_whiteboywithbluebag` (`-0.1498`) and `carcomingfromlight` (`-0.1468`).

Relative to P1, P1 minus P2 had mean AUC `+0.003429`, median `+0.000064`, and a 50.0% win rate for each condition. Thus P2 does not outperform the Haar adapter and has lower aggregate metrics than both P0 and P1.

## Interpretation

The raw cross-modal Value adapter is a matched negative control for the Haar design. In this single-seed, 3-epoch proxy screening protocol, it provides no positive evidence of a general Value-injection gain and does not support promoting P2 or the Value-injection route to full confirmation. P1 remains only diagnostic-positive-inconclusive; the P1 checkpoint audit found nonzero lambdas (`0.1111` RGB and `0.1051` TIR), but event-level injection `rho` was not recorded. These results do not establish an independent Haar contribution or justify M2.

## Evidence

- P2 metrics: `output/experiments/cfvi_parent_raw_value_20261001/P2/analysis/eval_data.pkl`
- P2 serial log: `output/experiments/cfvi_parent_raw_value_20261001/serial.log`
- P1 contribution audit: `output/experiments/cfvi_parent_raw_value_20261001/P1_audit/injection_ratio_summary.json`
- P0/P1/P2 configs and parent checkpoint are recorded in the experiment registry.
