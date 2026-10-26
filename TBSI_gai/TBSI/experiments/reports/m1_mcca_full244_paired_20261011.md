# M1 MCCA One-Time Full-244 Paired Evaluation (2026-10-11)

## Decision and scope

Following the fresh96 continuation-holdout result, the user requested one full-test paired confirmation. This is a one-time evaluation of the existing matched G0 and bidirectional G1 epoch-3 checkpoints, not k-fold cross-validation and not a new training run. The LasHeR test-244 set is already an inspected test set; this evaluation must not be used to tune MCCA or select another method. No other arm is included.

## Locked comparison

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, explicit epoch-15 checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- G0: `M1_MCCA_G0_MATCHED_seed42`, explicit epoch-3 checkpoint SHA256 `d0fdb062f05217a388d82d6f1eda9523ac588f1f738f9ba754d53f6efc1f5e2f`.
- G1: `M1_MCCA_G1_BIDIR_seed42`, explicit epoch-3 checkpoint SHA256 `3595f7da6e1634cc2a506d6f1eda9525bb20a88f8dae55c0d2ca48fd51f8db498`.
- Same official `testingsetList.txt`; tracker excludes the registered corrupt/incomplete sequences and must resolve exactly 244 unique sequences. Dataset list SHA256: `16eef326975aacccb94a7e133e8ef0aceb734ece4264caf6f82bc94442c20e46`.
- Serial single-GPU FP32 closed-loop inference, `threads=0`, no retraining. New run IDs: G0 `810`, G1 `811`. Paired sequence bootstrap: 50,000 resamples, seed `20261011`.
- Output: `output/experiments/M1_MCCA_FULL244_PAIRED_20261011/`; tracker predictions are stored under the corresponding unique run IDs.

## Pre-run storage cleanup

Three explicitly stopped LCGA/Strip epoch-3 proxy checkpoints were removed after recording byte size and SHA256. Total reclaimed: 2,319,173,731 bytes. Their reports, logs, configs, manifests, predictions and metrics remain. No protected M1, 4090D reproduction-group, MCCA G0/G1, or status-ambiguous DBF/TAM checkpoint was removed. Full list: `experiments/reports/m1_stopped_local_candidates_checkpoint_cleanup_20261011.md`.

## Results

Both arms completed 244/244 sequences with saved predictions, timing files, and per-sequence metrics. The G0 then G1 serial order was preserved.

### Three-way view including the fixed M1 parent

The protected M1 epoch-15 benchmark is added below as a descriptive reference. All three rows report the same official LasHeR full-244 benchmark metrics; M1 values are from the registered parent result (`56.2077 / 70.2073 / 66.1980 / 68.3339 / 47.1349`).

| Arm | Training/checkpoint meaning | AUC | PR | NPR | OP50 | OP75 | Delta AUC vs M1 | Delta PR vs M1 | Delta NPR vs M1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | Protected epoch-15 checkpoint; no continuation | 56.2077 | 70.2073 | 66.1980 | 68.3339 | 47.1349 | 0.0000 | 0.0000 | 0.0000 |
| G0 | Same M1 initialization, then 3-epoch matched continuation with MCCA off | 55.8083 | 69.6715 | 65.8821 | 67.8442 | 46.8714 | -0.3994 | -0.5358 | -0.3159 |
| G1 | Same M1 initialization and 3-epoch continuation, with bidirectional MCCA | 56.2571 | 70.1067 | 66.4096 | 68.3041 | 47.2272 | +0.0494 | -0.1006 | +0.2116 |

This resolves the missing-baseline view, but does not make M1 a matched training control: M1 is the uncontinued epoch-15 checkpoint, while G0/G1 received three extra epochs on the registered train176 subset. The clean MCCA estimate within this experiment remains G1 minus G0: `+0.4488` AUC, `+0.4352` PR, `+0.5275` NPR; all paired bootstrap 95% intervals cross zero. Relative to the original M1, G1 is effectively flat on AUC (`+0.0494`) and slightly lower on PR (`-0.1006`). Thus the defensible reading is that MCCA may offset some degradation from this particular continuation, not that it has demonstrated a reliable gain over M1.

| Arm | AUC | PR | NPR | OP50 | OP75 | Evaluator FPS |
|---|---:|---:|---:|---:|---:|---:|
| G0 matched continuation | 55.8083 | 69.6715 | 65.8821 | 67.8442 | 46.8714 | 71.5524 |
| G1 bidirectional MCCA | 56.2571 | 70.1067 | 66.4096 | 68.3041 | 47.2272 | 67.3948 |

Paired G1-G0 changes over 244 sequences: AUC `+0.4488` (135 wins / 109 losses; median `+0.0776`), PR `+0.4352` (99/85/60 wins/losses/ties), NPR `+0.5275` (110/84/50). Sequence bootstrap with 50,000 resamples and seed `20261011` gives 95% intervals AUC `[-0.5310, 1.4521]`, PR `[-0.7809, 1.6730]`, and NPR `[-0.6464, 1.7420]`; all cross zero. Removing the largest AUC gain leaves `+0.6956`, so the positive mean is not explained by one maximum-gain sequence alone, but uncertainty remains broad.

The single-pass reported FPS is 71.5524 for G0 and 67.3948 for G1 (-5.81%). Summed per-frame tracker timings give 69.91 and 65.94 FPS respectively (219,292 timed frames each). Since arms were run sequentially rather than in alternating repeated rounds, this is an observed run-level difference, not a stable causal overhead estimate. Wall time was approximately 1:51:26 for G0 and 1:53:34 for G1; summed tracker timing is 1.795 hours, excluding setup and evaluator overhead.

## Decision

The full-244 point estimates are directionally positive, unlike the negative fresh96 fixed-checkpoint attribution, but the paired confidence intervals cross zero and the median AUC difference is small. This is not a robust confirmation of MCCA and does not overturn the failed promotion gate. Record the result as a mixed, uncertain single-seed outcome; stop the current MCCA candidate and do not tune on test-244. This fixed-checkpoint comparison is not k-fold cross-validation, not a new training replication, and does not establish multi-seed robustness. The three-way table is descriptive; only G1-vs-G0 controls for the three-epoch continuation, while G1-vs-M1 includes continuation effects.

## Execution provenance

- Runner: `tools/run_m1_mcca_full244_paired_20261011.sh`, SHA256 `0fdc27e705cec189b56d12df86b0571155e9376c2909361d56d3ceb7fe1eda75`.
- Analyzer: `tools/analyze_mcca_full244_paired.py`, SHA256 `795d80a40e9da8f3288ed4443c732a2e813dab35f94eabf939e547bdba17845c`.
- Metrics: `output/experiments/M1_MCCA_FULL244_PAIRED_20261011/metrics/G0.json`, `G1.json`, and `paired_comparison.json`. Logs and 244-member manifest are in the same run directory. Predictions: `output/test/tracking_results/tbsi_track/M1_MCCA_G0_MATCHED_seed42_810/` and `M1_MCCA_G1_BIDIR_seed42_811/`.
- Environment: 1x NVIDIA GeForce RTX 4090 D 24 GiB; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1; FP32, tracker `threads=0`.
- Commands per arm: `python -u tracking/test.py tbsi_track <config> --dataset_name lasher_test --threads 0 --num_gpus 1 --runid <810|811>`, then `python -u tools/analyze_dbf_short_eval.py --config <config> --runid <810|811> --dataset-name lasher_test --output <metric-json>`. Final analysis: `python -u tools/analyze_mcca_full244_paired.py --g0 .../G0.json --g1 .../G1.json --manifest .../test_manifest.json --output .../paired_comparison.json`.
- Preflight incident: the first launcher invocation stopped before inference because the G1 checkpoint hash literal had a typo. The checkpoint matched the registry; the literal was corrected and both protected M1 hashes and both MCCA checkpoint hashes passed before the successful run. No partial predictions were generated by the failed invocation.
- Cleanup before this run: three stopped LCGA/Strip candidate checkpoints were removed after hash capture; maintenance evidence is `experiments/reports/m1_stopped_local_candidates_checkpoint_cleanup_20261011.md`. All protected anchors and MCCA comparison weights were retained.
