# M1 adaptive post-fusion router P0: normal + low-AUC cohort (2026-10-13)

## Question and decision contract

Does the one-epoch P0 adaptive post-fusion router preserve performance on normal sequences while improving M1's selected low-AUC sequences? This is a matched current-source closed-loop comparison, not an archive parity run or a full LasHeR test result.

The fixed cohort is 20 normal-control sequences followed by 62 sequences selected as archived-M1 AUC < 40. The two lists are disjoint. Both arms use exactly the same ordered 82-sequence manifest, tracker/evaluator code, FP32 inference, one tracker process, and independent output directories. M1 uses its protected epoch-15 checkpoint; P0 uses the quick epoch-1 checkpoint. No parameters or thresholds were changed after cohort selection.

## Results

All metrics are sequence-macro percentages from the project's standard evaluator. Delta is P0 minus M1, computed before display rounding. AUC win/loss/tie compares exact per-sequence AUC, with ties at absolute difference <= 1e-12; displayed condition values may not subtract exactly to the displayed delta due to rounding.

| Cohort | Condition | AUC | PR | NPR | OP50 | OP75 | AUC W/L/T |
|---|---|---:|---:|---:|---:|---:|---:|
| Normal control (20) | M1 | 85.83 | 98.42 | 98.78 | 98.93 | 94.65 | Baseline |
| Normal control (20) | P0 quick | 85.84 | 98.45 | 98.83 | 98.96 | 94.62 | 6/6/8 |
| Normal control (20) | Delta | +0.01 | +0.03 | +0.05 | +0.02 | -0.04 | |
| Archived-M1 low-AUC (62) | M1 | 22.85 | 28.74 | 25.54 | 26.31 | 14.17 | Baseline |
| Archived-M1 low-AUC (62) | P0 quick | 23.21 | 29.08 | 26.01 | 26.90 | 14.38 | 19/24/19 |
| Archived-M1 low-AUC (62) | Delta | +0.37 | +0.33 | +0.47 | +0.59 | +0.22 | |
| Union (82) | M1 | 38.21 | 45.74 | 43.41 | 44.03 | 33.80 | Baseline |
| Union (82) | P0 quick | 38.49 | 46.00 | 43.77 | 44.48 | 33.95 | 25/30/27 |
| Union (82) | Delta | +0.28 | +0.26 | +0.37 | +0.45 | +0.15 | |

Sequence-level paired bootstrap (10,000 resamples; fixed seed `20261013`) gives the following 95% intervals for AUC delta:

| Cohort | Mean AUC delta | 95% paired sequence-bootstrap CI |
|---|---:|---:|
| Normal control (20) | +0.008 | [-0.038, +0.058] |
| Archived-M1 low-AUC (62) | +0.367 | [-0.789, +1.817] |
| Union (82) | +0.280 | [-0.592, +1.342] |

All intervals include zero. The apparent low-AUC mean gain is heterogeneous: largest gains are `runningcameragirl` (+31.47 AUC), `shinycarcoming` (+13.66), and `redmidboy` (+12.01); largest drops are `rightbottlecomes` (-11.19), `whiteboyrightcoccergoal` (-10.19), and `foldedfolderatlefthand` (-6.82). These per-sequence extremes are descriptive, not confirmatory.

## Interpretation

- **Normal-sequence retention:** supported only as a small-cohort non-degradation observation. Mean metrics move by less than 0.05 points; the AUC interval includes zero. This does not prove that normal sequences are protected generally.
- **Low-AUC benefit:** not established. Mean metrics are positive, but P0 loses on more sequences than it wins (19 wins, 24 losses, 19 ties), and the paired interval is wide and includes zero. A few very large improvements coexist with material regressions.
- **Combined result:** exploratory positive mean deltas (+0.28 AUC, +0.26 PR, +0.37 NPR, +0.45 OP50, +0.15 OP75), all statistically inconclusive at the sequence level.
- **Runtime:** tracker-reported FPS was 55.81 for M1 and 52.78 for P0, a 5.4% reduction for P0 in this run. This is a descriptive single-run comparison.
- **Research conclusion:** `INCONCLUSIVE / supports a larger, properly trained single-variable validation only`. P0 was trained for one epoch on 16 train samples using ordinary tracking loss, not the intended utility/no-op supervision. The present result cannot establish that adaptive routing caused the mean gain, that its need gate protects good sequences, or that the effect generalizes. Do not tune P0 on these test cohorts.

## Limits and remaining confounds

The 62-sequence cohort was selected using archived M1 performance, so it is a targeted stress set, not an independent random sample. This run replays the fixed M1 weight under current working-tree code; it is directly comparable to P0 here, but not a strict reproduction of the archive's per-frame predictions. The normal cohort has only 20 sequences. There is one M1 seed and one tiny P0 training run. Sequence-bootstrap intervals treat sequences as sampling units but do not cover training-seed variation or selection uncertainty. Therefore the cleanest supported statement is: **P0 showed near-zero change on this normal-20 cohort and a small, inconsistent positive mean on the selected low-AUC-62 cohort; neither effect is statistically resolved.**

The intervention changes only the post-fusion adaptive router. MHSA, the CENTER head, tracker state update, and evaluation definitions were not modified for this comparison. Because the training run is tiny and not utility/no-op supervised, attribution to the learned gate itself remains unresolved.

## Provenance and commands

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15, seed 42.
- Git HEAD: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`; relevant code/config snapshot SHA256: `395e4cd99da9ef7cb5ef7e70641b5ce39bf4e8631acc652cf315c2fbf0f8972f`.
- M1 YAML SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e` (unchanged).
- M1 checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef` (unchanged).
- P0 quick YAML SHA256: `d4d5b98bf6483a46dc8529eb523d075d689124fb1d036616a3795714b54e999e`.
- P0 quick checkpoint SHA256: `be382cacc364d8a82647159f3c98073a53552ea15daf5e4d3bda0a2aea2bcb35`.
- Union manifest SHA256: `817660efc89284460341f2ce52a31c3758e366e3bf73638fd20fb9beac0e531a`; normal manifest SHA256: `635905661a3be4ea8e9a9a5a28c01268d36137800c0a73b21f2b83b884023a2b`; low-AUC manifest SHA256: `99c07aabc0d9b6891cf848622bd7238f47d1bdf432edfc0072e409900b96c6a9`.
- Environment: 1x NVIDIA GeForce RTX 4090 D 24 GiB; Python 3.8.20; PyTorch 2.0.0+cu118; CUDA 11.8; FP32. M1 evaluation 59m24s; P0 evaluation 59m19s; approximately 0.99 GPU-hours each.
- M1 command: `TBSI_RPP_TEST_LIST=experiments/manifests/m1_normal20_lowauc62_union_20261013.txt TBSI_INFER_FP16=0 TBSI_CHECKPOINT_OVERRIDE=output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar python3.8 tracking/test.py tbsi_track TBSI_E_MHSA_FULL_4090D_seed42 --runid 9904 --dataset_name rpp_lasher_test --threads 0 --num_gpus 1`.
- P0 command: same command and environment, with parameter `M1_ADAPTIVE_POST_ROUTER_P0_QUICK_seed42`, runid `9905`, and checkpoint `output/experiments/M1_ADAPTIVE_POST_ROUTER_P0_QUICK_V2_20261013/experiments/M1_ADAPTIVE_POST_ROUTER_P0_QUICK_seed42/checkpoints/TBSITrack_ep0001.pth.tar`.
- Standard evaluator commands: `python3.8 tracking/analysis_results.py --tracker_name tbsi_track --tracker_param <M1-or-P0-config> --dataset_name rpp_lasher_test --runid <9904-or-9905>`, with the corresponding cohort manifest in `TBSI_RPP_TEST_LIST`.
- Analysis script: `experiments/tools/analyze_m1_adaptive_post_router_union82.py`; outputs: `output/experiments/M1_ADAPTIVE_POST_ROUTER_P0_UNION82_20261013/paired_analysis/`.
- Tracking results: `output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_9904/` and `output/test/tracking_results/tbsi_track/M1_ADAPTIVE_POST_ROUTER_P0_QUICK_seed42_9905/`; both contain 82/82 prediction files.
- Full command logs are in `output/test/logs/m1_adaptive_router_p0_union82_*`.

## Next action

Do not expand this into another router variant or tune on these 82 sequences. If the route continues, train the same frozen-M1 router with the intended train-side utility/no-op objective and a sequence-disjoint held-out cohort, then run one preregistered matched comparison containing both normal and difficult strata. Keep a no-op/identity control and report paired per-sequence uncertainty and runtime.
