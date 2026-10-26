# Early-Gate Quick3 Control Repeatability

## Decision

**Do not rank another operator from a single quick3 control pair yet.** The seed-42 control repeat produced a `+11.15 pp` quick7 AUC spread even though both runs consumed the same ordered worker/sample/frame trace. Test scheduling is already ruled out by the prior same-checkpoint `threads=0/1` exact-box comparison. This supports short-run training/closed-loop instability, but does not yet separate augmented-input variation from CUDA numerical variation.

The fixed-state BN swap gives a narrower result: BN running statistics materially affect some sequences, but they do not explain the main `ab_bikeoccluded` split. This calibration is diagnostic only; it does not support a performance claim for Early-gate or any candidate operator.

## Contract

- Two fresh Early-conditioned MSHA gate controls, both initialized from `TBSITrack_SOT_Pretrained.pth.tar` (SHA256 `1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b`), seed 42, identical YAML contents (SHA256 `ab3173ad5034df2ec48576ac881649f4dc934d02e19913c09ba70d760e0252ab`).
- LasHeR training: 512 samples/epoch x 3 epochs, batch 16, accumulation 8, 12 optimizer updates, LR `1e-4`, fresh AdamW/StepLR, AMP, worker 4, epoch-3 checkpoint.
- No model/operator, optimizer, seed, schedule, data definition, or deterministic-kernel setting changed. The only source instrumentation was a default-off `TBSI_SAMPLE_TRACE_DIR` hook after processing; it makes no random calls and leaves returned tensors unchanged.
- Both checkpoints were evaluated serially on the same retrospective seven-sequence manifest, FP32, template update off, threads 1. The full manifest has SHA256 `44247b6dbc9259c3e17c61a2d513ca1dff9588ec977946934c3ec4e87bc8d087`.

## Results

Macro means across the fixed seven-sequence development cohort:

| Metric | Repeat A | Repeat B | B - A |
|---|---:|---:|---:|
| AUC / success | 45.51 | 56.65 | +11.15 pp |
| AO | 46.36 | 58.80 | +12.43 pp |
| Precision | 57.73 | 71.72 | +13.99 pp |
| Normalized precision | 56.02 | 69.79 | +13.77 pp |
| OP50 | 56.99 | 71.02 | +14.03 pp |
| OP75 | 39.11 | 49.43 | +10.33 pp |

| Sequence | A AUC | B AUC | B - A |
|---|---:|---:|---:|
| 3rdfatboy | 76.49 | 76.46 | -0.03 pp |
| 4thboywithwhite | 10.99 | 11.20 | +0.21 pp |
| ab_bikeoccluded | 5.96 | 66.05 | +60.08 pp |
| boydownplatform | 74.71 | 75.09 | +0.38 pp |
| leftfarboycomingpicktheball | 54.22 | 54.26 | +0.04 pp |
| shinycarcoming | 37.62 | 38.12 | +0.50 pp |
| womanback2car | 58.55 | 75.39 | +16.84 pp |

The ordered sample trace matched exactly for all 1,664 accepted samples per run: 1,536 training samples and 128 validation samples, with identical worker assignment, sample index, dataset, sequence, template/search frame IDs, and retry count. The records cover selected frames, not post-augmentation tensor bytes.

Epoch-3 training losses were nearly equal: train total `1.11076` vs `1.11085`; validation total `1.59872` vs `1.59937`. Checkpoint model state had 282 tensor keys. 267 keys were not bitwise equal, but the differences were mostly tiny non-buffer updates: maximum absolute non-buffer delta `3.64e-4`, RMS `4.22e-6`. Twenty-six BatchNorm running buffers differed; the largest state difference was `2.45886` in `box_head.conv1_ctr.1.running_var`.

## BN Counterfactual

Only `ab_bikeoccluded` and `womanback2car` were selected for the BN-buffer cross replay. Model parameters stayed with their original run; only 39 BatchNorm running mean/variance and batch-count buffers were replaced by the other checkpoint's values. Evaluation remained FP32, update off, threads 1.

| Sequence | A raw | A parameters + B BN | B raw | B parameters + A BN |
|---|---:|---:|---:|---:|
| ab_bikeoccluded AUC | 5.96 | 5.98 | 66.05 | 66.71 |
| womanback2car AUC | 58.55 | 74.39 | 75.39 | 75.04 |

The BN swap substantially changed A's womanback2car trajectory, while it did not transfer the catastrophic bike outcome between checkpoints. This is evidence that running statistics can contribute in a sequence-dependent way, not that BN is the general cause of the control variance.

## Attribution

- **Excluded for this pair:** different selected sequence/frame identities or worker sample ordering; different quick7 manifest; tracker thread scheduling as a general explanation (same-checkpoint `threads=0/1` replay was previously frame-exact).
- **Supported:** repeat-to-repeat checkpoint and closed-loop prediction variability under the same sampled frame trace. The A/B split begins by frame 21 on `ab_bikeoccluded`; per-frame output and metrics are saved in `snapshots/m1_early_gate_control_repeatability_20261026/`.
- **Partly implicated:** BatchNorm running statistics explain a large change on `womanback2car` under counterfactual swapping, but not the bike split.
- **Still unresolved:** whether random crop/color/flip tensors differed despite matching selected frame IDs, whether CUDA kernels or optimizer updates diverged, and how much tiny weight changes are amplified by the tracking feedback loop.

The close aggregate train/validation losses do not establish numerical reproducibility. Since post-augmentation input tensors were not hashed, this result does not yet isolate a CUDA-kernel cause.

## Next Experiment

Before adding another model operator, run the smallest discriminative repeatability audit: record a hash of the successful post-processing template/search tensors for each sample, alongside the existing index/frame trace, in two identical controls. If input hashes match but weights diverge, isolate deterministic-kernel settings in a new pair. If input hashes differ, audit worker RNG initialization and crop/color/flip sampling first. Keep the model, optimizer, seed, data, and evaluation manifest fixed.

This is one seed, three epochs, and a retrospective seven-sequence cohort. It supports only a run-repeatability diagnosis, not full LasHeR performance or general seed variance.
