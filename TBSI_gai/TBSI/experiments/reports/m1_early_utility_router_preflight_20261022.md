# M1 Early Utility Router Preflight (2026-10-22)

## Decision

`CALIBRATION-GATE-FAILED / DO NOT START THE 1,264-UPDATE SCREEN` under the frozen candidate rule. A temporary 32-update router fit reduced calibration utility regret relative to random modality selection, but did not beat the constant class-prior predictor on utility cross-entropy. The candidate has not been tracking-trained or evaluated for AUC, OP50, OP75, precision, sequence metrics, or runtime.

## Candidate contract

The candidate predicts crop-level RGB/TIR utility from block02 features: RGB/TIR template foreground, template background, and search descriptors, plus four cross-modal/same-modal cosine relations. One shared router emits two logits. `softmax` probabilities become bounded multiplicative scales `exp(0.25 * (2p - 1))`, broadcast uniformly over each MSHA template-token axis and used only on cross-modal Value messages at loc3/6/9. Uniform logits produce exact scale 1. CENTER, search fusion, template state, and tracker update are not changed.

Counterfactual RGB-only/TIR-only CENTER outputs provide detached utility labels from GT-region maximum minus strongest non-GT response. The only added loss is `0.10 * CE(router_logits, argmax(utility_rgb, utility_tir))`; existing tracking losses remain unchanged. For the preflight fit, all M1 parameters are frozen and only 38,882 router parameters are trainable. No preflight weights were persisted.

## Implementation checks

The final preflight used protected M1 checkpoint SHA256 `1302de3b...e04ef`, candidate YAML SHA256 `f0775920...f33a2`, and parent YAML SHA256 `74dd5d89...179e`. Synthetic inputs used separate RGB/TIR tensors `[1,3,128,128]` and `[1,3,256,256]`. Harness failures and fixes are listed in `output/experiments/M1_EARLY_UTILITY_ROUTER_QUICK_V1_20261022/preflight_attempts.json`.

- Parent/candidate score map max absolute difference: `1.3113e-5`.
- Parent/candidate box max absolute difference: `8.7619e-6`.
- Output shapes: score map `[1,1,16,16]`, box `[1,1,4]`, utility logits `[1,2]`.
- Zero-init utility logits max absolute value: `0`.
- Router parameters: `38,882`; all and only router parameters trainable.
- Final-layer gradient L1: `2.48193`; descriptor-projection gradient L1 after output-layer warm-start step: `0.05182`.
- Frozen M1 BatchNorm running statistics changed: `false`.

The harness exposed and repaired four implementation issues before the passing run: wrong synthetic input structure; failure to broadcast the crop-level gate to `[B,N,2]`; counterfactual teacher suppression when the parent is held in eval mode; and an out-of-scope actor flag. The exact passing output is `output/experiments/M1_EARLY_UTILITY_ROUTER_QUICK_V1_20261022/preflight.json`.

## Utility calibration gate

Temporary fit: 32 optimizer updates from quick-train only. Calibration: 768 sampled crops from the sequence-disjoint 48-sequence calibration list; seed `42022`; loader workers `0` for this bounded diagnostic. The temporary model was evaluated on calibration and then discarded. This is a sequence-held-out random-crop check, not a replay of the archived fixed states.

| Metric | Router | Baseline | Result |
|---|---:|---:|---|
| Utility cross-entropy | `0.686705` | Train-derived constant class prior: `0.669720` | Worse |
| Mean utility regret | `0.084712` | Random modality: `0.144097` | Better |
| Modality-label accuracy | `0.6172` | Train class-prior majority: `0.6621` | Worse |

The metrics disagree: the router does not reliably classify the higher-utility modality, while its mistakes are less costly on average than random choice. The preregistered gate requires both lower utility CE and lower regret; therefore it fails. No main quick training or tracking evaluation was launched. Development, locked holdout, and full-244 candidate evaluation remain unopened.

Environment: Python `3.8.20`, PyTorch `2.0.0+cu118`, CUDA `11.8`, 1x NVIDIA GeForce RTX 4090 D. This differs from M1's protected recorded PyTorch `1.9.0+cu111` / CUDA `11.1`; the preflight is not a strict M1 environment reproduction.

## Provenance

- Pre-change source archive: `experiments/snapshots/M1_EARLY_UTILITY_ROUTER_V1_20261022_prechange.tar.gz`, SHA256 `a783afd9...10dcbe3`.
- Exact post-change implementation snapshot: `experiments/snapshots/M1_EARLY_UTILITY_ROUTER_V1_20261022_implementation.tar.gz`, SHA256 `c7def7e4ffa45e7f0ceada3846abadf84b7f69cc0139fab8b818c79bc0913a36`.
- Candidate and parent configurations: `experiments/tbsi_track/M1_EARLY_UTILITY_ROUTER_{QUICK_V1,PARENT_V1}_20261022.yaml`.
- Implementation preflight: `tools/preflight_m1_early_utility_router_20261022.py`.
- Temporary fit/calibration: `tools/preflight_m1_early_utility_calibration_20261022.py`.
- Calibration output: `output/experiments/M1_EARLY_UTILITY_ROUTER_QUICK_V1_20261022/utility_calibration.json`.
- Run manifest and attempt history: `output/experiments/M1_EARLY_UTILITY_ROUTER_QUICK_V1_20261022/{run_manifest.json,preflight_attempts.json}`.
- Protected M1 YAML/checkpoint hashes were verified unchanged.

## Next permitted step

Do not train this candidate further or tune it against the already-used calibration cohort. Analyze the CE/regret mismatch and determine whether the current hard class target is the wrong decision objective. Any new objective or router mapping must be registered as a new candidate version, use a new train-side calibration cohort or a frozen one-time development rule, and retain the locked holdout until its design is frozen.
