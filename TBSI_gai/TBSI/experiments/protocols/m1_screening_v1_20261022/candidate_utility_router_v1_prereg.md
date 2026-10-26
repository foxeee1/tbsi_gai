# Candidate Preregistration: Early Utility-Conditioned MSHA Router V1

Status: `IMPLEMENTATION-PREFLIGHT-PASSED / CALIBRATION-GATE-FAILED / QUICK-SCREEN-NOT-ADMITTED`  
Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15  
Candidate ID: `M1-EARLY-UTILITY-ROUTER-QUICK-V1-20261022`

Pre-change source snapshot: `experiments/snapshots/M1_EARLY_UTILITY_ROUTER_V1_20261022_prechange.tar.gz`, SHA256 `a783afd9efde0ea09a3c01f5351a306f5765ab211240f2003ac8b2ef610dcbe3` (captured before candidate implementation changes).

## Evidence and problem

The calibration and development cohorts independently show crop-contained first persistent failures where CENTER GT top-1 falls from 22/23 to 11/23 and from 14/15 to 7/15. Calibration reversible search-fusion interventions show a per-event modality oracle has headroom (16/23 onset states exceed baseline; mean/median IoU delta `+0.1402/+0.0773`), with RGB selected on 13 events and TIR on 10. Existing hand-designed early-margin inputs do not predict that utility: the early-margin-to-gate correlation is near zero, and the early block02 margin versus RGB/TIR intervention utility has Spearman `-0.079` (`p=0.72`).

The supported problem is missing target-conditioned modality utility in frame-local MSHA interaction. The first causal operation and a hand-designed online reliability signal are not established. The oracle statistic is used only as a train-side teacher and is never available to inference.

## Single change

Add one shared predictor, evaluated once after backbone block02, from the six pooled descriptors `[RGB-template-FG, RGB-template-BG, RGB-search, TIR-template-FG, TIR-template-BG, TIR-search]`. Inputs are `[B,64,768]` for each template, `[B,256,768]` for each search, and the existing soft foreground masks `[B,64]` from the initialized template box. Each descriptor is pooled with its mask (background uses `1-mask`), passed through one shared `LayerNorm(768)+Linear(768,32)+GELU`, then concatenated with four cosine terms: RGB FG/search, TIR FG/search, RGB/TIR FG, and RGB/TIR search. The resulting 196-vector feeds `Linear(196,64)+GELU+Linear(64,2)`. The final layer is zero-initialized. The router is shared across all three MSHA interaction locations.

The two-way softmax `p=softmax(logits)` controls only source-modality cross-modal Value messages:

\[
g_m=\exp(0.25(2p_m-1)),\quad m\in\{RGB,TIR\}.
\]

Thus `g_m ∈ [exp(-0.25),exp(0.25)]`; uniform probability gives exact identity. `g_RGB` scales RGB values sent to TIR queries, and `g_TIR` scales TIR values sent to RGB queries. Attention logits, within-modality values, residuals, search-neck fusion, CENTER decoding, state updates, and template updates are unchanged.

For each modality, a training-only counterfactual readout applies the existing `tbsi_fuse_search` and CENTER head to `[RGB,0]` or `[0,TIR]`. The teacher utility is computed from the resulting CENTER score map using the existing focal-target region (`Gaussian target >= 0.5`) and the strongest response outside that region:

\[
u_m=\max_{j\in GT}s_m(j)-\max_{j\notin GT}s_m(j).
\]

The detached class label is `argmax(u_RGB,u_TIR)`; exact ties within `1e-6` are omitted. The only added objective is `0.1 * CrossEntropy(router_logits, teacher_class)`. The M1 tracking losses remain unchanged. The counterfactual heads run in eval mode and under `no_grad` so they cannot update shared BatchNorm state or receive gradients. At inference, the router uses only block02 features and initialized-template masks; it receives no GT, label map, future frame, or state-update outcome.

This is a crop-level modality-utility decision. It does not claim to produce a tokenwise target/background mask. Its value must be demonstrated by CENTER ranking and tracking outcomes; better teacher-class accuracy alone is insufficient.

## Frozen data and compute contract

- Candidate updates: only `quick_train_sequences.txt` (176/977 LasHeR official train sequences; 18.01%, attribute-marginal deviation at most 1.02 sequences).
- Training budget: 1,264 optimizer updates; schedule is scaled by optimizer-step ratio to M1's 7,020 updates. Record actual sample exposures and schedule values.
- Candidate and direct-parent control: same seed 42, sample order, augmentation, resolution, optimizer, effective batch, precision, update count, and checkpoint rule.
- Calibration: 48 frozen train sequences; one pre-update predicted-utility ranking/regret check. No candidate gradient updates on calibration.
- Development: 24 frozen train sequences; one evaluation after candidate freeze.
- Locked holdout: 24 sequences; remain sealed until design and success rule are frozen.
- Full LasHeR test: 244 sequences for historical paired comparability only; selection-exposed, not independent confirmation.
- Paper cases: four retrospective test cases for illustration only; no threshold or candidate selection.
- Template updating remains disabled as a research variable; no template-update claim is made.

## Preflight and decision rules

1. Uniform router probability must match M1 within FP32 tolerance; tensor shape, checkpoint save/load, and frozen-parent state must match.
2. With the parent frozen, router gradients and the utility loss must be finite and nonzero. Fit the router on a fixed small subset drawn only from `quick_train`; this is a feasibility preflight, and its weights are discarded afterward.
3. Evaluate the frozen preflight router once on calibration fixed states. It must beat a constant-prior predictor on utility cross-entropy and achieve lower utility regret than random modality selection. Otherwise stop and do not launch the 1,264-update screen.
4. If preflight passes, initialize from the protected M1 checkpoint and train exactly one candidate plus a matched direct-parent continuation control, both on the same 176 sequences for 1,264 optimizer updates. Do not add a second loss, alter router location, or tune a gate threshold during this run. This is a short M1-parent screen, not full retraining from the common public initialization.
5. If the screen is directionally useful, run the full matched training schedule from the same public pretrained initialization for parent and candidate before making a from-initialization performance claim. Do not use short-run results as a substitute for this comparison.
6. Report utility-prediction quality, fixed-state CENTER rank/margin, development results, full-244 paired AUC/PR/NPR/OP50/OP75, per-sequence changes, strata, runtime, and parameter/FLOP costs. A short-run gain is exploratory only.

## Falsification and stop conditions

- Stop before training if the router cannot predict held-out calibration utility better than the constant prior after implementation checks.
- Stop the candidate if its auxiliary objective falls but predicted modality utility does not improve on calibration fixed states; this is supervision/representation mismatch.
- Stop promotion if development and full-244 disagree materially, if the gain is concentrated in a few sequences, or if stable controls regress without a measured subgroup benefit.
- Do not open locked holdout until one candidate, checkpoint, and success rule are frozen. Do not claim multiple-seed or external generalization from this single-seed screen.

## Implementation and calibration addendum (2026-10-22)

The implementation contract was made explicit before any candidate checkpoint was saved: initialize from the protected M1 epoch-15 checkpoint, freeze every M1 parameter and normalization buffer, and train only the 38,882-parameter early utility router. The frozen parent is the direct control; it receives no optimizer updates. This isolates whether the early descriptor can select a useful cross-modal Value scale before considering joint or from-initialization training.

The first preflight verifies implementation only. It uses synthetic tensors and therefore cannot satisfy the calibration gate. The utility generalization check then performs a temporary 32-update fit on samples drawn only from the 176 quick-train sequences and evaluates 768 deterministic-seed sampled crops from the 48 calibration sequences. This is a sequence-held-out random-crop test, not a fixed-state replay; no calibration sequence contributes optimizer updates. The environment used for this preflight is Python 3.8.20, PyTorch 2.0.0+cu118, CUDA 11.8, on 1x NVIDIA GeForce RTX 4090 D, so it is not a strict reproduction of M1's recorded PyTorch/CUDA stack.

The gate result is `CALIBRATION-GATE-FAILED`: utility cross-entropy is `0.686705`, worse than the quick-train class-prior baseline `0.669720`; mean utility regret is `0.084712`, better than random modality choice at `0.144097`. The two metrics disagree. Under the frozen rule, this candidate is not admitted to the 1,264-update quick screen. Temporary-fit weights were discarded. Calibration data are now exposed to this utility-prediction check and must not be reused as an untouched evaluation set. The 24-sequence development and locked holdout cohorts remain unopened.

During implementation preflight, four harness/contract failures were found and repaired before the final pass: the synthetic harness passed a concatenated six-channel tensor where the model expects separate RGB/TIR tensors; crop-level `[B,2]` gates were not broadcast over the MSHA template-token axis; router-only eval mode suppressed counterfactual teacher generation; and the actor loss referenced a forward-local flag. The final implementation preflight passed zero-gate M1 parity (maximum score-map difference `1.31e-5`, box difference `8.76e-6`), output shapes, finite router gradients, descriptor gradients after one output-layer warm-start step, and frozen BatchNorm state. These are implementation checks, not performance evidence.
