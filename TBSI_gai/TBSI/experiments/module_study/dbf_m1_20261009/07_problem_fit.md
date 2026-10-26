# Can current DBF solve an unresolved TBSI/M1 problem?

Status: `STATIC + REUSE OF EXISTING RESULTS / NO NEW RUN` (2026-10-09). Answer: **not demonstrated**. The only defensible research hypothesis is narrower: token-dependent differential cross-modal reading *might* reduce local search-stage interference when the target remains inside the search crop. The present DBF is neither an illumination-reliability estimator nor a temporal-template method. Its completed one-seed proxy result is negative against both Vanilla and M1 Legacy.

## Evidence and boundaries

- **Paper A, direct source:** `papers/TBSI_Target_Preserved_Template_Updating.pdf`, SHA256 `8d21db472dcdb1eda98f4b88c3cbef5d1ceae05e6fc1965b96e7351d596b4efe`. Introduction pp. 634-636; TBSI Section III-B pp. 638-639; IGF Section III-C/Fig. 5 pp. 639-640; TPTU Section III-D p. 640; component discussion Section IV-D pp. 642-645. The paper itself already proposes TBSI for target-guided interaction/background noise, IGF for illumination-guided reliability, and TPTU for template drift. These are not unaddressed problems in the paper's design.
- **Paper B, direct source:** `papers/Differential_Transformer_arxiv_2410.05258v1.html`, Section 2.1/Eqs. (1)-(3) and Section 3.8/Table 6. Two independently projected attention maps are subtracted to reduce attention noise in language-model experiments. No RGB-T tracking validation is reported there.
- **Current model code:** `lib/models/layers/differential_bridge.py` uses one Q, two K, sigmoid scalar lambda, shared V and a learned scalar gamma per direction; `lib/models/layers/dbf_igf_replacement.py` takes templates plus searches and averages both updated searches with fixed 0.5 weights. It does not read the RGB frame or estimate current illumination. It also omits the ICLR paper's headwise output normalization.
- **Existing experiment:** `reports/dbf_igf_replacement_short_20261009.md`, single seed/3 epochs/60 LasHeR test-proxy sequences. AUC L/V/D is `52.4663/50.5768/49.0681`. The test proxy has already influenced design, so every new subgroup calculation below is exploratory diagnosis, never a model-selection criterion.
- **M1 failure evidence:** `reports/m1_full_test_failure_map_20261005.md` records `136/244` sequences with at least one sustained 30-frame IoU<0.5 period, but does not identify illumination, distractors, crop-out, or modality conflict as the cause. It lacks per-frame crop transforms. Failure count alone cannot establish a DBF opportunity.

## Problem-to-mechanism judgment

| Proposed problem | Paper/repository evidence | What current DBF can actually do | Verdict |
|---|---|---|---|
| Coarse RGB/TIR interaction and background distractors | Paper A Introduction/Fig. 1 and Section III-B explicitly assign target-guided background suppression to TBSI; M1 enables TBSI-E. | Two token-dependent cross-modal maps could learn different spatial reads. Their subtraction has no target/background guarantee and may also cancel useful target agreement. | **Possible incremental hypothesis, not a newly identified unresolved paper problem.** Old D-V and clutter subgroup are negative. |
| Low-light RGB unreliability and bright-scene TIR noise | Paper A Section III-C/Fig. 5 uses current RGB frame -> GIE global factor -> weighted search fusion -> CA/SA. GIE is frozen from reference [23]. | Current DBF ignores the frame and ends with fixed 0.5 modality averaging. Token-dependent residuals may compensate locally, but there is no input-conditioned modality-reliability weight or illumination supervision. | **Cannot claim IGF's illumination function is replaced.** Old low-illumination subgroup has a negative average and large failures. |
| Spatially varying modality conflict, thermal crossover, or localized corruption | A single GIE factor is global, but IGF also has CA/SA; the paper does not establish that its local adaptation is inadequate. M1 failure map has no causal labels. | Cross-attention is token-dependent, so a local-interference hypothesis is technically plausible. Shared Q and a second softmax do not by themselves distinguish target from noise or align displaced modalities. | **Most plausible new question, but unverified and currently contradicted by the old thermal-crossover proxy average.** Do not call it an established IGF weakness. |
| Appearance change, template contamination, disappearance/reappearance | Paper A Section III-D and IV-D assign temporal-template decisions and reverse-tracking checks to TPTU/TPTU+. | Post-backbone DBF is stateless on each search crop. It has no update decision, first-frame reverse check, or global search. | **Out of scope.** It cannot directly fix TPTU-type failures or recover a target already outside the crop. |
| M1's persistent tracking failures | M1 full-244 result quantifies long failures but not their causes. | DBF could change one-step search features only while relevant target information is present. | **No attribution yet.** Fixed-crop, same-state error analysis would be needed before assigning these failures to fusion. |

## Exploratory attribute audit of the completed proxy

Read-only recomputation from the saved per-sequence `metrics/*.json`, the frozen `proxy_test_manifest.json`, and the LasHeR `Attributes&Order.zip` labels. AUC values are sequence-equal means inside each overlapping attribute subset; `D>V` counts sequence wins. These are descriptive slices of the **same** already-used 60 test sequences, not new tests or independent evidence.

| Attribute | n | L AUC | V AUC | D AUC | D-V | D-L | D>V |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low illumination (LI) | 21 | 49.315 | 46.601 | 42.483 | -4.118 | -6.832 | 12/21 |
| Background clutter (BC) | 27 | 53.492 | 51.528 | 48.171 | -3.357 | -5.321 | 11/27 |
| Thermal crossover (TC) | 43 | 47.006 | 45.557 | 43.286 | -2.271 | -3.720 | 15/43 |
| Heavy occlusion (HO) | 4 | 37.470 | 40.892 | 37.551 | -3.341 | +0.081 | 1/4 |

LI is heterogeneous: D-V median is `+0.085` and D wins 12/21, but two sequences lose `58.32` and `37.91` AUC respectively. Both also carry BC and TC labels, so the subgroup averages are not independent and are strongly affected by the same severe losses. The evidence does **not** support a robust low-light gain or a causal claim about why those sequences failed. The four-sequence HO slice is too small for a useful conclusion.

## Decision for the next investment

1. **Stop the exact old template-context/raw-mean DBF replacement.** Its result is below Vanilla and Legacy, and the trained M1 fusion readout was bypassed.
2. Keep one falsifiable opportunity: search-only, token-level cross-modal interference control at the fusion slot. Diagnose output distribution, per-head behavior, optimizer LR, effective residual, and gradient/update health before another run. A search-only change is a new experimental version, not an explanation of the old result.
3. If pursuing this opportunity, use the `05_experiment_plan.md` G1 L/V/D matched short screen with M1's readout preserved as a **diagnostic augmentation**. Require D to beat V and matched L without concentrated catastrophic losses and with acceptable latency before G2 standalone replacement. The original IGF comparison additionally needs the faithful implementation and GIE weights.
4. A frame-conditioned reliability gate would address IGF's illumination question more directly, but it is a **different proposed module**. Do not attribute its possible future gains to current DBF or add it to the first short screen.

Paper-facing claim today: *The existing Differential-inspired DBF has not demonstrated a solution to M1's unresolved tracking errors; its plausible contribution is local search-stage interference control, pending a controlled, head-compatible comparison.*
