# M1 MSHA Discriminative Evidence Router Design (2026-10-22)

## Decision

Do not replace the failed local-cosine-margin branch with another generic attention block. The next candidate should borrow the discriminative target-vs-background scoring operation used by DiMP-style trackers, then use that score as a bounded control signal for the existing MSHA cross-modal Value path. SiamCDA supports explicitly treating distractors as part of RGB-T tracking, but its located public GitHub repository is a dataset repository rather than a verified reusable implementation.

This is a design brief only. No code, configuration, or training run was changed or started.

## Evidence from M1

- `EarlyTargetMarginGate` labels each template token's global highest cosine match as its peak and compares it with four high responses outside a 3x3 neighborhood. It does not use template foreground masks to distinguish target tokens from context tokens, and it has no GT-defined search foreground/background distinction.
- The resulting correction is added to MSHA gate logits and then exponentiated to scale cross-modal Value messages. It does not directly impose GT-vs-hard-background ordering on the CENTER response.
- On the fixed 12 train-side clips, margin correction versus the early-gate parent has 4 wins / 8 losses and mean AUC delta -0.892 points; `motofromdark` falls by 8.844 points. Predictions first diverge in frames 1-3. This shows the operator has leverage but its evidence does not reliably identify the target.
- Existing 50-sequence diagnostics show that M1's current template-query similarity and modality agreement do not reliably predict CENTER target-vs-hard-background margin. A new router therefore needs a target-discriminative score trained from labels, not another transformation of the same unsupervised agreement signal.

## Mature mechanisms checked

1. **DiMP (ICCV 2019), discriminative model prediction for tracking.** The tracker learns a dense target classifier from positive and negative samples and treats hard-negative localization as a distinct failure condition; the official PyTracking repository contains the DiMP tracker implementation and explicit hard-negative handling. Transferable operation: train a target-vs-background score map with hard negatives. Do not transplant its online optimizer wholesale into M1.
   - Paper: https://arxiv.org/abs/1904.07220
   - Official implementation: https://github.com/visionml/pytracking/tree/master/pytracking/tracker/dimp
2. **SiamCDA (IEEE TCSVT 2022), complementarity- and distractor-aware RGB-T tracking.** It treats complementarity and distractors as explicit modeling concerns rather than assuming cross-modal agreement means target reliability. The located public repository is `LSS-Dataset`; it does not provide a verified plug-in implementation, so use the paper as a task-specific design precedent, not copied code.
   - Paper: https://doi.org/10.1109/TCSVT.2021.3072207
   - Located repository: https://github.com/Tianlu-Zhang/LSS-Dataset
3. **OSTrack (ECCV 2022), one-stream template-search interaction with candidate elimination.** The official repository is mature and reproducible. Candidate elimination is useful evidence that token selection can be built into tracking transformers, but its token-importance pruning is not itself a supervised foreground-vs-distractor score. Treat it as a secondary implementation reference, not the primary operator.
   - Paper/code: https://github.com/botaoye/OSTrack

## Proposed operator: Discriminative Evidence Router (DER)

### Placement and inputs

Keep the parent as the protected M1 plus the trained Early-Conditioned MSHA Gate checkpoint. Add DER only as a default-off adapter at the existing MSHA gate input at TBSI locations 3, 6, and 9. It reads block-2 RGB/TIR template and search features plus the initial template box's patch-coverage mask. The parent MSHA, CENTER head, and tracker state/update path remain fixed.

### Score-map operation

Use a small learned per-modality projection to form dense template-to-search target score maps from foreground template tokens. Unlike the failed branch, do not independently treat every template token's maximum as a target. During training, derive positive search cells from the GT box and hard negatives from the highest-scoring cells outside a dilated GT region:

\[
R_m(j)=\operatorname{LSE}_{i}\big[\langle \phi_m(t_i^m),\phi_m(s_j^m)\rangle/\tau+\log(w_i^{fg}+\epsilon)\big]-\log\sum_i(w_i^{fg}+\epsilon),\qquad
\mathcal L_{rank}=\operatorname{softplus}\!\left(\mu-\operatorname{LME}_{j\in P}R_m(j)+\operatorname{LME}_{j\in H}R_m(j)\right).
\]

Here `P` is the GT target region, `H` is the top-K hard-negative set outside the dilated GT region, and `w_fg` is the known template foreground coverage. Use this as one auxiliary ranking objective, with a small predeclared weight; retain the original tracking losses. This follows DiMP's discriminative target/background principle without importing its online filter-optimization loop.

At inference, the score map has no GT input. Define each modality's confidence from its trained map's top candidate versus the strongest spatial competitor outside the candidate neighborhood. Calibrate this margin on held-out train-side sequences before it is allowed to modulate fusion. Estimate RGB and TIR confidence independently; do not punish them merely for disagreeing, since disagreement can carry complementary target evidence. A low or ambiguous margin must reduce the new correction toward zero, not amplify it.

### Bounded MSHA control

Use modality-specific discriminative confidence and the foreground token mask to produce a direction-specific additive correction to the existing gate logits:

\[
\ell'_{i,m\to n}=\ell_{i,m\to n}+\alpha\,w_i^{fg}\,c_m\,\tanh(h_m(e_i)),
\qquad \alpha(0)=0,\quad |\alpha|\leq\alpha_{max}.
\]

`c_m` is the calibrated target-vs-distractor confidence for the source modality; `h_m` is a small residual predictor; `alpha` is bounded and initialized to zero. Apply the result only to cross-modal Value messages. Do not gate self messages, change attention Q/K ranking, or alter CENTER directly in this candidate. Feature-gate-off and zero-residual modes must exactly reproduce the parent.

This keeps the existing MSHA contribution in place while changing what controls its cross-modal strength: a supervised target-discriminative signal rather than raw agreement or an unsupervised peak uniqueness statistic.

## Risks and limits

- A score map trained with GT can still be miscalibrated on unseen distractors; held-out sequence calibration and per-sequence margins are required.
- The leading score at inference is still a candidate proposal, not ground truth. If its margin is high on a distractor, DER can make the wrong MSHA correction confidently. Therefore the first acceptance test is score-map GT rank and hard-negative margin, before closed-loop AUC.
- A weak modality can provide complementary evidence even when its standalone target-vs-distractor margin is low. Independent confidence scaling may suppress that useful contribution; test this by attribute and modality-ablation strata before considering any reliability interpretation.
- Template foreground masks from the initial template identify target support but not all appearance variation; overly tight masks can discard useful context. Use fractional patch coverage, not a hard binary token cutoff.
- SiamCDA's task-specific result does not prove this M1 integration will help. DiMP's online classifier and M1's frozen template/Center pipeline have different training and update contracts.
- This is an integration of established discriminative tracking ideas, not a claim that the underlying target-vs-background objective is new.

## Minimal validation gate

1. Fixed M1/gate parent, identical crop/template/state: measure block-2 per-modality GT rank, positive-vs-hard-negative score margin, and CENTER GT rank on at least 50 fixed sequences. Include a non-target template-token control.
2. Before closed-loop training, pass the repository's module contracts: default-off identity parity, shape/save-load, finite single-batch gradients, frozen-parent state equality, and fixed-16-sample learnability. If the score map cannot overfit the tiny set or does not rank GT above hard negatives, stop before quick training.
3. If step 1-2 pass, run one three-epoch quick train with only DER parameters trainable, then evaluate the same frozen 12-sequence train-side cohort against both the gate-only parent and M1. Report per-sequence AUC and fixed-state GT rank/margin; do not promote from the proxy alone.
4. Stop if GT score rank does not improve across the held-out fixed states, if non-target tokens receive comparable correction, or if the bounded branch causes broad first-frame prediction changes without corresponding GT-vs-background rank gains.

## References

- Bhat et al., *Learning Discriminative Model Prediction for Tracking*, ICCV 2019, arXiv:1904.07220.
- Zhang et al., *SiamCDA: Complementarity- and Distractor-Aware RGB-T Tracking Based on Siamese Network*, IEEE TCSVT 32(3), 2022, DOI: 10.1109/TCSVT.2021.3072207.
- Ye et al., *Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework*, ECCV 2022 (OSTrack), official repository above.
