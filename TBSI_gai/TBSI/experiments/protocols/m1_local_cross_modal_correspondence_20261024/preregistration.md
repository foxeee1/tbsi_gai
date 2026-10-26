# M1 Local Cross-Modal Correspondence Exchange

## Decision context

This is a proposed experiment only. No model code, config, training, or tracking test has been changed or run in this turn.

The current M1 failure chain is crop-internal center-selection error followed by feedback drift and weak recovery. This is a system-level observation; it does not establish MSHA as the first cause. Early-conditioned MSHA gate has a small heterogeneous full-244 point estimate, while its paired sequence interval crosses zero. The recent template-search relation-bias screen regressed on its 12-sequence train-side cohort. Sparse spatial-token bridge learnability and amplitude diagnostics showed that increasing bridge messages did not improve localization-facing ranks or replay. Therefore this experiment changes the interaction location and the transported quantity.

## Hypothesis and alternatives

**H1, candidate:** At early search features, RGB and TIR contain complementary target evidence that is spatially displaced within a small neighborhood. Same-position or template-only interaction does not transfer that evidence reliably. A local, target-conditioned correspondence operator that transports the opposite modality's feature values should improve target-versus-hard-background localization on crop-inside failures without broadly harming stable controls.

**H0:** The search streams do not need local cross-modal correspondence; the candidate will either learn near-zero messages, make similar updates at target and hard-background locations, or change features without improving localization.

**Alternative explanations:** (a) The failure is dominated by CENTER/head readout or crop feedback rather than cross-modal interaction; (b) current training losses do not direct the exchange toward target evidence; (c) the two modalities are sufficiently aligned and a spatially flexible operator adds noise; (d) a fixed small neighborhood cannot cover the relevant displacement.

## Operator

At the early paired search feature map, reshape each modality from `[B,256,768]` to `[B,16,16,768]`. For each modality `m` and search position `p`, form a query from `x_m[p]` and nine keys/values from the other modality `n` at `p + delta`, where `delta` spans the 3x3 neighborhood and out-of-grid positions are masked:

`q_m,p = Wq_m LN(x_m[p])`

`k_n,p,d = Wk_n LN(x_n[p + d])`

`a_m,p,d = softmax_d(q_m,p dot k_n,p,d / sqrt(d_k) + b_m,d)`

`c_m,p = sum_d a_m,p,d Wv_n x_n[p + d]`

Use a foreground-weighted pooled template vector `t_m` from the existing 8x8 template annotations only to form a bounded target relevance coefficient per search location:

`r_m,p = sigmoid(w_m^T [LN(x_m[p]); LN(c_m,p); cos(x_m[p],t_m); cos(c_m,p,t_n)])`

The coefficient is spatial and channel input-dependent; it is not a global RGB/TIR choice. The update is:

`x'_m[p] = x_m[p] + r_m,p * Wo_m(c_m,p)`

Initialize only final `Wo_m` to zero, giving exact initial identity while allowing the output projection to receive a gradient on the first step. Once `Wo_m` moves, gradients can reach the correspondence and relevance branches. Do not zero-initialize a second multiplicative scalar, which would deaden this path. Clamp the per-token residual norm to at most 5% of the source token norm. The other stream is updated symmetrically. No online template update, auxiliary localization head, new ranking loss, or CENTER-head change is included in the first candidate.

This is a proposal, not a claim of novelty. It differs from the tested template-token bridge by operating on paired 16x16 search maps and exchanging Value features; it differs from template-search relation-bias by changing search representations rather than attention logits; it differs from post-TBSI global cross-attention by restricting correspondence to a local spatial neighborhood and conditioning the message on target templates. Related work still needs full-paper and implementation-level comparison before any manuscript novelty claim.

## Insertion and contracts

- Insert once after transformer block index 2, where the existing code already captures paired early search maps, and before the first TBSI-E fusion call (if TBSI is configured at index 2, insert immediately before that call). Capture/modify search tokens only; leave template tokens and protected M1 default path unchanged.
- Reuse the existing per-modality foreground mask from `template_box_to_token_weights`; it is derived from `visible.template_anno` in training and is not available from the search annotation at inference. No search GT enters the forward path.
- New behavior must be behind an independent config feature gate, default off. The protected M1 config/checkpoint/output paths remain untouched.
- With feature gate off and with zero-initialized final projections, output tensors must match parent exactly. Check shape, finite values, save/load, first-step projection gradients, multi-step gradients to Q/K/V/relevance parameters, and frozen parent parameter and buffer equality.
- For a frozen-parent contract diagnostic, keep the parent in eval mode to prevent BatchNorm/running-buffer drift. For joint training, use the normal matched parent recipe and make the new branch the only architectural difference.

## Experiment sequence

### A. Contract and learnability

Use the Early-conditioned MSHA gate full-15 checkpoint as the immediate candidate parent because this experiment investigates its next interaction step. Keep protected M1 epoch 15 as the fixed reference. Parent/candidate comparisons must use the same code snapshot, initialization, seed, data definition, training progress, checkpoint epoch, and evaluator; the Early-gate parent result is not a substitute for the protected M1 comparison.

Run zero-gate parity, exact tensor-contract checks, gradient checks, frozen-state checks, and a deterministic 32-sample fit probe. The fit probe is diagnostic only and does not select operator settings. Stop if the branch cannot overfit this fixed batch or if gradients are absent/non-finite; first audit data axes, crop/label coordinates, and loss path rather than changing the operator.

### B. Matched quick continuation on the 176-sequence proxy

If A passes, make a paired 3-epoch continuation comparison initialized from the exact Early-conditioned MSHA gate full-15 checkpoint: control = gate parent continued with the new operator disabled; candidate = the same checkpoint with the local correspondence operator enabled. Both arms use the locked 176-sequence LasHeR train proxy manifest `experiments/protocols/m1_screening_v1_20261022/quick_train_sequences.txt` (manifest SHA256 `410f7197a6d9a6d9e1886c0511e93d5a4ee1b4729fe77541e9d911aeaa82d8c3`), with 10,112 sampled examples per epoch. Each arm runs 3 epochs (237 optimizer updates), batch 16 with gradient accumulation 8, seed 42, AdamW, LR `1e-4`, AMP, worker 4, prefetch 2, persistent workers false. Keep crop policy, loss, trainable-parameter policy and evaluation cadence identical between arms. This paired continuation isolates the incremental operator on the strongest gate parent. Do not substitute the 12-sequence relation-bias cohort.

Train the candidate and operator-off control jointly with the backbone using the same trainable parameter policy. Do not train on fixed-state calibration events as a separate small event set. Save epoch 1 and final checkpoints. Select the final checkpoint by the pre-existing quick protocol, not by an added per-event tuning pass. This continuation screen does not replace a from-pretraining full-run comparison if the candidate is promoted.

### C. Matched quick tracking and mechanism readout

Evaluate candidate and its operator-off continuation control on the same locked 60-sequence RPP-v2 manifest, SHA256 `ab587f801525c2e7e0fc7a8465fb90f3a2d99b67764eb2bc19cac2b422a938a8`, with identical FP32 tracker settings, serial `threads=0`, template-update off, initialization and evaluator. Use same-state paired replay for 23-sequence/46-state calibration only as a mechanism diagnostic, not as a replacement for closed-loop tracking.

Record overall AUC/Success, Precision, Normalized Precision, OP50, OP75 and FPS; per-sequence paired deltas and sequence bootstrap interval; failure/control strata fixed before evaluating candidate outputs; first divergence; GT rank/margin and actual Hann-windowed top-1; offset/size decoded at the selected cell; replay IoU; per-token residual norm and target-vs-hard-background residual energy. Report all sequences, including losses.

### D. Promotion

Only a quick result with positive candidate-parent point estimates on Success/AUC and Precision, majority-sequence nonnegative paired IoU, no severe-control regression pattern, and an operator-specific signature (target-location exchange exceeds matched hard-background exchange and improves same-state localization readout) may be considered for a separate matched full-15 experiment. A quick result is not a full-data performance claim. Full test must compare the exact candidate to both the matched parent and protected M1 with the same official 244-sequence definition.

## Stop/go rules

- Contract or deterministic fit failure: stop and audit the training/representation contract.
- Learned residual remains negligible, or target/background residual selectivity is absent: stop the current operator; do not add a gate or extra loss to rescue it in the same run.
- Features change but target rank/readout and same-state IoU do not improve: classify as representation/readout coupling not supported; stop this form.
- Fixed-state readouts improve but closed-loop quick tracking regresses: classify as crop-feedback interaction risk; do not promote to full training.
- Quick tracking is mixed or its paired interval spans zero: exploratory/inconclusive; do not claim improvement. One next run may only change a single preregistered design variable that distinguishes a named alternative, such as 3x3 versus fixed aligned correspondence. Do not tune on the same quick manifest.

## Risks and interpretation limits

- Local cross-attention may still be redundant with earlier spatial modeling and may add noise where modalities are already aligned.
- Template masks are coarse 8x8 annotation-derived weights; target pooling can be noisy for tiny objects. Keep this mask path fixed and log effective foreground mass.
- Zero initialization of the final projection intentionally delays gradients into Q/K/V and relevance parameters until the output projection moves. Verify first-step projection gradients and multi-step upstream gradient flow; do not add a second zero multiplier.
- The 176-sequence proxy reduces but does not remove overfitting risk. Quick train/test evidence is exploratory and cannot establish LasHeR full-set gain or generalization to RGBT234.
- The candidate combines local correspondence and target-conditioned relevance in one operator family. If it passes, a follow-up ablation must separate local correspondence from target relevance before making a causal module claim.

## Provenance to fill before implementation/run

- Experiment IDs/config/output directories: `M1_LOCAL_CROSS_MODAL_CORRESPONDENCE_CONTROL_QUICK3_20261024` and `M1_LOCAL_CROSS_MODAL_CORRESPONDENCE_QUICK3_20261024`.
- Parent checkpoint SHA256 and matched quick-parent config/checkpoint: pending exact asset selection.
- Git commit or immutable source snapshot SHA256: pending; current worktree is dirty and must be snapshotted before code changes.
- Quick tracking manifest and SHA256, software/hardware, command lines, and measured process budget: pending; no run has been authorized or performed in this design turn.
