# M1 MSHA Foreground Role Pair Bias Preflight

Status: `TRAINING-SMOKE-COMPLETE / NOT-ADMITTED`; no development or locked-holdout evaluation was started.

## Question and hypothesis

Does favoring cross-modal attention edges between foreground template tokens cause a trainable native MSHA parameter to improve the existing tracking objective on a fixed small training sample? The hypothesis is that source-token selection by target role is more useful than scaling an entire cross-modal Value message.

## Parent and single change

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15, config SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`, checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Add a per-head scalar gain, initialized to zero, at TBSI location 9 only.
- For cross-modal logits only, add `tanh(gain[h]) * fg_query * fg_key` in both RGB-query/TIR-key and TIR-query/RGB-key blocks. No self-attention logits, Value vectors, fusion neck, CENTER head, state update, or template update change.
- The soft foreground masks come only from the initialized template box. Current-frame GT is not an inference input.
- Train only the 12 per-head gains; freeze M1 parameters and buffers. Use ordinary parent tracking losses without added auxiliary loss.

## Tensor contract

At the selected TBSI location, each modality has 64 template tokens and 256 search tokens. `V2TemplateFusion` performs joint attention on the 128 concatenated template tokens. Logits are `[B,12,128,128]`; RGB and TIR template masks are `[B,64]`. The bias is applied only to the two `[B,12,64,64]` cross-modal blocks. The trainable parameter count is 12; its zero initialization must give parent parity.

## Data and tiers

- Preflight budget: 16 sampled pairs/epoch for 3 epochs from the frozen M1 screening quick-train manifest (176 LasHeR official training sequences). The standard sampler may draw different pairs between epochs; this is a local trainability smoke, not a strict repeated-example overfit test or a performance estimate.
- If implementation and fit checks pass, a separate quick screen may use 512 samples/epoch for 3 epochs on the same 176-sequence manifest, seed 42, batch 2, FP32, worker 4, and an independent run directory.
- The 48-sequence calibration cohort has already been exposed to invalid prior analysis and will not be used as an untouched gate. The 24-sequence development cohort may be used once, only after the candidate checkpoint and rule are frozen. The 24-sequence locked holdout remains sealed.
- The protected M1 checkpoint is the initialization and fixed parent comparator. This is a short parent-initialized adaptation, not an from-initialization reproduction.

## Checks and stop rules

1. Disabled gate and enabled zero-gain must match the M1 score map and decoded box within FP32 tolerance.
2. A nonzero gain must alter cross-modal attention logits and downstream output while preserving tensor shapes and finite values.
3. Tracking loss must produce finite, nonzero gradients for all intended gains; one optimizer step must change a gain. Frozen M1 parameters and normalization buffers must not move.
4. The 16-pair fit must reduce the same training objective and show a nonzero response in CENTER/readout diagnostics. If it cannot fit, inspect gradients and the masked attention path, then stop; do not launch quick training.
5. If preflight passes, train the separate 3-epoch quick screen. Development is evaluated once after freezing the checkpoint and success rule. No threshold tuning on development.

## Outcomes and limits

The preflight claim is implementation and local trainability only. A good small-sample fit does not establish generalization. A development result is one-seed screening evidence; full-244 is required for historical tracking comparison, and locked holdout remains unopened until a single candidate and acceptance rule are frozen.

## Snapshot

Pre-change source archive: `experiments/snapshots/m1_msha_role_pair_bias_20261023_prechange.tar.gz`, SHA256 `4b30c8291e6386228e34329cc6bb9ecacd4843f3608e0b573d9a862e6993098e`.

## Execution addendum (2026-10-23)

- The first launch failed before a training batch: `TBSITrackActor.forward_pass` only derived template masks for older feature flags and omitted `MSHA_ROLE_PAIR_BIAS_ENABLE`. This is an implementation-contract failure, not evidence against the operator. Its output directory is retained as `M1_MSHA_ROLE_PAIR_BIAS_PREFLIGHT_20261023`.
- A narrow actor fix now derives the existing 8x8 soft mask from `visible.template_anno` when the role-pair candidate is enabled. It does not use search annotations and is inactive with the default M1 configuration. The fix was preceded by snapshot `experiments/snapshots/m1_msha_role_pair_bias_mask_plumbing_fix_20261023_prechange.tar.gz`, SHA256 `b0f1762d7773689ef896bff69a66dbfbfc6b126f7dba0960cd80e369019d443f`.
- The corrected 3-epoch smoke completed under new config `M1_MSHA_ROLE_PAIR_BIAS_RETRY1_20261023`, config SHA256 `b8de9949b8cafacb0a6d2879cab738693667757a9953597280891c503535d6a2`. The protected M1 config and checkpoint hashes remained unchanged. All 12 intended gains received gradients and changed; the final maximum absolute gain was about `1.11e-4`.
- Protocol deviation: the run config used `LasHeR_train`, which loads all 977 official training sequences. It did not pass the preregistered 176-sequence manifest (SHA256 `141cbce02e760c262c9e54f9a63378ef537b7c15dae461d2833646c0cc00fc06`). Only 16 random pairs per epoch were sampled, 48 total; the sample identities were not frozen. Treat this as an exploratory smoke, not the preregistered 176-sequence fit check.
- Logged epoch means were total loss `0.6023 / 1.0166 / 0.9300`, location loss `0.2489 / 0.3601 / 0.2346`, and IoU `0.8510 / 0.7438 / 0.7602`. Because examples differed across epochs, these do not establish a controlled overfit failure; they also do not meet the planned positive fit/readout evidence. No CENTER rank/readout comparison or validation was performed.
- Decision: `TRAINING-COMPLETE / PRE-FIT-NOT-SUPPORTED / PROTOCOL-DEVIATION`. Do not start the 512-sample quick screen or open development/locked holdout. A follow-up, if retained, must first repair the data contract with an explicit 176-sequence loader and frozen 16-pair manifest, then compare parent and candidate on identical repeated batches while logging CENTER peak/rank/box readouts. This is a new experiment, not a retry of this run.
- Checkpoint: `output/experiments/M1_MSHA_ROLE_PAIR_BIAS_RETRY1_20261023/checkpoints/TBSITrack_ep0003.pth.tar`, SHA256 `c26afa63588bb87119a951ea9f5a30c0552372a0726ea38f7327526d736c98af`. Full log and diagnostic CSVs are under the same run directory. No tracking metrics were produced.
