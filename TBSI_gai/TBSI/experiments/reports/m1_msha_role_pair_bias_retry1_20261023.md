# M1 MSHA Role-Pair Bias Retry1 (2026-10-23)

## Question and decision

This run asks whether a zero-initialized, per-head bias on cross-modal foreground-template query/key pairs can be trained through the ordinary TBSI tracking objective. The operator changes only loc9 MSHA attention logits; only its 12 gains are trainable.

Decision: `TRAINING-COMPLETE / PRE-FIT-NOT-SUPPORTED / PROTOCOL-DEVIATION`. The interface fix works and the gains receive gradients and update. The run does not meet the preregistered fit/readout evidence needed to authorize a larger screen. It does not establish that the operator is ineffective.

## Provenance

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15.
- Protected config SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`.
- Protected checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Candidate config: `experiments/tbsi_track/M1_MSHA_ROLE_PAIR_BIAS_RETRY1_20261023.yaml`, SHA256 `b8de9949b8cafacb0a6d2879cab738693667757a9953597280891c503535d6a2`.
- Code snapshot before mask-plumbing fix: `experiments/snapshots/m1_msha_role_pair_bias_mask_plumbing_fix_20261023_prechange.tar.gz`, SHA256 `b0f1762d7773689ef896bff69a66dbfbfc6b126f7dba0960cd80e369019d443f`.
- Runtime: 1x NVIDIA GeForce RTX 4090 D; Python 3.8.20; PyTorch 2.0.0+cu118; CUDA 11.8; FP32; batch 2; worker 4; prefetch 2; persistent workers false; seed 42.
- Actual command: `python tracking/train.py --script tbsi_track --config M1_MSHA_ROLE_PAIR_BIAS_RETRY1_20261023 --save_dir output --mode single`.

## Contract failure and repair

The first launch failed before the first batch because `TBSITrackActor.forward_pass` did not derive foreground masks for the new role-pair flag. The actor already had the correct conversion from `visible.template_anno` to 64 soft template-token weights, but its feature-enable predicate omitted this flag. The repaired predicate enables this conversion only for the candidate; default M1 remains unchanged. The initial failed output is retained at `output/experiments/M1_MSHA_ROLE_PAIR_BIAS_PREFLIGHT_20261023/` and is registered separately as an implementation-contract failure.

After repair, implementation preflight passed: zero-gain parent parity was exact for the synthetic score map and box; the active gate changed both; all 12 gain parameters were trainable with finite nonzero gradient; frozen parameters and BatchNorm buffers did not change. This is an implementation check, not data-fit evidence.

## Training result

The retry completed 3 epochs, 8 batches per epoch, with no mask-contract exception. Logged per-epoch means:

| Epoch | Total loss | CENTER location loss | IoU |
|---:|---:|---:|---:|
| 1 | 0.602301 | 0.248876 | 0.851021 |
| 2 | 1.016627 | 0.360133 | 0.743796 |
| 3 | 0.930034 | 0.234614 | 0.760156 |

The sampler drew different pairs each epoch, so the non-monotonic means cannot distinguish optimization failure from sample difficulty. Loss and IoU do not show a controlled positive fit trajectory. Gradient-flow logging recorded finite nonzero gain gradients; the final largest absolute gain was `1.11e-4`. No parent replay, CENTER rank/peak comparison, validation, AUC/PR/NPR/OP, or tracking FPS was run. Checkpoint SHA256: `c26afa63588bb87119a951ea9f5a30c0552372a0726ea38f7327526d736c98af`.

## Protocol deviation and interpretation

The preregistration specified the frozen 176-sequence quick-train manifest. The run config instead used `LasHeR_train`; `build_datasets` loads the full official 977-sequence training set, and no manifest was supplied. The 16 samples per epoch were random rather than a frozen repeated set, for 48 sampled pairs total. Thus this run is an exploratory trainability smoke on the full training population, not the preregistered 176-sequence fit check. No development or locked holdout was opened.

What is established: the missing template-mask interface was repaired; the operator participates in the real training graph and gets nonzero gradients; this short run did not produce the required fit/readout evidence. What remains unresolved: whether the gains can fit a fixed set of examples, whether they move CENTER peak/rank/box readouts in a useful direction, and whether any effect generalizes.

Do not launch the planned 512-sample quick screen from this result. A justified follow-up must use a new run identity, explicitly load the 176-sequence manifest, freeze a small set of sample identities, and compare parent/candidate on those same repeated batches while recording CENTER peak, rank, and decoded-box diagnostics. Only a positive controlled fit response should authorize a broader screen. Locked holdout remains sealed.

## Evidence

- Preregistration and execution addendum: `experiments/protocols/m1_msha_role_pair_bias_20261023/preregistration.md`.
- Implementation preflight: `output/experiments/M1_MSHA_ROLE_PAIR_BIAS_RETRY1_20261023/implementation_preflight.json`.
- Training log, gradient flow, loss components, and internal signals: `output/experiments/M1_MSHA_ROLE_PAIR_BIAS_RETRY1_20261023/logs/`.
- Checkpoint: `output/experiments/M1_MSHA_ROLE_PAIR_BIAS_RETRY1_20261023/checkpoints/TBSITrack_ep0003.pth.tar`.
