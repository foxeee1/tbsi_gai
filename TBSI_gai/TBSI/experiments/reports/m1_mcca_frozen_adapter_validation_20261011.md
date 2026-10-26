# M1 Frozen-Backbone MCCA Adapter Validation (2026-10-11)

## Question

Can the existing bidirectional MCCA adapters, trained by themselves from the protected M1 epoch-15 checkpoint, improve closed-loop tracking without continuation updates to the M1 backbone, original fusion, or CENTER head?

This is a new training contract, not a replication of joint-continuation G1. Its purpose is to separate the effect of learning the adapter from the degradation observed in the matched continuation G0. It does not establish that the adapter is a reliable-modality gate.

## Falsifiable hypothesis

H1: Starting from exact zero-output MCCA adapters, optimizing only their parameters against the existing tracking loss can create an adapter correction that improves paired sequence-level validation metrics over the fixed M1 checkpoint, without changing any M1 parameter or normalization buffer.

H0: Under this frozen-parent contract, adapter-only training does not yield a reproducible cross-sequence localization benefit. A positive train loss change or nonzero adapter gradient alone does not support H1.

## Locked protocol

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15; config and checkpoint remain read-only.
- Candidate: bidirectional MCCA at the existing three template MHSA locations; initialize missing adapter tensors with the implementation's zero-output initialization; train only the six adapters (both directions x three layers).
- Frozen state: every non-MCCA parameter has `requires_grad=False`; network remains in `eval()` during adapter optimization, so dropout and normalization behavior match inference. Autograd remains enabled through frozen operations to the adapter parameters. Save and compare the parent state dict after training; only adapter keys may differ.
- Objective: unchanged `2*GIoU + 5*L1 + focal`; no auxiliary reliability, ranking, or identity loss. This experiment tests the frozen-adapter hypothesis before adding supervision.
- Train: existing RPP-v1 176-sequence train manifest, 3 epochs, 10,752 sampled pairs/epoch, batch 16, accumulation 8 (252 optimizer steps), seed 42, FP32, worker 1, LR `1e-4`, AdamW and weight decay `1e-4`.
- Validation: the outcome-blind 96-sequence LasHeR train-side cohort in `experiments/proxy/mcca_confirm_two_seed_20261011/validation_sequences.txt`; it has zero overlap with the 176 training sequences and no candidate prediction directory existed at preflight. The protected M1 pretraining has seen the official train split, so this is adapter-training-held-out validation, not an unseen-to-M1 dataset.
- Evaluation: one serial FP32 closed-loop run for fixed M1 and one for the epoch-3 adapter candidate on identical 96 sequences, explicit checkpoint paths, `threads=0`. Report paired AUC/PR/NPR/OP50/OP75, sequence wins/losses/ties, paired bootstrap intervals, center error and normal-frame regressions if those measurements are available, plus observed FPS (not as a promotion target).
- No test-244 use, no hyperparameter search, no second seed, and no M1 config/checkpoint write. Any change to this contract requires a new experiment ID.

## Gates and interpretation

Before training: (1) M1 versus zero-output candidate max output difference <= `1e-6`; (2) all six adapter output projections receive finite nonzero gradients; (3) only MCCA parameters are in the optimizer; (4) one optimizer step changes adapter state and leaves all other parameters unchanged; (5) the base model stays in eval mode while gradients are enabled.

After training, compare the saved checkpoint tensor-by-tensor with the protected M1 checkpoint. Only MCCA adapter keys may differ. A positive mean with a paired interval crossing zero is exploratory, not a confirmed gain. A negative or near-zero result across AUC and localization measures rejects this frozen-adapter contract; it does not disprove all supervised or search-stage fusion approaches. If H1 is unsupported, do not add a loss to this same test cohort; first close the candidate or design a new train-side contract.

## Results and provenance

Status: complete; frozen-adapter hypothesis not supported.

P0 on the registered RTX 4090 D environment passed: M1 versus zero-output MCCA max absolute output difference `0.0`; six adapters found; all six final projection gradient norms finite and nonzero (`0.00538` to `0.01390`); 18 adapter parameter tensors updated after one optimizer step; frozen parameter changes `0`; frozen buffer changes `0`. This is a synthetic-input connectivity/state-integrity gate, not a task-performance result. Full output: `output/experiments/M1_MCCA_FROZEN_ADAPTER_3EP_20261011/p0/p0.json`.

Training completed 3/3 epochs (2,016 batches; 252 optimizer steps) without non-finite loss/gradient warnings. Gradient diagnostics at optimizer steps 200 within each epoch report zero gradient parameters `0/24`; the final checkpoint SHA256 is `6e808d2b350aa1527d76d9cd395d4f60528a1e56b915d1f74afc65964e294a67`. A post-training state comparison passed: all shared non-adapter tensors match M1 exactly, and the candidate contains exactly 24 adapter tensors.

The first validation launch failed during sequence-list construction, before any frame inference: `whiteboycome/visible` uses filenames such as `000001v.jpg`, while the historical parser only handled a leading modality marker. The validation cohort was not changed. A numeric-prefix sort now accepts both filename forms; a CPU smoke loaded all 96 sequences (120,485 frames) and verified each visible/infrared/GT count matches. This is an evaluation-loader compatibility fix, not a model or data-membership change. The failed preflight log is retained at `output/experiments/M1_MCCA_FROZEN_ADAPTER_3EP_20261011/logs/M1_eval.log` if present.

- Config: `experiments/tbsi_track/M1_MCCA_FROZEN_ADAPTER_3EP_seed42.yaml`.
- Training sequence list: existing `experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt`.
- Validation manifest: `experiments/proxy/mcca_confirm_two_seed_20261011/manifest.json`; validation list SHA256 `87c3b09c5c688f43e370d35ea027247ad387cb69717fa159fc72a71f33eb05e0`.
- Parent checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Output root: `output/experiments/M1_MCCA_FROZEN_ADAPTER_3EP_20261011/`.
- Source snapshot SHA256: `104f8c4300a3765ed1a62943e96fd52d6f62e9ecf54fbf4175f0a15f81a20b63`.
- Evaluation source snapshot SHA256: `a2929bcb7c8926ccdcf653458da2c44bda932917aac8fbfad73d13e3dfa3cc65`; LasHeR loader SHA256: `46d25426107c755dd6eba0184980d83140460fff38052487f108ee3f273a2e9c`; resumed runner SHA256: `aa714baa4cf4c48985b54531b01452ae01915df8567fb82b6c69eb03991fdf46`.

## Results

Both fixed-checkpoint arms completed 96/96 sequences and 120,485 frames. The protected M1 parent scored AUC/PR/NPR `65.6715/79.1348/76.7483`; adapter-only MCCA scored `64.6416/77.7137/75.5188`. Candidate minus parent was AUC `-1.0300`, PR `-1.4211`, NPR `-1.2296`, OP50 `-1.2743`, and OP75 `-0.8045`. Sequence-level AUC had 56 wins and 40 losses, but the mean was pulled down by large regressions: `man2startmoto -50.71`, `boyleave -48.17`, `firstleftrunning -14.04`; largest gains were `redbackpackgirl +18.75`, `blkbikefromnorth +12.49`, and `mirrorfront +10.10`.

Paired 50,000-resample bootstrap 95% intervals crossed zero for all metrics: AUC `[-2.8091, 0.3797]`, PR `[-3.7027, 0.3656]`, NPR `[-3.3389, 0.4372]`, OP50 `[-3.4508, 0.4431]`, OP75 `[-2.4035, 0.4607]`. The candidate ran at observed `66.84 FPS` versus M1 `73.25 FPS`; this is a single serial pass, not a repeated latency estimate.

## Interpretation

The result rejects H1 for this exact contract: standard tracking supervision can update MCCA while the M1 parent is frozen, but it does not produce a reliable held-out closed-loop gain. It also separates two issues. This is not a connectivity failure: P0, nonzero gradients, adapter updates, and frozen-parent integrity all passed. The likely remaining problem is objective/interface alignment: the adapter receives only indirect focal/L1/GIoU supervision at the template message stage, while the measured failure appears in search-time peak selection and closed-loop state transitions. The few large gains and losses show that the learned correction is scene-sensitive but not safely controlled. Adding an auxiliary loss is therefore a plausible next hypothesis, not a conclusion; it should only be evaluated under a new preregistered contract with a supervision target tied to message reliability or target-vs-hard-negative localization. Do not tune that loss on these 96 sequences or test-244.
- Training-only implementation file hashes: `base_functions.py` `ad219acce1793c7485a53d09c043c1f2e1290ae0e9f3b5cc6e6ae05f801b80de`; `train_script.py` `c48ac7f5c71f5b09aee7cb67ed7d2927aa8cf0e618813561b418c3d453a15e9d`; `ltr_trainer.py` `b58ddb3944e79e66f0731aac9353ecc3edbeae02eaafdee738279fd08c4c198a`; `tbsi_track.py` `663b7c4e051bc6bd0845d2e00ca4752b3479e6bc6b0f8827a66ec17b6be97554`; config defaults `config.py` `ab7cef8be8fecdf1c8eaac5010ac7af7a5347561e7657c8ceeeef5580100564c`.
- Runtime environment, exact hashes, commands, P0, training, paired evaluation, and complete metrics will be appended after execution.
