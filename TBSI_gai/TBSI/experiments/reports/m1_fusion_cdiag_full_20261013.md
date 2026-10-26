# M1 Cdiag full confirmation (2026-10-13)

## Contract

- Parent: `TBSI_E_MHSA_FULL_4090D_seed42`, fixed epoch-15 checkpoint, seed 42.
- Candidate: `M1_FUSION_CDIAG_FULL_15EP_B64_seed42`.
- Initialization: exact M1 checkpoint. `TRAIN.CPLAIN_ONLY=True` freezes M1 and leaves only four `post_tbsi_residual` tensors trainable (`gamma_rgb`, `gamma_tir`, `rgb_from_tir.weight`, `tir_from_rgb.weight`).
- Data: full `LasHeR_train`, `SAMPLE_PER_EPOCH=59904`, 15 epochs; full 244-sequence `lasher_test` official evaluation after training.
- Effective batch: 128 in both parent contract and candidate; candidate uses batch 64 with accumulation 2 for throughput. `NUM_WORKER=1` is recorded as a resource deviation from historical M1 worker 4.

## Attribution evidence

Training logs record, at the same sampled optimizer steps:

- `gradient_flow.csv`: total and trainable-parameter gradient norms, zero-gradient count.
- `loss_components.csv`: total, GIoU, L1, localization and IoU terms.
- `internal_signals.csv`: TBSI RGB/TIR template/search feature norms.
- `feature_changes.csv`: Cdiag input/message/output norms, residual delta norms and both gamma values.

The runner executes candidate training, candidate tracking, and the official `calc_seq_err_robust` audit serially. It refuses to overwrite the protected parent or an existing candidate result directory.

## Interpretation gate

The final claim is allowed only when the official audit has finite AUC, Precision, Norm Precision, OP50, OP75 and per-sequence deltas for all 244 sequences. A positive result must also show that M1 tensors remain unchanged and that the gain is not confined to a selected proxy cohort. If the candidate loses or the residual saturates without full-test gain, the result is recorded as a clean refutation of this post-TBSI Cdiag contract, not as evidence against all fusion methods.

Current run artifact: `output/experiments/M1_FUSION_CDIAG_FULL_15EP_B64_20261013/`.
