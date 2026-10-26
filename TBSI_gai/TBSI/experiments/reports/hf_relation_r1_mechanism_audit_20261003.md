# HF-Relation-R1 mechanism audit and route decision

## Scope

The selected route retained the HF-Relation MHSA-logit interface and tested whether query/key reliability calibration should precede a second full training run. R0 was audited with the existing epoch-15 `HF_RELATION_LOC9_seed42_retry1` checkpoint on 24 frozen LasHeR train sequences and two fixed events per sequence, for 48 events total.

Conditions were evaluated from the same tracker state and RNG snapshot at each event: `normal`, `beta0`, `shuffle`, and `sign_flip`.

The proposed R1 implementation was checked independently with synthetic tensors. It uses fixed Haar high-pass reconstruction, top-8 query/key reliability calibration, scalar bounded direction coefficients, and cross-modal logits only. It does not modify Q/K/V, Value, output residuals, masks, or future-IoU signals.

## P0

`tools/p0_hf_relation_r1.py` passed: bias shape `[2, 12, 128, 128]`; finite output; same-modal blocks exactly zero; zero-beta output exactly zero; beta gradients nonzero (`8.15e-5` and `9.41e-5`); and bounded nonzero bias (`max_abs=0.2012`, configured bound `0.25`). This validates the operator contract only, not tracking performance.

## R0 audit

All 48 events completed. The learned R0 bias was numerically present but small: mean absolute bias `0.000483`, maximum absolute bias `0.01334` across active-layer snapshots.

| Condition | Mean delta IoU vs normal | Median | Positive / negative events |
|---|---:|---:|---:|
| normal | 0 | 0 | 0 / 0 |
| beta0 | +0.00000218 | 0 | 22 / 20 |
| shuffle | -0.00000230 | -0.00000313 | 15 / 27 |
| sign_flip | +0.00000030 | 0 | 21 / 21 |

The perturbation responses are effectively zero at event level. `normal` is not better than `beta0`, and sign reversal does not produce a meaningful counter-response. The shuffle difference is at numerical-noise scale relative to the tracking output.

## Decision

The serial campaign gate is `R0_DENIED`. The R0 full-run improvement cannot currently be attributed to active use of the learned frequency relation; it may instead be a training trajectory or another small optimization effect. The campaign therefore stopped before R1 full training. No second seed, spatial control, FACMA/WDA integration, Value injection, message residual, or additional frequency module was launched.

## Evidence and provenance

- Serial entrypoint: `tools/run_hf_relation_r1_serial.sh`
- R0 audit: `tools/audit_hf_relation_r0.py`
- R0 summary: `output/experiments/hf_relation_r0_mechanism_audit_20261003/summary.json`
- R0 event rows: `output/experiments/hf_relation_r0_mechanism_audit_20261003/event_results.csv`
- R1 P0: `output/experiments/hf_relation_r1_p0_20261003.json`
- R1 config (implemented but not trained): `experiments/tbsi_track/HF_RELATION_R1_LOC9_FULL_MATCHED_seed42.yaml`
- R0 checkpoint SHA256: `8936b1b69e27d56fac92bf7da829177aa22336d38163a055d95da3545f270a0b`
- Git: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` plus working tree
- Hardware: 1 x RTX4090D 24GB; Python 3.8.20; PyTorch 1.9.0+cu111; FP32 tracker inference
- Canonical baseline YAML remained unchanged.
