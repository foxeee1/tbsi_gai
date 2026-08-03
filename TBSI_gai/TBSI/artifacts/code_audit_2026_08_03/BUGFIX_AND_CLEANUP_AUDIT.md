# TBSI Experiment Harness Bugfix and Cleanup Audit

Date: 2026-08-03

## Conclusion

The previous negative-result pattern was not solely a modeling issue. The
repository contained several experiment-harness hazards that could silently
invalidate training or evaluation. The major issues have been fixed in the
current code path, but historical results produced before these fixes should be
treated as low-trust unless rerun.

## Fixed High-Risk Issues

- Evaluation no longer reuses existing per-sequence result files unless
  `--resume_results` is explicitly passed.
- Sequence failures now raise by default instead of being silently skipped.
- Sequence timeouts now raise by default instead of writing first-frame fallback
  boxes, which could corrupt metrics.
- Tracker checkpoint loading reports missing/unexpected keys and fails by
  default on config/checkpoint mismatch.
- Training resume now reports missing/unexpected keys and fails by default on
  config/checkpoint mismatch. Use `TBSI_STRICT_TRAIN_RESUME=0` only for an
  intentional architecture-mismatch warm start.
- Stage-2 fine-tuning now freezes by explicit module whitelist and fails if an
  enabled module has no trainable parameters.
- Stage-2 optimizer now assigns all unfrozen lightweight-module parameters to
  `TRAIN.LR`, preventing accidental backbone-multiplier learning rates.
- Trainer `fail_safe` is disabled by default and no longer reports successful
  completion after a crash.
- GIoU failures now raise by default instead of silently replacing tracking loss
  with zeros.
- `tracking/test_exp.py` is now a compatibility wrapper around the safe
  `tracking/test.py` entrypoint instead of a hard-coded unrelated-dataset
  script.

## Removed Deprecated Artifacts

- Removed old/unfixed v5-v7 experiment output directories:
  `v5.0.0-cutr-lite-routeronly-*`, `v6.0.0-rtm-only`,
  `v6.1.0-rtm-aligned-local`, `v7.0.0-egir-only-smoke`,
  `v7.1.0-rsm-only-smoke`, and `v7.1.1-rsm-v2-imbalance`.
- Removed abandoned HPF Gate-1 helper scripts.
- Removed superseded v5.0.0 YAML configs. The fixed rerun configs are kept.

## Remaining Non-Blocking Notes

- Some `strict=False` calls remain in model initialization because warm-starting
  from a pretrained/base checkpoint into a model with newly added lightweight
  modules necessarily creates expected missing keys. These paths now validate
  unexpected or non-module missing keys where they affect Stage-2 experiments.
- Diagnostic logging failures in FCC are still ignored intentionally because
  they do not alter model outputs.
- Git still shows many deleted tracked `output/experiments/.../test_results`
  files from previous cleanup. These are experiment artifacts, not source-code
  changes; do not mix them into a clean code commit unless the repository policy
  is to stop tracking generated result files.

## Verification

- `python -m py_compile` passed for the core training, testing, model, actor,
  and tracker files after the new fixes.
- Fixed v5-v7 configs remaining under `experiments/tbsi_track`:
  `v5.0.1-cutr-lite-routeronly-fixed.yaml`,
  `v6.0.1-rtm-only-fixed.yaml`,
  `v6.1.1-rtm-aligned-local-fixed.yaml`,
  `v7.0.1-egir-only-fixed.yaml`,
  and `v7.1.2-rsm-v2-imbalance-fixed.yaml`.
