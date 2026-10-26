# Candidate 2 Learnability Diagnostic Plan

## Objective

Estimate whether Candidate 2's negligible fixed-state effect came from only eight optimizer updates or from a weak coupling between the learned bridge message and localization outputs.

## Contract

- Run ID: `M1-SPATIAL-TOKEN-BRIDGE-LEARNABILITY-10EP-20261024`.
- Parent: protected M1 epoch15; its YAML and checkpoint remain read-only.
- Train only the enabled `SpatialTokenBridge`; no neck, backbone, or head parameter updates.
- LasHeR train proxy: 16 samples/epoch, batch 2, 10 epochs, 80 optimizer steps, seed 42, FP32, 4 workers.
- Save epochs 1, 5, and 10 under an independent output directory.
- Fixed-state follow-up: same 23-sequence/46-state manifest, FP32 serial, bridge off/target/shifted for each saved checkpoint.

## Hypotheses

- H0: Additional optimization exposure does not create useful target-conditioned messages.
- H1: Eight steps underfit the branch; longer bridge-only optimization increases useful messages and improves target-over-shifted readouts.
- H2: Messages grow but do not affect CENTER/query rank or replay, indicating weak coupling under the current insertion/readout contract.

## Decision Gates

- Stop on non-finite losses/gradients, missing bridge gradients, frozen-parent drift, or failed checkpoint loading.
- Do not claim performance from this proxy or train-side calibration cohort.
- Advance to a separate M1-aligned full experiment only if the 10-epoch checkpoint has a target-over-shifted margin gain above `1e-3`, positive on a majority of sequence means, with no CENTER-rank or replay-IoU regression.
- If residual norms stay near the previous `~0.001` token norm and all readouts remain tied, stop the current operator. If residuals grow but readouts stay tied, classify the current interface as weakly coupled and stop before full training.

## Commands

Training:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
/root/autodl-tmp/conda_envs/tbsi/bin/python -u tracking/train.py \
  --script tbsi_track --config M1_SPATIAL_TOKEN_BRIDGE_LEARNABILITY_10EP_20261024 \
  --save_dir output --mode single --use_lmdb 0 --use_wandb 0
```

Fixed-state evaluation will use `tools/run_m1_screening_fixed_state_flow_20261022.py` with explicit config/checkpoint, fixed event and sequence manifests, and modes `off`, `target`, `shifted`.

## Provenance

- Config SHA256: `f4294e536f55e7060f3baa60efc6a8446d3c49ed8922fbe8ee284d524b161dce`.
- Protected M1 YAML SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`.
- Protected M1 checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Code source bundle SHA256: `3ccce72120ac9bef00828f6b3cc5224304d543e581fe37b4375d22300af70051`; its six tracked source/config members were hash-checked against the current tree before launch.
- Preregistration: `experiments/protocols/m1_spatial_token_bridge_learnability_20261024/preregistration.md`.

## Outcome

- The 10-epoch bridge-only run completed 80 optimizer steps and saved epochs 1, 5, and 10. Gradients and parameter updates were present.
- On the same 46 fixed states, mean RGB/TIR bridge delta norms increased from `0.000994/0.000483` at epoch 1 to `0.002016/0.001264` at epoch 5 and `0.002633/0.001563` at epoch 10.
- At epoch 10, target-minus-shifted means were `-1.308e-6` for CENTER GT margin, `-1.046e-6` for post-fusion query GT margin, and `-1.183e-6` for replay IoU. CENTER rank and query rank were unchanged on all 46 states. Target-minus-off was also negligible: `-1.018e-6`, `+5.701e-8`, and `+2.325e-7`, respectively; ranks remained unchanged.
- The target-over-shifted margin admission threshold (`>1e-3`) was missed by roughly three orders of magnitude. This supports H2 for the current injection/readout contract and stops this operator before full training. It does not establish a full tracking performance result or prove that all spatial correspondence fusion is ineffective.
- Detailed paired analysis: `experiments/reports/m1_spatial_token_bridge_learnability_20261024.md`.
