# AMCM boundary signal audit (2026-10-20)

## Scope and identity

- Parent: M1 epoch 15; candidate under audit: `M1_AMCM_TMATCH_RESIDUAL_CENTER_QUICK15_3EP_20261019`, epoch 3.
- Data: the same 10 selected LasHeR testingset sequences/frames, identical M1 `state_before`, crop and template. This is fixed-state analysis, not closed-loop sequence performance.
- Source SHA256: `a0f4ddc1197f2277267a544551ec630c51d711d84c610df25463225bcbed0550`.
- Config SHA256: `0cbf8e014fb1abba4c4dfe05ea9d1464f7b9be18f93b219e89ed1b28a931fdd3`.
- Checkpoint SHA256: `3601738bf2ffe0ba382ebb1084a35f0dddabbb43b613937e205d0bad958f70df`.
- Existing fixed-state capture script SHA256: `b775ffb89d96f20ebbd523e4b31705b942fa5abd6ee5d491a572e89732aef3bf`. Audit was run by in-memory hooks around this path; no model/tracker/actor source was changed.
- Environment: 1x NVIDIA GeForce RTX 4090 D; Python 3.8; PyTorch 2.0.0+cu118; CUDA 11.8; FP32 serial. No training or full-sequence test was run.

## Measurement definitions

1. Template-map AUC ranks every GT-region cell against every non-GT cell using the live AMCM `search-to-all-template-token` similarity map. AUC `0.5` is chance; below `0.5` means the map scores GT cells lower than background. The max margin is `max(sim in GT) - max(sim outside GT)`.
2. Loss direction is the first-order derivative `sum((dL/d corrected_feature) * current_residual)` separately over GT and non-GT cells, divided by the number of cells. Negative means scaling the present residual infinitesimally lowers the local tracking objective; positive means it raises it. The objective matches the actor terms for this config: focal + `2 * GIoU` + `5 * L1`, with no AMCM auxiliary loss.
3. Fusion transfer compares the exact candidate fused map against the same candidate model with AMCM bypassed, using identical backbone tokens and TBSI fusion weights. Relative perturbation is RMS(`delta fused`)/RMS(`base fused`) by GT/background locations. This measures the actual finite-difference effect at the fusion boundary, not a Jacobian estimate.

## Results

**Template signal is systematically reversed on this event set.** RGB and TIR map AUC means are `0.144` and `0.106`; no one of the 20 modality/event maps is above chance. Mean `max(GT)-max(background)` is `-0.284` RGB and `-0.274` TIR. The AMCM has template and search tokens, but this implementation compares each search location to all 64 template tokens without an explicit foreground/background token mask. On these frames, background tokens dominate the target-conditioned signal.

**The tracking objective reaches AMCM but does not give a consistent modality correction direction.** RGB residual direction lowers the local loss at GT in 7/10 events; TIR does so in only 4/10. At non-GT cells, TIR direction lowers loss in 10/10 and RGB in 6/10, while the GT directional derivatives are inconsistent in sign and often opposite across modalities. Training logs separately confirm nonzero gradients for all 16/16 AMCM parameter tensors, but this local directional test shows that nonzero gradients do not mean target-selective supervision. The trained residual magnitude was also below the feature scale: from the paired fixed-state trace, mean residual/input RMS was `0.015%` RGB and `0.028%` TIR.

**TBSI fusion does not erase the correction.** Input perturbation RMS relative to the original concatenated stream was `0.0094%` at GT and `0.0250%` outside GT. After `tbsi_fuse_search`, perturbation relative to the parent fused feature was `0.0201%` and `0.0537%`, respectively. The measured RMS transfer factors were `5.00x` at GT and `3.73x` outside GT. Thus the signal is already tiny and background-heavy before fusion; the fusion operator amplifies its relative effect rather than attenuating it. The fused change is still very small in absolute feature scale and is larger outside the target (GT/background delta ratio `0.60` on average).

| Sequence | Match-map AUC RGB/TIR | Max margin RGB/TIR | Loss derivative `grad·res` at GT RGB/TIR | Loss derivative at BG RGB/TIR | Fused delta relative RMS GT/BG | Fused delta GT/BG ratio |
|---|---:|---:|---:|---:|---:|---:|
| `7rightorangegirl` | 0.318/0.256 | -0.132/-0.241 | -1.70e-05/+7.53e-05 | -5.92e-07/-3.22e-06 | 2.80e-04/4.22e-04 | 0.785 |
| `ab_bolstershaking` | 0.042/0.112 | -0.470/-0.237 | +5.73e-06/-3.39e-06 | +7.38e-08/-1.87e-06 | 9.09e-05/6.75e-04 | 0.216 |
| `blkboydown` | 0.285/0.081 | -0.181/-0.143 | -6.66e-06/+9.26e-05 | +9.81e-09/-1.80e-06 | 3.28e-04/4.36e-04 | 0.794 |
| `boyss` | 0.083/0.056 | -0.256/-0.305 | +4.05e-06/+4.53e-05 | -6.79e-07/-9.73e-07 | 1.97e-04/5.10e-04 | 0.595 |
| `foamatgirl\`srighthand` | 0.105/0.069 | -0.467/-0.379 | +1.10e-06/+8.28e-07 | +4.43e-07/-1.07e-06 | 7.94e-05/8.60e-04 | 0.132 |
| `girldownstairfromlight` | 0.154/0.003 | -0.156/-0.362 | -4.95e-07/-4.15e-07 | -5.90e-07/-1.41e-06 | 1.97e-04/4.31e-04 | 0.756 |
| `leftopenexersicebook` | 0.178/0.062 | -0.230/-0.352 | -3.08e-06/-2.81e-05 | -6.23e-07/-2.71e-06 | 1.99e-04/5.02e-04 | 1.050 |
| `rightbluewhite` | 0.171/0.132 | -0.245/-0.167 | -1.12e-06/+2.02e-05 | +5.00e-07/-3.24e-06 | 2.43e-04/5.40e-04 | 0.592 |
| `rightdarksingleman` | 0.006/0.031 | -0.434/-0.375 | -2.29e-06/-3.06e-06 | -4.87e-07/-7.56e-07 | 1.61e-04/4.10e-04 | 0.545 |
| `umbrella` | 0.104/0.263 | -0.271/-0.176 | -2.00e-05/+4.05e-05 | -2.74e-06/-2.21e-06 | 2.31e-04/5.86e-04 | 0.539 |

## Decision

Decision: `CURRENT-DIRECTED-RESIDUAL-AMCM-NOT-FIT-FOR-TARGET/BACKGROUND-LOCALIZATION`. Three sequential variants already failed to change GT rank/top-1; this audit locates the common issue before the fusion block. Do not add a downstream module to compensate for fusion attenuation: that attenuation is not observed. Stop promoting or expanding the current AMCM branch for this goal. Keep the code/checkpoints as historical evidence rather than deleting them.

This does not prove every AMCM design or the separate legacy gate mode is useless. It does show that the current all-template-token similarity plus two-stream residual design is a poor target/background estimator under the tested data and loss. If the research continues, replace this estimator with a target-vs-context signal that has an explicit foreground token definition and train it against the same hard-background competition measured at inference. That would be a new design and objective, not another scale/routing adjustment to this AMCM. Validate that proposal first with fixed-frame map AUC and gradient-direction diagnostics before any training.

Artifacts: `../output/experiments/M1_AMCM_BOUNDARY_SIGNAL_AUDIT_20261020/{per_sequence.csv,summary.json}`; parent fixed-state tensors and event manifest under `../output/experiments/M1_AMCM_TMATCH_RESIDUAL_CENTER_FIXED10_DIAG_20261019/`.
