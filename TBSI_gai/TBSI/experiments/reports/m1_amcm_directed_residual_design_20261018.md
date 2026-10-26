# AMCM directed residual redesign (2026-10-18)

## Problem in the previous E2 path

The previous post-fusion path predicted RGB/TIR deltas, averaged them, and multiplied the already fused feature. That operation could not change the relative modality contribution. Its supervision trained a signed modality difference while inference used the unsigned mean.

## New formulation

Let `x_r, x_t` be RGB/TIR search feature maps and `t_r, t_t` their template tokens. A shared context encoder produces `c_r, c_t`. The reliability predictor receives:

`z = [c_r, c_t, |c_r-c_t|, cos(x_r,t_r), cos(x_t,t_t)]`

and outputs logits `q_r, q_t`. The modality reliability distribution is:

`p = softmax([q_r,q_t])`.

Two learned message networks produce directional residuals:

`m_r = tanh(M_r(z)) * alpha`, `m_t = tanh(M_t(z)) * alpha`.

The corrected streams are:

`x'_r = x_r + p_t * m_r`

`x'_t = x_t + p_r * m_t`.

The corrected streams are then concatenated and passed to the original `tbsi_fuse_search`. A modality with lower estimated reliability receives a message routed from the other modality; the original fusion operator still determines the final representation.

At the ground-truth cell, the frozen similarity scores define a target distribution:

`p* = softmax(tau [s_r,s_t])`.

The reliability objective is:

`L_rel = lambda KL(p* || p)`.

This target supervises exactly the probability used in the residual routing. Tracking losses remain active, so the residual must also preserve localization performance.

## Implementation

- New feature gate: `MODEL.AMCM_RELIABILITY.MODE: directed_residual`.
- Legacy E2 mode remains the default and is unchanged.
- New config: `experiments/tbsi_track/M1_AMCM_DR_DIRECTED_RESIDUAL_PROXY_20261018.yaml`.
- New parameters are isolated under `amcm_reliability.*`.
- Contract check passed: shape preservation, probability normalization, message and gate gradients.
- Proxy training: 3 epochs, 256 samples/epoch, seed42, FP32, worker1.
- Training `Loss/amcm_supervision`: `0.01870` at epoch1, `0.01042` at epoch3.
- Validation IoU: `0.65301` at epoch1, `0.67032` at epoch3.

These proxy observations establish trainability only. A fixed-state gate trace and matched tracking comparison are still required before interpreting sequence-level effects.
