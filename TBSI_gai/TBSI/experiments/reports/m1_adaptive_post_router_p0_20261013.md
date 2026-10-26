# M1 adaptive post-fusion router P0 (2026-10-13)

## Scope

Implemented the first bounded adaptive post-fusion prototype. It keeps the M1 fused feature as the identity expert, predicts a `16x16` RGB/TIR route and a global need gate, and adds only zero-initialized, tanh-bounded RGB/TIR residuals. It does not modify MHSA, CENTER, or the M1 backbone.

The independent configuration is `experiments/tbsi_track/M1_ADAPTIVE_POST_ROUTER_P0_seed42.yaml`. The feature is disabled by default in the base configuration.

## Contract checks

- Router module shape check: passed.
- Router backward/gradient check: passed.
- Zero-initialized residual identity: passed (`residual_norm=0`).
- Disabled M1 model vs enabled P0 model on identical random feature input: `score_map max_abs_diff=0.0`.
- M1 legacy model construction and P0 model construction both passed.

The initial need gate is intentionally not treated as a result: it is only a learnable route input and the residual is exactly zero before training.

## Quick training contract

The first quick launch was invalidated before training because the incremental YAML omitted the M1 backbone fields. A second launch also exposed BatchNorm buffer drift under the nominal router-only freeze. The trainer was corrected to keep the full M1 network in `eval()` and switch only the router to `train()`.

The valid V2 run used 16 train samples, 4 validation samples, one epoch, batch size 2, and four DataLoader workers. Training and validation completed with finite losses and IoU values. The saved checkpoint contains exactly 12 changed keys, all under `adaptive_post_fusion.*`; all M1/backbone/head parameters and buffers match the protected epoch-15 checkpoint bitwise.

This is an optimization and state-contract test, not a tracking-performance result. The router was trained with ordinary tracking loss on a tiny proxy and has not been evaluated on held-out tracking sequences.

V2 artifact: `output/experiments/M1_ADAPTIVE_POST_ROUTER_P0_QUICK_V2_20261013/`.

## Evidence boundary

No performance claim is made yet. Evaluating this module before utility/no-op supervision would be an untrained-random or zero-residual comparison. The next experiment must train only the router on independent train-side states with three-way labels `{no-op, RGB, TIR}` and abstention, then evaluate on held-out sequences with no M1 parameter updates.
