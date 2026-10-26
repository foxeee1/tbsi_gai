# M1 AMCM NLformer Contract Audit

## Scope

The official NLformer three-branch attention core was added as a new
`MODE=nlformer` behind the existing AMCM interface. No training, tracking
evaluation, backbone, TBSI, `tbsi_fuse_search`, CENTER head, tracker state, or
external loss was changed.

The wrapper packs RGB/TIR search maps into `2C` channels, applies the official
local first-order attention, second-order channel covariance, and low-rank
complementary branches, then splits the projected output back into two `C`
channel streams. A zero-initialized template summary projection is exposed as
the minimal conditioning adapter. The residual scale `nlformer_alpha` starts
at zero.

## Contract results

The deterministic CPU audit used `C=768`, search `[1,768,16,16]` for both
modalities and template `[1,64,768]` for both modalities.

| Check | Result |
|---|---:|
| Output shapes | `[1,768,16,16]` for both streams |
| Identity max absolute error at `alpha=0` | `0.0` |
| Save/load max absolute error | `0.0` |
| Parameter count | `26,719,659` |
| Local branch gradient | `2.11e-7` mean absolute |
| Covariance branch gradient | `1.55e-7` |
| Complementary branch gradient | `2.39e-7` |
| Template adapter gradient | `1.28e-5` |
| Residual-scale gradient | `1.48e-4` |
| Enabled residual L2 (`alpha=0.1`) | `1.3522` |
| Spatial standard deviation of residual | `3.98e-5` |
| Status | `PASS` |

All intended paths are differentiable and the disabled path is exact identity.
The three branch outputs are nonzero, so this is a valid operator contract
rather than a dead or constant gate.

## Cost decision

The initial direct packed implementation introduced `26.72M` parameters at
`2C=1536`, so it was not admitted to training. A fixed-width internal
projection was then added inside AMCM: `2C -> 256`, the official core runs at
256 channels, and the output is projected back to `2C`.

The bottleneck audit now reports `2,576,811` parameters, identity error `0`,
reload error `0`, nonzero local/covariance/complementary/template gradients,
enabled residual L2 `1.2798`, and spatial residual standard deviation
`5.37e-5`. It passes the same contract without retaining the direct 26.72M
cost. This bottleneck version is the only version eligible for the next fixed
failure-frame signal diagnostic; no training has started yet.

Evidence: `output/experiments/M1_AMCM_NLFORMER_CONTRACT_AUDIT_20261020/contract_audit.json`.
