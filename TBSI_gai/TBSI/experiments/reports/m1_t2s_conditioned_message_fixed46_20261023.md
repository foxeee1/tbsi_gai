# M1 T2S Conditioned Message Fixed-State Diagnostic (2026-10-23)

## Research question

Can early foreground-versus-local-background evidence modulate the search-position-aligned template-to-search message and improve CENTER selection under the fixed M1 calibration states?

## Contract

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42` epoch15 checkpoint.
- Cohort: 23 calibration events, pre-onset and severe-onset states, 46 states total.
- FP32, one GPU, torch threads 1, identical frame/crop/state/template inputs.
- Arms: parent alpha=0; target-conditioned alpha=0.25; equal-density shifted control alpha=0.25.
- Locked holdout and training were not opened.

## Execution result

The first implementation smoke exposed that `tbsi_e` returns immediately after `V2TemplateFusion`; the `CASTBlock(mode='t2s')` path is not executed for the protected M1 route. The later helper therefore operated on the returned search features before the downstream neck. It is a spatial residual diagnostic, not a valid T2S-message intervention.

The corrected helper was run on all 46 states to quantify this contract failure:

| Arm | CENTER margin mean | Fixed replay IoU mean | CENTER rank change |
|---|---:|---:|---:|
| Parent | 0.190498 | 0.469170 | reference |
| Target residual | 0.190552 | 0.469108 | 0 / 46 |
| Shifted residual | 0.190029 | 0.468958 | 0 / 46 |

Paired to parent, target changed margin by `+5.37e-05` and IoU by `-6.26e-05`; shifted changed margin by `-4.69e-04` and IoU by `-2.12e-04`. These are diagnostic perturbations only and do not establish the proposed T2S mechanism.

## Decision

`DIAGNOSTIC-COMPLETE / INJECTION-CONTRACT-FAILED`. The target and shifted controls are numerically distinguishable, but the experiment did not reach the intended CAST T2S message. No training or full tracking evaluation is admitted.

## Next allowed action

Refactor the `tbsi_e` route so that the intended template-to-search message is explicitly computed on the 256 search positions after template fusion, then repeat alpha=0 parity, target, and shifted fixed-state controls. Keep Q/K, CENTER, size/offset decoding, top-k, dilation, and locked holdout unchanged.

## Native MSHA follow-up

The native `V2TemplateFusion` cross-value gate was then implemented and screened. At alpha `0.25` over all 46 states, target-conditioned gating changed CENTER margin by `-2.80e-06` and fixed replay IoU by `-5.67e-06`; shifted control changed them by `+2.19e-05` and `-9.74e-06`. CENTER rank changed in `0/46` states. A stronger alpha `1.0` two-state smoke produced the same target and shifted output because the direction-level scalar uses peak-minus-hard-background, which is invariant to spatial permutation.

This is an interface representation failure: native template MHSA has 64 template Value positions, while the evidence that should affect CENTER is defined on 256 search positions. A scalar direction gate cannot provide a meaningful spatial shifted control or directly alter the search peak location. Do not train this branch.

## Fixed directional message controls

The native cross-value interface was then tested without any evidence gate. Each direction was independently scaled by `0.75` and `1.25` over the same 46 states:

| Direction | Scale | Margin delta | Fixed replay IoU delta | CENTER rank changes |
|---|---:|---:|---:|---:|
| RGB -> TIR | 0.75 | -0.000276 | -0.01025 | 1 |
| RGB -> TIR | 1.25 | +0.000139 | -0.00870 | 3 |
| TIR -> RGB | 0.75 | +0.000663 | -0.00030 | 2 |
| TIR -> RGB | 1.25 | -0.000846 | +0.00030 | 0 |

The message interface is therefore causally active, but a direction-level amplitude change does not provide a stable localization improvement. Margin and IoU often move in different directions. This rejects the next hypothesis that a simple learned or hand-tuned direction strength is sufficient; it does not reject MSHA as an insertion point.

The next MSHA experiment must condition the message on a target-versus-competitor relation that is mapped to the native template-query axis, or use a query-level intervention with an explicit 64-token contract. Do not start short training from the current scalar gate.

## Route decision

Priority 1 should branch explicitly:

1. Preserve M1 and abandon native MSHA-only conditioning as a route to CENTER spatial ranking; focus on the CENTER peak-selection/Hann mechanism already supported by the geometry crossover.
2. If a true target-conditioned T2S message remains the research objective, add a new explicit search-position message path as a new architecture, with its own identity contract and training experiment. It cannot be described as a modification of the existing native MSHA Value message.

Evidence directories:

- `output/experiments/M1_T2S_CONDITIONED_PARENT_CALIB46_20261023/`
- `output/experiments/M1_T2S_CONDITIONED_TARGET025_CALIB46_20261023/`
- `output/experiments/M1_T2S_CONDITIONED_SHIFTED025_CALIB46_20261023/`
- `output/experiments/M1_T2S_CONDITIONED_TARGET025_SMOKE_20261022/` (pre-fix no-op path)
- `output/experiments/M1_T2S_CONDITIONED_TARGET025_SMOKE3_20261023/` (post-fix smoke)
