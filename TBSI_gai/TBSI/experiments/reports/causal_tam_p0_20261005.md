# Causal TAM Pilot P0 (2026-10-05)

## Scope

This is the first repository-side integration check for the tracker-specific
causal TAM adaptation. It is not a reproduction of the TAM paper and it does
not claim a tracking result. The canonical M1 YAML and its output directory
were not modified.

## Contract

- Module: `lib/models/layers/causal_tam_adapter.py`
- Injection: fused search representation immediately before `box_head`
- Input: `[B, 256, 768]` current and previous head features
- Geometry: actual current/previous crop rectangles in image-space `xywh`
- Causality: history frame ID must be strictly smaller than current frame ID
- Gate: scalar `gamma`, initialized to zero
- Default: `MODEL.CAUSAL_TAM.ENABLE=False`
- Experiment config: `experiments/tbsi_track/G0_CAUSAL_TAM_PILOT_seed42.yaml`

## Results

The isolated P0 passed in the repository Python 3.8 / PyTorch 1.9 environment
on CPU. Results are in `output/experiments/causal_tam_p0_20261005/p0.json`:

- zero-gate identity: pass
- no-history identity: pass
- nonzero gate changes output: pass
- no geometric overlap returns identity: pass
- future history is rejected: pass
- gate gradient at zero: pass
- Q/K branch gradient after gate update: pass
- finite output and bounded message ratio: pass (`0.10`)
- adapter parameters: `2,129` for the synthetic `dim=32` check

The actual TBSI model construction smoke test also passed with the M1 epoch-15
checkpoint. The enabled model reported `tam_raw_head_feature=[1,256,768]`,
`tam_raw_head_map=[1,768,16,16]`, `pred_boxes=[1,1,4]`, and TAM parameters
`591,169` for `dim=768, hidden=192`.
The expected missing checkpoint keys were only the new `causal_tam.*` keys.

The first real tracker attempt exposed and fixed a cache contract bug: the
tracker initially cached the `[B,768,16,16]` map while TAM requires
`[B,256,768]` tokens. After changing the explicit cache output to tokens, the
same `bike2left` sequence completed successfully. The matched disabled and
enabled-gamma-zero runs both produced 257 frames and had maximum and mean
absolute box differences of `0.0`.

The enabled run measured about `38.85 FPS`; the disabled matched control was
about `53.33 FPS`. This is a one-sequence implementation observation, not a
performance result.

## Boundary

This is a single-sequence tracker parity result, not a benchmark result. No
training, G0 information-source comparison, or full-244 evaluation was run.
The next authorized step is crop-coverage/state-preservation checking over the
fixed train-side P0 sample, followed by the fixed train-side
`Hreal/Hcopy/Htime/Hxy` G0 only if those checks pass.

## Provenance

- Python: `/root/autodl-tmp/conda_envs/tbsi/bin/python`
- PyTorch: 1.9.0+cu111 environment
- Hardware contract: 1x RTX4090D 24GB; this P0 used CPU for the isolated check
- FP32 tracker contract: `TBSI_INFER_FP16=0`
- Canonical YAML SHA256: `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`
