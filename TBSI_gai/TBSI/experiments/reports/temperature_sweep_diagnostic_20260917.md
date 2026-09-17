# B3 Temperature Calibration Diagnostic

Date: 2026-09-17

## Protocol

T0 raw-logit temperature sweep on the same corrected B3 path, using the frozen
S1 epoch-3 checkpoint, the frozen 60-sequence RPP-v2 test manifest, FP32
inference, and at most 20 frames per sequence. No training was performed.

The only intervention was `logits / tau` in V2 template MHSA. The canonical
baseline YAML was unchanged.

## Results

| tau | entropy | RT R8 | RT R16 | RT R32 | target-pair purity | top-10 target purity |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 4.62364 | 0.29750 | 0.47964 | 0.73019 | 0.06800 | 0.31523 |
| 0.7 | 4.73341 | 0.24032 | 0.41073 | 0.67071 | 0.06449 | 0.34348 |
| 1.0 | 4.79355 | 0.20093 | 0.35960 | 0.62186 | 0.06180 | 0.35877 |
| 1.3 | 4.81741 | 0.18130 | 0.33280 | 0.59442 | 0.06035 | 0.36721 |
| 1.5 | 4.82604 | 0.17297 | 0.32112 | 0.58203 | 0.05972 | 0.37034 |
| 2.0 | 4.83743 | 0.15995 | 0.30251 | 0.56170 | 0.05869 | 0.37320 |

## Decision

Temperature changes concentration as expected, but the target-pair purity
gain at the sharpest setting is only `+0.62` percentage points over tau=1.0,
below the pre-registered `2-3` percentage-point threshold. Top-10 target
purity also declines at tau=0.5. The result is therefore diagnostic-only and
does not justify fixed-temperature or learnable-temperature training.

## Evidence

- `output/experiments/temperature_sweep/logs/tau_0.5.jsonl` through `tau_2.0.jsonl`
- `output/experiments/temperature_sweep/logs/tau_*_summary.json`
- `output/experiments/temperature_sweep/logs/tau_*.log`
- Checkpoint SHA256: `b4260363f9159b0c2e89790457ca53d3c9cff639e80bfc2bd426afc8d47f67cb`
- Test manifest SHA256: `ab587f801525c2e7e0fc7a8465fb90f3a2d99b67764eb2bc19cac2b422a938a8`
- Canonical baseline YAML SHA256: `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`
