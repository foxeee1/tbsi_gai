# Fusion Residual Channel Selectivity Diagnostic

Date: 2026-09-17

## Protocol

The corrected B3 V2 fusion path was evaluated with the frozen S1 epoch-3
checkpoint, frozen 60-sequence RPP-v2 test manifest, FP32 inference, and at
most 20 frames per sequence. No training was performed. For each active layer,
the residual `Delta = fused - input` was summarized using channel variance
`Var_i(Delta_i,c)`, followed by descending cumulative variance ratios.

## Results

| Layer | RGB Top-10 | RGB Top-20 | RGB Top-25 | TIR Top-10 | TIR Top-20 | TIR Top-25 |
|---:|---:|---:|---:|---:|---:|---:|
| L3 | 0.2085 | 0.3559 | 0.4206 | 0.2131 | 0.3621 | 0.4272 |
| L6 | 0.1990 | 0.3440 | 0.4084 | 0.2018 | 0.3471 | 0.4115 |
| L9 | 0.1955 | 0.3384 | 0.4023 | 0.1989 | 0.3431 | 0.4072 |

## Decision

Residual variance is mildly non-uniform but not strongly concentrated in a
small channel subset. Top-20% channels explain only 34-36% of variance, and
Top-25% explains 40-43%. This does not support adding a channel-selective
residual projector. Innovation 1 is therefore closed as B3 V2 Joint Fusion;
the next research budget should move to Template Update.

## Evidence

- `output/experiments/fast_fusion_snapshot/logs/fusion_snapshot.jsonl`
- `output/experiments/fast_fusion_snapshot/logs/summary.json`
- Checkpoint SHA256: `b4260363f9159b0c2e89790457ca53d3c9cff639e80bfc2bd426afc8d47f67cb`
- Test manifest SHA256: `ab587f801525c2e7e0fc7a8465fb90f3a2d99b67764eb2bc19cac2b422a938a8`
- Canonical baseline YAML SHA256: `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`
