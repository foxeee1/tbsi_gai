# BridgeTokenPruner strict 10-step decision

## Scope

This report covers the strict protocol with pruner-only training, seed 42,
the frozen 72-event RGB-blur manifest, and independent M2/M3 output roots.
The original TBSI model parameters were frozen by `TRAIN.PRUNE_ONLY=True`;
only the eight parameters under `backbone.tbsi_layers.2.bridge_token_pruner`
were optimized.

## Step status

1. Migration: completed previously.
2. keep=1.0 identity parity: passed, max absolute difference 0.
3. TBSI location=9 only: passed; layer id 2 only.
4. Freeze original model, train pruner only: completed and printed exactly 8 trainable parameter tensors.
5. 1000-pair smoke: passed; 1000/1000 nonzero-gradient pairs and no nonfinite gates.
6. M2 keep=0.75: 15 epochs completed; final train IoU 0.76293.
7. M3 keep=0.50: 15 epochs completed; final train IoU 0.76321.
8. Clean + 72 RGB-blur events: completed for both M2 and M3; 72/72 events, no skips, offsets 0-5.
9. Gate-collapse audit: passed for both; no group had >95% zero or >95% one gates on 1000 deterministic probe batches.
10. Directional bridge budget decision: STOP / do not promote.

## RGB blur diagnostic

`drop = IoU(clean) - IoU(RGB blur)`; this is a controlled train-sequence
diagnostic, not an unseen validation result.

| Model | offset 0 | offset 1 | offset 2 | offset 3 | offset 4 | offset 5 |
|---|---:|---:|---:|---:|---:|---:|
| M2 keep 0.75 | 0.098107 | 0.009554 | 0.003409 | 0.002617 | -0.005523 | -0.005599 |
| M3 keep 0.50 | 0.080988 | 0.006219 | 0.006801 | 0.001218 | 0.002237 | -0.009167 |

The immediate RGB-blur sensitivity remains positive across the fixed events,
but the experiment does not establish a positive tracking gain versus the
direct parent. M3 does not improve the diagnostic consistently over M2, and
no clean/full tracking metric was produced for this strict pruner-only line.

## Decision

`STOP_NO_DIRECTIONAL_BRIDGE_BUDGET`. The mechanism is numerically trainable
and its gate is not collapsed, but the current evidence is insufficient for
directional budget promotion. A future branch would need a pre-registered
parent-comparable clean tracking result and a specified bridge-budget metric;
this run does not justify more architecture or full confirmation.

Evidence roots:

- `output/experiments/bridge_token_pruner_p0/`
- `output/experiments/bridge_token_pruner_pair_smoke/`
- `output/experiments/BTP_PRUNE_ONLY_M2_seed42/`
- `output/experiments/BTP_PRUNE_ONLY_M3_seed42/`
