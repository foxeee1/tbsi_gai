# FMR full matched M1 analysis

## Result

`FMR_TBSI_E_FULL_MATCHED_M1_seed42` completed P0, 15-epoch training, serial FP32 tracking on all 244 LasHeR test sequences, and analysis. The candidate applies zero-initialized high-band refinement to the projected joint-MHSA message only at TBSI location 9. It does not change Q/K/V, the attention route, search processing, masks, loss, or TBSI-E topology.

| Metric | FMR | TBSI-E M1 | Delta |
|---|---:|---:|---:|
| AUC | 55.91 | 56.21 | -0.30 |
| OP50 | 67.93 | 68.33 | -0.40 |
| OP75 | 46.92 | 47.13 | -0.21 |
| Precision | 69.81 | 70.21 | -0.40 |
| Norm Precision | 66.08 | 66.20 | -0.12 |

The pre-registered success threshold was AUC >= 57.21 without an obvious OP50/precision tradeoff. FMR fails this threshold and is below the parent on all five headline metrics. It is a valid full negative result, not an infrastructure failure.

## Paired analysis

The comparison uses the same 244 sequence names and sequence-level paired curves.

- Mean sequence AUC delta: `-0.3008` percentage points.
- Median sequence AUC delta: `+0.0397` percentage points.
- Sequence win rate: `52.87%` (129/244).
- Bootstrap 95% interval for mean delta: `[-1.4230, +0.7330]` percentage points.
- Sequences with delta <= -5: `11`; delta >= +5: `11`.

The positive win rate therefore does not imply improvement: most wins are small, while a limited number of regressions are large enough to lower the overall mean.

## Attribute analysis

Descriptive AUC deltas versus M1 were: `LR +1.02` (n=68), `DEF +0.29` (n=46), `FL +2.15` (n=20), `SV +0.20` (n=221), and `ARV +0.09` (n=108). Negative groups included `BC -0.63` (n=77), `SA -0.74` (n=106), `CM -0.77` (n=132), `FM -0.45` (n=192), and `OV -1.79` (n=7).

These are not validated attribute-specific effects. LasHeR attributes are multi-label, subgroup sizes vary substantially, and the same full test set has been used throughout method selection. The complete table is in `fmr_vs_m1_analysis.csv`.

## Code and optimization evidence

The new module is `MessageFrequencyRefine` in `lib/models/layers/cfvi_module.py`. It decomposes the already projected message into Haar bands, reconstructs only LH/HL/HH, and adds the result with learned modality-specific gamma. Final epoch-15 checkpoint values were:

- `gamma_rgb = 0.05489`, `gamma_tir = 0.04396`;
- RGB high-band scales: `[1.03795, 1.03749, 1.01893]`;
- TIR high-band scales: `[1.02107, 1.02862, 1.01593]`.

The branch was active and learned; this was not a zero-use failure. P0 passed on CPU and CUDA, including Haar reconstruction, zero-gamma parity, finite forward, and nonzero gamma/band gradients. Training logs still reported two zero-gradient parameters at diagnostic checkpoints, consistent with the scale parameters being gated by zero-initialized gamma early in training.

## Decision

Stop this FMR candidate and do not run a second seed, dynamic gate, or another full frequency module. The result does not support the claim that projected-message high-frequency refinement improves TBSI-E. The earlier corrected response audit remains useful as measurement evidence, but it does not identify a validated frequency mechanism.

Evidence:

- `output/experiments/fmr_tbsie_full_matched_m1/serial.log`
- `output/experiments/FMR_TBSI_E_FULL_MATCHED_M1_seed42/logs/train.log`
- `output/experiments/FMR_TBSI_E_FULL_MATCHED_M1_seed42/logs/test.log`
- `output/experiments/FMR_TBSI_E_FULL_MATCHED_M1_seed42/checkpoints/TBSITrack_ep0015.pth.tar`
- checkpoint SHA256: `adf7fefa774ec37beb1a1ec6aeadee72779b1dd998ad8fdf8c19d10e19926337`
- `output/experiments/fmr_tbsie_full_matched_m1/eval_data.pkl`
- `output/experiments/fmr_tbsie_full_matched_m1/fmr_vs_m1_analysis.json`
