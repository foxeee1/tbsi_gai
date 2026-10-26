# M1 utility/no-op rho sweep (2026-10-13)

## Question

Before implementing a post-fusion residual, can the existing S1/S2/S3 signals select a non-zero RGB/TIR action that improves the original CENTER decoded box over no-op on held-out sequences?

## Contract

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15.
- Same saved crop, tracker state, template, feature tensor, CENTER head and decode for every arm.
- 72 sequences / 144 fixed states; 48 calibration and 24 sequence-held-out sequences.
- Arms: no-op, fixed RGB, fixed TIR, random direction, S1, S2, S3, S1+S2, S1+S2+S3, low-confidence no-op.
- Symmetric RGB/TIR action magnitudes: `rho=0.05`, `0.10`, `0.15`.
- This is fixed-state decoded-IoU utility, not closed-loop tracking AUC.

## Held-out result

| rho | arm | Delta IoU vs no-op | 95% sequence CI | Delta vs fixed TIR |
|---:|---|---:|---:|---:|
| 0.05 | fixed TIR | -0.00125 | [-0.00306, +0.00067] | 0 |
| 0.05 | S1 | -0.00199 | [-0.00359, -0.00026] | -0.00074 |
| 0.05 | S2 | -0.00135 | [-0.00318, +0.00047] | -0.00010 |
| 0.05 | S3 | +0.00035 | [-0.00157, +0.00227] | +0.00160 |
| 0.05 | S1+S2+S3 | -0.00209 | [-0.00381, -0.00036] | -0.00084 |
| 0.10 | fixed TIR | -0.00266 | [-0.00623, +0.00118] | 0 |
| 0.10 | S1 | -0.00401 | [-0.00712, -0.00062] | -0.00135 |
| 0.10 | S2 | -0.00637 | [-0.01662, +0.00058] | -0.00371 |
| 0.10 | S3 | -0.00311 | [-0.01379, +0.00401] | -0.00045 |
| 0.10 | S1+S2+S3 | -0.00780 | [-0.01820, -0.00101] | -0.00514 |
| 0.15 | fixed TIR | -0.00193 | [-0.01006, +0.00725] | 0 |
| 0.15 | S1 | -0.00387 | [-0.01112, +0.00488] | -0.00193 |
| 0.15 | S2 | -0.00862 | [-0.01998, +0.00033] | -0.00669 |
| 0.15 | S3 | -0.00390 | [-0.01602, +0.00542] | -0.00197 |
| 0.15 | S1+S2+S3 | -0.01064 | [-0.02206, -0.00200] | -0.00870 |

## Decision

**STOP current signal-to-fixed-scaling route.** No signal/magnitude pair passed the precondition for training a residual: positive held-out utility with a non-negative paired interval and superiority to fixed-TIR. S3@0.05 is the closest candidate, but its interval crosses zero and it is not a reliable admission signal. Larger rho increases harmful intervention without producing a stable gain.

The next valid experiment is not a larger fusion module. It is a new outcome-blind train-side collection that predicts the three-way utility target `{no-op, RGB, TIR}` and includes an abstention loss; only a held-out positive utility gate should unlock a zero-initialized post-fusion residual.

Artifacts:

- `output/experiments/M1_UTILITY_NOOP_RHO_0.05_20261013/`
- `output/experiments/M1_UTILITY_NOOP_RHO_0.10_20261013/`
- `output/experiments/M1_UTILITY_NOOP_RHO_0.15_20261013/`

