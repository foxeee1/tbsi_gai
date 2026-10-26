# M1 decoded feature utility replay (2026-10-14)

## Purpose

Test whether the earlier target-background margin proxy overstated the usefulness of local RGB/TIR feature intervention. The protected M1 checkpoint and configuration were used read-only. No model training or checkpoint modification was performed.

## Contract

- Parent: `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15.
- Data: 72 train-side sequences, 144 fixed states from the existing utility cohort.
- Replay: same crop, tracker state, template, fusion output, CENTER head and decode path as M1.
- Intervention: 3x3 target-cell feature replacement toward RGB-only or TIR-only feature, `rho=0.05`.
- Estimand: decoded box IoU change relative to no-op at the fixed state.
- Trace repair: trace files were indexed by `(sequence, frame)` and merged across the three archived trace batches. All 144 requested keys were found.

## Results

| Action | Sequence-macro utility | State mean utility | Positive states | Harmful states |
|---|---:|---:|---:|---:|
| RGB local feature | -0.00029 | -0.00046 | 45.1% | 52.8% |
| TIR local feature | -0.00073 | -0.00113 | 40.3% | 57.6% |
| Readout score oracle | +0.01067 | +0.01940 | 41.7% | 39.6% |
| Readout offset oracle | +0.00817 | +0.00753 | 12.5% | 0.0% |
| Readout size oracle | +0.00571 | +0.00544 | 9.7% | 4.2% |
| Joint readout oracle | +0.25572 | +0.32245 | 95.1% | 0.0% |

The readout oracle rows are upper bounds because they use ground-truth cell or regression values. They are not deployable gains.

## Decision

`DIAGNOSTIC-COMPLETE / FEATURE-ACTION-NOT-SUPPORTED`.

The decoded metric confirms that local feature replacement is not a safe intervention: both modality-specific actions have slightly negative average utility and more harmful than beneficial states. This rejects the explanation that the previous proxy alone hid a useful feature-side action. It also supports keeping the investigation on readout/state recovery, but the small independent offset/size headroom still requires a deployable selector before any training.

Do not start feature residual or adaptive router full training from this result. The next permitted experiment is a sequence-held-out selector for a bounded readout offset/size action using decoded IoU utility, with explicit no-op abstention and risk-coverage reporting.

## Evidence

- Replay output: `output/experiments/M1_DECODED_FEATURE_UTILITY_144_20261014/oracle_replay5/`.
- Per-state CSV SHA256: `ffa9107a049cc6b0f45327974cedd61db2e9d60f72a2a1fde5a97c476dbe3241`.
- Summary JSON SHA256: `2fe211c5beaf317afae78b175ab324a4cfca4ef222a6066f3064e4ce9d60d2c71`.
- Protected M1 config/checkpoint hashes matched the registered values during replay.

## Limitations

This is fixed-state offline replay, not closed-loop tracking. The selected cohort is train-side and exploratory. Oracle readout values cannot be used as expected final-test gains.
