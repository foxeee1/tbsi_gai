# M1 Fusion A/B/C Serial Campaign (2026-10-13)

## Contract

This campaign compares three fusion arms on one explicit 82-sequence cohort:

- A: protected M1 `TBSI_E_MHSA_FULL_4090D_seed42` original fusion;
- B: `G0_FFM_CDIAG_seed42`, pointwise cross-modal residual;
- C: `G0_FFM_CSPATIAL_seed42`, bidirectional spatial cross-modal residual.

The cohort is the union of the registered 62 low-AUC/failure sequences and 20 normal sentinel sequences. Every arm uses the same list, FP32 inference, one tracker process, and an independent result directory. The runner is serial and refuses to overwrite any existing result.

## Runner

`tools/run_m1_fusion_abc_serial_20261013.py`

Dry-run command:

```bash
python tools/run_m1_fusion_abc_serial_20261013.py \
  --output-dir output/experiments/M1_FUSION_ABC_SERIAL_20261013 \
  --dry-run
```

Actual command, once B/C checkpoints are restored:

```bash
TBSI_PYTHON=/root/autodl-tmp/conda_envs/tbsi/bin/python \
python tools/run_m1_fusion_abc_serial_20261013.py \
  --output-dir output/experiments/M1_FUSION_ABC_SERIAL_20261013
```

The runner writes `sequence_list.txt`, `categories.json`, `contract.json`, per-arm logs, per-sequence metrics, paired deltas, and `summary.json`. It reports all five tracking metrics plus mean IoU and separates `LOW_AUC_FAILURE` from `NORMAL_SENTINEL`.

## Current status

The dry-run completed and created the cohort contract without loading a model. The real campaign was not started because the explicit B/C checkpoints are missing:

- `output/experiments/G0_FFM_CDIAG_seed42/checkpoints/TBSITrack_ep0001.pth.tar`
- `output/experiments/G0_FFM_CSPATIAL_seed42/checkpoints/TBSITrack_ep0001.pth.tar`

Only logs and prior analysis remain for those arms. A partial A-only run would not answer the A/B/C question and is therefore not treated as an experiment result.

## Retry result

The missing B/C adapters were retrained for one epoch in an isolated save root and the campaign was completed serially. The result root is `output/experiments/M1_FUSION_ABC_SERIAL_20261013_retry1_eval5/`; B and C checkpoints are under the corresponding `retry1/training` root.

| Arm | 82-seq AUC | Low-AUC/failure AUC | Normal sentinel AUC | Low-AUC delta vs A | Normal delta vs A | Wins/losses vs A |
|---|---:|---:|---:|---:|---:|---:|
| A C0 | 38.5292 | 22.8226 | 87.2197 | 0 | 0 | baseline |
| B Cdiag | 40.6638 | 25.6432 | 87.2279 | +2.8206 | +0.0082 | 54/28 |
| C Cspatial | 39.8444 | 24.6001 | 87.1018 | +1.7775 | -0.1179 | 52/30 |

The same ordering holds for the supplementary metrics. On the low-AUC subset, B improves Precision/NormPrec/OP50/OP75 by `+3.9768/+3.5479/+3.5884/+2.5199` points; C improves them by `+2.4763/+2.2211/+2.2132/+1.3641`. On the normal sentinel set, changes remain within `-0.12` AUC points for both arms.

Decision: both arms pass the exploratory promotion gate, with B clearly preferred over C. This supports a simple pointwise cross-modal residual as the next fusion candidate; it does not establish a full-244 or multi-seed result. The campaign uses a custom sequence-macro metric implementation for this cohort, so official evaluator confirmation is required before any final claim.

## Decision rule

Promote an arm only if its low-AUC failure subset improves by more than `+0.5` AUC points, its normal sentinel subset is no worse than `-0.25`, and it wins more sequences than it loses against A. Otherwise the automatic conclusion is `NO_PROMOTION`; the per-sequence and subgroup values remain available for diagnosis.

This is a matched exploratory campaign, not a full-244 claim. Any promoted arm requires a fresh independent seed or full-test confirmation under the repository experiment rules.
