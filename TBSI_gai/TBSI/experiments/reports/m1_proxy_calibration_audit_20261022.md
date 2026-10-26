# M1 Historical Proxy Calibration Audit (2026-10-22)

## Decision

**The existing proxy workflow is not calibrated to predict full-244 candidate ranking.** The historical evidence is useful as a warning but cannot validate the new 176/48/24/24 protocol. Do not use RPP-v1's 60-sequence test subset or the prior union82 cohort as a trusted candidate-ranking gate.

## Audited evidence

| Candidate | Historical proxy delta | Full-244 delta | Direction | Contract issue |
|---|---:|---:|---|---|
| Cdiag feature residual | +2.8206 AUC points on union82 | -0.0274 | Reversed | Proxy and full runs do not match training scope/progress. |
| S1 bounded readout | +2.83 on union82 | +0.2522 | Same sign | Proxy cohort was selected/exposed; it is not the new locked evaluation cohort. |
| Cspatial | +1.78 on union82 | NA | Not assessable | No comparable full-244 result was found. |
| Early-conditioned MSHA gate | No paired RPP60 delta | -0.22 | Not assessable | The candidate has an RPP60 absolute metric, but M1 was not evaluated on that manifest. |

Across the two historical candidates with proxy and full deltas, only 1/2 directions agree. Their aggregate candidate ordering happens to match, but `n=2` is too small and the contracts differ. Sequence-level proxy/full paired files were not recovered for both candidates, so no sequence rank correlation or uncertainty interval is justified. These values must not be represented as a valid Spearman estimate.

## Protocol change

The new screening protocol uses:

- 176/977 LasHeR train sequences for candidate updates, selected to preserve full-population marginal frequencies rather than over-sampling rare labels;
- 48 calibration, 24 development and 24 locked holdout train sequences, disjoint from candidate training;
- all 244 official test sequences for historical comparability, explicitly marked selection-exposed;
- the 60-sequence RPP-v1 subset only as an optional difficulty stress set.

The candidate correction must not train on the 96 event-mining cohort sequences. Parent M1 was trained on all 977 train sequences, so these cohorts are held out from candidate adaptation, not unseen to M1. Test-set outcomes can be reported, but independent generalization requires a new benchmark that has not been used for idea selection.

## Recalibration rule

Before batch screening with a proxy, collect at least three positive, negative and null candidate comparisons that match parent, initialization, train/eval manifests, optimizer updates, checkpoint rule, seed policy and evaluator. Recompute paired sequence deltas on common sequences and calculate candidate ranking Spearman with bootstrap uncertainty. If contracts cannot be matched or the interval remains broad, use proxy results only for diagnostics and continue full evaluation; do not infer that a failed candidate or cohort is the cause until the mechanism strata and learning curves are compared.

## Sources

- `experiments/reports/m1_fusion_cdiag_full_20261014.md`
- `experiments/reports/m1_s1_full244_confirmation_20261014.md`
- `experiments/reports/m1_early_conditioned_msha_gate_proxy_20261021.md`
- Machine-readable audit: `experiments/protocols/m1_screening_v1_20261022/historical_proxy_calibration.csv`
