# Evaluation Audit

## Controlled Comparison

Both historical and current predictions were evaluated with the current `extract_results.py` implementation over their 244 shared sequences.

| Prediction source | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| Historical files generated July 13 | 55.4566 | 67.2762 | 46.7347 | 68.9763 | 65.2298 |
| Current v1.0.0 ep15, restricted to same 244 | 54.4680 | 66.1864 | 45.7235 | 68.1856 | 64.3593 |

The metric code matches the initial repository version. The only evaluation-protocol change is that `advancedredcup` was added, increasing the sequence count from 244 to 245. Restricting current outputs to the original 244 still leaves a 0.9886 AUC gap, so the added sequence is not the cause.

All 244 shared prediction files differ. Historical predictions date to July 13, while the currently stored v1.0.0 ep15 checkpoint dates to July 30. This is an artifact-lineage mismatch, not an evaluation-formula regression.
