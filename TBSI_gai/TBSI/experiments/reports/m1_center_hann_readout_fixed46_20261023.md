# M1 CENTER Hann Readout Fixed-State Diagnostic (2026-10-23)

## Question

Does the Hann window correct CENTER peak selection on the fixed calibration states, or is the localization error already present in the raw score argmax? The comparison changes only the readout window from all ones to the saved Hann window. Score, size/offset maps, crop, frame, and tracker state remain fixed.

## Contract and provenance

- 23 LasHeR training calibration sequences, with one pre-onset and one severe-onset state each (46 states).
- Inputs are the saved parent and candidate head maps from the 2026-10-22 matched sparse-neck geometry crossover. The parent is the M1-initialized matched quick-training parent at epoch 16; the candidate is the sparse-neck retry2 epoch 16. These are not the protected M1 epoch15 checkpoint and are not full tracking runs.
- Parent maps SHA256: `afac5b6cf0c4f9bc0bb004832a7fb6bef0f17a725489c9e72f3688c66ffbada1`.
- Candidate maps SHA256: `40e93ebe9eb67a6548b3f248adb33e51893d2a8994e9ed0ccc7352e51d796a6b`.
- Offline CPU analysis only; no model inference, training, tracker replay, or locked holdout access. Decoder reuses the geometry-crossover decode function and uses each arm's own size/offset at the selected cell.
- Analysis script SHA256: `4cb9362f47a590d16fee9c7cf309f68be5be0e3f6ae25eca753e348ad5c83414`.

## Results

| Arm / phase | All-ones IoU | Hann IoU | Mean delta | Median delta | IoU win/tie/loss | Peak cell changed | Mean GT rank, ones -> Hann | Sequence bootstrap 95% CI, mean delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Parent, pre-onset (n=23) | 0.51189 | 0.59970 | +0.08781 | 0 | 4 / 19 / 0 | 4/23 | 1.43 -> 1.22 | [+0.00752, +0.18342] |
| Parent, severe onset (n=23) | 0.31192 | 0.35324 | +0.04133 | 0 | 3 / 20 / 0 | 4/23 | 3.26 -> 1.96 | [0, +0.10088] |
| Parent, both phases (46 states / 23 sequences) | 0.41190 | 0.47647 | +0.06457 | 0 | 7 / 39 / 0 | 8/46 | 2.35 -> 1.59 | [+0.01210, +0.13026] |
| Candidate, pre-onset (n=23) | 0.51985 | 0.62897 | +0.10913 | 0 | 5 / 18 / 0 | 5/23 | 1.39 -> 1.13 | [+0.02382, +0.21427] |
| Candidate, severe onset (n=23) | 0.30132 | 0.30374 | +0.00243 | 0 | 1 / 22 / 0 | 2/23 | 3.17 -> 2.22 | [0, +0.00728] |
| Candidate, both phases (46 states / 23 sequences) | 0.41058 | 0.46636 | +0.05578 | 0 | 6 / 40 / 0 | 7/46 | 2.28 -> 1.67 | [+0.01191, +0.10849] |

The sequence-cluster bootstrap resamples the 23 sequence-level paired deltas (10,000 draws, seed 42); the all-phase value averages each sequence's pre-onset and onset deltas before resampling. The intervals are descriptive for this small, selected calibration cohort.

## Interpretation

**Supports Hann as a useful center prior on these fixed states, but does not support it as a severe-onset repair.** Its overall mean IoU gain comes mainly from pre-onset states. At severe onset, the candidate gains only `+0.00243` mean IoU, with 22/23 states unchanged and a median delta of zero. The Hann window changes the selected cell in only 2/23 candidate onset states. The parent also has a zero lower bootstrap bound at onset.

The score ranking can improve without changing the decoded box: Hann reduces mean GT rank for both arms, while the median IoU delta remains zero because most selected cells stay the same. This separates rank reweighting from actual peak switching and geometry readout. On the tested candidate's severe-onset states, the raw score field and/or the peak-local geometry remain the larger unresolved limitation; the present data do not identify which is causal.

This is a readout counterfactual on a small LasHeR calibration cohort, not a general M1 result or tracking-performance claim. It does not justify tuning the Hann shape or changing the CENTER head. For the fusion line, use this as a diagnostic constraint: a new fusion candidate must move the useful search-space peak and its decoded box on fixed severe-onset states, not merely improve margin or GT rank.

## Run note and evidence

The first aggregation attempt wrote the per-state CSV but failed while serializing NumPy integer counts to JSON. It is retained at `output/experiments/M1_CENTER_HANN_READOUT_FIXED46_20261023/`. The integer conversion was fixed and the complete result was rerun in a separate directory.

- Complete output: `output/experiments/M1_CENTER_HANN_READOUT_FIXED46_RETRY2_20261023/summary.json` and `per_state.csv`.
- Script: `tools/analyze_m1_center_hann_readout_20261023.py`.
- Parent and candidate source diagnostic: `experiments/reports/m1_center_geometry_crossover_20261022.md`.
