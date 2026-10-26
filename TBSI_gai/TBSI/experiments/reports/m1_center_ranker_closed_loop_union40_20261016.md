# M1 candidate ranker closed-loop attribution (2026-10-16)

## Question

Does a sequence-held-out CENTER candidate ranker help by changing only the current output, or only after its candidate is committed to the next crop/state?

## Contract

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15, FP32, fixed checkpoint.
- Data: frozen `m1_ranker_closed_loop_union40_20261016.txt` (normal20 followed by difficult20), 40 sequences.
- Ranker: frozen OOF ridge model from `center_ranker_oof/summary.json`; top-8 CENTER candidates, no-op when predicted utility <= 0.
- `output_only` (run 102): candidate changes current returned box; next frame uses M1 baseline state.
- `coupled` (run 103): candidate changes current returned box and next frame state.
- M1 baseline: protected full-244 result directory, evaluated on the same union40 subset.

## Results

Canonical AUC is computed with the repository evaluator (mean sampled overlap success curve).

| split | method | AUC | avg IoU | OP50 | Precision | Norm Precision |
|---|---|---:|---:|---:|---:|---:|
| all40 | M1 | 65.339 | 0.6644 | 75.873 | 77.838 | 75.006 |
| all40 | output_only | 65.208 | 0.6630 | 75.594 | 77.627 | 74.497 |
| all40 | coupled | 61.979 | 0.6302 | 71.840 | 72.849 | 70.962 |
| normal20 | M1 | 85.805 | 0.8758 | 98.941 | 98.370 | 98.777 |
| normal20 | output_only | 85.718 | 0.8751 | 98.922 | 98.288 | 98.737 |
| normal20 | coupled | 83.406 | 0.8517 | 96.434 | 95.659 | 96.168 |
| difficult20 | M1 | 44.873 | 0.4530 | 52.804 | 57.306 | 51.236 |
| difficult20 | output_only | 44.698 | 0.4509 | 52.267 | 56.967 | 50.257 |
| difficult20 | coupled | 40.553 | 0.4088 | 47.247 | 50.040 | 45.757 |

Relative AUC to M1 is `-0.131` for output-only and `-3.360` for coupled. Output-only wins/losses by sequence are `14/24`; coupled is `13/27`. The largest output-only gains are `shinycarcoming +0.411`, `rightblkfatboyleftwhite +0.308`, and `carcomingfromlight +0.280`; largest losses are `manoncall -1.956`, `boyss -1.029`, and `girldownstairfromlight -0.802`. Coupled's largest losses include `blueboy -50.130`, `whiteridingbike -26.499`, and `redmidboy -15.156` AUC points.

## Conclusion

**STATE-FEEDBACK-DOMINANT / FIXED-STATE-RANKER-NOT-SUPPORTED.** Committing the candidate state is demonstrably harmful under this contract and explains most of the gap between output-only and coupled. However, output-only itself has no positive aggregate or split gain, so the earlier fixed-state OOF utility signal does not establish deployable CENTER readout gain. This rules out promoting the current ranker or adding more state machinery.

The remaining explanations are ranker-to-runtime contract mismatch, candidate response boosting not being a faithful learned box correction, or insufficient candidate information. A next experiment, if needed, should be one minimal parity audit on the same frames (selected candidate and decoded box before/after runtime) rather than another closed-loop module. Do not start full training from this result.

Evidence:

- `output/experiments/M1_RANKER_CLOSED_LOOP_UNION40_20261016/closed_loop_summary.json`
- `output/experiments/M1_RANKER_CLOSED_LOOP_UNION40_20261016/closed_loop_summary.csv`
- `output/experiments/M1_RANKER_CLOSED_LOOP_UNION40_20261016/output_only/tracking.log`
- `output/experiments/M1_RANKER_CLOSED_LOOP_UNION40_20261016/coupled/tracking.log`
- protected M1 result directory `output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42/`
