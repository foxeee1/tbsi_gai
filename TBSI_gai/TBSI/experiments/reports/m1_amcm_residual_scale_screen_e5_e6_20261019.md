# AMCM directed residual scale screen and E5 scale follow-up (2026-10-19)

## Fixed-state screen

E5 and E6 epoch-3 checkpoints were evaluated on the same 123 saved train-side events from 60 LasHeR sequences. Each event reused the recorded frame and `tracker_state_before`; only the hook scale on the AMCM residual changed (`alpha=0, 0.5, 1, 2`). No tracker state was fed back between events. This is mechanism evidence from a previously inspected train-side event set, not held-out or closed-loop sequence performance.

| Checkpoint | alpha | mean fixed-state IoU | delta from alpha 0 | top-1 peak changes |
|---|---:|---:|---:|---:|
| E5 | 0 | 0.62282 | 0 | 0 |
| E5 | 0.5 | 0.62314 | +0.00031 | 0 |
| E5 | 1 | 0.62463 | +0.00180 | 1 |
| E5 | 2 | 0.62510 | +0.00228 | 1 |
| E6 | 0 | 0.62282 | 0 | 0 |
| E6 | 0.5 | 0.62240 | -0.00042 | 3 |
| E6 | 1 | 0.61722 | -0.00561 | 7 |
| E6 | 2 | 0.61367 | -0.00916 | 10 |

E5 alpha 2 has 55 event wins and 61 losses against alpha 0; its mean peak margin changes by `-0.000016`. The average IoU gain is small and does not show a consistent improvement in peak separation. Its largest sequence gain is `4two` (`+0.0488`), while the largest loss is `manaftercars` (`-0.0179`).

E6 alpha 2 has 56 wins and 60 losses, with a mean peak margin increase of `+0.026736`, yet IoU decreases and top-1 changes more often. This is direct evidence that increasing score separation alone is not aligned with selecting the target: the changed peaks can be wrong. E6's largest losses include `left4thblkboyback` (`-0.2438`), `rightwhitegirl` (`-0.1293`), and `shotmaker` (`-0.0926`); gains on `AQraincarturn2` and `rightblkboy2` do not offset them.

At alpha 0, E5 and E6 produced identical fixed-state IoU for all 123 events, confirming identity behavior at the prediction level. The learned residual direction, rather than the scalar strength alone, differs materially between E5 and E6. Do not promote E6 by scaling it. E5 supports only a narrowly positive, inconsistent fixed-state response to larger residuals, not a performance claim.

The generated event table contains fixed-state IoU, Hann peak rank/margin, peak indices, and crop containment. It does not log hook-adjusted RGB/TIR feature delta norms; the AMCM diagnostic norms written by the tracker are pre-hook values and must not be read as scaled feature changes. The subsequent trained E5 scale-2 run does log its actual feature changes and gradients.

## E5 scale-2 short train and test

One serial training run is testing whether E5 can use a larger residual bound when trained under it. Relative to E5, the only config change is `AMCM_RELIABILITY.MAX_DELTA: 0.10 -> 0.20`; the run keeps M1 epoch 15 initialization, E5's no-auxiliary-loss objective, 3 epochs, LasHeR train sampling at 512 samples/epoch, batch 2, FP32, seed 42, worker 4, and the same RPP 15-sequence test list. Output is isolated at `output/experiments/M1_AMCM_DR_LOSS_E5_SCALE2_QUICK15_3EP_20261019/`.

The run completed 3 epochs and the RPP 15-sequence test. Metrics are `AUC/OP50/OP75/Precision/Norm Precision = 55.29/67.48/44.99/70.58/65.72`; compared with E5 (`54.87/66.62/44.99/70.33/64.84`), deltas are `+0.42/+0.86/0/+0.25/+0.88`. It is `+0.16 AUC` over E6 but `-0.61 AUC` below M1 (`55.90`). Mean per-sequence AO is `0.56183`, versus M1 `0.56780`, E5 `0.55725`, and E6 `0.55995`. The candidate beats E5 in 7 sequences and loses in 8; it beats E6 in 6 and loses in 9; it beats M1 in 6 and loses in 9.

The gains are concentrated: `bluegirlbiketurn` improves from E5 `0.5204` to `0.5931` (`+0.0727`), and `whiteboyrightcoccergoal` from `0.4019` to `0.5397` (`+0.1378`). The largest regression is `large`, from `0.7279` to `0.6024` (`-0.1255`), followed by `3rdgrouplastboy` (`-0.0308`). This matches the scale screen's warning: a stronger residual can fix a few sequences while moving the representation away from useful features on others. The full 15-sequence table is `output/experiments/M1_AMCM_DR_LOSS_E5_SCALE2_QUICK15_3EP_20261019/per_sequence_comparison.csv`.

Training diagnostics show a live path: all 16/16 AMCM parameters had nonzero gradients at 48 logged steps, with mean gradient norm `0.0815`. Mean actual feature delta/input norm was `1.36%` for RGB and `1.12%` for TIR. Thus the new scale bound increases correction magnitude and the module receives tracking gradients, but the objective does not consistently choose a useful correction direction across scenes. Mean tracking FPS on the 15 sequences was `64.91`.

Decision: the experiment supports a small proxy improvement over E5 and E6, but rejects promoting residual scale 2 as a general solution. It remains below M1 on both AUC and mean AO and has a severe `large` regression. The primary unresolved issue is target-aligned direction/conditioning under scene changes, not simply insufficient amplitude. No full-data run is justified from this single-seed, 15-sequence result.
