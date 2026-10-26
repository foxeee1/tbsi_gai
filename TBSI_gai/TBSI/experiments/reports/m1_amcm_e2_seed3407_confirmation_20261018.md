# M1 AMCM E2 second-seed confirmation (2026-10-18)

## Question

Does the full-test gain observed for E2 with seed 42 persist under the matched seed 3407 training run?

## Fixed contract

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch15 checkpoint.
- Candidate: E2 post-fusion AMCM with GT cell RGB/TIR similarity supervision.
- Train: seed 3407, 10 epochs, 512 samples/epoch, batch 2, FP32, one DataLoader worker; only `amcm_reliability.*` trainable.
- Test: LasHeR `lasher_test`, target 244/244, FP32, same search/template settings as M1, two independent sequence workers.
- Candidate checkpoint: `output/experiments/M1_AMCM_E2_SEED3407_10EP_FULL244_20261018/checkpoints/TBSITrack_ep0010.pth.tar`.
- Checkpoint SHA256: `8cedeb340f880ae75c51e2e302b5079c960ad143af6b19460e164a5b6905af3e`.
- Code snapshot: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` (working-tree experiment files are also recorded in the config/output directory).

## Execution status

The first serial test attempt was manually interrupted after 8/244 sequences because its measured throughput was too low. Its partial outputs are retained under `output/test/tracking_results/tbsi_track/M1_AMCM_E2_SEED3407_10EP_FULL244_20261018_SERIAL_ABORTED/` but are not evidence for aggregate metrics. A fresh run with `--threads 2 --runid 20261018p2` completed in:

`output/test/tracking_results/tbsi_track/M1_AMCM_E2_SEED3407_10EP_FULL244_20261018_20261018p2/`

The run completed with 244/244 sequences. The official evaluator reported:

| Method | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 protected reference | 56.21 | 68.33 | 47.13 | 70.21 | 66.20 |
| E2 seed42 | 56.68 | 69.03 | 47.60 | 70.83 | 66.88 |
| E2 seed3407 | 56.13 | 68.26 | 47.15 | 70.08 | 66.24 |
| seed3407 - M1 | -0.08 | -0.07 | +0.01 | -0.13 | +0.04 |

The seed3407 result is therefore not a confirmation of the seed42 gain. It is effectively M1-level, with small negative aggregate deltas on AUC, OP50, and Precision.

## Paired per-sequence analysis

The reproducible analysis is in `output/experiments/M1_AMCM_E2_SEED3407_10EP_FULL244_20261018/analysis/per_sequence_comparison.csv` and `summary.json`.

For AUC, relative to M1:

- seed42: 125 improved, 117 degraded, 2 tied; mean delta `+0.47`.
- seed3407: 134 improved, 110 degraded, 0 tied; mean delta `-0.08`.
- seed3407 versus seed42: 128 improved, 116 degraded; mean delta `-0.55`.

Seed3407's largest AUC gains over M1 were `3rdfatboy` (+53.68), `leftfarboycomingpicktheball` (+47.66), and `leftuphand` (+19.88). Its largest losses were `blackdown` (-57.76), `blkboydown` (-38.97), and `carcomingfromlight` (-14.81). Seed42's largest gains were `ab_bikeoccluded` (+57.44), `3rdfatboy` (+54.11), and `leftfarboycomingpicktheball` (+47.10); its largest losses were `rightblkfatboyleftwhite` (-25.15), `bluegirlbiketurn` (-15.51), and `redmidboy` (-15.18).

## Decision

`FULL244-COMPLETE / SECOND-SEED-CONTRADICTORY / NOT-PROMOTED`. The positive seed42 observation is seed-sensitive and does not support a robust E2 improvement claim. The shared gain on a few sequences is outweighed by changed failure sequences and a negative seed3407 aggregate AUC. Do not expand AMCM module search from this result; treat supervision/optimization and sequence-specific failure causes as unresolved.
