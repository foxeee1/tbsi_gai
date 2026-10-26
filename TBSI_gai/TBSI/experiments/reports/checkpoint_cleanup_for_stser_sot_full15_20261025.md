# Checkpoint cleanup before STSER SOT full15

Date: 2026-10-25

The following obsolete or confounded checkpoints were removed to make room for the single selected full-data run. Configs, logs, reports, metrics, and tracking outputs were retained.

| Path | Reason | SHA256 before deletion |
|---|---|---|
| `output/experiments/M1_STSER_COMPETITION_FULL15_FROM_M1_20261024/checkpoints/TBSITrack_ep0015.pth.tar` | M1-epoch15 continuation; invalid as the selected SOT-from-zero comparison | `bb81b1b32015fc478ce6a4077f0a9d3d37210165b1818baf67defb5332dc1a52` |
| `output/experiments/M1_STSER_COMPETITION_QUICK3_20261024/checkpoints/TBSITrack_ep0003.pth.tar` | superseded STSER proxy with mixed initialization contract | `61f36630a3a3ff0d7c032bde9a7046860c1c8676722d57702ae450f1c7658f25` |
| `output/experiments/M1_STSER_PLAIN_T2S_QUICK3_20261024/checkpoints/TBSITrack_ep0003.pth.tar` | superseded STSER proxy control | `4071491e964eea23fe64feeabf699a4f248a887d677d787f192f45e7ddae98d2` |
| `output/experiments/M1_STSER_QUICK3_20261024/checkpoints/TBSITrack_ep0003.pth.tar` | superseded STSER proxy | `pending: hash command exceeded the shell time limit; file size 1,384,514,437 bytes` |

Protected M1 and 4090D equivalent reproduction checkpoints were not touched. The SOT checkpoint and the SOT-initialized quick3 control/candidate checkpoints were retained as provenance and selection evidence.

Three dangling historical result symlinks were removed after verifying that their targets did not exist:

- `output/test/tracking_results/tbsi_track/v1.0.0-完美基线-全量`
- `output/test/tracking_results/tbsi_track/v1.0.0-完美基线-全量-rerun-20260810`
- `output/test/tracking_results/tbsi_track/v1.5.0-基线+dgs-v4-diffproj-全量v2`

Test directories containing reports, partial-run evidence, or registered metrics were retained.
