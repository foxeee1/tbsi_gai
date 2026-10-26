# Stopped Local-Candidate Checkpoint Cleanup (2026-10-11)

## Scope

User-authorized storage cleanup before the one-time MCCA full-244 paired evaluation. Only checkpoints from candidates explicitly marked stopped in the experiment records were removed. Reports, logs, YAML configs, manifests, predictions, metrics, source snapshots, and asset rows remain.

| Experiment | Removed checkpoint | Bytes | SHA256 | Recorded decision |
|---|---|---:|---|---|
| `M1-LCGA-FAITHFUL-SHORT-20261010` | `output/experiments/M1_LCGA_FAITHFUL_PROXY_seed42/checkpoints/TBSITrack_ep0003.pth.tar` | 1,378,989,619 | `f628599219d954e4b9f4b57263b04c03baa5f2f0e49c95a88f39018ddbae97ff` | `STOP-NO-LOCALIZATION-GAIN`; AUC -1.0643, speed -17.46% |
| `M1-LCGA-FROZEN-ONLY-20261010` | `output/experiments/M1_LCGA_FROZEN_ONLY_seed42/checkpoints/TBSITrack_ep0003.pth.tar` | 498,200,827 | `0f24b0bbd6f66e43725be5896a2b6b7de2a1186a0e1e5084a276e4ba400b6547` | `STOP-NO-LOCALIZATION-GAIN`; AUC -0.2984, center error improved on 10/30, speed -19.99% |
| `M1-STRIP-SCORE-SHORT-20261010` | `output/experiments/M1_STRIP_SCORE_ONLY_seed42/checkpoints/TBSITrack_ep0003.pth.tar` | 441,983,285 | `40854438c28aa7c22e72e3d649fa952d00811591761119260d8aac917b2d874d` | `STOP-PROXY`; AUC +0.0045, only 1/30 center-error improvements |

Total removed: 2,319,173,731 bytes (2.1599 GiB). The LCGA matched control checkpoint was retained. MCCA G0/G1 checkpoints, protected M1, 4090D reproduction-group checkpoints, DBF and TAM artifacts were not touched. The hashes above preserve identity of the deleted artifacts; the files themselves are intentionally absent.

## Verification

Verified after deletion: all three checkpoint paths are absent; protected M1 config/checkpoint hashes remain unchanged; available space increased from about 3.1 GiB to 5.3 GiB. No model process was started by this maintenance action.
