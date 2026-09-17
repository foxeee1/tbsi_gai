# Selective Cross-Modal Top-K RPP Screening

Date: 2026-09-16

## Contract

- Parent: `B3_v2_fusion_softmax_corrected` (dense V2 template fusion)
- Candidates: S1 K=32, S2 K=16, S3 K=8
- Intended train/test protocol: frozen RPP-v2 sequence manifests, 3 epochs, `rpp_lasher_test`, FP32 tracker inference (`TBSI_INFER_FP16=0`), independent output directories
- Canonical baseline: read-only; no baseline YAML or result path was modified

## Implementation

`v2_topk` applies Top-K selection only to the RGB-to-TIR and TIR-to-RGB template attention blocks. Same-modality template attention and the standard TBSI search/template path remain dense. The feature is disabled by default and controlled by `MODEL.BACKBONE.TBSI_FUSION` and `TBSI_CROSS_MODAL_TOPK`.

## B3 mass-concentration analysis

Source: `output/experiments/B3_v2_fusion_softmax_corrected/fusion_diagnostics/logs/fusion_directionality.jsonl`

- 60 frozen test sequences, 14,220 layer/frame rows, TBSI layers 3/6/9
- RGB to TIR: R4 0.289397, R8 0.435710, R16 0.618023, R32 0.820981
- TIR to RGB: R4 0.288581, R8 0.433258, R16 0.616693, R32 0.824180

The two directions are nearly symmetric. R16 is moderate rather than strongly concentrated, so Top-K is a diagnostic screening route, not a prior-supported guaranteed improvement.

## P0 and run status

Layer-level forward/gradient checks passed for K=0/32/16/8: finite loss, unchanged output shapes, and nonzero QKV gradients.

The first launch correctly refused because the filesystem had only 2.6 GiB free, below its 10 GiB safety reserve. After space was freed, S1 trained successfully for 3 epochs and produced `TBSITrack_ep0003.pth.tar` (SHA256 `b4260363f9159b0c2e89790457ca53d3c9cff639e80bfc2bd426afc8d47f67cb`). The initial `threads=6` test stalled during multiprocessing worker handling after writing 43/60 sequences. Investigation showed `boylefttheNo_9boy` itself was complete (4921 bbox lines and 4921 time lines); the issue was evaluation scheduling/worker shutdown rather than corrupt sequence data. Serial checkpoint resume (`--threads 0`) skipped existing files and completed all 60 sequences.

## S1 proxy result

Official analysis over all 60 frozen test sequences:

| configuration | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| B3 V2 dense parent | 50.24 | 59.52 | 42.45 | 61.32 | 58.57 |
| S1 Top-K=32 | 49.97 | 59.26 | 41.30 | 60.75 | 57.76 |
| S1 - parent | -0.27 | -0.26 | -1.15 | -0.57 | -0.81 |

S1 is a proxy-screening regression on every reported metric. S2 and S3 were not started; S1 does not qualify for full confirmation.

Recorded environment facts: Python entrypoint `/root/autodl-tmp/conda_envs/tbsi/bin/python`; baseline YAML SHA256 `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fb83d696538251a2d8fadd24b`.

Next action: stop the Top-K route at K=32 under this 3-epoch proxy result; do not spend S2/S3 compute unless a separate reason justifies testing smaller K. For future RPP tests use serial resume or add an explicit worker timeout/recovery mechanism. Do not delete existing experiment outputs without an explicit retention decision.
