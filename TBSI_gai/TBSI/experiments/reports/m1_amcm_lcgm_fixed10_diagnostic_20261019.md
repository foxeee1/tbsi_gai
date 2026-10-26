# AMCM LCGM fixed-frame diagnostic (2026-10-19)

## Question and protocol

Test whether a short-trained LCGM-inspired AMCM changes target/background representations at the same ten low-AUC M1 failure frames, and whether the changed representation affects `tbsi_fuse_search` and decoded CENTER localization. The M1 and candidate arms used the same frame, first-frame template initialization, and saved M1 `state_before`; scores were multiplied by the tracker Hann window. This is a fixed-state diagnostic, not sequence-level evaluation.

The candidate checkpoint is `M1_AMCM_LCGM_QUICK15_3EP_20261019`, trained from M1 epoch 15 for 3 epochs on 512 LasHeR samples/epoch, seed 42, batch 2, FP32, worker 4. The config had AMCM-only optimization and no AMCM-specific auxiliary loss. Training used the project Python 3.8.20 / PyTorch 1.9.0+cu111 / CUDA 11.1 environment. Fixed-frame inference used Python 3.8 / PyTorch 2.0.0+cu118 / CUDA 11.8 on 1x NVIDIA GeForce RTX 4090 D.

The training loss was the existing tracking objective (GIoU, L1 and CENTER focal); no new LCGM or modality-reliability labels were used. Training logs show finite AMCM gradients on all 48 logged intervals (12/20 parameters had nonzero gradients at sampled steps; gradient norm ranged 0.0145-0.1000 after clipping). The output projection started at zero and learned a small nonzero update.

## LCGM operation checked

The AMCM `lcgm` branch projects RGB and TIR separately to 32 channels, self-gates each projected stream, applies depthwise 3x3 local convolution, multiplies the resulting local features elementwise, and maps the shared product through two stream-specific output projections. Bounded residuals are added to the original RGB/TIR features before `tbsi_fuse_search`. Template tokens are shape-validated by the AMCM interface but are not used by this branch.

This is an LCGM-inspired adaptation, not a faithful reproduction of the paper's complete model: the AMCM interface requires two outputs, so a shared cross-modal product is projected into both streams. The candidate therefore does not test template-conditioned matching, and the paper's task loss/training protocol is not reproduced here.

## Results

Across the ten fixed events, mean IoU changed from `0.301330` to `0.301230` (delta `-0.000101`); five events rose and five fell. No event improved the GT rank, and no top-1 peak changed ownership. Mean GT-minus-strongest-background margin changed from `-0.000958` to `-0.001053`, slightly worse.

| Sequence | Frame | M1 IoU | LCGM IoU | Delta | GT rank M1 -> LCGM | Margin M1 -> LCGM |
|---|---:|---:|---:|---:|---:|---:|
| 7rightorangegirl | 343 | 0.32536 | 0.32536 | +0.00000 | 1 -> 1 | 0.21311 -> 0.21306 |
| ab_bolstershaking | 2952 | 0.29974 | 0.29995 | +0.00021 | 1 -> 1 | 0.09299 -> 0.09272 |
| blkboydown | 49 | 0.27652 | 0.27582 | -0.00070 | 6 -> 6 | -0.07721 -> -0.07690 |
| boyss | 218 | 0.30582 | 0.30595 | +0.00013 | 5 -> 5 | -0.16551 -> -0.16550 |
| foamatgirl`srighthand | 67 | 0.29698 | 0.29705 | +0.00007 | 2 -> 2 | -0.02024 -> -0.02075 |
| girldownstairfromlight | 708 | 0.30070 | 0.30055 | -0.00015 | 1 -> 1 | 0.06006 -> 0.06011 |
| leftopenexersicebook | 120 | 0.29999 | 0.30000 | +0.00001 | 7 -> 7 | -0.71187 -> -0.71170 |
| rightbluewhite | 236 | 0.29838 | 0.29807 | -0.00031 | 1 -> 1 | 0.38312 -> 0.38320 |
| rightdarksingleman | 883 | 0.30742 | 0.30738 | -0.00004 | 1 -> 1 | 0.25404 -> 0.25332 |
| umbrella | 197 | 0.30280 | 0.30259 | -0.00022 | 3 -> 3 | -0.03807 -> -0.03810 |

The AMCM residual was very small: for the first three saved events its RGB/TIR relative RMS was about `0.061%-0.087%`; the full per-event tensors are retained. Target/background residual norm ratios averaged `0.992` for both modalities, so the residual was not target-selective. Across the ten events, RGB target-to-hard-background cosine shifted by only `+0.000045` and TIR by `+0.000037`; feature distances changed by less than `0.0001` on average. The fused feature distance to the M1 hard-background location changed from mean `18.9532` to `18.9493`, with no associated rank or ownership change.

## Exact actor-loss gradient audit (2026-10-20)

Replayed the same ten frames with the saved M1 state, template and crop in normal autograd mode. The candidate network remained in eval mode; no weights were changed. Loss came from the project `TBSITrackActor.compute_losses`: weighted GIoU + L1 + CENTER focal, with no rank or AMCM auxiliary loss in this config. At each AMCM output, measured the spatial inner product `gradient(loss, corrected_feature) * actual_residual`, summed over channels. Positive means applying the learned residual locally moves in a direction that increases the current tracking objective. This is a first-order attribution of the actual residual, not a claim about an optimizer step or sequence-level effect.

| Measure | RGB | TIR |
|---|---:|---:|
| GT region: residual direction increases loss | 10/10 | 9/10 |
| GT region mean directional loss change | +0.000138 | +0.000102 |
| Strongest background cell: positive / negative direction | 5 / 5 | 5 / 5 |
| Mean residual RMS / input RMS | 0.0695% | 0.0801% |

The gradient is present at the `tbsi_fuse_search` output (mean gradient norm `0.3630`), so the loss path is connected through the fusion layer. The target-region residual direction is consistently counterproductive under this loss, while the strongest-background direction is inconsistent. Together with near-unity target/background residual amplitude ratios and unchanged GT rank, this supports a stronger conclusion than the forward-only result: **the trained LCGM residual is not aligned with the localization objective on these fixed failures**. The residual is also very small, but simply increasing its scale is not justified because its measured GT direction mostly raises loss.

## Decision

`STOP THIS LCGM-AMCM ADAPTATION` for the present M1 objective and interface. The paper-inspired operator reaches the correct path and receives gradients, yet learned residuals are near-spatially-uniform, weak, and mostly move GT features in a loss-increasing direction. Fusion is not the blocked gradient path. Do not continue with scale-only tuning or another paper-module substitution at this interface. This does not rule out the paper's method in its native segmentation setup; it rejects this adaptation as a promising M1 localization correction based on the current evidence.

The reusable M1 precision-screen selector is `tools/select_m1_precision_screen_sequences_20261020.py`. It reads the M1 full-244 sequence audit and closed-loop traces, ranks persistent low-overlap/low-AUC failures, selects an auditable in-crop frame, and adds high-AUC controls. Its default output contains 20 failure sequences and 5 controls. The selected failure group has mean AUC `7.57` and frame IoU `0.269`; controls have mean AUC `89.31` and frame IoU `0.776`. This is intentionally a high-contrast mechanism screen, not a representative test sample. Freeze and hash each generated manifest before using it to compare candidates.

The gradient audit implementation is `tools/audit_m1_amcm_lcgm_gradient_fixed10_20261020.py`. Gradient results are in `output/experiments/M1_AMCM_LCGM_GRADIENT_AUDIT_20261020/per_sequence.csv` and `summary.json`. Do not use this result to claim full-sequence performance.

## Provenance

- Config SHA256: `2443d8b2b768b63462fa65659d2cbe8e1a34c28928526d3d824f731d96421701`
- Candidate checkpoint SHA256: `589760e12c1303667678cb1d519a36a3bfa713ba0655adb8985ed1c5e9b69df2`
- Diagnostic script SHA256: `eec3a89988f5de9edb3939e6f77d8987edd23fd2206ab808d9612e8d4d4985b8`
- Code commit at run: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`; AMCM source snapshot before LCGM: `experiments/snapshots/amcm_reliability_pre_lcgm_20261019.py`, SHA256 `a0f4ddc1197f2277267a544551ec630c51d711d84c610df25463225bcbed0550`.
- Outputs: `output/experiments/M1_AMCM_LCGM_QUICK15_3EP_20261019/` and `output/experiments/M1_AMCM_LCGM_FIXED10_DIAG_20261019/`.
