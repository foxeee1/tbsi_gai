# M1 AMCM NLformer Quick Training/Test

## Experimental contract

Only the AMCM internal mode changed to the audited NLformer bottleneck
operator. The official local first-order, second-order covariance, and
low-rank complementary branches run at 256 channels between `2C -> 256 ->
2C` projections. The output is split into RGB/TIR streams before the existing
`tbsi_fuse_search` path. Backbone, TBSI, CENTER head, tracker state and losses
were unchanged.

Training used the published `pretrained_models/TBSITrack_SOT_Pretrained.pth.tar`
only, 512 samples/epoch x 3 epochs, batch 2, FP32, `NUM_WORKER=4`. M1 epoch15
was not loaded into this candidate. Closed-loop evaluation used the same fixed
10-event cohort as earlier AMCM diagnostics and compared event IoU with the
protected M1 results.

## Event results

| Sequence | M1 IoU | NLformer IoU | Delta |
|---|---:|---:|---:|
| 7rightorangegirl | 0.2559 | 0.0000 | -0.2559 |
| ab_bolstershaking | 0.3501 | 0.3269 | -0.0232 |
| blkboydown | 0.2895 | 0.0000 | -0.2895 |
| boyss | 0.3421 | 0.5697 | +0.2276 |
| foamatgirl`srighthand | 0.3215 | 0.3091 | -0.0125 |
| girldownstairfromlight | 0.2927 | 0.3818 | +0.0891 |
| leftopenexersicebook | 0.3029 | 0.3004 | -0.0025 |
| rightbluewhite | 0.2403 | 0.5994 | +0.3591 |
| rightdarksingleman | 0.3827 | 0.0487 | -0.3339 |
| umbrella | 0.0000 | 0.0773 | +0.0773 |

Mean event delta is `-0.01644`, median `-0.00747`, with 4 wins and 6 losses.
The candidate has large positive changes on `boyss` and `rightbluewhite`, but
large failures on `blkboydown` and `rightdarksingleman`.

## Optimization signal

The module is trainable under the existing tracking loss. Logged mean residual
norm/input norm is approximately `3.30%` for RGB and `3.50%` for TIR, rising to
about 6% near the end of the short run. This is materially larger than the
previous hand-built orthogonal module, whose residual was below 0.4%.

The result therefore supports **operator activation and optimization**, but not
target/background correctness. No GT-rank or target/background residual map was
recorded in this run, so the reason for the sequence-level sign changes is not
yet identified. In addition, the candidate and M1 use different training
initialization histories, so the event comparison is exploratory rather than a
matched performance claim.

## Decision

`DIAGNOSTIC-ONLY / NOT-PROMOTED`.

NLformer is a viable mature operator and its branches are being used, but this
quick run does not establish that it fixes background peak selection. Before a
longer or full-data run, run the fixed-frame target/background diagnostic with
the trained epoch-3 checkpoint: compare branch residuals, GT rank, target-vs-
strongest-background margin, and the same quantities after `tbsi_fuse_search`.
Only if those signals improve on the two severe failure sequences should this
candidate receive a matched training comparison.

Evidence: `output/experiments/M1_AMCM_NLFORMER_QUICK15_3EP_20261020/`.
