# M1 Early Search Relation Bias RPP-v2 60-sequence test

The epoch-3 relation-bias checkpoint was evaluated on the historical RPP-v2 60-sequence proxy used by the earlier Early-conditioned gate experiment. The manifest hash is `ab587f801525c2e7e0fc7a8465fb90f3a2d99b67764eb2bc19cac2b422a938a8` and coverage is 60/60.

| metric | relation bias epoch 3 | historical Early-gate proxy |
|---|---:|---:|
| AUC | 49.739 | 51.46 |
| OP50 | 58.766 | 61.10 |
| OP75 | 41.276 | 43.59 |
| Precision | 60.853 | 62.71 |
| Normalized Precision | 57.454 | 59.82 |

The historical proxy did not retain a matched parent prediction on this manifest, so the comparison is absolute and directional only. Still, the larger cohort does not reproduce a positive signal: all five metrics are lower than the previous gate proxy. The lowest-AUC sequences include `2girlup`, `lastleftgirl`, `girl`sblkbag`, `boyunder2baskets`, and `leftunderbasket`.

**Conclusion: the 12-sequence negative result is not adequately explained by small-cohort overfitting.** The relation-bias branch remains unsupported under this training and inference contract. Do not promote it to full training. A future retry should first create a matched 60-sequence parent replay and run gain-zero/frozen-projection controls; otherwise move to a different operator family.
