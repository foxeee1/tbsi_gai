# AMCM directed residual loss screen and template/search audit (2026-10-19)

## Template/search audit

The current AMCM implementation modifies the **search branch**. In `forward_fusion_only`, it reads RGB/TIR search maps and initial template tokens. The template tokens are used only to compute two global-mean cosine similarity maps. In directed-residual mode, the corrected RGB/TIR search maps replace the original maps before `tbsi_fuse_search`. The stored template `self.z_dict1` is created once in `initialize()` and is not updated by AMCM during `track()`.

The current path is therefore:

```text
initial template tokens -> similarity condition
current RGB/TIR search -> AMCM residual correction -> tbsi_fuse_search -> CENTER head -> box/state update
```

It is not a template updating method. The paper's TBSI uses template bridging to improve downstream search interaction, while TPTU changes the inference-time template state. The current experiment only tests the search-side part of that idea.

## Serial loss screen

All runs start from the protected M1 epoch-15 checkpoint and keep the same residual injection, 3-epoch proxy training, 512 samples/epoch, batch 2, FP32, seed 42, worker 4, and 15-sequence manifest. Only the auxiliary objective changes.

| run | auxiliary objective | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---|---:|---:|---:|---:|---:|
| KL control | modality probability KL at GT center | 51.16 | 61.75 | 42.36 | 65.77 | 60.34 |
| E5 | no AMCM auxiliary loss; tracking loss only | 54.87 | 66.62 | 44.99 | 70.33 | 64.84 |
| E6 | target/background feature margin, weight 0.5 | 55.13 | 67.24 | 44.98 | 70.48 | 65.44 |
| E7 | failure-conditioned Hann-aware center rank | 50.31 | 60.51 | 42.20 | 64.06 | 58.82 |

E5 improves the KL residual control by `+3.71` AUC and exceeds the post-fusion gate control (`54.36`) by `+0.51`. E6 is the best aggregate result in this screen, exceeding the gate by `+0.77`, but its gain is small and proxy-only. E7 is rejected under this contract.

## Group and per-sequence AO

| group | E5 mean AO | E6 mean AO | E7 mean AO |
|---|---:|---:|---:|
| all 15 | 0.5572 | 0.5600 | 0.5110 |
| low AUC | 0.5363 | 0.5135 | 0.4240 |
| difficult | 0.6618 | 0.6661 | 0.6787 |
| normal | 0.4737 | 0.5002 | 0.4304 |

E5 recovers `bluegirlbiketurn` to `0.5204` and `leftboyoutofthetroop` to `0.5340`, while E6 gives the strongest normal-group average and improves `whiteboyrightcoccergoal` to `0.529`. E6 still loses on `bluegirlbiketurn` relative to E5 (`0.423` vs `0.520`). E7's aggregate failure is driven by large losses on `boyleftblkrunning2crowd` (`0.254`), `bluegirlbiketurn` (`0.265`) and `leftboyoutofthetroop` (`0.388`), despite gains on `boyss` and `manoncall`.

## Interpretation and direction

The KL objective is the wrong first target for the residual path: it supervises a reliability probability, while the deployed effect is `probability * cross-modal message`. Removing it produces the largest recovery. The representation margin is closer to the stated goal of changing fused features to separate target from background, but its short-run gain is not yet stable on the low-AUC group. The center rank objective is applied to the final response and is therefore too indirect for a frozen-backbone residual adapter in this setting.

The research direction should remain feature fusion, but the conditioning source should become target-preserving template context rather than only an initial-template global mean. A suitable next architecture is an initial-template anchor plus a confidence-gated candidate template memory, used to condition the pre-fusion residual; the memory update can follow the paper's target-preserved template principle, while the residual remains the fusion contribution. This addresses the observed crop drift and recovery chain without turning the experiment into a pure search-factor or head modification. The next test should be E5 versus E6 with a small template-memory condition and no new reliability KL; do not combine E6 with E7.

Evidence: configs and checkpoints under `output/experiments/M1_AMCM_DR_LOSS_E5_NOSUP_QUICK15_3EP_20261019/`, `...E6_REPR.../`, and `...E7_RANK.../`; tracking results under `output/test/tracking_results/tbsi_track/`; manifest `experiments/manifests/m1_amcm_prefusion_quick15_20261019.txt`.
