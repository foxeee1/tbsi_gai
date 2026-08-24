# TBSI Original Repository Comparison

## Compared Sources

- Upstream repository: `https://github.com/RyanHTR/TBSI`
- Upstream snapshot: `6456fdd` (`2023-05-29`)
- Local baseline commit: `c79204832e6ec1a99665114f5ab87d6bdb595ea3`
- Local working tree: `/root/autodl-tmp/TBSI_gai/TBSI`

## Main Differences

| Area | Upstream | Local baseline | Impact |
|---|---|---|---|
| Inference precision | FP32 network, input and Hann window | `network.cuda().half()`, half input and half Hann window | **High: direct prediction change** |
| TBSI bridge | 61-line original bridge | 158-line degradation-aware bridge | High; disabled flags still require parity verification |
| TBSI model wrapper | 140 lines | 467 lines; temporal, DA and stage-2 branches | High; baseline flags are off, but state construction/forward paths differ |
| Attention blocks | Original `CASTBlock` | Optional attention gate and quality-mask arguments | Medium; expected neutral only when disabled and `quality_mask=None` |
| Training actor | Original single-frame path | Temporal branch and reorganized loss path | Medium; temporal branch is off, but must be parity tested |
| Training entry | No global matmul tuning | Enables `cudnn.benchmark` and `set_float32_matmul_precision('high')` | Medium: changes numerical algorithms under PyTorch 2 |
| Optimizer grouping | Original SOT grouping | Adds stage-2 and temporal-LR branches | Low for current flags, because `TEMPORAL_LR=None` and stage-2 is empty |
| Dataset package | Tracked in upstream repository | Ignored locally and later copied from a changed local version | Medium for provenance; low for current single-search semantics after comparison |
| LasHeR train dataset | No skip set, no cache | Skips 3 sequences and adds file/image cache | Protocol and performance change; cache should not change values |
| Data sampler | Original frame order | Sorts sampled search IDs only when `num_search_frames > 1` | No effect for current `SEARCH.NUMBER=1` |
| Data loader | Shared-memory tensor stacking | Plain `torch.stack`, prefetch/persistent worker options | Mainly performance/memory; not expected to change sample values |
| Evaluation protocol | Upstream does not silently skip the three local sequences | Local legacy protocol skips `advancedredcup`, `cameraman_1202`, `mirroratleft` | Explains 244 vs 245, not a 1.91 AUC drop within 244 |
| Checkpoint loading | Loads checkpoint per tracker instance | Adds process-level checkpoint cache | Should be value-neutral |

## Most Important Finding

The local inference code is not the upstream inference code. In
`lib/test/tracker/tbsi_track.py`, the local version changes all of the following:

```text
network.cuda()                    -> network.cuda().half()
preprocessor.process(...)         -> process(..., half=True)
Hann window                       -> half()
torch.no_grad()                   -> torch.inference_mode()
```

The first three changes directly alter the numerical path used to generate
boxes. The model is trained with AMP but the upstream tracker evaluates in
FP32. This is the highest-priority explanation for the stable `53.55 AUC`
versus the historical `55.46 AUC`.

## Data Package Finding

The upstream repository contains the complete `lib/train/data` package. The
local Git history does not contain it because the repository ignores `data`.
The current local copies are not identical to upstream:

- `sampler.py`: local sorts multi-search frame IDs; inactive for `SEARCH.NUMBER=1`.
- `loader.py`: local removes shared-memory output allocation and adds worker
  compatibility options; this changes throughput/memory behavior, not tensor
  values in the normal path.
- `lasher.py`: local adds three skipped sequences and image/file caches. The
  skip set affects protocol; the cache should be semantically neutral.
- `processing.py` and the core crop/augmentation implementation match the
  upstream snapshot by SHA-256.

Therefore the missing data package is a reproducibility defect, but it is not
currently the strongest explanation for the 244-sequence score gap.

## Configuration Finding

The upstream repository publishes two official settings in its README:

- ImageNet-1k: `vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k.yaml`, learning rate
  `4e-4`, `SOT_PRETRAIN=False`.
- SOT: `vitb_256_tbsi_32x1_1e4_lasher_15ep_sot.yaml`, learning rate `1e-4`,
  `SOT_PRETRAIN=True`.

The local `完美基线全量.yaml` is a later experiment configuration derived from
the SOT setting. It additionally changes worker/prefetch/checkpoint behavior
and embeds the 244-sequence claim. It is not an upstream configuration file.

## Conclusion

The reproduction gap is not explained by one YAML parameter. The evidence
supports this priority order:

1. **Inference FP16 conversion in the local tracker.**
2. **Changed TBSI bridge/model implementation, even with optional modules off.**
3. **Different runtime/data package provenance.**
4. 244/245 evaluation protocol and stochastic training differences.

The next decisive test is a fixed-checkpoint inference A/B on the same
sequences: upstream FP32 tracker versus local FP16 tracker. This costs far
less than another training run and directly tests the highest-risk difference.
