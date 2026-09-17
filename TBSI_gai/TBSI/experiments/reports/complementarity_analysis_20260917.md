# Cross-modal Residual Complementarity Analysis

Date: 2026-09-17
Parent claim: B3 joint fusion may contain a learnable cross-modal complementary residual.
Decision question: does the fusion residual align with the source modality more strongly in target regions, so that a complementary adapter is justified?

## Decision

Stop the complementary-adapter route. The residual norm is somewhat larger on target regions, but the residual is not more source-aligned there and is not separated from the target representation. The aggregate cosine values show stronger alignment with the existing target representation than with the source representation at every inspected layer and direction. This is insufficient evidence for a source-complementary adapter.

B3 V2 joint fusion remains a retained fusion contribution from the preceding experiments. This analysis does not establish a new trainable module and does not claim a new tracking score.

## Protocol

- Checkpoint: `S1_v2_crossmodal_topk32_rpp/checkpoints/TBSITrack_ep0003.pth.tar`, used as a frozen diagnostic proxy for the dense V2 fusion path.
- Test set: frozen RPP-v2 sequence-level test manifest, 60 sequences; no sequence selection was changed after seeing results.
- Inference: FP32 (`TBSI_INFER_FP16=0`), serial execution, maximum 80 frames per sequence.
- Layers and directions: layers 3, 6, 9; RGB -> TIR and TIR -> RGB.
- Regions: target and background tokens.
- Observables: residual norm, residual/source cosine, residual/target cosine, and source-minus-target cosine.
- Run command: `TBSI/tools/run_complementarity_analysis.sh`.
- Code snapshot: Git commit `7f6cb012ac56de245dbb5e52d409469af5fa074c`.
- This is a diagnostic proxy, not a full 15-epoch RPP training or full LasHeR confirmation.

Raw evidence is in:

- `TBSI/output/experiments/complementarity_analysis/logs/complementarity.jsonl`
- `TBSI/output/experiments/complementarity_analysis/logs/summary.json`
- `TBSI/output/experiments/complementarity_analysis/logs/pipeline.log`

## Results

All values below are target minus background means unless stated otherwise.

| Layer / direction | Delta residual norm | Delta source cosine | Delta target cosine | Delta source-target cosine |
|---|---:|---:|---:|---:|
| L3 RGB -> TIR | +1.197 | -0.038 | -0.003 | -0.035 |
| L3 TIR -> RGB | +1.213 | -0.033 | +0.006 | -0.039 |
| L6 RGB -> TIR | +1.295 | -0.012 | -0.002 | -0.010 |
| L6 TIR -> RGB | +1.066 | -0.009 | +0.001 | -0.010 |
| L9 RGB -> TIR | +1.997 | -0.010 | +0.000 | -0.010 |
| L9 TIR -> RGB | +1.391 | -0.010 | +0.001 | -0.011 |

The unconditioned means show the same structural pattern. For example, at L9 RGB -> TIR, residual/source cosine is 0.649 while residual/target cosine is 0.983; at L6 RGB -> TIR the corresponding values are 0.574 and 0.967. The other layers and reverse direction follow the same ordering.

## Interpretation and boundary

The residual has a modest target/background magnitude difference, especially at L9, so it is not numerically inert. However, target-region residuals do not become more aligned with the source modality. The source-minus-target cosine remains negative in all six slices, while target cosine is high in the aggregate means. Thus the residual is better described as target-representation reinforcement or a coupled fusion correction than as clean source-modality complementarity.

This analysis cannot prove that no adapter can ever help: it is a frozen checkpoint diagnostic, uses a 60-sequence proxy test set, and does not measure downstream learned-adapter performance. It does establish that the current residual evidence does not justify spending another training run on that adapter under the present storage and compute budget.

## Reproducibility and safety checks

- Canonical baseline YAML SHA256: `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fb83d696538251a2d8fadd24b`.
- The canonical baseline YAML diff was empty.
- No training, test, or analysis process remained after completion.
- No baseline output directory, baseline log, or canonical experiment name was used as an output target.

## Next route

Close the complementary-adapter branch and move to the separate template-update / Innovation 2 investigation. Any future fusion experiment must be motivated by a different, explicitly testable mechanism rather than by this residual-complementarity hypothesis.
