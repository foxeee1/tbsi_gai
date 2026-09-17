# Progressive Joint Template Fusion Mechanism Analysis

Date: 2026-09-17
Question: where and how frequently should V2 joint template fusion be applied?
Parent: B3 V2 joint fusion, retained as the current positive fusion direction.

## Protocol and comparability

Seven active-layer schedules were evaluated: F3, F6, F9, F3+F6, F3+F9, F6+F9, and F3+F6+F9. The model used the same frozen S1 epoch-3 checkpoint as a dense-V2 proxy, the same frozen 60-sequence RPP-v2 test manifest, FP32 inference, serial execution, and a maximum of 80 frames per sequence. Only the active fusion-layer gate changed. This is a representation diagnostic, not a trained ablation and not a tracking-score comparison.

The implementation uses `MODEL.BACKBONE.TBSI_FUSION_ACTIVE`; the module list and checkpoint tensor shapes remain unchanged. The empty value preserves the existing all-layer behavior. The unified entry point is `TBSI/tools/run_fusion_mechanism_analysis.sh`.

Evidence:

- `TBSI/output/experiments/fusion_mechanism_analysis/logs/summary.json`
- `TBSI/output/experiments/fusion_mechanism_analysis/logs/A1_F3.jsonl` through `A1_F369.jsonl`
- `TBSI/output/experiments/fusion_mechanism_analysis/logs/pipeline.log`

## Results

The following values are the RGB2TIR/background slice, included as a compact depth comparison. The reverse direction follows the same qualitative pattern.

| Schedule | Layer | Cross cosine before -> after | Delta | RGB preserve | TIR preserve |
|---|---:|---:|---:|---:|---:|
| F3 | 3 | 0.6255 -> 0.6434 | +0.0179 | 0.9761 | 0.9733 |
| F6 | 6 | 0.6409 -> 0.6533 | +0.0124 | 0.9838 | 0.9831 |
| F9 | 9 | 0.7375 -> 0.7453 | +0.0078 | 0.9876 | 0.9878 |
| F3+F6 | 3,6 | layer-local; see raw summary | | | |
| F3+F9 | 3,9 | layer-local; see raw summary | | | |
| F6+F9 | 6,9 | layer-local; see raw summary | | | |
| F3+F6+F9 | 3,6,9 | layer-local; see raw summary | | | |

For each active layer, RGB/TIR preservation remains high, approximately 0.973 to 0.990. Fusion increases cross-modal cosine at the inspected layer, but the measured increment decreases from early to late depth in this frozen proxy. The target/background residual norm remains non-zero, while residual/source cosine does not become positive or source-dominant; this agrees with the previous complementarity analysis.

## Interpretation

The evidence supports a cautious mechanism description: V2 joint fusion increases shared representation while retaining substantial modality-specific representation. It does not support the stronger claim that later fusion is intrinsically better, because these schedules were not separately trained and the checkpoint was trained with the dense schedule. The layer-local diagnostic suggests early fusion produces the largest immediate representation change, whereas later layers operate on already more aligned representations.

The seven schedules therefore establish a useful analysis axis, not a selected new training recipe. No schedule is promoted to a full confirmation run from this diagnostic alone. A paper-facing claim should use “progressive joint template interaction” as a mechanism hypothesis and keep the depth result labeled proxy/diagnostic until trained schedule ablations are available.

## Safety and reproducibility

- Canonical baseline YAML SHA256: `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fb83d696538251a2d8fadd24b`.
- Canonical baseline YAML diff is empty.
- Checkpoint SHA256: `b4260363f9159b0c2e89790457ca53d3c9cff639e80bfc2bd426afc8d47f67cb`.
- Frozen test manifest SHA256: `ab587f801525c2e7e0fc7a8465fb90f3a2d99b67764eb2bc19cac2b422a938a8`.
- No training process or baseline output was used.

## Route decision

The requested zero-training Fusion Depth plus Information Preservation analysis is complete. The fusion-module search remains closed. The evidence is sufficient to write the mechanism hypothesis and to guide the separate Template Update / Innovation 2 line. Do not claim a new best schedule or new AUC without a separately trained, multi-seed confirmation.
