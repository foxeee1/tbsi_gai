# Token-routing diagnosis

## Parent and protocol

Parent is B3 dense V2 fusion. No independently saved B3 dense checkpoint was available, so this diagnostic uses the existing S1 epoch-3 checkpoint with dense `v2` configuration as an explicitly labeled proxy. It evaluates the frozen 60-sequence RPP-v2 test manifest, FP32 inference, serial scheduling, and at most 80 frames per sequence. No training was performed.

## Question

Do target and background template tokens receive materially different fusion updates, supporting token-wise conditional routing?

For each RGB/TIR template token at TBSI layers 3, 6, and 9, the diagnostic records:

`d_i = ||z_hat_i - z_i||_2`

with oracle target/background token labels derived from the initial bounding box.

## Result

The run produced 56,880 valid aggregate rows over all 60 sequences.

| Layer | RGB target/background | TIR target/background |
|---|---:|---:|
| 3 | 0.99890 | 0.99815 |
| 6 | 0.99843 | 0.99870 |
| 9 | 0.99797 | 0.99754 |

Ratios are target mean gain divided by background mean gain. They are effectively 1.0 at every layer and modality. Absolute mean gain increases with depth, but the increase is not target-selective.

## Diagnosis

The tested observable does not support the hypothesis that target versus background tokens require different fusion strength. This is negative evidence against a target-guided token router based on the current fusion-gain signal. It does not rule out token-wise routing based on modality reliability, spatial correspondence, or learned content features, but those variants need a distinct falsifiable signal before training.

Attribution: coordination/distribution hypothesis not identified by target/background gain; do not proceed directly to R1/R2 dense routing on this evidence alone.

## Artifacts

- `tools/run_token_gain_analysis.sh`
- `tools/analyze_token_gain.py`
- `output/experiments/C1_token_gain_analysis/logs/token_gain.jsonl`
- `output/experiments/C1_token_gain_analysis/logs/summary.json`

Canonical baseline SHA256 remains `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fb83d696538251a2d8fadd24b`; its diff is empty.

## Decision

Do not launch R1/R2 token routing yet. Stop the target/background routing hypothesis. Any next routing experiment must first define and validate a different observable, preferably modality disagreement or cross-modal correspondence, and must remain independent of the protected baseline.
