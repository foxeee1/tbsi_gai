# V2 Attention Replacement: Real Code and Full-Run Comparison

Date: 2026-09-15

This report separates executable code paths from experiment labels. The
canonical baseline YAML was not modified. Its SHA256 is
`4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`.

## Result status

All values below come from full LasHeR test evaluation over 244 sequences,
except where explicitly marked invalid for attribution.

| Group | Actual path | AUC | OP50 | OP75 | Precision | Norm Precision | Status |
|---|---|---:|---:|---:|---:|---:|---|
| B0 | Canonical baseline | 54.82 | 66.52 | 45.80 | 68.53 | 64.59 | valid reference |
| B3 old | V2 template fusion, early return | 55.88 | 67.98 | 46.99 | 69.92 | 66.07 | valid V2 result, old path |
| B4 old | Same early-return V2 path; SMSA bypassed | 56.81 | 69.05 | 47.56 | 71.22 | 67.13 | invalid SMSA attribution |
| B3 corrected | V2 template fusion + full TBSI + Softmax | 55.98 | 67.84 | 46.97 | 69.87 | 65.97 | valid parent |
| B4 corrected | V2 template fusion + full TBSI + SMSA | 55.20 | 66.87 | 46.46 | 68.79 | 65.00 | valid SMSA run, regression |

The causally valid SMSA comparison is corrected B4 versus corrected B3:

| Metric | Delta from SMSA |
|---|---:|
| AUC | -0.78 |
| OP50 | -0.97 |
| OP75 | -0.51 |
| Precision | -1.08 |
| Norm Precision | -0.97 |

Therefore the supported conclusion is: V2 fusion improves the baseline, but
SMSA does not improve the corrected V2-plus-TBSI path under this configuration
(`ENTMAX_ALPHA=1.3`). The old B4 gain must not be reported as an SMSA gain.

## Actual forward path

The implementation is in `lib/models/layers/tbsi_layer.py`.

`V2TemplateFusion` concatenates RGB and TIR template tokens, applies joint
multi-head self-attention, adds a residual, and splits the streams again. The
corrected path then continues into the original TBSI interaction sequence:

```text
RGB/TIR template tokens
        |
        v
joint V2 template MHSA + residual
        |
        v
fused template + original search tokens
        |
        +--> TIR search -> fused template (ca_s2t_i2f)
        +--> fused template -> RGB search (ca_t2s_f2v)
        +--> RGB search -> fused template (ca_s2t_v2f)
        +--> fused template -> TIR search (ca_t2s_f2i)
        +--> template interactions (ca_t2t_f2v/f2i)
```

The critical corrected behavior is in `TBSILayer.forward`: V2 no longer
returns immediately after template fusion. It only changes template
initialization and leaves all subsequent interaction blocks active.

SMSA is implemented in `lib/models/layers/attn.py` inside `Attention_st`:

```python
attn = (q @ k.transpose(-2, -1)) * self.scale
if self.use_smsa:
    with torch.cuda.amp.autocast(enabled=False):
        attn = _entmax_bisect(attn.float(), alpha=self.smsa_alpha, dim=-1)
else:
    attn = attn.softmax(dim=-1)
```

`use_smsa` is restricted to `mode == 's2t'`. With corrected B4, the six
SMSA blocks are active: three TBSI locations `[3, 6, 9]` times two modality
directions (`ca_s2t_i2f` and `ca_s2t_v2f`).

## Structural validation

The one-batch validation before the corrected full runs confirmed:

- corrected B3 and B4 output shape: `(1, 1, 4)`;
- finite loss for both configurations;
- nonzero V2 fusion gradients for both configurations;
- all six corrected B4 SMSA blocks had attention maps and nonzero gradients;
- SMSA zero ratio was approximately `0.966-0.971`;
- effective support was approximately `7.4-8.8` search tokens;
- attention normalization error was approximately `1.2e-7`;
- no NaN/Inf was observed in the one-batch SMSA attention check.

The full corrected runs produced 4,860 V2 gradient rows each. The aggregate
gradient-to-weight diagnostic sum was `57451.4` for corrected B3 and `57464.7`
for corrected B4. These are diagnostics, not significance tests.

## Reproduction commands

The actual full-run shell is:

```bash
cd /root/autodl-tmp/TBSI_gai/TBSI
./tools/run_corrected_b3_b4_full.sh
```

The script runs, in order:

```bash
/root/autodl-tmp/conda_envs/tbsi/bin/python -u tracking/train.py \
  --script tbsi_track --config B3_v2_fusion_softmax_corrected \
  --save_dir ./output --mode single

TBSI_INFER_FP16=0 /root/autodl-tmp/conda_envs/tbsi/bin/python -u tracking/test.py \
  tbsi_track B3_v2_fusion_softmax_corrected \
  --dataset_name lasher_test --threads 0 --num_gpus 1

/root/autodl-tmp/conda_envs/tbsi/bin/python -u tracking/analysis_results.py \
  --tracker_name tbsi_track \
  --tracker_param B3_v2_fusion_softmax_corrected \
  --dataset_name lasher_test
```

It repeats the same three commands for
`B4_v2_fusion_smsa_corrected`. Output is isolated under:

- `output/experiments/B3_v2_fusion_softmax_corrected/`
- `output/experiments/B4_v2_fusion_smsa_corrected/`

The authoritative full-run log is
`output/experiments/B3_v2_fusion_softmax_corrected/logs/corrected_full_pipeline.log`.
The final corrected B4 line is:

```text
B4_v2_fusion_smsa_corrected | 55.20 | 66.87 | 46.46 | 68.79 | 65.00
```

## Not yet run

B5 temporal memory has a configuration file but no valid full training/test
result. B1 and B2 are also not valid completed comparison groups in the
current evidence package. They must not be included in a performance claim.

Environment for the corrected full runs: Python 3.8.20, PyTorch 1.9.0+cu111,
CUDA 11.1, one NVIDIA GeForce RTX 3090, LasHeR train/full test, seed 42,
15 epochs, batch 32, gradient accumulation 4, and FP32 tracker inference.
