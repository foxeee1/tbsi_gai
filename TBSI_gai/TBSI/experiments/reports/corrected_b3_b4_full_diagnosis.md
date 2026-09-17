# Corrected B3/B4 Full Diagnostic Report

Date: 2026-09-15

## Scope and evidence boundary

The only valid causal comparison in the current package is:

- Parent: `B3_v2_fusion_softmax_corrected`, V2 template fusion followed by the full TBSI interaction path with Softmax.
- Candidate: `B4_v2_fusion_smsa_corrected`, the same path with SMSA/Entmax in the six `s2t` blocks.

Both runs used full LasHeR training, 15 epochs, batch 32, gradient accumulation
4, seed 42, one RTX 3090, and full 244-sequence FP32 test. The canonical
baseline YAML was unchanged.

Old B3/B4 are retained as historical artifacts. Old B4 is not evidence for an
SMSA gain because its V2 branch returned before the SMSA blocks.

## Main performance result

| Group | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| Canonical baseline | 54.82 | 66.52 | 45.80 | 68.53 | 64.59 |
| Corrected B3: V2 + Softmax | **55.98** | **67.84** | **46.97** | **69.87** | **65.97** |
| Corrected B4: V2 + SMSA | 55.20 | 66.87 | 46.46 | 68.79 | 65.00 |
| B4 - B3 | **-0.78** | **-0.97** | **-0.51** | **-1.08** | **-0.97** |

Corrected B3 improves the canonical reference by AUC `+1.16`. Corrected B4
retains only `+0.38` AUC over the canonical reference. The B4 regression is
therefore real under this one-seed full-run protocol, but it is not yet a
multi-seed statistical claim.

## 1. Implementation validity

The corrected forward path is:

```text
joint RGB/TIR template MHSA + residual
    -> fused template construction
    -> TIR search -> fused template
    -> fused template -> RGB search
    -> RGB search -> fused template
    -> fused template -> TIR search
    -> template interaction blocks
```

Corrected B4 activates six SMSA blocks: locations `[3, 6, 9]` multiplied by
`ca_s2t_i2f` and `ca_s2t_v2f`. The durable validation command is:

```bash
cd /root/autodl-tmp/TBSI_gai/TBSI
/root/autodl-tmp/conda_envs/tbsi/bin/python tools/validate_attention_paths.py
```

The output is saved in
`output/experiments/attention_path_validation.json`. It confirmed finite loss,
correct output shape, nonzero V2 gradients, six captured SMSA maps, and
nonzero gradients in the SMSA path.

## 2. Optimization and update health

The full diagnostic files contain 270 gradient snapshots and 4,860 V2
parameter rows per run.

| Signal | Corrected B3 | Corrected B4 | Interpretation |
|---|---:|---:|---|
| Mean total gradient norm | 0.100000 | 0.100000 | clipping is active at the configured 0.1 limit |
| Mean backbone gradient diagnostic | 0.002221 | 0.002228 | no broad gradient disappearance |
| Mean head gradient diagnostic | 0.001711 | 0.001696 | essentially matched |
| Mean V2 parameter gradient | 0.000758 | 0.000762 | V2 path learns in both groups |
| V2 grad/weight diagnostic | 0.001127 | 0.001124 | comparable update scale |
| Nonfinite full-run events | 0 | 0 | corrected runs show no recorded event |

The total norm being exactly `0.1` is a post-clipping diagnostic, so it must not
be interpreted as the raw gradient norm. The evidence rules out “SMSA did not
receive gradients” as the primary explanation for the regression.

## 3. SMSA representation behavior

Corrected B4 training ended at approximately:

- entropy: `1.27243`;
- zero ratio: `0.94670`;
- effective support: `13.64384` of 256 search tokens;
- normalization error: `0.00000` at logged precision;
- NaN/Inf flag: `0.00000`.

Thus SMSA is genuinely sparse and numerically normalized. However, sparsity is
not target correctness. The current SMSA path has no target-support loss or
box-aware support constraint. It selects a sparse distribution from feature
similarity alone, so a high-confidence background region can win the support
competition.

## 4. Training dynamics and representation coordination

At epoch 15, the two runs have almost identical training behavior:

- corrected B3 final total loss: about `0.77007`, IoU `0.81787`;
- corrected B4 final total loss: about `0.77015`, IoU `0.81801`;
- corrected B3 mean V2 feature deltas: RGB `217.87`, TIR `221.02`;
- corrected B4 mean V2 feature deltas: RGB `218.04`, TIR `222.58`.

The small matching training differences, combined with the test regression,
point away from simple optimization failure. The likely issue is downstream
coordination: SMSA changes which search tokens update the fused template, but
the later `t2s` and `t2t` blocks have no explicit mechanism to verify that the
selected support remains target-aligned.

SMSA also costs compute. The corrected training forward time is approximately
`0.637 s` for B3 versus `0.655 s` for B4, about a 2.8% increase; measured
training throughput is approximately 50.0 versus 48.7 FPS.

## 5. Sequence-level evidence

The durable comparison command is:

```bash
cd /root/autodl-tmp/TBSI_gai/TBSI
/root/autodl-tmp/conda_envs/tbsi/bin/python tools/compare_corrected_sequences.py
```

Output:

```text
sequences=244 wins=120 ties=0 losses=124 mean_delta=-0.8087 median_delta=-0.0139
```

The per-sequence file is:

`output/experiments/B4_v2_fusion_smsa_corrected/logs/per_sequence_overlap.csv`

Largest observed regressions in average overlap include:

| Sequence | Delta AO |
|---|---:|
| `ab_girlcrossroad` | -49.52 |
| `whitegirlinlight` | -46.23 |
| `right2ndflagformath` | -22.83 |
| `runningcameragirl` | -20.96 |
| `basketballathand` | -20.53 |
| `bikefromlight` | -20.30 |

Largest gains include:

| Sequence | Delta AO |
|---|---:|
| `ab_whiteboywithbluebag` | +29.74 |
| `carcominginlight` | +23.26 |
| `rightblkfatboyleftwhite` | +17.23 |
| `bikeboyintodark` | +14.30 |

The near-balanced win/loss count with a negative mean indicates a heavy-tail
failure pattern rather than a uniform degradation. The next analysis needed is
attribute mapping for these sequences: OCC/PO, FM, TC, LI, DEF, LR, HI, BC/SA,
SV, and ordinary sequences. Current logs do not yet provide that join.

## 6. Diagnosis using L1-L4

### L1: learning failure

Not supported as the primary cause. SMSA maps, gradients, normalization, and
training loss are healthy.

### L2: information bottleneck

Plausible. A support of roughly 13.6/256 tokens may discard useful weak target
evidence, especially under blur, low light, partial occlusion, or thermal
confusion.

### L3: position/scale mismatch

Plausible but unproven. SMSA is applied in the search-to-template attention at
all three TBSI locations, while the supervision is only the final box loss.
There is no explicit mechanism ensuring that token support is spatially
consistent with the target box at each intermediate location.

### L4: coordination/distribution mismatch

Most supported diagnosis. Entmax changes the support distribution entering the
fused-template update, but later interaction blocks remain unchanged and do
not receive modality reliability or target-support state. The large negative
outliers are consistent with occasional wrong sparse support followed by
irreversible template contamination.

## 7. DPBA -> TMR -> RTMM gap map

The proposed hierarchy maps cleanly onto the measured failures:

| Level | Missing capability in corrected B3/B4 | Proposed role |
|---|---|---|
| Token | sparse support is not target-aware | DPBA: target-supervised support selection |
| Modality | V2 fuses RGB/TIR without explicit reliability routing | TMR: estimate and gate modality contribution |
| Temporal | template state is initialized but not conservatively updated | RTMM: confidence-controlled memory write/update |

This ordering is better motivated than adding another feature enhancer. It
matches the observed failure chain: wrong token support can corrupt modality
fusion, and unverified fusion can corrupt future memory.

## Recommended next experiment

Do not start a combined DPBA+TMR+RTMM run. The first follow-up should be one
variable only:

**DPBA-only:** retain corrected B3's Softmax parent and add target-supervised
support guidance to the existing `s2t` attention. Keep TMR and RTMM disabled.

Expected falsifiable signature:

- support overlap with the ground-truth search region increases;
- catastrophic negative AO outliers decrease;
- mean performance improves without requiring a much denser support;
- V2 fusion and training schedule remain unchanged.

If DPBA reduces catastrophic loss but not mean AUC, the next intervention is
TMR. RTMM should only be tested after token and modality reliability signals
are available and logged.

## Limitations

- Results are single-seed full runs; seed variance is not estimated.
- No attribute-level join has yet been computed for the per-sequence outliers.
- The corrected B4 result establishes a regression under the current SMSA
  settings, not a universal statement that all sparse attention is harmful.
- The working tree contains the exact research code used by the runs, but the
  changes are not yet committed; the run logs record HEAD
  `7f6cb012ac56de245dbb5e52d409469af5fa074c` plus the working-tree configs.
