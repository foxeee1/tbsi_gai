# IG-CAB search-only prescreen

Status: gate evaluator created; no IG-CAB module, P0/P1, training, or tracking run has been performed. Result status remains **未验证基线**.

## Interface audit

The current working tree routes transformer block `i` through `TBSILayer` after that block. At the requested location 9, the `fusion_mode == 'tbsi_e'` branch runs `v2_template_fusion` on template tokens, then concatenates the original search tokens and returns. The proposed adapter boundary is therefore inside this location-9 branch, after template fusion and before the return, changing only `x_v[:, lens_z:]` and `x_i[:, lens_z:]`. Do not attach it to `backbone_feat`.

The Windows `IG-CAB.py` and the author repository's exact `CrossAttention_S` implementation are not present in this workspace. The operator has not been implemented or claimed equivalent. Before adding it, compare Q/K/V projections, normalization, reshape/transpose, scaling, softmax axis, bias, and output projection against the supplied implementation. Keep only the required code in the PyTorch 1.9 tracker repository.

## Gate tool

`tools/igcab_search_screen_gate_v1.py` evaluates saved evidence only. It does not build models, launch training/testing, or generate measurements. Each phase writes a JSON decision artifact; later phases require earlier passing artifacts.

### P0

Command:

```bash
python tools/igcab_search_screen_gate_v1.py p0 \
  --evidence PATH/p0_evidence.json --output PATH/p0_gate.json \
  --max-extra-memory-mb PREDECLARED_VALUE \
  --max-latency-overhead-pct PREDECLARED_VALUE
```

Evidence must record the actual location-9 boundary, exactly one adapter call in train and serial inference, `lens_z=64`, 256 search tokens, 768 channels, 16x16 layout, checkpoint missing/unexpected keys, full-tracker and box parity at gamma zero, unchanged template tokens, stable CPU FP32 and warmed CUDA FP32, and measured baseline/candidate peak memory and latency. Limits must be declared before collecting measurements.

### P1

Command:

```bash
python tools/igcab_search_screen_gate_v1.py p1 \
  --evidence PATH/p1_evidence.json --p0-gate PATH/p0_gate.json \
  --output PATH/p1_gate.json --min-write-ratio PREDECLARED_VALUE \
  --max-write-ratio PREDECLARED_VALUE
```

Evidence records exactly three optimizer steps using the actual tracking loss, finite loss, both gamma values, and gradient norms for both down projections, both cross-attention directions, and both up projections. The third-step internal gradients must be finite and nonzero. Each step's `||gamma*delta||/||search||` must lie within the predeclared interval.

### S1

Command:

```bash
python tools/igcab_search_screen_gate_v1.py s1 \
  --evidence PATH/s1_evidence.json --csv PATH/s1_per_sequence.csv \
  --p0-gate PATH/p0_gate.json --p1-gate PATH/p1_gate.json \
  --rpp-admission PATH/rpp_admission.json --output PATH/s1_gate.json \
  --max-catastrophic-losses PREDECLARED_COUNT
```

The CSV columns are `seed,sequence,arm,auc`; arms are `C0`, `Cplain`, `Cig`. It requires seeds 42/3407, all three arms, and the exact frozen 60-sequence RPP-v1 test set. The evidence must state 2 or 3 epochs, the same M1 epoch-15 parent, matching optimizer/schedule and sample order, all original trainable parameters enabled, and both parameter counts. `Cplain` and `Cig` must differ by at most 5% in added parameters. For each seed, `Cig-C0` mean AUC must be >=0.7 points, `Cig-Cplain` >=0.3, both 10% trimmed means must be positive, and catastrophic losses (<= -5 AUC) must not exceed the predeclared count.

RPP-v1 historical admission and serial/parallel consistency are prerequisites. The admission artifact must record at least four anchors, Pearson, Spearman >=0.60, direction agreement >=0.75, catastrophic-regression detection rate, rank consistency, SMSA above baseline, old CTVM negative/later-group, and passed serial/parallel consistency. The current workspace has frozen manifests but no qualifying admission artifact; S1 will remain blocked until that evidence exists. The 2-3 epoch continuation is treated only as this explicitly scoped diagnostic prescreen. Any full confirmation must train all matched arms from the same SOT initialization for the full 15 epochs; S1 may not be used as a paper comparison to M1 epoch-15.

### S2

Command:

```bash
python tools/igcab_search_screen_gate_v1.py s2 \
  --csv PATH/s2_per_sequence.csv --s1-gate PATH/s1_gate.json \
  --output PATH/s2_gate.json --min-mean-drop PREDECLARED_VALUE \
  --min-trimmed-drop PREDECLARED_VALUE
```

CSV arms are `Cig`, `Cig_GAMMA_OFF`, and `Cig_UNIFORM`, with the same seeds and sequence list. For each seed, normal Cig must exceed both perturbations by the declared mean and 10% trimmed AUC drops. This tests sensitivity to the branch; it is not causal proof by itself.

No `PREDECLARED_VALUE` should be chosen after observing the corresponding result. This gate script does not invoke the canonical baseline entry or write the protected baseline YAML/output paths.
