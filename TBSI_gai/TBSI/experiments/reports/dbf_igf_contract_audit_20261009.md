# DBF Post-Backbone Contract Audit (2026-10-09)

## Scope

Static review of the attached interpretation, the four short-run configs,
model routing, optimizer grouping, and existing logs. No training, tracking,
checkpoint loading, or code changes were performed for this audit.

## Findings

### 1. Confirmed output-contract discontinuity

`legacy` returns `tbsi_fuse_search(cat([rgb_search, tir_search]))`; the helper
is a default 3x3 Conv2d + BatchNorm2d + ReLU. `mean` instead returns the raw
arithmetic mean, and both DBF modes return a mean of two residual-updated
streams. All have shape `[B,C,H,W]`, but the latter routes bypass the learned
M1 fusion module and have different normalization, activation, and spatial
filtering. This is the strongest established confound behind Legacy versus
the replacement arms; shape equality does not establish head-input parity.

### 2. Confirmed lower learning rate for the new module

The short configs set `SOT_PRETRAIN=True`, `LR=1e-4`,
`BACKBONE_MULTIPLIER=0.1`, `LR_DROP_EPOCH=2`, and leave
`STAGE2_BASELINE` empty. In the active optimizer branch, names containing
`tbsi` receive the optimizer default LR (`1e-4`), while other trainable names
receive `LR * BACKBONE_MULTIPLIER` (`1e-5`). Thus the Legacy
`tbsi_fuse_search` gets 10x the LR of the new `post_fusion.*` parameters. The
StepLR then decays both groups by another 10x after epoch 2. Optimizer
membership was verified, but that assertion does not check the LR group.
Three epochs therefore give the newly initialized bridge a short, lower-LR
adaptation window. This makes the clean comparison insufficient to decide
whether the Differential mechanism itself is unsuitable.

### 3. Template-context DBF is not paper IGF input semantics

The replacement receives `[rgb_template, rgb_search, tir_template,
tir_search]`; each search stream attends over the other modality's concatenated
template and search tokens. There is no explicit template/search mask or new
segment embedding. Original token positional information remains in the
features, so this is a risk hypothesis about context use, not evidence that
attention cannot distinguish positions. The module also does not receive the
RGB search frame or implement the original paper's GIE/CA/SA path. It is
accurately described as a template-context replacement at M1's post-backbone
fusion slot, not a faithful original-IGF reproduction. Search-only DBF would
be a different experimental contract.

### 4. Training state: some claims can be narrowed

- `TEMPORAL_TOKENS=False`, `DEGRADATION_AWARE=False`, layer-9 `DBF_ENABLE=False`,
  and the Stage-2 baseline is empty. The suspected temporal-token slice error
  and Stage-2 freeze whitelist do not explain this run.
- The token layout assertion expects 640 tokens. The slices used by the
  replacement select RGB template `0:64`, RGB search `64:320`, TIR template
  `320:384`, and TIR search `384:640`; there is no evident indexing error.
- Startup logs show M1 checkpoint loading with no unexpected keys; for
  Vanilla/Differential, the missing keys are only the newly initialized
  `post_fusion.*` entries. Mean and Legacy have no missing keys.
- Differential/Vanilla gamma initialization is `0.001`; gamma zero applies
  only to the Mean config, which has no bridge parameters. The claim that the
  tested Differential started at gamma zero is false. Existing logs do not
  contain gamma/lambda trajectories or `||gamma*delta|| / ||mean||`, so their
  learned magnitudes cannot be established now.
- `gradient_flow.csv` logs aggregate parameter-tensor counts and group norms,
  not per-parameter values. Its `zero_grad_count` counts `p.grad is None`, not
  numerically zero gradients. Differential has 6 such tensors versus 12 for
  Vanilla at sampled steps; the six-count difference is consistent with
  Differential's two `k2` weight/bias pairs and two `lambda_raw` parameters
  participating in autograd. This establishes graph connectivity, not useful
  gradient magnitude or learned behavior.

## Interpretation Boundary

The proxy result establishes that this trained, template-context,
lower-LR implementation underperformed Vanilla and Legacy in this one-seed
60-sequence screen. It does not isolate a general failure of differential
attention. The likely confounds are, in order: (1) bypassing M1's learned
3x3-BN-ReLU fusion and changing the head input distribution; (2) the new
module's 10x smaller optimizer LR plus three-epoch schedule; (3) template
context differing from paper IGF semantics. Signed attention, shared V, and
fixed bidirectional averaging remain design risks but are not isolated causes
by the current evidence.

The attached note calls the learned fusion a 1x1 convolution; the actual
default helper uses a 3x3 convolution. Its warning about distribution shift is
valid, but the operator description should be corrected. Its gamma-zero
warning is conditional and does not describe the tested Vanilla/Differential
configs.

## Source Evidence

- Fusion routing: `lib/models/tbsi_track/tbsi_track.py` lines 384-435.
- Model initialization/load and Stage-2 gate: same file, lines 621-684.
- `conv` defaults: `lib/models/layers/head.py` lines 8-23.
- Optimizer groups and membership assertion:
  `lib/train/base_functions.py` lines 248-331.
- StepLR: same file, lines 336-337; epoch stepping is in
  `lib/train/trainers/base_trainer.py` lines 87-91.
- DBF template/search context: `lib/models/layers/dbf_igf_replacement.py`
  lines 25-41.
- Differential formula and gamma: `lib/models/layers/differential_bridge.py`
  lines 22-70.
- Aggregated gradient diagnostic semantics:
  `lib/train/trainers/diagnostics.py` lines 91-124.
- Run artifacts: `output/experiments/DBF_IGF_REPLACE_SHORT_{LEGACY,MEAN,VANILLA,DIFFERENTIAL}_seed42/logs/`.

## Next Diagnostic Design (Not Run)

Before attributing the score gap to Differential, make the optimizer group
explicit for `post_fusion.*` and inspect the LR contract; collect per-direction
gamma, lambda, delta norm, and residual-to-mean ratio; and preserve a matched
head-input distribution control. Treat search-only versus template-context
as a separately named factor. Any follow-up should use a new experiment ID and
must retain the current no-go proxy result as prior evidence.
