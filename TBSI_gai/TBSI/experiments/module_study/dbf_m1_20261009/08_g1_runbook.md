# DBF G1 post-backbone screen: ready for AutoDL terminal

Status: P0 passed; three-arm training and closed-loop evaluation have **not** run.
Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15, FP32.
This G1 experiment augments M1's trained `tbsi_fuse_search` readout. It does not
replace or compare against the original paper IGF/GIE/CA/SA.

## Contract and comparison

`G1_LEGACY`: original two-search-stream concat -> trained 3x3-BN-ReLU -> head.
`G1_VANILLA`: search-only bidirectional one-softmax residuals -> same trained readout -> head.
`G1_DIFFERENTIAL`: search-only bidirectional two-softmax difference residuals -> same readout -> head.
Templates remain upstream in TBSI-E but are not read directly by the G1 bridge.
All three start from the pinned M1 checkpoint. V/D start at gamma=0.001 and
share identical initial q/k1/v/out projections. Gamma=0 gives exact legacy
fused features and boxes. The old layer-9 DBF, temporal tokens, degradation
aware fusion and Stage-2 freeze are disabled in all three.

Training: 176 frozen LasHeR training sequences; 10,752 samples/epoch, batch 16,
accumulation 8, 3 epochs = 252 planned optimizer steps. AdamW, base LR 1e-4,
backbone multiplier 0.1, G1 module LR 1e-4, no LR drop before epoch 3, AMP off,
one DataLoader worker. The global two-process ceiling makes one worker the
maximum while a trainer is active. Training uses four OMP/MKL threads in the
main process. Arms run serially and each keeps only its epoch-3 checkpoint.
Expected checkpoint size is about 1.3 GiB per arm; free space was 6.4 GiB
at preflight. The launcher checks space before every arm and refuses to reuse
an existing output directory.

Evaluation: fixed 30 LasHeR **training** sequences sampled without replacement
from the full train split after removing the 176 short-train sequences; seed
42009, manifest SHA256 `37ad5c642c246d49dbdf7d05bcaab8551bff0a0c75aaa92aa57b5570779c33f2`.
M1 previously trained on the full training split, so this is an exploratory
continuation screen, not independent generalization. The old 60 test-proxy
sequences do not enter this selection. Tracking uses FP32 and `--threads 0`
because past sharded inference changed predictions. The report includes paired
AUC differences, wins/losses, LI/PO/TO/HO/BC/TC subset means, and a paired
sequence bootstrap interval. FPS is total tracked frames divided by summed
reported frame time. Individual attributes overlap and small subsets can be
unstable; inspect their counts.

## Preflight result

`tools/preflight_dbf_g1.py` loaded the exact protected config/checkpoint
hashes, one real short-train batch, and all three models. Shared M1
backbone/head/readout weights matched the checkpoint. Head input was
`[1,768,16,16]`, final prediction `[1,1,4]`. V/D gamma-zero fused features
and boxes were bitwise equal to Legacy on the fixed token input. At gamma
0.001 both changed fused features. FP32 q/k1/v/out/gamma gradients were
finite and nonzero; D also had finite nonzero k2/lambda gradients. The G1
optimizer contained the module at LR 1e-4 and updated gamma. Each arm
completed a full forward from actual LasHeR RGB/TIR crops. Evidence:
`output/experiments/DBF_G1_P0_20261009/preflight.json`.
On that one real crop, the M1 legacy fused feature had mean `0.4214` and
RMS `0.7812`; raw RGB/TIR mean had mean `-0.0130` and RMS `0.3216`.
At gamma 0.001, V/D differed from the legacy feature by relative norms
`0.000461` / `0.000418`. These are one-batch initialization diagnostics,
not evidence that either mechanism improves tracking.

## AutoDL web-terminal sequence

From `/root/autodl-tmp/TBSI_gai/TBSI`:

```bash
bash tools/run_dbf_g1_train.sh --check-only
tmux new -s dbf_g1_train
bash tools/run_dbf_g1_train.sh
```

Detach from tmux with `Ctrl-b`, then `d`. Reattach with
`tmux attach -t dbf_g1_train`. In a second web terminal:

```bash
tail -f output/experiments/DBF_G1_LEGACY_seed42/logs/launcher.log
nvidia-smi
ps -eo pid,ppid,stat,cmd | grep -E '[r]un_training.py|[t]racking/test.py'
df -h /root/autodl-tmp
```

The active arm changes from LEGACY to VANILLA to DIFFERENTIAL. Stop a run
with `Ctrl-c` **inside its tmux pane**; do not delete a partial run directory.
The launcher records each completed arm in both asset tables.

After all three checkpoints exist, start tracking separately:

```bash
tmux new -s dbf_g1_eval
bash tools/run_dbf_g1_eval.sh
```

Monitor `output/experiments/DBF_G1_EVAL_20261009/logs/` and inspect
`output/experiments/DBF_G1_EVAL_20261009/comparison.json` after completion.
The evaluation runner also updates both asset tables. Do not start training
and evaluation together.

## Decision gate

Read D-V first, then D-L, then V-L. D must show positive paired AUC against
both V and matched L, acceptable FPS cost, and no systematic LI/occlusion/
conflict collapse before considering a standalone replacement G2. A small
single-parent, training-side gain is only a reason to verify on an untouched
evaluation set. Any claim versus the original IGF requires a faithful IGF
arm with its frozen pretrained GIE weights and matched protocol.
