# STSER content redistribution: executable protocol

Status on 2026-10-25: code and scripts prepared; no preparation, preflight,
training, tracking test, or paired analysis has been run for this pair.

Execution update: preparation and preflight completed. The first control
training was interrupted before the first batch after more than ten minutes.
Two diagnostic retries also stopped before the first batch: one worker with
OpenCV, then one worker with the existing jpeg4py fallback. No checkpoint was
produced and no candidate training or tracker test started. The phase logs
classify this as a data-loader first-batch block, not a model result. Before
restarting, isolate the problematic sampler/image path with no training
processes active and record any loader change as a new protocol version.

## Question and single variable

The direct parent is the STSER competition host. The candidate adds a
target-anchored, search-relative template Value source redistribution inside
its existing search-side message path. Native M1 template MSHA, CENTER head,
losses, optimizer family, and tracker state policy remain configured as before.
The two arms have the same architecture, SOT-initialized parameter tensor
values, data, seed, and schedule. Their only config difference is
`SEARCH_CONDITIONED_T2S_CONTENT_STRENGTH=0` versus `1`.

Training foreground weights come from the sample's template annotation;
tracking weights come from the first-frame init box. Both use patch coverage
on the 8x8 template grid. No search ground truth enters the new operator.
The object-scale excluded neighborhood estimates the first-frame box aspect
ratio from covered patches, so this is a grid approximation, not the exact
continuous box ratio.

The main claim is conditional: if the candidate gains against its matched
control, and disabling or breaking its target/location correspondence removes
that gain, then content source redistribution is supported as the mechanism.
A full LasHeR gain by itself would not prove this. A negative quick3 result
is diagnostic only and cannot veto a full15 experiment on model capacity.

## Entry points

From the repository root, use the project Python environment. Each command
invokes exactly one phase and returns when that phase ends. There is no
default `all` phase. The commands below are a future execution order, not a
record of executed work:

```bash
/root/autodl-tmp/conda_envs/tbsi/bin/python tools/run_m1_stser_content_campaign_20261025.py --scope quick3 --phase prepare
/root/autodl-tmp/conda_envs/tbsi/bin/python tools/run_m1_stser_content_campaign_20261025.py --scope quick3 --phase preflight
/root/autodl-tmp/conda_envs/tbsi/bin/python tools/run_m1_stser_content_campaign_20261025.py --scope quick3 --phase train-control
/root/autodl-tmp/conda_envs/tbsi/bin/python tools/run_m1_stser_content_campaign_20261025.py --scope quick3 --phase train-candidate
/root/autodl-tmp/conda_envs/tbsi/bin/python tools/run_m1_stser_content_campaign_20261025.py --scope quick3 --phase test-control
/root/autodl-tmp/conda_envs/tbsi/bin/python tools/run_m1_stser_content_campaign_20261025.py --scope quick3 --phase test-candidate
/root/autodl-tmp/conda_envs/tbsi/bin/python tools/run_m1_stser_content_campaign_20261025.py --scope quick3 --phase analyze
```

For the matched full experiment, repeat these phases with `--scope full15`.
The two scopes have independent common initial states, configs, outputs,
checkpoints, and locked manifests. They must never be directly compared as
though data and training progress were identical. A failed main test can be
continued with `resume-test-control` or `resume-test-candidate`; the script
checks the original partial result and unchanged explicit checkpoint.

After a completed candidate test, optional same-checkpoint interventions use
`probe-off`, `probe-roll-mask`, `probe-roll-competition`, and
`probe-scale-only`, one phase per invocation, followed by `analyze-probes`.
They are inference sensitivity checks, not separately retrained ablations.
`roll-mask` shifts the first-frame target prior by one 8x8 patch row;
`roll-competition` shifts the location-specific relative evidence by one
16x16 search row; `scale-only` replaces the source redistribution direction
with a norm-matched base message direction before output projection. The
final bounded residual norm may differ, so the logged final norm must also
be inspected before claiming an equal-amplitude comparison. The learned checkpoint and
tracker crop/state decisions remain identical at the start of each sequence;
later trajectories can diverge because tracking is closed loop.

For one-sequence same-crop attribution, use
`tools/probe_m1_stser_content_fixed_state_20261025.py` with `--scope`,
`--sequence`, and `--mode`. It reads the matched control's saved boxes and
forces each frame to start from that identical external state before both
candidate forwards. This isolates the immediate prediction difference on a
shared search crop; it is an offline diagnostic and never a deployable
tracking result. A positive `--max-frames` restricts the scope and must be
reported as such. Run it only after both main tests finish.

## Contracts and evidence

Preparation creates a common SOT-initialized state with tensor hashes and a
manifest, then locks config, source, SOT, data-list, and initial-state hashes.
It does not inherit an M1 or previous STSER optimizer/checkpoint. The
preflight checks operator shape, disabled exact parity, finite FP32/AMP
behavior, content and Value gradients, strict full-model load, and zero-gamma
identity. Both gates must complete before training. Training uses worker4
alone; the runner checks active experiments before resource phases and never
overwrites existing runs or results. Tracker phases use FP32, threads0,
explicit ep0003/ep0015 checkpoint, and template update off.
After training, the checkpoint audit stores the recorded epoch, optimizer
state count, distinct actual AdamW step values, and the checkpoint hash.

The analysis requires complete paired boxes and timings for every resolved
sequence. It reports AUC, Precision, Norm Precision, OP50/75, per-sequence
differences, bootstrap intervals, FPS, and three-frame IoU<0.2 onset windows
with predicted and ground-truth geometry. It also counts contiguous
IoU<0.2 episodes lasting at least three valid frames, their duration, and
whether a later valid frame reaches IoU>=0.5. These windows do not establish
whether the target was inside the crop; that needs separate crop traces.
Training diagnostics record the content source shift, foreground mass,
relative message direction, route probabilities, and sparse preclip gradient
norms. For control, content metrics should be absent because the branch is
disabled; this is expected, not missing evidence of a zero-valued effect.

The quick3 data contract is RPP176 train, RPP-v2 60 test, 3 epochs. Full15
uses the resolved LasHeR training/validation lists and LasHeR full test,
15 epochs. The protected M1 reference is a fixed single-seed aggregate only;
its saved per-sequence predictions are not used as a paired control. Report
candidate versus matched control as the primary causal comparison and label
any M1 aggregate comparison with its exact protocol limits. AUC greater
than 0.5 percentage point on one seed is an observation, not a robust claim.

The operator uses float64 summed-area accumulation for the excluded
neighborhood. PyTorch 1.9 AMP compatibility, runtime, memory, and backward
performance remain unverified until the preflight and actual runs execute.
If a phase fails, preserve its logs and partial artifacts; diagnose the
contract or numerical path before changing the operator or retrying.

After each completed, interrupted, or failed phase, update both
`experiments/实验资产数据库.md` and
`experiments/experiment_asset_registry.csv` with the real command, scope,
config and checkpoint hashes, environment, progress, result, and conclusion.
Do not register these prepared configurations as completed experiments.
