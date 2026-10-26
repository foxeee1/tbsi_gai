# M1 Local Cross-Modal Correspondence Exchange: Design Decision

## Decision

Proceed to implementation planning for one new interaction family: local, bidirectional cross-modal correspondence on the early 16x16 search feature maps, transporting feature values into the receiving modality stream. Do not extend the recent template-search relation-bias branch, the sparse template-token bridge, or another scalar modality gate.

This is a hypothesis worth one matched quick-training experiment, not a conclusion that spatial misalignment is the M1 root cause. The observed M1 failure chain remains crop-internal center-selection error followed by feedback drift and weak recovery.

## Why this is a distinct test

The relevant paths divide into three different operations:

| Path | Changed object | Current evidence | How this proposal differs |
|---|---|---|---|
| Early-conditioned MSHA gate | Scale template cross-modal Value messages | Full-244 point estimate is modest and heterogeneous; paired interval crosses zero | This proposal does not predict a global/tokenwise gain on the template Value path |
| Early search relation-bias | Add template-search relation values to cross-modal attention logits | 12-sequence quick screen regressed and had an extreme sequence failure | This proposal changes search token representations by moving projected feature values across modalities |
| Spatial token bridge | Correspondence-weighted template feature residual | Learnability and amplitude screens changed message magnitude without meaningful rank/replay change | This proposal operates on paired search maps, not template-token reconstruction |
| Post-TBSI cross-modal residual / N-gram fusion | Global cross-attention or local context residual at later search/fusion boundary | Prior evidence is mixed or negative; N-gram residual was weakly selective and did not improve GT rank | This proposal uses explicit 3x3 spatial matching at the earliest paired search representation and is target-conditioned from template features |

The mechanism is still adjacent to established cross-attention and RGB-T feature exchange. Novelty cannot be claimed from this distinction alone; literature and implementation comparison is required before paper framing.

## Experiment contract

The equation, tensor shapes, insertion contract, parent controls, quick-train protocol, metrics, stop/go rules, and risks are preregistered in [preregistration.md](/root/autodl-tmp/TBSI_gai/TBSI/experiments/protocols/m1_local_cross_modal_correspondence_20261024/preregistration.md).

Implementation should touch only an independent new layer, a default-off config entry, and the feature-gated call path in `lib/models/tbsi_track/vit_tbsi_care.py`. If token splitting or data plumbing needs adjustment, isolate it in the new branch. The existing feature gate and actor already provide template foreground weights via `template_box_to_token_weights`; no search annotation should be passed to the model. Before modifying these shared files, save an immutable source snapshot because the current worktree contains extensive unrelated/uncommitted edits.

The insertion point must be confirmed against the selected candidate configuration: earliest location where both 16x16 search streams exist, before the first TBSI-E template-fusion call. The default-off path must leave protected M1 outputs exactly unchanged.

## Recommended next action

First inventory the existing broad quick-training group and select its exact matched parent config/checkpoint, optimizer/schedule, trainable-parameter policy, tracking manifest, and evaluator. Then snapshot the implementation sources and build this candidate as an independent feature-gated branch. Do not launch training until the full matched pair and process budget are recorded. No experiment run, model edit, or asset-registry row was created in this design step.
