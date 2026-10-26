# M1 Joint Local Signal Decision Experiment (2026-10-13)

## Question

Does an explicit S1 location map plus an S2 utility decision over `{no-op, RGB, TIR}` improve M1 tracking when used for a small local post-fusion correction? Does adding S3 image-degradation/geometry inputs improve that decision beyond S1+S2?

## Parent and cohort

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15, seed 42.
- Protected checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Protected YAML SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`.
- Closed-loop stress cohort: the existing 82-sequence manifest `experiments/manifests/m1_normal20_lowauc62_union_20261013.txt`, SHA256 `817660efc89284460341f2ce52a31c3758e366e3bf73638fd20fb9beac0e531a`; normal20 + archived-M1-low-AUC62.
- This cohort has been observed in prior experiments and is an exploratory matched stress comparison, not an independent confirmation set or full-244 result.
- Signal fitting/utility labels use only the existing LasHeR train-side 72-sequence fixed-state set. It is development data previously used for signal diagnostics; M1 training exposure is unknown. Sequence split is frozen at 48 fit / 24 held-out sequences by the existing recorded assignment, yielding 100 fit / 44 held-out states. No test82 data is used to fit or tune the controller.

## Action and signal contract

M1 output is `F0 = C_M1([R,T])`. The fixed local action for modality `m` is

`F_m = C_M1([R,0])` for RGB or `C_M1([0,T])` for TIR,

`F'(p) = F0(p) + 0.05 * G_m(p) * (F_m(p) - F0(p))`,

where `G_m` is a binary 3x3 patch centered on that modality's S1 template-conditioned response peak. The operator uses the existing frozen M1 fusion convolution and CENTER head; no MHSA, trained parent weights, template update, or tracker-state rule changes.

- S1: frozen learned template-conditioned response maps for each modality; each modality's peak determines its local action support.
- S2: sequence-held-out ridge utility predictor trained on the decoded-box IoU deltas of the exact local RGB/TIR actions. It predicts `u_RGB` and `u_TIR`; `u_no-op=0` is explicit. Decision is `argmax(0,u_RGB,u_TIR)`, so non-positive predictions abstain.
- S3: crop-visible brightness, clipping, sharpness, thermal contrast, and current-state scale/aspect features appended only in the three-signal model.

## Arms

1. Matched M1 / no-op (existing run 9904; rerun only if disabled parity fails).
2. Fixed local RGB action using S1 location.
3. Fixed local TIR action using S1 location.
4. S1+S2 local utility decision with no-op.
5. S1+S2+S3 local utility decision with no-op.
6. Deterministic random-direction local action, same S1 locations and 5% magnitude, as a chance control.

All new arms use the same frozen checkpoint, current source after an exact disabled-parity check, FP32, same manifest/evaluator, one tracker process, and separate output directories. Existing M1 predictions are reused only if per-frame disabled parity is exact on the preflight sequence under the modified source.

## Primary analysis and stop rules

- Report AUC, PR, NPR, OP50, OP75, FPS, sequence-macro paired deltas, AUC win/loss/tie, and sequence-bootstrap 95% intervals for normal20, low-AUC62, and union82.
- Report action coverage, RGB/TIR/no-op frequencies, and per-sequence extremes for the two decision arms.
- The primary mechanism comparison is S1+S2+S3 versus S1+S2; the primary baseline comparison is each learned decision versus matched M1 and the strongest fixed local control.
- No threshold or model change after opening union82 results. Positive mean deltas with sequence intervals spanning zero are inconclusive.
- Stop before closed-loop evaluation if S1 map or S2 no-op/action labels fail shape, finite-value, no-op parity, or sequence-held-out contract checks. Failed/partial runs remain recorded.

## Provenance

- Pre-change source snapshot: `experiments/snapshots/m1_joint_local_signal_20261013_before/source_before.tar.gz`, SHA256 `3c4f0ec0fa7b28b21f85dd93f87c51e2eb43fd601997fa5448f3ce391f5839b9`.
- Full command, output, runtime environment, source hashes, and completed analysis will be appended to this protocol and the asset registry.
