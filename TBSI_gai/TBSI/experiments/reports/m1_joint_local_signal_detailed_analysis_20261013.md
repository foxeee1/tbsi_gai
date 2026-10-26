# M1 joint local signal: detailed comparison and attribution (2026-10-13)

## 1. Estimand

This experiment estimates a **fixed-state, one-step decoded-IoU utility**. It does not estimate closed-loop AUC, recovery, or long-term tracking.

For state (i), let (F_0) be the protected M1 fused search feature and let (F_m) be the same frozen M1 fusion convolution fed with only modality (m\in\{R,T\}). The local intervention is

\[
F_{i,m}'(p)=F_{i,0}(p)+\rho G_{i,m}(p)\,[F_{i,m}(p)-F_{i,0}(p)],\qquad \rho=0.05,
\]

where (G_{i,m}) is a binary (3\times3) support centered at the frozen S1 response peak for modality (m). The CENTER head and decode are unchanged.

Let (b_i) be the decoded IoU from (F_{i,0}), and (q_{i,m}) the decoded IoU after the local action. The observed action utility is

\[
u_{i,m}=q_{i,m}-b_i,\qquad u_{i,0}=0.
\]

The oracle upper bound used here is the hindsight action selector

\[
u_i^*=\max(0,u_{i,R},u_{i,T}).
\]

It is not deployable and is only a ceiling for the three-action decision problem.

## 2. Data and separation

- 72 train-side sequences, 144 fixed states.
- Sequence split fixed before fitting: 48 fit sequences / 100 states and 24 held-out sequences / 44 states.
- The held-out unit is the sequence, so the primary macro estimate is

\[
\hat U_{macro}=\frac1{|S|}\sum_{s\in S}\frac1{|I_s|}\sum_{i\in I_s}u_{i,a(i)}.
\]

- No test-82 state was used to fit the utility model. The 82-sequence closed-loop comparison was not run after this gate failed.

The source cohort was previously used in signal diagnostics and M1 pretraining exposure is unknown. Therefore the result is a development diagnostic, not an independent generalization claim.

## 3. Held-out comparison

| Decision rule | No-op | RGB | TIR | Mean selected utility | Sequence-macro utility | Oracle fraction |
|---|---:|---:|---:|---:|---:|---:|
| S1+S2 | 26/44 (59.09%) | 13/44 (29.55%) | 5/44 (11.36%) | +0.000013 | +0.000121 | 11.20% |
| S1+S2+S3 | 25/44 (56.82%) | 10/44 (22.73%) | 9/44 (20.45%) | -0.000095 | +0.000018 | 1.62% |

The oracle fraction is (\hat U/\hat U^*\), using state-level mean utility. The S1+S2 rule captures only about 11% of the hindsight ceiling; the three-signal rule captures about 1.6% and is negative at state level.

For the same held-out states:

| Fixed action | Mean utility | Interpretation |
|---|---:|---|
| RGB | +0.000293 | Strongest fixed action in this cohort |
| TIR | -0.000761 | Harmful on average |
| S1+S2 choice | +0.000013 | 0.000280 below fixed RGB; 0.000774 above fixed TIR |
| S1+S2+S3 choice | -0.000095 | 0.000388 below fixed RGB; 0.000666 above fixed TIR |

The correct comparison is therefore not “learned selector versus zero”; it is “learned selector versus the best fixed action available under the same frozen local operator.” On this held-out cohort both learned selectors lose to fixed RGB.

## 4. Signal-to-utility noise

The ridge model is fit as

\[
\hat W=(X_c^\top X_c+\lambda I)^{-1}X_c^\top(Y-\bar Y),
\qquad
\hat Y=\bar Y+X_c\hat W,
\]

with standardized and clipped features (X_c), (\lambda=10), and (Y_i=[u_{i,R},u_{i,T}]). The no-op is not regressed; it is an explicit zero action in the argmax:

\[
a_i=\arg\max_{a\in\{0,R,T\}}(0,\hat u_{i,R},\hat u_{i,T}).
\]

Held-out utility scale and model residual scale are:

| Model | RMSE RGB | RMSE TIR | Held-out fixed RGB mean | Held-out fixed TIR mean |
|---|---:|---:|---:|---:|
| S1+S2 | 0.005364 | 0.005696 | +0.000293 | -0.000761 |
| S1+S2+S3 | 0.004991 | 0.005097 | +0.000293 | -0.000761 |

The residual RMSE is approximately 7.7--21.8 times the absolute fixed-action mean utility, depending on arm and model. Thus the model can reduce regression error slightly after adding S3 while still making worse discrete decisions. Lower continuous MSE is not equivalent to better action selection when the decision boundary is the zero utility threshold.

## 5. Clean attribution

### Supported attribution

1. **The local operator is not admitted as a useful intervention under this contract.** Fixed RGB is positive, but neither learned selector beats it; this is a statement about the complete frozen local-action-plus-signal contract.
2. **The selector fails to identify the better modality reliably.** S1+S2 selects TIR on 5/44 held-out states despite fixed TIR being negative on average; S1+S2+S3 increases TIR selections to 9/44 and lowers state-level utility.
3. **The available S1/S2/S3 features have insufficient predictive signal at the action boundary.** Utility magnitudes are around (10^{-4}) to (10^{-3}), while held-out model RMSE is around (5\times10^{-3}).
4. **S3 has no demonstrated incremental value.** Its addition changes the action mix and worsens mean selected utility; it does not establish that image/geometry features are intrinsically useless outside this contract.

### Not supported

- Not supported: “the adaptive router architecture is invalid.” The router was not trained with these utility labels and was not evaluated here.
- Not supported: “RGB is universally better than TIR.” The fixed RGB result is cohort-specific and fixed-state.
- Not supported: “post-fusion residuals cannot improve tracking.” The tested residual is local, fixed-magnitude, one-step, and uses an unvalidated predictor.
- Not supported: “S1, S2, or S3 is causally irrelevant.” The experiment changes the complete decision rule; it does not isolate each feature with a matched nested causal intervention.
- Not supported: any closed-loop AUC, FPS, recovery, or full-244 claim.

## 6. Failure decomposition

The observed negative result can be decomposed into three non-exclusive terms:

\[
U_{sel}-U_{RGB}
=\underbrace{(U_{oracle}-U_{RGB})}_{\text{available action headroom}}
-\underbrace{(U_{oracle}-U_{sel})}_{\text{selection regret}}.
\]

For S1+S2, (U_{sel}=0.000013), (U_{RGB}=0.000293), and (U_{oracle}=0.001083). The selector leaves most oracle headroom unused and is below the fixed RGB policy. This identifies **selection regret / weak utility prediction** as the immediate bottleneck. It does not distinguish whether the weakness comes from S1 localization, S2 feature construction, ridge capacity, label noise, or the local action itself.

The S1+S2+S3 comparison is a nested feature test under the same model family and labels. Its state-level change is

\[
\Delta_{S3}=U_{S1+S2+S3}-U_{S1+S2}=-0.000108.
\]

This is evidence against adding S3 in this exact contract, not evidence against all geometry or degradation features.

## 7. Decision

The clean decision is **SIGNAL-NOT-ADMITTED**. Do not train the adaptive residual and do not run the preplanned 82-sequence closed-loop evaluation from these labels. A valid continuation would first need an independent outcome-blind train-side cohort and a predeclared action-class evaluation reporting balanced action accuracy, utility regret, abstention precision, and sequence-bootstrap uncertainty. Until that exists, the remaining explanation is unresolved between weak deployment-visible signal, noisy utility labels, and an unsuitable fixed local action.

Evidence: `output/experiments/M1_JOINT_LOCAL_SIGNAL_DECISION_20261013/model_artifacts_retry2/summary.json`, `local_action_utility.csv`, and `local_utility_models.json`.
