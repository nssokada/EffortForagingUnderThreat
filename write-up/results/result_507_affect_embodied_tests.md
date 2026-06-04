---
result_id: 507
class: metacognition_affect
title: Affect tracks raw threat and distance, not model-derived embodied survival — the embodied-affect claim does not survive direct test
status: refuted
prereg_h: []
internal_h: []
samples: [exploratory_290, confirmatory_281]
notebooks: []
scripts: [scripts/analysis/affect_embodied_tests.py]
outputs: [results/stats/affect_analysis/embodied_tests_exploratory.csv, results/stats/affect_analysis/embodied_tests_confirmatory.csv, results/stats/affect_analysis/embodied_tests_summary.csv]
figures: [TODO]
created: 2026-06-04
last_run: 2026-06-04
---

# Result 507 — Affect tracks raw threat and distance, not model-derived embodied survival

> **A direct test of the "affect as embodied readout" claim.** [[result_501]] showed that affect tracks `S(u*)` with tight effect sizes (β ≈ ±0.55–0.68). That single-predictor result is necessary but not sufficient for the embodied affect claim: `S(u*)` is a nonlinear transform of (T, D) weighted by per-subject (ω, κ), so the bare correlation could just reflect that affect tracks raw threat. Three tests here ask whether `S(u*)` carries embodied predictive content beyond raw (T, D), and the answer is **no**. A simple `T + D` model beats `S(u*)` alone by ΔAIC ≥ 57 in every test cell; controlling for (T, D), the model-derived embodied survival quantity is either null or wrong-signed; subject-specific (ω, κ) at fixed (T, D) predicts affect only weakly and channel-specifically. The "anxiety is an interoceptive readout of embodied value computation" claim — Frame C in the paper's outline — does not survive direct testing.

## Overview

The H5/501 result shows that probe-trial affect (anxiety, confidence) tracks the M4-derived survival probability `S(u*) = exp(−h · T^γ · D / speed(u*))` at the trial level, with β ≈ −0.58 (anxiety) and β ≈ +0.65 (confidence) on `S_probe_z` in mixed-effects models. This is consistent with two readings: (a) affect is an embodied readout of model-derived survival prospects under the subject's planned optimal motor output `u*` (the embodied affect claim), or (b) affect tracks raw task conditions (T, D), and `S(u*)` looks predictive because it is itself a nonlinear function of (T, D). Distinguishing these requires direct head-to-head tests. We ran three: a within-model incremental-variance test (A), a between-model AIC comparison (B), and a between-subject embodied-parameter test at fixed (T, D) (C). In all three, the embodied reading fails. `S(u*)` has either null or wrong-signed coefficient after controlling for T and D; the (T, D) model wins decisively over `S(u*)` on AIC in both samples and both channels; per-subject (ω, κ) effects on affect at fixed (T, D) are mostly null with one ω → confidence exception in exploratory only. The paper-level implication is that affect tracks raw threat and distance directly, not through the W(u)-derived survival representation; the "interoceptive readout of embodied value" framing is not warranted.

## Hypothesis

**Statement.** Subjective anxiety and confidence are real-time readouts of the body's predicted survival prospects under planned motor execution, computed via the W(u) framework's `S(u*)`. If true, three predictions follow:
- **(A)** `S(u*)` should remain a significant predictor of affect after controlling for raw threat and distance.
- **(B)** A model with `S(u*)` as a single predictor should fit affect at least as well as a model with raw `T + D` (otherwise `S(u*)` adds nothing the simpler predictors don't already capture).
- **(C)** At fixed task conditions (T, D), subject-specific (ω, κ) — which determine `S(u*)` via the body's predicted optimal action — should predict affect.

**Predicted direction (under the embodied reading):**
- Test A: β(S_probe_z) significant and in the direction that S(u\*) should drive affect (negative for anxiety, positive for confidence).
- Test B: ΔAIC favors the `S(u*)` model over the `T + D` model.
- Test C: high κ → more anxiety / less confidence (high κ → lower predicted u\* → lower S → worse survival → more anxiety); high ω → less anxiety / less confidence (ω mobilises execution → higher predicted u\* → higher S → less anxiety, but with capture-aversion still in the mix).

**Preregistered criterion.** This is an exploratory follow-up to the H5/501 result; it does not appear in the formal prereg H# list. Qualitative criterion: the embodied reading is supported if all three tests reject the null in the predicted directions; the embodied reading is undercut if any of them fails, and especially if Test B (the direct model comparison) favors the raw conditions.

**Source of the hypothesis.** The "affect as interoceptive readout of embodied value computation" claim — Frame C in the project's outline (2026-06-04 working frame). The framing positions the paper against cognitive-appraisal theories of emotion (Lazarus, Scherer) by claiming affect is computed from the body's action-survival prospects rather than from abstract decision context. This result is the direct empirical test of that claim.

## Data Source

- **Samples:** Exploratory N = 290, confirmatory N = 281.
- **Inputs:**
  - `data/{sample}_350/processed/stage5_filtered_data_*/feelings.csv` — per-probe anxiety and confidence ratings (1–10).
  - `results/stats/joint_optimal/{sample}/mcmc_m4_params.csv` — per-subject (ω, κ) from M4.
  - `results/stats/joint_optimal/{sample}/mcmc_convergence_diagnostics.csv` — M4 population (γ, h, σ_sp) posterior means.
- **Unit of analysis:** Probe trial × channel (anxiety or confidence). Per sample × channel: 5,068–5,220 probe-trial × subject observations, 281–290 subjects.

## Method

Per probe trial, compute `u* = argmax_u W(u; T, D, ω_i, κ_i, R, req, γ, h, σ_sp)` via a 40-point grid search over u ∈ [0.1, 1.5]. Then `S_probe = exp(−h · T^γ · D / speed(u*))` where `speed(u) = sigmoid((u − 0.25·req) / σ_sp)`. All using each subject's M4 (ω, κ) posterior means and population γ, h, σ_sp posterior means (exactly the convention in [[result_501]]).

Standardise all predictors per sample: `S_probe_z`, `T_z`, `D_z`, `omega_z`, `kappa_z` (log-then-z for the parameters). Fit mixed-effects models with `statsmodels.MixedLM`, ML estimation, random intercept by subject.

**Test A — Incremental variance:**

`response ~ T_z + D_z + S_probe_z + (1 | subj)`

Question: does `S_probe_z` remain significant after controlling for `T_z` and `D_z`?

**Test B — Model comparison:**

Fit two single-predictor models:
- `M_S`: `response ~ S_probe_z + (1 | subj)`
- `M_TD`: `response ~ T_z + D_z + (1 | subj)`

Compare via log-likelihood and AIC.

**Test C — Between-subject embodied content at fixed (T, D):**

`response ~ T_z + D_z + omega_z + kappa_z + (1 | subj)`

Question: do per-subject (ω, κ) predict affect after controlling for trial-level (T, D)? This tests whether the *embodied* component of `S(u*)` (the part driven by individual differences in W(u)'s parameters) carries any unique predictive content for affect.

All three tests run separately for anxiety and confidence, in both samples. MixedLM warnings about singular random-effects covariance are benign at this N (290 subjects × ≈ 18 probes per subject).

**Script:** `scripts/analysis/affect_embodied_tests.py`. Mirrors the `S_probe` computation in `scripts/analysis/affect_survival_lmm.py` exactly.

## Result

**Test A — Incremental variance (β with z, p):**

| Channel | Sample | β(T_z) | β(D_z) | β(S_probe_z) |
|---|---|---|---|---|
| Anxiety | Exploratory | +0.712 (z=8.5, p=2e-17) | +0.313 (z=5.6, p=3e-8) | **+0.161 (z=1.65, p=0.10)** |
| Anxiety | Confirmatory | +0.886 (z=11.3, p=2e-29) | +0.511 (z=9.2, p=2e-20) | **+0.441 (z=4.7, p=2e-6)** |
| Confidence | Exploratory | −0.645 (z=−7.6, p=2e-14) | −0.334 (z=−5.9, p=3e-9) | **−0.077 (z=−0.8, p=0.44)** |
| Confidence | Confirmatory | −0.657 (z=−8.1, p=4e-16) | −0.251 (z=−4.4, p=1e-5) | **+0.017 (z=+0.2, p=0.86)** |

**S_probe_z is null or wrong-signed in three of four cells.** For anxiety, the embodied prediction is β(S\*) < 0 (more survival → less anxiety), but the data show β(S\*) > 0 in both samples — wrong direction, and significantly so in confirmatory (p = 2e-6). For confidence, the embodied prediction is β(S\*) > 0 (more survival → more confidence), but the data show β(S\*) ≈ 0 in both samples. The S_probe sign-flip after controlling for T, D is a classic suppressor pattern: S_probe is dominated by its (T, D) content, and once T and D are in the model the residual S_probe carries either no signal or anti-predicted variance via collinearity.

**Test B — Model comparison (log-likelihood and AIC):**

| Channel | Sample | logL(M_S) | logL(M_TD) | AIC(M_S) | AIC(M_TD) | **ΔAIC (M_S − M_TD)** |
|---|---|---|---|---|---|---|
| Anxiety | Exploratory | −10736.0 | −10695.8 | 21480.0 | 21401.5 | **+78.5 (TD wins)** |
| Anxiety | Confirmatory | −10327.2 | −10275.4 | 20662.3 | 20560.8 | **+101.5 (TD wins)** |
| Confidence | Exploratory | −10757.0 | −10727.4 | 21522.0 | 21464.7 | **+57.3 (TD wins)** |
| Confidence | Confirmatory | −10445.3 | −10400.7 | 20898.7 | 20811.4 | **+87.3 (TD wins)** |

**The raw `T + D` model beats `S(u*)` alone by ΔAIC ≥ 57 in every case, replicating across both samples and both affect channels.** This is decisive on standard model-comparison criteria. `S(u*)` is a strictly worse predictor of affect than the two raw task variables it derives from.

**Test C — Between-subject (ω, κ) at fixed (T, D):**

| Channel | Sample | β(T_z) | β(D_z) | β(ω_z) | β(κ_z) |
|---|---|---|---|---|---|
| Anxiety | Exploratory | +0.580 (z=23.8, p<1e-100) | +0.230 (z=9.4, p=5e-21) | +0.094 (z=1.1, p=0.25) | +0.012 (z=0.15, p=0.88) |
| Anxiety | Confirmatory | +0.532 (z=21.9, p<1e-100) | +0.276 (z=11.4, p<1e-29) | −0.073 (z=−0.9, p=0.36) | +0.017 (z=0.22, p=0.83) |
| Confidence | Exploratory | −0.582 (z=−23.7, p<1e-100) | −0.295 (z=−12.0, p<1e-32) | **−0.221 (z=−2.7, p=0.007)** | −0.135 (z=−1.6, p=0.10) |
| Confidence | Confirmatory | −0.671 (z=−27.0, p<1e-160) | −0.260 (z=−10.5, p=1e-25) | −0.138 (z=−1.7, p=0.08) | −0.144 (z=−1.8, p=0.07) |

**At fixed (T, D), per-subject (ω, κ) effects on affect are mostly null.** The only consistent significant effect is ω → confidence in exploratory (β = −0.22, p = 0.007), which marginally fails to replicate in confirmatory (β = −0.14, p = 0.08). κ does not predict either channel in either sample. ω does not predict anxiety in either sample (replicating the appraisal dissociation noted in [[result_503]]).

**Verdict:** All three tests run against the embodied affect claim. The single-predictor [[result_501]] β ≈ −0.58 for anxiety on `S_probe_z` reflects the fact that `S(u*)` is itself a nonlinear function of (T, D); the smooth nonlinearity captures roughly the same variance the raw (T, D) predictors do, but no more. The embodied parameter component of `S(u*)` — the part driven by individual (ω, κ) — adds essentially no unique predictive content for affect.

## Interpretation

The result is a clean falsification of the strong embodied affect claim. Three points:

**1. Affect tracks raw task conditions, not model-derived embodied survival.** Test B settles this directly. Across four (channel × sample) cells, the (T, D) model beats `S(u*)` by ΔAIC ≥ 57. The model-derived survival quantity is a worse predictor of affect than the two task conditions it nonlinearly summarises. There is no AIC-favourable reading in which `S(u*)` is the operative quantity affect is monitoring.

**2. The result_501 finding is correct but not discriminating.** The single-predictor regression `anxiety ~ S_probe_z + (1|subj)` produces β ≈ −0.58 because `S(u*)` is a smooth (and largely accurate) summary of how (T, D) compose into survival probability. When T and D are entered directly into the model, they absorb all of this variance and leave S\* with either nothing to explain or wrong-signed residual variance. The embodied wrap (the (ω, κ)-mediated subject-specific component of `S(u*)`) adds no measurable predictive content for affect.

**3. Per-subject (ω, κ) at fixed (T, D) does not shape affect.** Test C generalises this: even when we look directly for embodied parameter effects on affect after controlling for raw conditions, the effects are mostly null. ω has a modest negative effect on confidence in exploratory only (β = −0.22, p = 0.007), which partially replicates in confirmatory (β = −0.14, p = 0.08) — consistent with the [[result_503]] appraisal dissociation that ω predicts confidence but not anxiety, but well below the magnitude the embodied affect framing predicted. κ effects are null throughout. The body's parameters that shape behaviour (208's β_ωv, β_κv on vigor; β_ωc, β_κc on choice) do not analogously shape affect at the trial level.

The substantive conclusion is that subjective affect in this task is best read as tracking the *task's* objective threat/distance conditions, not the *body's* model-derived survival prospects under planned action. If the embodied affect claim were correct, controlling for (T, D) should leave a residual signal carried by `S(u*)` — and there is no such residual. Affect appears to function more like a cognitive appraisal of the displayed task condition than like an interoceptive readout of the embodied value computation.

This narrows what the paper can honestly claim. The behavioural embodied story — ω dissociated across decision and execution, κ aligned, the channel-specific signs predicting marginal coupling — survives (cf. [[result_207]], [[result_208]], [[result_401]]). The *subjective experience* extension of that story does not. Affect monitors task conditions; it does not monitor the embodied computation that drives action.

**For the paper outline:** Frame C (affect as embodied readout) is no longer supported. The paper should be pitched around Frame B (channel-specific behavioural signatures of a shared embodied computation), with affect as a complementary description (it tracks threat coherently, but not through the model's S(u\*) representation). The translational angle (Frame A) retains the κ → apathy mapping and the ω-side metacognitive miscalibration story from [[result_506]], neither of which depends on the embodied-affect reading.

## Caveats & Limitations

- **`S(u*)` is built from M4 posterior-mean (ω, κ) per subject + population (γ, h, σ_sp).** A full propagation of posterior uncertainty in (ω, κ) into `S(u*)` would widen its effective distribution and could (in principle) recover incremental predictive value. We do not expect this to flip the result: per-subject (ω, κ) recovery is r ≈ 0.92 ([[result_205]]), so the posterior-mean approximation is reasonable.

- **Test A's sign-flip is a known consequence of collinearity, not a substantive sign reversal.** `S(u*)` correlates negatively with T (high T → low S) in both samples; T does most of the work in the joint model; the residual S\* coefficient is driven by partial-correlation arithmetic, not by a substantive "more survival → more anxiety" reading. The honest characterisation is "null incremental content with collinearity-induced sign-flip," not "more survival predicts more anxiety."

- **The result_501 finding is not invalidated by this test.** Affect *does* track S(u\*) — it just tracks it because S(u\*) is a good summary of (T, D), not because affect tracks the embodied W(u)-derived quantity per se. The 501 result remains a useful descriptive characterisation of how affect maps onto task structure; what fails is the stronger embodied interpretation.

- **Test C's null on κ → anxiety might reflect operationalisation.** κ shapes the body's effort-cost expectations; if anxiety is driven by anticipated capture rather than anticipated effort, κ wouldn't be expected to predict it. The fact that κ also doesn't predict confidence (which is more execution-relevant) is the more telling null.

- **The Test A confirmatory β(S\*) = +0.44 is significant at p = 2e-6 in the wrong direction.** This is consistent with the strong T effect saturating the model and S\* picking up suppression variance; it should not be read as substantive evidence for "more survival → more anxiety." A figure showing the marginal vs partial S\*-affect relationship would clarify this for readers if the result is shipped.

- **Affect was elicited as a *prospective* judgement on probe trials,** before pressing began. Subjects rated anticipated anxiety/confidence based on the displayed condition. This is task-design-dependent: a retrospective or within-trial affect probe might tap a different signal more reliably tied to embodied execution. The current result speaks to *prospective affect about the displayed condition*, not affect during execution.

- **The single-channel `S(u*)` representation may be wrong.** We compute `S(u*)` for the cookie presented in the probe trial. If subjects' affect is shaped by the *choice* between heavy and light (each with their own S), the right embodied quantity might be `max(S_H, S_L)` or `expected S under the subject's choice policy` rather than `S(u*)` for the forced probe option. Re-running with a choice-policy-conditioned S would be a more generous test of the embodied claim but is unlikely to flip the headline (ΔAIC of 57+ leaves substantial room before the embodied model becomes competitive).

## Replication

```bash
python scripts/analysis/affect_embodied_tests.py
```

**Expected runtime:** ~30 s on CPU (12 MixedLM fits via statsmodels).

**Expected outputs:**
- `results/stats/affect_analysis/embodied_tests_exploratory.csv` — Tests A, B, C coefficients per channel.
- `results/stats/affect_analysis/embodied_tests_confirmatory.csv` — same for confirmatory.
- `results/stats/affect_analysis/embodied_tests_summary.csv` — concatenated.

## References

**Related results:**
- [[result_501]] — Single-predictor `affect ~ S_probe_z + (1|subj)`. The result this test contextualises and partially supersedes.
- [[result_102]] — `affect ~ T_z + D_z + (1+T|subj)`. The H1b result this test shows is the operative description of affect.
- [[result_503]] — ω → confidence, not anxiety (appraisal dissociation). Test C's modest ω → confidence effect in exploratory is the only consistent signal here and aligns with this.
- [[result_502]] — Anxiety calibration → optimality. This is a *between-subject* index of how well anxiety tracks threat, not a within-trial embodied claim, and is not affected by the present result.
- [[result_207]] — Embodied joint W(u) framework — survives for behaviour; the affect extension does not.

**Literature:**
- Lazarus, R. S. (1991). Emotion and adaptation. (Cognitive-appraisal theory of emotion — the standard view that this result is consistent with.)
- Seth, A. K. (2013). Interoceptive inference, emotion, and the embodied self. (The embodied/interoceptive view that this result *does not* support, against the predictions of which it was tested.)
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. (Model comparison framework underlying Test B.)
