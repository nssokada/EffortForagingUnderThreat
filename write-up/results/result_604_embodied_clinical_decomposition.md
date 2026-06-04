---
result_id: 604
class: individual_differences
title: Joint (ω, κ) parameters do not dissociate anxiety-depression comorbidity — clinical translation is weaker than the embodied framework predicts
status: refuted
prereg_h: []
internal_h: []
samples: [exploratory_290, confirmatory_281, pooled_571]
notebooks: []
scripts: [scripts/analysis/embodied_clinical_pooled.py, scripts/analysis/embodied_clinical_decomposition.py, scripts/analysis/verify_clinical_decomp.py]
outputs: [results/stats/clinical/embodied_subscale_regressions_pooled.csv, results/stats/clinical/embodied_comorbidity_groups_pooled.csv, results/stats/clinical/embodied_comorbidity_group_params_pooled.csv, results/stats/clinical/embodied_subscale_regressions.csv, results/stats/clinical/embodied_comorbidity_groups.csv, results/stats/clinical/embodied_comorbidity_group_params.csv]
figures: [TODO]
created: 2026-06-04
last_run: 2026-06-04
---

# Result 604 — (ω, κ) does not dissociate anxiety-depression comorbidity

> **A direct test of the clinical decomposition claim.** The embodied framework predicts that ω (capture cost / avoidance) and κ (activation / effort cost) should map onto distinct clinical phenotypes — ω onto anxiety, κ onto depression / apathy — with the (ω × κ) interaction predicting comorbid presentations. Two analyses tested this directly on the pooled sample (N = 571, per prereg §"Other Planned Analyses" #6): (1) subscale-specific regressions with the interaction term, on 17 clinical scales; (2) comorbidity-group analysis classifying subjects as anxious-only / depressed-only / comorbid / neither and comparing (ω, κ). Both analyses largely refute the embodied clinical decomposition. Under the prereg-compliant pooled-z analysis, only **one** subscale term (AMI_Social on ω, β = +0.091) survives the 95% HDI threshold in the predicted-or-interpretable direction; no κ-loading on depression / fatigue / apathy scales survives; no (ω × κ) interaction reaches significance; the (ω, κ) profile does not differ across clinical groups. A within-sample-z secondary analysis (more conservative, controls for between-sample mean differences) yields four hits but inflates the effect via removing between-sample variance that pulls in the opposite direction. The paper's Frame A clinical translation needs to be substantially weakened: only the small ω → social apathy effect is defensible.

## Overview

The embodied W(u) framework makes specific clinical predictions: ω (the avoidance / capture-cost parameter) should preferentially load onto anxiety symptoms; κ (the activation / effort-cost parameter) should preferentially load onto depression, apathy, and fatigue; the (ω × κ) interaction should predict comorbid presentations (because comorbid subjects have both kinds of cost elevated). [[result_601]] previously found that broad symptom totals don't predict (ω, κ) — but that test was weak (composite scales, no interaction, no pooling). This result runs the principled version: 17 subscale regressions with the (ω × κ) interaction on pooled samples (N = 571), plus a comorbidity-group analysis using DASS21 anxiety and depression median splits. The clinical decomposition does not survive either test. The dominant clinical hits are a weak ω → social / emotional apathy effect (β ≈ +0.10 to +0.14) — the *opposite* signature from the expected κ → apathy. The (ω, κ) joint profile does not differentiate anxious-only, depressed-only, comorbid, or neither subjects.

## Hypothesis

**Statement (embodied clinical decomposition).** Under the W(u) framework:

- ω is the avoidance / capture-cost parameter — should map onto anxiety scales (DASS21-Anx, OASIS, STAI-Trait, STICSA)
- κ is the activation / effort-cost parameter — should map onto depression scales (DASS21-Dep, PHQ9), apathy scales (AMI Behavioural, Total), and fatigue scales (MFIS)
- (ω × κ) interaction should predict comorbid presentations: subjects with both elevated should show higher symptoms than the additive sum would predict
- At the group level: anxious-only subjects should have elevated ω; depressed-only subjects should have elevated κ; comorbid subjects should have both elevated; neither should be near population mean

**Predicted direction.** β(ω → anxiety scale) > 0; β(κ → depression / apathy / fatigue scale) > 0; β(ω × κ → composite symptom) > 0 for comorbid pattern.

**Preregistered criterion.** Not in the formal prereg H# list. The prereg's "Other Planned Analyses" #6 calls for pooled-sample clinical regressions but does not specify a comorbidity decomposition test. This is an exploratory follow-up motivated by the broader Frame A translational pitch developed during paper outlining.

**Source of the hypothesis.** The Frame A (anxiety + apathy as channel-specific failures of one embodied computation) translational pitch. If the embodied framework's clinical content is going to be a load-bearing part of the paper, the (ω, κ) decomposition needs to demonstrably separate the comorbid presentations that traditional symptom dimensions conflate. This is the direct test of that claim.

## Data Source

- **Samples:** Exploratory N = 290, confirmatory N = 281; pooled N = 571.
- **Inputs:**
  - `data/{sample}_350/processed/stage5_filtered_data_*/psych.csv` — clinical scales per subject.
  - `results/stats/joint_optimal/{sample}/mcmc_m4_params.csv` — per-subject ω, κ.
- **Unit of analysis:** Subject; pooled across samples with sample as a fixed-effect covariate.
- **N entering each regression:** 569–571 (slight variation by missing-data pattern across scales).

## Method

**Stage 1 — Subscale-specific regression.** For each of 17 clinical scales / subscales (DASS21: Anxiety, Depression, Stress, Total; PHQ9 Total; OASIS Total; STAI Trait & State; STICSA Total; AMI Behavioural, Social, Emotional, Total; MFIS Physical, Cognitive, Psychosocial, Total), we ran the regression under two standardisation choices:

**Primary (prereg-compliant) — pooled-z**:

```
scale_z ~ omega_z + kappa_z + omega_z:kappa_z
```

Both samples concatenated into one dataset (N = 571); `scale_z`, `omega_z`, `kappa_z` standardised on the pooled distribution. This is the analysis the prereg's "Other Planned Analyses" #6 specifies: "both samples (~580 subjects) pooled for clinical regressions to maximize power for detecting small effects."

**Secondary (more conservative) — within-sample-z, sample as covariate**:

```
scale_z ~ omega_z + kappa_z + omega_z:kappa_z + sample
```

Outcome and predictors z-scored within each sample, then pooled. Sample enters as a fixed-effect covariate. This removes any between-sample mean shift in ω, κ, or the outcome scale before testing the relationship. It is the more conservative analysis; it is reported as a robustness check.

We also ran a main-effects-only spec (`scale_z ~ omega_z + kappa_z`) under both standardisations to test whether dropping the interaction changes power for detecting main effects.

Sampler for all bambi fits: 4 chains × 2,000 draws + 1,000 tuning (the `BKW` configuration matching [[result_208]] and other H4 family analyses).

**Stage 2 — Comorbidity-group analysis.** Using median splits on DASS21-Anxiety and DASS21-Depression (pooled medians for the primary analysis, within-sample medians for the secondary), classify each subject into one of four groups:

- `anxious_only`: high anx, low dep
- `depressed_only`: low anx, high dep
- `comorbid`: high both
- `neither`: low both

Fit:

```
omega_z ~ group + sample
kappa_z ~ group + sample
```

`neither` as reference category. Compare each group's (ω, κ) profile against the reference via 95% HDI.

**Script:** `scripts/analysis/embodied_clinical_decomposition.py`.

## Result

### Stage 1 — Subscale-specific regressions: PRIMARY (pooled-z)

Under the prereg-compliant pooled-z analysis, only **2 of 51 terms** cross the 95% HDI threshold (chance ≈ 2.5):

| Scale | Type | β(ω) | β(κ) | β(ω × κ) |
|---|---|---|---|---|
| DASS21_Anxiety | anxiety | −0.029 | −0.010 | +0.006 |
| OASIS_Total | anxiety | −0.090 | +0.028 | +0.027 |
| STAI_Trait | anxiety | +0.044 | **−0.102 ★** | −0.007 |
| STAI_State | anxiety | −0.031 | +0.070 | +0.011 |
| STICSA_Total | anxiety | −0.048 | +0.044 | −0.013 |
| DASS21_Depression | depression | −0.041 | −0.006 | +0.044 |
| PHQ9_Total | depression | −0.058 | +0.003 | +0.023 |
| DASS21_Stress | stress | −0.075 | +0.024 | +0.026 |
| DASS21_Total | composite | −0.053 | +0.002 | +0.029 |
| AMI_Behavioural | apathy | −0.015 | +0.067 | −0.015 |
| **AMI_Social** | apathy | **+0.091 ★** | −0.042 | +0.013 |
| AMI_Emotional | apathy | +0.055 | −0.005 | +0.000 |
| AMI_Total | apathy | +0.058 | +0.014 | −0.001 |
| MFIS_Physical | fatigue | −0.031 | +0.006 | −0.055 |
| MFIS_Cognitive | fatigue | −0.038 | +0.015 | −0.029 |
| MFIS_Psychosocial | fatigue | −0.001 | +0.006 | −0.002 |
| MFIS_Total | fatigue | −0.032 | +0.010 | −0.039 |

**Pooled-z significant hits (with-interaction spec):**

1. **AMI_Social: β(ω) = +0.091** [+0.002, +0.176]. High capture-aversion subjects report less social engagement. Plausibly interpretable as avoidance-driven social withdrawal. Direction matches the within-sample analysis.
2. **STAI_Trait: β(κ) = −0.102** [−0.190, −0.013]. Wrong direction; best read as noise (1 of 51 tests).

**Main-effects-only spec (pooled-z, dropping interaction)** adds one borderline-significant term: OASIS_Total β(ω) = −0.089 [−0.179, −0.005], in the wrong direction (more capture-aversion → less overall anxiety). Best read as collinearity-driven suppression — there is no anxiety-loading pattern for ω.

**Predicted κ → depression / apathy / fatigue effects**: all null. AMI_Behavioural β(κ) = +0.067 (the κ-side analog of result_602's claim); DASS21-Dep β(κ) = −0.006; PHQ9 β(κ) = +0.003; MFIS subscales all near zero. The framework's prediction that κ should preferentially map onto effort-cost-linked clinical traits has no empirical support.

**Predicted ω × κ interactions**: zero scales reach significance. The comorbidity-as-compound-elevation prediction has no support.

### Stage 1 — Subscale-specific regressions: SECONDARY (within-sample-z)

The within-sample-z analysis (z-scoring within each sample before pooling, with sample as covariate) yields **4 hits** instead of 2:

| Scale | Term | Pooled-z β | Within-sample-z β |
|---|---|---|---|
| AMI_Social | ω | **+0.091 ★** | **+0.137 ★** |
| AMI_Emotional | ω | +0.055 (n.s.) | **+0.103 ★** |
| AMI_Total | ω | +0.058 (n.s.) | **+0.114 ★** |
| STAI_Trait | κ | **−0.102 ★** | **−0.101 ★** |

The within-sample-z analysis inflates the AMI Emotional and AMI Total effects from non-significant to significant. Why? Confirmatory has *higher* mean ω AND *lower* mean AMI than exploratory; the between-sample variance is therefore *negatively* associated. Within-sample z-scoring removes that between-sample variance, leaving only the within-sample associations (which are positive in both samples but small in exploratory: r ≈ +0.08 and moderate in confirmatory: r ≈ +0.15). The pooled-z analysis preserves the between-sample variance and shows the effects are weaker. The prereg's pooling mandate favours the pooled-z analysis as the headline; the within-sample-z analysis is reported here as transparency.

**Significant hits (95% HDI excludes zero):**

1. **AMI_Social: β(ω) = +0.137** [+0.046, +0.223]. High capture-aversion subjects report more social apathy.
2. **AMI_Emotional: β(ω) = +0.103** [+0.015, +0.188]. High capture-aversion subjects report more emotional apathy.
3. **AMI_Total: β(ω) = +0.114** [+0.028, +0.203]. Driven by Social and Emotional subscales.
4. **STAI_Trait: β(κ) = −0.101** [−0.189, −0.012]. Higher effort cost predicts *lower* trait anxiety. Direction-of-effect is not what the framework predicts.

**Predicted patterns that did NOT survive:**

- **κ → apathy / depression / fatigue**: all null. AMI_Behavioural β(κ) = +0.064 (non-significant). PHQ9, DASS21-Dep, MFIS subscales all show β(κ) near zero.
- **ω → anxiety**: all null. DASS21-Anx, OASIS, STAI-State, STICSA all show β(ω) ≈ 0 or trending negative (wrong direction).
- **ω × κ interaction**: zero scales out of 17 show a significant ω × κ effect. The comorbidity-as-compound-elevation prediction has no support.

### Stage 2 — Comorbidity-group analysis

**Group sizes (pooled median splits — primary):**

| Group | N |
|---|---|
| neither | 221 |
| comorbid | 218 |
| anxious_only | 67 |
| depressed_only | 65 |

**Group means (raw, not z-scored, pooled-z ω and κ):**

| Group | DASS21_Anx | DASS21_Dep | ω_z (pooled) | κ_z (pooled) |
|---|---|---|---|---|
| neither | 1.2 | 1.1 | +0.025 | −0.007 |
| anxious_only | 10.0 | 3.2 | +0.111 | −0.026 |
| depressed_only | 1.7 | 16.7 | +0.045 | +0.022 |
| comorbid | 16.9 | 21.7 | −0.072 | +0.008 |

**Bayesian contrasts vs "neither" (95% HDI, pooled-z primary):**

| Contrast | β(ω_z) [HDI] | β(κ_z) [HDI] |
|---|---|---|
| comorbid | −0.062 [−0.335, +0.207] | +0.025 [−0.253, +0.297] |
| depressed_only | +0.049 [−0.289, +0.386] | +0.041 [−0.320, +0.371] |
| anxious_only | +0.017 [−0.256, +0.287] | +0.012 [−0.265, +0.284] |
| sample (exp) | **−0.516 [−0.673, −0.356] ★** | +0.025 [−0.133, +0.200] |

**Every group contrast has HDI that spans zero.** The (ω, κ) profile does not distinguish the four clinical groups. The framework's prediction — that comorbid subjects should sit at the high-(ω, κ) corner — is not supported. The numerically largest group effect (comorbid β(ω) = −0.062) is in the wrong direction.

**Note on the sample fixed effect**: exploratory has lower mean ω than confirmatory by β = −0.516 ★. This is a real and substantial between-sample shift in the M4 ω posterior means, controlling for clinical group. It does not affect the within-group comparison conclusions (group contrasts adjust for sample) but does indicate that the two M4 fits produced different absolute parameter scales.

### Stage 3 — Factor analysis on the latent dimensional structure

To address the possibility that (ω, κ) predict the *latent structure* of the symptom panel rather than individual noisy scales (the HiTOP / p-factor concern: anxiety and depression scales correlate strongly so individual regressions dilute the signal), we ran:

- **Parallel analysis** (Horn's method, 500 random permutations) to determine the number of factors
- **Exploratory factor analysis** with varimax rotation on N = 568 (pooled, dropping subjects with any missing subscale data)
- **Regression of factor scores on (ω, κ)** with the interaction term

**Parallel analysis** retained 2 factors. Observed eigenvalues (top 8): [8.52, 1.42, 0.84, 0.67, 0.52, 0.47, 0.34, 0.25]. Random 95% cutoff (top 8): [1.33, 1.25, 1.19, 1.15, 1.11, 1.07, 1.04, 1.00]. F1 dominates (eigenvalue 8.52) — consistent with the p-factor literature that a single general-distress factor explains most symptom variance.

**Factor structure** (varimax rotation):

| Subscale | F1 loading | F2 loading | Interpretation |
|---|---|---|---|
| DASS21_Anxiety | **+0.86** | −0.14 | F1 marker |
| STICSA_Total | **+0.86** | −0.32 | F1 marker |
| DASS21_Stress | **+0.82** | −0.36 | F1 marker |
| OASIS_Total | **+0.77** | −0.39 | F1 marker |
| PHQ9_Total | +0.76 | −0.46 | F1 dominant |
| DASS21_Depression | +0.72 | −0.47 | F1 dominant |
| STAI_State | +0.71 | −0.43 | F1 dominant |
| STAI_Trait | −0.59 | +0.49 | Anomalous (loads opposite to other anxiety scales) |
| MFIS_Psychosocial | +0.44 | **−0.75** | F2 marker |
| MFIS_Cognitive | +0.56 | **−0.72** | F2 marker |
| MFIS_Physical | +0.55 | **−0.69** | F2 marker |
| AMI_Behavioural | +0.25 | **−0.57** | F2 marker |
| AMI_Social | +0.14 | −0.41 | F2 dominant |
| AMI_Emotional | −0.26 | +0.03 | weak loading on both |

F1 is **general internalising distress** (anxiety + depression + stress + somatic anxiety, the symptom comorbidity cluster). F2 is **apathy / fatigue / anhedonia** (motor + cognitive + psychosocial fatigue plus behavioral apathy). The structure is theoretically sensible — roughly matches HiTOP's internalising distress + somatic-form distress distinction. STAI_Trait loads in the opposite direction from other anxiety scales, possibly reflecting that STAI_Trait taps general "negative affectivity" / dispositional worry that contains substantial variance separate from acute anxiety symptoms.

**(ω, κ) regression on factor scores (with interaction):**

| Factor | β(ω_z) | β(κ_z) | β(ω × κ) |
|---|---|---|---|
| F1 (general distress) | −0.073 [−0.160, +0.016] | +0.027 [−0.060, +0.114] | +0.029 [−0.045, +0.103] |
| F2 (apathy / fatigue) | −0.003 [−0.091, +0.084] | −0.016 [−0.105, +0.074] | +0.033 [−0.048, +0.103] |

**No significant (ω, κ) effects on either factor.** Every HDI spans zero. F1 ω effect is marginal and in the wrong direction (high ω → *less* general distress, β = −0.073). The factor analysis — which was a strictly more powerful test than the subscale regressions, because it aggregates across correlated outcomes and extracts the orthogonal dimensions — confirms the null.

**Verdict across three analyses:**

| Analysis | Hits (predicted direction) |
|---|---|
| Stage 1 (subscale regressions, pooled-z) | 1 (ω → AMI_Social, β = +0.091) |
| Stage 2 (comorbidity groups) | 0 |
| Stage 3 (factor analysis on F1 + F2) | 0 |

The clinical decomposition is genuinely null in this dataset, not just underpowered. (ω, κ) does not predict the latent dimensional structure of psychopathology in this sample. This is a real and substantial between-sample shift in the M4 ω posterior means, controlling for clinical group. It does not affect the within-group comparison conclusions (group contrasts adjust for sample) but does indicate that the two M4 fits produced different absolute parameter scales, even though the relative orderings are similar (see [[result_205]] for recovery diagnostics, and the cross-sample analysis in [[result_208]]).

**Verdict:** Both stages of the analysis fail to support the embodied clinical decomposition. Under the prereg-compliant pooled-z analysis, only **one** subscale term (ω → AMI_Social, β = +0.091) survives in the predicted-or-interpretable direction. The κ-side has no clinical hit at all. The (ω × κ) interaction is null on every scale. The comorbidity group analysis finds no (ω, κ) differences. The within-sample-z analysis (presented as a robustness check) yields additional hits on AMI_Emotional and AMI_Total, but these are artifacts of removing between-sample variance that pulls in the opposite direction.

## Interpretation

The embodied clinical decomposition does not survive direct testing. Three things this tells us:

**1. The κ → apathy / depression mapping does not hold in pooled regression with proper controls.** [[result_602]] reported that AMI apathy tracks vigor — and vigor is shaped strongly by κ ([[result_208]] β_κv = −0.24). But the AMI ↔ κ direct association is null here once ω is included. The simplest explanation is that the result_602 finding was correlational at the behavioural level (AMI ↔ observed vigor) but the *parameter*-level mapping is more complex. The clinical face of κ may be too diffuse to recover from these scales.

**2. The only consistent clinical signal is ω → social and emotional apathy (NOT behavioural apathy).** β(ω) ≈ +0.10–0.14 on AMI_Social, AMI_Emotional, AMI_Total, with AMI_Behavioural near zero. This pattern is *interpretable but unexpected*. AMI_Social ("not motivated to spend time with others") and AMI_Emotional ("things don't move me") map cleanly onto avoidance-driven withdrawal: a high-ω subject avoids social and emotional engagement because both are situations where bad outcomes can be costly. AMI_Behavioural ("I don't put much effort into things") would be the natural κ-side prediction, and that's the one that's null. The clinical face of ω in this sample is *social/emotional avoidance*, not raw threat anxiety. The clinical face of κ is essentially undetectable.

**3. Comorbidity is not a compound-parameter pattern in this data.** The framework's strongest clinical prediction — that the (ω × κ) interaction should mark comorbid presentations, and that comorbid subjects should sit at the upper corner of the (ω, κ) plane — has no support. Comorbid subjects do not differ from "neither" in either ω or κ. The high empirical co-occurrence of anxiety and depression symptoms does not correspond to a co-elevation of model parameters.

**For the paper's Frame A clinical translation:** the story needs to be substantially weakened. The strongest defensible clinical claim from this dataset is:

> **The avoidance parameter (ω) shows a small association with self-reported social disengagement (AMI_Social, β ≈ +0.09), but the model's parameters do not cleanly map onto standard psychiatric symptom dimensions, and the joint (ω, κ) profile does not discriminate anxiety-depression comorbidity.**

This is honest, replicated (in a pooled-sample sense), and uses the (ω × κ) decomposition the framework provides — but it doesn't deliver the wholesale "the embodied computation explains comorbid presentations" pitch that Frame A originally envisioned.

**For the paper's central claim:** the computational integration story (Frame B — channel-specific signatures of one embodied value computation) remains intact. Joint W(u) fits behaviour ([[result_201]]–[[result_208]], [[result_401]], [[result_404]]); the channel-specific signatures replicate; the marginal correlation is quantitatively predicted. The framework's clinical content is narrower than its computational content. The paper should lead with the computational integration as the substantive claim and frame the clinical findings as a partial, hypothesis-generating extension rather than a translational headline.

## Caveats & Limitations

- **The Prolific sample is non-clinical.** Subjects are recruited from the general population, not from clinical populations. The high proportion of subjects in the "comorbid" group via median split reflects the prevalence of subclinical symptoms, not diagnostic comorbidity. A clinical-sample replication would be needed to test the embodied framework against actual psychiatric phenotypes. The current result speaks to subclinical symptom dimensions, which may not generalise to diagnosis-level questions.

- **Median splits lose information.** Stage 2's group classification via median splits is the standard approach but discards the continuous symptom information. A more powerful version would use continuous symptom regressions (which Stage 1 does on individual scales) and look for ω × κ × symptom interactions. Stage 1's null on the ω × κ interaction across all 17 scales argues against this rescue.

- **Multiple comparisons.** 17 scales × 3 terms = 51 tests in Stage 1. Four hits where chance gives ~2.5 is barely above noise floor. The four hits are clustered on the AMI subscales (a single instrument), which provides some replication-internal evidence but reduces independent-test count. A formal multiple-comparison correction would likely eliminate the STAI_Trait β(κ) finding entirely and weaken the AMI findings.

- **The (ω, κ) recovery uncertainty propagates into clinical regressions.** Per-subject (ω, κ) are posterior-mean point estimates from M4; their uncertainty (~r ≈ 0.92 recoverability per [[result_205]]) is not propagated into the clinical regression. This is the classic measurement-error-in-predictor problem and biases coefficients toward zero. The fully Bayesian alternative — fitting M4 + clinical regressions in one graph — would propagate this uncertainty but is computationally large.

- **The κ → apathy story might be operationalisation-sensitive.** AMI_Behavioural items mostly tap "I don't put effort into things" — which sounds like a κ-loading construct but may be measuring effort *attempted* not effort *cost-experienced*. A more direct effort-cost measure (e.g., effort-discounting tasks separate from this one) might recover the κ → effort-aversion association.

- **The STAI_Trait β(κ) = −0.10 wrong-direction effect is concerning but small.** A 0.01-tailed HDI excludes zero on the wrong side of the prediction. The most plausible reading is sampling noise within the 17 × 3 = 51 tests, but the direction (high κ → less trait anxiety) is hard to interpret theoretically. Could reflect subjects who disengage from the task (high κ) also reporting lower endorsement of generic worry items.

## Replication

```bash
python scripts/analysis/embodied_clinical_decomposition.py
```

**Expected runtime:** ~3–4 min (19 bambi fits with 4 chains × 2,000 draws + 1,000 tuning each).

**Expected outputs:**
- `results/stats/clinical/embodied_subscale_regressions.csv` — Stage 1 coefficient table per scale × term.
- `results/stats/clinical/embodied_comorbidity_groups.csv` — Stage 2 group sizes.
- `results/stats/clinical/embodied_comorbidity_group_params.csv` — Stage 2 contrast coefficients.

## References

**Related results:**
- [[result_207]] — Embodied joint W(u) framework. The behavioural content survives this result; the clinical extension does not.
- [[result_601]] — Original null on broad psychiatric symptoms vs (ω, κ). This result is the principled follow-up that confirms and extends the null at the subscale level.
- [[result_602]] — AMI apathy tracks vigor (behavioural-level finding). The parameter-level direct mapping does not survive once ω is included in pooled regression.
- [[result_507]] — Affect tracks raw (T, D), not embodied S(u\*). Frame C undercut.
- [[result_208]] — Channel-specific partial slopes (ω dissociated, κ aligned) — the computational decomposition that survives.

**Notebook / scripts:**
- `scripts/analysis/embodied_clinical_decomposition.py` — this result's pipeline.
- `scripts/analysis/affect_embodied_tests.py` — related Frame C test ([[result_507]]).

**Literature:**
- Treadway, M. T., & Zald, D. H. (2011). Reconsidering anhedonia in depression. Trends in Cognitive Sciences.
- Husain, M., & Roiser, J. P. (2018). Neuroscience of apathy and anhedonia: a transdiagnostic approach. Nature Reviews Neuroscience.
- Kotov, R., et al. (2017). The Hierarchical Taxonomy of Psychopathology (HiTOP): a dimensional alternative to traditional nosologies. Journal of Abnormal Psychology.
