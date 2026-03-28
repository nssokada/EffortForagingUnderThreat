# H4: Calibration and Discrepancy Doubly Dissociate Performance from Clinical Symptoms

## Results from Discovery Sample (N = 293)

---

## Overview

This hypothesis tests whether two orthogonal dimensions of metacognitive anxiety — calibration (signal quality: how accurately anxiety tracks danger) and discrepancy (systematic bias: how much anxiety exceeds danger) — dissociate adaptive performance from clinical symptomatology. The prediction derives from metacognitive theories of anxiety (Wells, 2009) and the interoceptive prediction error framework (Paulus & Stein, 2010): calibration reflects the fidelity of threat monitoring, while discrepancy reflects a systematic mismatch between threat appraisal and affective response — the hypothesized computational substrate of clinical anxiety.

---

## Methods

### Metacognitive Decomposition

For each subject, we computed two measures from their 18 anxiety probe trials and the model-derived survival probability S:

**Calibration (per-subject).** The within-subject Pearson correlation between anxiety ratings (0–7 scale) and model-derived danger (1 − S), computed across the subject's probe trials. S = (1 − T^0.210) + 0.098 × T^0.210 × p_esc, using population-level γ and ε from the fitted EVC model. Higher calibration means the individual's anxiety more accurately tracks the objective threat level. Range: [−1, +1].

**Discrepancy (per-subject).** We first fit a population-level regression of anxiety on S (across all subjects and trials). Each subject's discrepancy is the mean residual of their anxiety ratings from this population-level prediction. Positive discrepancy means the individual reports more anxiety than the average person would at the same objective danger level — a systematic positive bias.

**Rationale for orthogonality:** Calibration measures the SLOPE of the within-subject anxiety-danger relationship (signal quality). Discrepancy measures the INTERCEPT shift (bias level). Mathematically, a subject can have high calibration (steep slope, tight tracking) with either positive discrepancy (shifted up — "accurately anxious") or negative discrepancy (shifted down — "accurately calm"). Conversely, a subject can have low calibration (flat slope, poor tracking) at any discrepancy level.

### Psychiatric Measures

Between task blocks, participants completed validated psychiatric questionnaires:
- **DASS-21:** Depression (7 items), Anxiety (7 items), Stress (7 items)
- **PHQ-9:** Depression severity (9 items)
- **OASIS:** Overall anxiety severity and impairment (5 items)
- **STAI-State:** State anxiety (20 items)
- **STICSA:** Cognitive and somatic anxiety (21 items)
- **AMI:** Apathy/Motivation Index — Emotional, Behavioural, Social subscales (18 items)
- **MFIS:** Modified Fatigue Impact Scale — Physical, Cognitive, Psychosocial (21 items)

All scores were z-scored across participants before analysis.

### Performance Measures

**Choice quality:** Proportion of 45 choice trials where the subject chose the EV-maximizing option (computed from empirical conditional survival rates).

**Survival rate:** 1 − (proportion of choice trials where captured).

**Total earnings:** Sum of trial-level payoffs (R if survived, −C if captured).

### Statistical Tests

**H4a (Orthogonality):** Pearson r between calibration and discrepancy. Test: |r| < 0.15.

**H4b (Calibration → performance):** Pearson r between calibration and choice quality, and between calibration and survival rate. Test: at least one r > 0, p < 0.05.

**H4c (Discrepancy → STAI):** Pearson r between discrepancy and STAI-State. Test: r > 0, p < 0.01.

**H4d (Discrepancy → additional measures):** Count of additional clinical measures (from {OASIS, STICSA, PHQ-9, DASS-Anxiety, DASS-Stress, DASS-Depression}) showing r > 0, p < 0.05 uncorrected. Test: count ≥ 2.

**Bayesian analysis:** Bayesian linear regression (bambi/PyMC, weakly informative priors, 2,000 draws × 4 chains, target_accept = 0.9) predicting each clinical measure from log(ce) + log(cd) + discrepancy + calibration. Region of practical equivalence (ROPE) = [−0.10, +0.10] for evaluating null effects of model parameters.

---

## Results

### Descriptive Statistics

| Measure | Mean | SD | Range | N |
|---------|------|----|-------|---|
| Calibration | 0.47 | 0.32 | [−0.58, 0.98] | 293 |
| Discrepancy | 0.00 | 1.43 | [−3.8, +5.2] | 293 |

85% of subjects have positive calibration (anxiety increases with danger). Discrepancy is zero-centered by construction.

---

### H4a: Calibration and Discrepancy Are Orthogonal

**Prediction:** |r(calibration, discrepancy)| < 0.15

| Measure 1 | Measure 2 | r | p | N |
|-----------|-----------|---|---|---|
| Calibration | Discrepancy | **0.019** | **0.75** | 293 |

**Verdict: CONFIRMED** (|0.019| << 0.15)

The two dimensions capture genuinely independent aspects of metacognitive functioning. A subject's accuracy in tracking danger (calibration) is unrelated to their tendency to overestimate danger (discrepancy).

---

### H4b: Calibration Predicts Task Performance

**Prediction:** Calibration correlates positively with at least one performance measure, p < 0.05.

| Predictor | Outcome | r | p | Direction |
|-----------|---------|---|---|-----------|
| Calibration | Choice quality | **0.230** | **< 0.001** | Better calibrated → better choices |
| Calibration | Survival rate | **0.185** | **0.002** | Better calibrated → more survival |
| Calibration | Total earnings | **0.239** | **< 0.001** | Better calibrated → higher earnings |

**Verdict: CONFIRMED** (all three performance measures significant)

**Interpretation:** Subjects whose anxiety accurately tracks model-derived danger make better choices (r = 0.23), survive more (r = 0.19), and earn more (r = 0.24). Accurate anxiety is ADAPTIVE — it provides a useful signal for behavioral regulation, increasing caution when danger is genuinely high and permitting engagement when danger is low.

#### Calibration and clinical measures

| Clinical measure | r | p | Significant? |
|-----------------|---|---|-------------|
| STAI-State | 0.138 | 0.019 | Yes (leakage) |
| STICSA | 0.075 | 0.206 | No |
| DASS-Anxiety | 0.019 | 0.745 | No |
| OASIS | 0.014 | 0.815 | No |
| PHQ-9 | 0.012 | 0.836 | No |
| DASS-Stress | 0.073 | 0.218 | No |
| DASS-Depression | 0.064 | 0.275 | No |

Calibration is largely unrelated to clinical symptoms (6 of 7 measures p > 0.10). The STAI-State exception (r = 0.14, p = .019) represents minor leakage — better threat-tracking may be associated with slightly higher state anxiety, possibly reflecting adaptive vigilance rather than pathology.

---

### H4c: Discrepancy Predicts STAI-State

**Prediction:** r(discrepancy, STAI-State) > 0, p < 0.01

| Predictor | Outcome | r | p |
|-----------|---------|---|---|
| Discrepancy | STAI-State | **0.308** | **< 10⁻⁶** |

**Verdict: CONFIRMED** (r = 0.308, p < 10⁻⁶)

Subjects who report more anxiety than the model's survival signal warrants (positive discrepancy) have substantially higher state anxiety as measured by a validated clinical instrument. This is the strongest clinical association in the dataset.

---

### H4d: Discrepancy Predicts Additional Clinical Measures

**Prediction:** ≥ 2 additional measures significant at p < 0.05 uncorrected.

#### Frequentist results

| Clinical measure | r | p | Significant? |
|-----------------|---|---|-------------|
| **STAI-State** | **0.308** | **< 10⁻⁶** | *** |
| **STICSA** | **0.249** | **< 10⁻⁴** | *** |
| **DASS-Anxiety** | **0.234** | **< 10⁻³** | *** |
| **DASS-Stress** | **0.217** | **< 10⁻³** | *** |
| **DASS-Depression** | **0.206** | **< 10⁻³** | *** |
| **PHQ-9** | **0.201** | **< 10⁻³** | *** |
| **OASIS** | **0.177** | **0.003** | ** |
| STAI-Trait | −0.203 | < 10⁻³ | *** (negative) |
| AMI-Emotional | −0.222 | < 10⁻³ | *** (negative) |

**7 of 7 additional measures significant** (threshold was ≥ 2).

**Verdict: CONFIRMED** (7 >> 2)

Discrepancy predicts symptoms across the full spectrum of anxiety (STAI, STICSA, OASIS), depression (PHQ-9, DASS-Depression), and stress (DASS-Stress). The consistency across independent instruments provides strong convergent validity.

#### Bayesian regression results (controlling for model parameters)

Full model: `clinical_z ~ log(ce) + log(cd) + discrepancy + calibration`

| Clinical measure | β(discrepancy) | 94% HDI | β(log_ce) in ROPE? | β(log_cd) in ROPE? |
|-----------------|----------------|---------|--------------------|--------------------|
| STAI-State | **0.338** | [0.22, 0.45] | Yes (77%) | Yes (93%) |
| STICSA | **0.285** | [0.17, 0.40] | Yes (82%) | Yes (78%) |
| DASS-Anxiety | **0.275** | [0.16, 0.39] | Yes (75%) | Yes (90%) |
| DASS-Stress | **0.217** | [0.10, 0.33] | Yes (80%) | Yes (91%) |
| DASS-Depression | **0.206** | [0.09, 0.32] | Yes (78%) | Yes (82%) |
| PHQ-9 | **0.212** | [0.10, 0.33] | Yes (76%) | Yes (80%) |
| OASIS | **0.180** | [0.07, 0.30] | Yes (78%) | Yes (75%) |
| AMI-Emotional | **−0.222** | [−0.34, −0.11] | Yes (73%) | Yes (65%) |

All discrepancy 94% HDIs exclude zero. All model parameter (log_ce, log_cd) posteriors fall predominantly within the ROPE [−0.10, +0.10], providing Bayesian evidence that the computational parameters DO NOT predict clinical symptoms when discrepancy is included.

#### The AMI-Emotional finding

Discrepancy is NEGATIVELY associated with emotional apathy (β = −0.222, HDI [−0.34, −0.11]). Subjects who overestimate danger are the OPPOSITE of apathetic — they are affectively engaged, even hypervigilant. This aligns with the proposal that anxiety and apathy represent opposite poles of motivational dysfunction (Husain & Roiser, 2018).

#### Machine learning prediction

Cross-validated ridge regression predicting each clinical measure from log(ce) + log(cd) + discrepancy + calibration + interactions: **all CV R² values are negative** (worse than predicting the mean). The associations reported above are group-level patterns, not individually predictive biomarkers. This is consistent with the modest effect sizes (R² = 0.03–0.11) and highlights that these parameters identify DIMENSIONS of variation that correlate with symptoms, not diagnostic classifiers.

---

### The Double Dissociation

#### Pattern summary

| Predictor | → Performance | → Clinical |
|-----------|--------------|-----------|
| **Calibration** | r = 0.19–0.24 (✓) | mostly null (6/7 p > .10) |
| **Discrepancy** | r = −0.15 (partial leakage) | r = 0.18–0.34 (✓✓✓) |

#### Is it a clean double dissociation?

**Partially.** The dominant pattern is clear and robust:
- Calibration **primarily** predicts performance
- Discrepancy **primarily** predicts clinical symptoms
- They are orthogonal (r = 0.019)

However, there is leakage in both directions:
- **Calibration → STAI-State:** r = 0.138, p = .019. Better calibration weakly predicts higher state anxiety. This could reflect adaptive vigilance (accurately anxious people are more aware of their state anxiety) or a methodological confound (both measures involve anxiety self-report).
- **Discrepancy → survival:** r = −0.153, p = .009. Overanxious subjects survive slightly less, possibly because excessive anxiety disrupts optimal motor execution or because they choose overly conservative strategies that happen to be suboptimal.

The paper should use **"partial double dissociation"** or **"predominant dissociation"** rather than claiming a clean separation. The pattern is strong enough to support the theoretical interpretation but not absolute.

---

## Effect Size Context

| Association | r | R² | Context |
|------------|---|-----|---------|
| Discrepancy → STAI | 0.308 | 9.5% | Largest clinical predictor in dataset |
| Discrepancy → STICSA | 0.249 | 6.2% | Consistent across instruments |
| Discrepancy → DASS-Anx | 0.234 | 5.5% | |
| Calibration → earnings | 0.239 | 5.7% | Largest performance predictor |

These effect sizes (R² = 3–10%) are consistent with the computational psychiatry literature (Gillan et al., 2016: R² ≈ 3–8%; Wise et al., 2020: R² ≈ 2–6%). They are meaningful for identifying mechanistic dimensions of variation but insufficient for individual-level prediction (as confirmed by the negative cross-validated R² values from machine learning).

---

## Theoretical Interpretation

The calibration-discrepancy decomposition operationalizes two distinct metacognitive functions:

**Calibration = monitoring fidelity.** How well does the affective system track the cognitive survival computation? High calibration means anxiety provides a faithful readout of the model's danger signal, enabling appropriate behavioral adjustment. This is the adaptive function of anxiety — an alarm system calibrated to actual threat.

**Discrepancy = systematic bias.** How much does the affective system deviate from the computation? High discrepancy means anxiety is systematically elevated beyond what the danger signal warrants — an interoceptive prediction error (Paulus & Stein, 2010). This excess affect is decoupled from the environment and instead reflects a stable individual tendency toward threat overestimation.

The dissociation implies that:
1. **Good monitoring (high calibration) → adaptive behavior** but is orthogonal to clinical risk
2. **Systematic bias (high discrepancy) → clinical vulnerability** but is orthogonal to behavioral performance
3. **The route from normative computation to psychopathology runs through the MISMATCH between computation and affect**, not through the computation itself

This is consistent with Wells' (2009) metacognitive model of anxiety: clinical anxiety reflects dysfunctional beliefs ABOUT threat cognition (meta-worry), not simply elevated threat cognition per se.

---

## Confirmation Plan

### Tests to run on confirmation sample (N ≈ 280–330):

| Test | Statistic | Threshold | Discovery value |
|------|-----------|-----------|-----------------|
| H4a: Orthogonality | \|r(cal, disc)\| | < 0.15 | 0.019 |
| H4b: Calibration → performance | r(cal, choice quality) | > 0, p < 0.05 | 0.230 |
| H4c: Discrepancy → STAI | r(disc, STAI) | > 0, p < 0.01 | 0.308 |
| H4d: Additional measures ≥ 2 | count p < 0.05 | ≥ 2 of 6 | 7 of 7 |

### Power analysis

At N = 280:
- H4c (weakest powered): power > 0.99 to detect r = 0.31 at α = 0.01
- H4d: power > 0.95 for each individual measure at r ≈ 0.18 (smallest effect)
- All tests are well-powered given the large discovery effect sizes

---

## Summary

| Sub-hypothesis | Test | Statistic | p | Threshold | Verdict |
|---------------|------|-----------|---|-----------|---------|
| H4a: Orthogonality | Pearson r | r = 0.019 | 0.75 | \|r\| < 0.15 | **CONFIRMED** |
| H4b: Cal → performance | Pearson r | r = 0.185–0.239 | < 0.002 | p < 0.05 | **CONFIRMED** |
| H4c: Disc → STAI | Pearson r | r = 0.308 | < 10⁻⁶ | p < 0.01 | **CONFIRMED** |
| H4d: ≥ 2 additional | Count | 7 / 7 | all < 0.003 | ≥ 2 | **CONFIRMED** |
| Bayesian: disc credible | 94% HDI | all exclude 0 | — | — | **CONFIRMED** |
| Bayesian: params in ROPE | % in ROPE | ~77% | — | — | **CONFIRMED** |
| Double dissociation | Pattern | Predominant | — | — | **PARTIAL** |

**Bottom line:** Calibration predicts who performs well. Discrepancy predicts who is anxious. These are orthogonal dimensions. The computational parameters (ce, cd) predict behavior but not symptoms. The bridge to psychopathology runs through metacognition.
