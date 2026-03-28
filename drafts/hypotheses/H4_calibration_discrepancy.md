# H4: Calibration-Discrepancy Double Dissociation

## Hypothesis

Metacognitive calibration (how accurately anxiety tracks danger) and discrepancy (how much anxiety exceeds danger) will be orthogonal dimensions that dissociate: calibration predicts task performance, while discrepancy predicts clinical symptoms.

---

## Motivation

The EVC model computes a survival signal S that predicts trial-level anxiety (H1c: t = −14.0). But the between-subject relationship between the model's survival signal and subjective anxiety may decompose into two independent dimensions:

1. **Calibration:** Does this person's anxiety accurately TRACK danger? (signal quality)
2. **Discrepancy:** Does this person's anxiety EXCEED danger? (systematic bias)

Metacognitive theories of anxiety (Wells, 2009; Paulus & Stein, 2010) propose that pathological anxiety reflects a mismatch between threat appraisal and affective response — not simply elevated threat sensitivity. The calibration-discrepancy decomposition operationalizes this computationally.

---

## Definitions

**Calibration (per-subject):**
Within-subject Pearson correlation between anxiety ratings and model-derived danger (1 − S), across the subject's 18 anxiety probe trials. Higher calibration means the individual's anxiety more closely tracks the objective threat level.

- Population mean calibration: 0.47 (SD = 0.32)
- Range: [−0.58, 0.98]
- 85% of subjects have positive calibration (anxiety increases with danger)

**Discrepancy (per-subject):**
Mean residual of a subject's anxiety ratings after subtracting the population-level prediction from a regression of anxiety on S. Positive discrepancy means the individual reports more anxiety than the average person would at the same survival level.

- Population mean discrepancy: 0.00 (by construction, SD = 1.43)
- Range: [−3.8, +5.2]
- Positive = overanxious relative to danger. Negative = underanxious.

---

## H4a: Orthogonality

**Test:** |r(calibration, discrepancy)| < 0.15

### Results

- r(calibration, discrepancy) = 0.019, p = .75

**Verdict: CONFIRMED** (|0.019| < 0.15)

The two dimensions are nearly perfectly orthogonal. Knowing how WELL someone tracks danger tells you nothing about how MUCH they overestimate it. This confirms they capture genuinely independent aspects of metacognitive functioning.

**Interpretation:** A person can be:
- Well-calibrated AND overanxious (anxiety tracks danger accurately but is shifted up)
- Well-calibrated AND calm (anxiety tracks danger accurately at low baseline)
- Poorly calibrated AND overanxious (anxiety is both noisy AND biased high)
- Poorly calibrated AND calm (anxiety is noisy but not elevated)

These four profiles have different behavioral and clinical implications.

---

## H4b: Calibration Predicts Performance

**Test:** r(calibration, choice quality) > 0 OR r(calibration, survival rate) > 0, p < 0.05

### Results

| Outcome | r | p | Direction |
|---------|---|---|-----------|
| Choice quality | 0.230 | < .001 | Better calibrated → better choices |
| Survival rate | 0.185 | .002 | Better calibrated → more survival |
| Total earnings | 0.239 | < .001 | Better calibrated → higher earnings |

**Verdict: CONFIRMED**

Subjects whose anxiety more accurately tracks model-derived danger make EV-optimal choices more often (r = 0.23) and survive at higher rates (r = 0.19). The combined effect on earnings (r = 0.24) is driven by both channels.

**Interpretation:** Accurate anxiety is ADAPTIVE. When anxiety faithfully tracks the survival computation, it provides a useful signal for behavioral regulation — increasing caution when danger is genuinely high and permitting engagement when danger is low.

### Calibration and clinical measures

| Measure | r | p | Significant? |
|---------|---|---|-------------|
| STAI-State | 0.138 | .019 | Yes (leakage) |
| STICSA | 0.075 | .206 | No |
| DASS-Anxiety | 0.019 | .745 | No |
| OASIS | 0.014 | .815 | No |
| PHQ-9 | 0.012 | .836 | No |
| DASS-Stress | 0.073 | .218 | No |
| DASS-Depression | 0.064 | .275 | No |
| AMI-Emotional | — | — | — |

Calibration is largely unrelated to clinical measures (6 of 7 measures p > .1). The exception is STAI-State (r = 0.14, p = .019), which may reflect that better threat-tracking is associated with higher state anxiety — a finding that, if replicated, could indicate adaptive vigilance rather than pathology.

---

## H4c: Discrepancy Predicts Clinical Symptoms — STAI

**Test:** r(discrepancy, STAI-State) > 0, p < 0.01

### Results

- r = 0.308, p < 10⁻⁶

**Verdict: CONFIRMED**

Subjects who report more anxiety than the model's survival signal warrants have higher state anxiety as measured by a validated clinical instrument.

---

## H4d: Discrepancy Predicts Additional Clinical Measures

**Test:** At least 2 additional measures from {OASIS, STICSA, PHQ-9, DASS-Anxiety, DASS-Stress, DASS-Depression} at p < 0.05

### Results — Frequentist

| Clinical measure | r | p | Significant? |
|-----------------|---|---|-------------|
| STAI-State | 0.308 | < 10⁻⁶ | *** |
| STICSA | 0.249 | < 10⁻⁴ | *** |
| DASS-Anxiety | 0.234 | < 10⁻³ | *** |
| DASS-Stress | 0.217 | < 10⁻³ | *** |
| DASS-Depression | 0.206 | < 10⁻³ | *** |
| PHQ-9 | 0.201 | < 10⁻³ | *** |
| OASIS | 0.177 | .003 | ** |
| STAI-Trait | −0.203 | < 10⁻³ | *** (negative) |
| AMI-Emotional | −0.222 | < 10⁻³ | *** (negative) |

**7 of 7 additional measures significant** (threshold was ≥ 2).

**Verdict: CONFIRMED** (7 > 2)

### Results — Bayesian (bambi regression)

Bayesian regression with weakly informative priors, controlling for log(ce) and log(cd):

| Clinical measure | β(discrepancy) | 94% HDI |
|-----------------|----------------|---------|
| STAI-State | 0.338 | [0.22, 0.45] |
| STICSA | 0.285 | [0.17, 0.40] |
| DASS-Anxiety | 0.275 | [0.16, 0.39] |
| DASS-Stress | 0.217 | [0.10, 0.33] |
| DASS-Depression | 0.206 | [0.09, 0.32] |
| PHQ-9 | 0.212 | [0.10, 0.33] |
| OASIS | 0.180 | [0.07, 0.30] |
| AMI-Emotional | −0.222 | [−0.34, −0.11] |

All 94% HDIs exclude zero. Effects survive controlling for model parameters.

### The AMI-Emotional finding

Discrepancy is NEGATIVELY associated with emotional apathy (β = −0.222). Subjects who overestimate danger are the OPPOSITE of apathetic — they are affectively engaged, perhaps hypervigilant. This echoes findings that anxiety and apathy are opposite poles of motivational dysfunction (Husain & Roiser, 2018).

---

## Model Parameters Are Clinically Inert

In the same Bayesian regressions, log(ce) and log(cd) posteriors fall largely within the ROPE [−0.1, 0.1]:

| Parameter | Mean ROPE containment | Interpretation |
|-----------|---------------------|----------------|
| log(ce) | ~77% | Evidence for null |
| log(cd) | ~77% | Evidence for null |

This is positive Bayesian evidence that the computational parameters THEMSELVES do not predict clinical symptoms. The bridge to psychopathology runs through METACOGNITION (discrepancy), not through the decision computation.

### Machine learning confirmation

Cross-validated ridge regression predicting each clinical measure from log(ce) + log(cd) + discrepancy + calibration + interactions: all CV R² values are negative (worse than predicting the mean). The associations are group-level patterns, not individually predictive biomarkers.

---

## The Double Dissociation

### Dissociation pattern

| Predictor | → Performance | → Clinical |
|-----------|--------------|-----------|
| Calibration | r = 0.19–0.24 (✓) | mostly null (6/7 p > .1) |
| Discrepancy | r = −0.15 (leakage) | r = 0.18–0.34 (✓✓✓) |

### Is it a clean double dissociation?

**Partially.** The dominant pattern is clear:
- Calibration primarily predicts performance
- Discrepancy primarily predicts clinical symptoms
- They're orthogonal (r = 0.019)

But there is leakage:
- Calibration → STAI-State: r = 0.138 (p = .019). Better calibration weakly predicts HIGHER state anxiety.
- Discrepancy → survival: r = −0.153 (p = .009). Overanxious subjects survive slightly less.

The paper should use "partial double dissociation" or "predominant dissociation" rather than claiming a clean separation.

---

## Theoretical Interpretation

The calibration-discrepancy decomposition operationalizes the metacognitive framework of Wells (2009) and the interoceptive prediction error framework of Paulus & Stein (2010):

- **Calibration** = signal quality. How well does the affective system track the cognitive computation? This is the metacognitive monitoring function.
- **Discrepancy** = systematic bias. How much does the affective system overreact relative to the computation? This is the interoceptive prediction error — a mismatch between expected and actual affective states.

The dissociation suggests that:
1. Good metacognitive monitoring (high calibration) produces adaptive behavior but is orthogonal to clinical risk.
2. Systematic metacognitive bias (high discrepancy) produces clinical vulnerability but is orthogonal to behavioral performance.
3. The route from normative computation to psychopathology runs through the MISMATCH between computation and affect, not through the computation itself.

---

## Summary

| Sub-hypothesis | Test | Result | Statistic |
|---------------|------|--------|-----------|
| H4a: Orthogonality | |r| < 0.15 | **CONFIRMED** | r = 0.019 |
| H4b: Calibration → performance | p < .05 | **CONFIRMED** | r = 0.19–0.24 |
| H4c: Discrepancy → STAI | p < .01 | **CONFIRMED** | r = 0.308 |
| H4d: ≥2 additional measures | p < .05 | **CONFIRMED** | 7/7 significant |
| Double dissociation | Pattern | **PARTIAL** | Leakage in both directions |
