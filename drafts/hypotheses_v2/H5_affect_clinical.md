# H5: Metacognitive Monitoring of the Foraging Computation

## Overview

This hypothesis tests how accurately metacognitive signals — anxiety and confidence — monitor the first-order survival computation, and whether monitoring accuracy predicts foraging efficiency beyond the computational model parameters. We decompose anxiety into calibration (signal quality) and reactivity (slope), and show that each predicts a different aspect of foraging performance. We further show that the computational capture-cost parameter (ω) maps onto confidence (a coping appraisal) rather than anxiety (a threat appraisal), and that confidence determines the type but not rate of foraging errors.

### Theoretical grounding

In the two-stage metacognitive architecture (Fleming & Daw 2017), a first-order process generates decisions while a second-order monitor evaluates their quality. Metacognitive accuracy — how well the monitor tracks the first-order process — determines whether behavior is appropriately adjusted. In our task, the first-order process is the survival computation (ω, κ → choice and vigor). The second-order monitors are anxiety (tracking threat, Lazarus's 1991 primary appraisal: "is this threatening?") and confidence (tracking coping capacity, Lazarus's secondary appraisal: "can I cope?"). We test whether monitoring accuracy predicts foraging efficiency, and whether anxiety and confidence monitor dissociable aspects of the computation.

---

## H5a: Anxiety calibration predicts foraging optimality beyond the computational model

### Prediction

After controlling for ω and κ, calibration (within-subject r between anxiety and threat) explains additional variance in foraging optimality, escape rate, and earnings. People whose anxiety is a better thermometer of danger make better decisions — the metacognitive monitor adds information that the first-order computation doesn't contain.

### Test

Bayesian model comparison (bambi + ArviZ LOO-CV):
- Base: pct_optimal ~ ω_z + κ_z
- Full: pct_optimal ~ ω_z + κ_z + calibration_z
- Compare via LOO-CV (PSIS-LOO). Report delta-ELPD and SE.
- Escape rate and earnings tested as supporting outcomes.

### Threshold

Calibration improves model fit for pct_optimal (delta-ELPD > 0, SE excludes zero).

### Exploratory benchmarks

| Outcome | R²(ω+κ) | R²(+calibration) | ΔR² | cal p |
|---------|---------|-------------------|-----|-------|
| % Optimal | 0.476 | 0.544 | **+0.068** | <.0001 |
| Earnings | 0.009 | 0.067 | **+0.058** | <.0001 |
| Escape | 0.056 | 0.094 | **+0.038** | .0006 |

---

## H5b: Anxiety reactivity predicts adaptive choice shifting

### Prediction

People whose anxiety reacts more strongly to threat (steeper slope) shift their choices more across threat levels. Anxiety reactivity drives choice adaptation but not vigor adaptation — the monitor guides the avoidance channel specifically.

### Test

Bayesian linear model (bambi): choice_shift ~ anxiety_slope_z.
- Choice shift = P(heavy at T=0.1) − P(heavy at T=0.9)
- Also check: vigor shift = mean vigor at T=0.9 − mean vigor at T=0.1 (expected null)

### Threshold

Posterior mean > 0, 95% HDI excludes zero.

### Exploratory benchmarks

- r(anxiety slope, choice shift) = **+0.389** (p < .0001)
- r(anxiety slope, vigor shift) = +0.004 (null)

Anxiety reactivity specifically predicts how much people adjust their avoidance strategy across threat levels, not how much they adjust their motor output.

---

## H5c: ω maps to confidence, not anxiety

### Prediction

The computational capture-cost parameter (ω) predicts subjective confidence but not anxiety. This dissociation maps onto Lazarus's (1991) distinction between primary appraisal (threat evaluation → anxiety) and secondary appraisal (coping evaluation → confidence). The computation informs the coping judgment; it does not generate the affective state.

### Test

Bayesian linear models (bambi): mean_confidence ~ ω_z and mean_anxiety ~ ω_z.

### Threshold

mean_confidence ~ ω_z: posterior mean < 0, 95% HDI excludes zero. mean_anxiety ~ ω_z: null tested via ROPE — 95% HDI falls entirely within [-0.10, +0.10] (standardized β), indicating the effect is practically equivalent to zero.

### Exploratory benchmarks

- r(ω, confidence) = **-0.216** (p = .0002)
- r(κ, confidence) = -0.153 (p = .009)
- r(ω, anxiety) = +0.071 (null)
- r(κ, anxiety) = +0.026 (null)

High ω (high capture cost) → lower confidence ("this is dangerous, I might not make it"). But ω does NOT increase anxiety. The survival computation produces a prospective coping appraisal, not an affective threat response.

---

## H5d: Confidence predicts error type, not error rate

### Prediction

Confident people don't make fewer errors — they make different errors. Confidence reduces overcautious errors but increases reckless errors. This is Pouget et al.'s (2016) confidence-as-action-signal: confidence determines what you commit to, not whether you succeed.

### Test

Bayesian linear models (bambi): n_overcautious ~ confidence_z and n_reckless ~ confidence_z.

### Threshold

n_overcautious ~ confidence_z: posterior mean < 0, 95% HDI excludes zero. n_reckless ~ confidence_z: posterior mean > 0, 95% HDI excludes zero.

### Exploratory benchmarks

- r(confidence, n_overcautious) = **-0.224** (p = .0001)
- r(confidence, n_reckless) = **+0.200** (p = .0006)

---

## Summary

| Signal | What it monitors | What it predicts | Framework |
|--------|-----------------|------------------|-----------|
| **Anxiety calibration** | Threat environment (quality) | Foraging optimality | Fleming & Daw 2017 |
| **Anxiety reactivity** | Threat changes (slope) | Adaptive choice shifting | Lazarus 1991 primary appraisal |
| **Confidence** | Coping capacity (ω) | Error type (cautious vs reckless) | Pouget et al. 2016; Lazarus 1991 secondary appraisal |

The first-order computation (ω, κ) governs foraging strategy — who avoids and who mobilizes. The second-order metacognitive monitors (anxiety, confidence) govern how wisely that strategy is deployed. Good anxiety calibration → appropriate avoidance → efficient foraging. Accurate confidence → appropriate risk-taking → balanced error profile. The monitors don't override the computation; they guide when and how it's applied.

---

## Confirmation Plan

| Test | Criterion | Threshold | Discovery value |
|------|-----------|-----------|-----------------|
| H5a: Cal → optimality | LOO delta-ELPD | > 0, SE excludes 0 | ΔELPD > 0 (all 3) |
| H5a: Cal → earnings | LOO delta-ELPD | > 0, SE excludes 0 | ΔELPD > 0 (all 3) |
| H5b: Slope → choice shift | posterior mean | > 0, HDI excludes 0 | HDI excludes 0 |
| H5c: ω → confidence | posterior mean | < 0, HDI excludes 0 | HDI excludes 0 |
| H5c: ω → anxiety | posterior mean | HDI includes 0 | HDI includes 0 |
| H5d: Conf → overcautious | posterior mean | < 0, HDI excludes 0 | HDI excludes 0 |
| H5d: Conf → reckless | posterior mean | > 0, HDI excludes 0 | HDI excludes 0 |
