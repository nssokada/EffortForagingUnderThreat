# H4: Foraging Profiles and Optimality

## Overview

Individual differences in capture cost (ω) and effort cost (κ) predict who survives, what kind of errors people make, and how they allocate motor resources. The balance between threat-driven and effort-driven avoidance determines foraging wisdom.

---

## H4a: ω predicts survival on attack trials

### Prediction

Higher ω predicts higher escape rate. People who perceive capture as costly adopt strategies that increase survival.

### Test

Bayesian linear model (bambi): escape_rate ~ ω_z + κ_z.

### Threshold

ω: posterior mean > 0, 95% HDI excludes zero.

### Exploratory benchmarks

- ω → escape: β = +0.060, p = .0002
- κ → escape: β = -0.006, p = .72 (null)

---

## H4b: Capture cost predicts overcautious errors

### Prediction

Higher ω predicts a greater proportion of overcautious errors — choosing light when heavy has higher expected reward. People who perceive capture as costly systematically avoid the high-reward option even when it's optimal.

### Test

1. Compute empirical expected reward per T×D cell. Define optimal choice. Classify errors.
2. Bayesian linear model (bambi): overcaution_ratio ~ ω_z.

### Threshold

ω → overcaution_ratio: posterior mean > 0, 95% HDI excludes zero. Overall overcaution percentage reported descriptively.

### Exploratory benchmarks

- 79% of errors are overcautious
- r(ω, overcaution ratio) = +0.810 (p < .0001)

---

## H4c: κ predicts pressing intensity

### Prediction

κ governs motor output — the effort cost parameter determines how hard people press. This is the activation side of the avoid-activate decomposition: ω drives avoidance (H4a-b), κ drives vigor.

### Test

Bayesian linear model (bambi): mean_vigor ~ κ_z.

### Threshold

κ: posterior mean < 0, 95% HDI excludes zero.

### Exploratory benchmarks

- r(κ, mean vigor) = -0.736 (p < .0001)
- r(ω, mean vigor) = +0.08 (null)

---

## H4d: Effort-driven avoidance is less optimal than threat-driven avoidance

### Prediction

The ω-κ balance predicts decision quality. People whose avoidance is primarily effort-driven (high κ relative to ω) are less optimal than those whose avoidance is primarily threat-driven (high ω relative to κ). Threat-sensitive avoidance is context-appropriate — you avoid when it's dangerous. Effort-sensitive avoidance is indiscriminate — you avoid heavy cookies regardless of threat level.

### Test

Bayesian linear model (bambi): pct_optimal ~ angle_z, where angle = atan2(κ_z, ω_z). Higher angle = more effort-focused.

### Threshold

angle: posterior mean < 0, 95% HDI excludes zero.

### Exploratory benchmarks

- r(angle, optimality) = -0.315 (p < .0001)
- r(angle, overcaution) = +0.436 (p < .0001)

---

## H4e: Consistency with W(u) across both channels predicts earnings

### Prediction

People who are more consistent with their own fitness function — choosing the cookie W(u) predicts AND pressing at the intensity W(u) predicts — will earn more. Both choice consistency and intensity pattern match will independently predict earnings.

### Test

Per-subject choice consistency = fraction of trials where actual choice matches model-predicted choice. Per-subject intensity deviation = RMSE between model-predicted u* and observed cell-mean rate across conditions (lower = closer to model). Bayesian linear model (bambi): earnings ~ choice_consistency_z + intensity_deviation_z.

### Threshold

choice_consistency posterior mean > 0, intensity_deviation posterior mean < 0; both 95% HDIs exclude zero.

### Exploratory benchmarks

- Choice consistency → earnings: r = +0.252
- Intensity pattern → earnings: r = +0.430
- Joint R² = 0.23, both p < .001
- The two consistencies are independent (r = +0.10)

---

## Descriptive: Four foraging profiles

Median-splitting on ω and κ produces four profiles (reported descriptively, not as a preregistered test):

| Profile | N | P(heavy) | Escape | Vigor | Earnings |
|---------|---|----------|--------|-------|----------|
| Strategic (Hi-ω Lo-κ) | 42 | 0.352 | 0.450 | 1.253 | +22.5 |
| Resilient (Lo-ω Lo-κ) | 103 | 0.608 | 0.337 | 1.026 | +16.1 |
| Helpless (Hi-ω Hi-κ) | 103 | 0.242 | 0.431 | 0.867 | -2.6 |
| Reckless (Lo-ω Hi-κ) | 42 | 0.529 | 0.317 | 0.796 | -10.8 |

Strategic foragers earned 33 points more than Reckless foragers, corresponding to real bonus payment differences.

---

## Confirmation Plan

| Test | Criterion | Threshold | Discovery value |
|------|-----------|-----------|-----------------|
| H4a: ω → escape | posterior mean | > 0, HDI excludes 0 | +0.060 |
| H4b: Overcaution % | % of errors | > 65% | 79% |
| H4b: ω → overcaution | posterior mean | > 0, HDI excludes 0 | +0.177 |
| H4c: κ → vigor | posterior mean | < 0, HDI excludes 0 | -0.194 |
| H4d: angle → optimality | posterior mean | < 0, HDI excludes 0 | -0.041 |
| H4e: choice cons → earnings | posterior mean | > 0, HDI excludes 0 | +14.3 |
| H4e: intensity → earnings | posterior mean | > 0, HDI excludes 0 | +36.6 |
