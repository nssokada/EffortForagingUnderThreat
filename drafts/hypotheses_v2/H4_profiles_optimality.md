# H4: Foraging Profiles and Optimality

## Overview

Individual differences in capture cost (ω) and effort cost (κ) predict who survives, what kind of errors people make, and how they allocate motor resources. The balance between threat-driven and effort-driven avoidance determines foraging wisdom.

---

## H4a: ω predicts survival on attack trials

### Prediction

Higher ω predicts higher escape rate. People who perceive capture as costly adopt strategies that increase survival.

### Test

OLS: escape_rate ~ ω_z + κ_z.

### Threshold

ω: β > 0, p < 0.01.

### Exploratory benchmarks

- ω → escape: β = +0.060, p = .0002
- κ → escape: β = -0.006, p = .72 (null)

---

## H4b: Overcaution is the dominant error, driven by ω

### Prediction

Among suboptimal choices, the majority (> 65%) are overcautious — choosing light when heavy has higher expected reward. ω predicts who is overcautious.

### Test

1. Compute empirical expected reward per T×D cell. Define optimal choice. Classify errors.
2. Correlate ω with overcaution ratio.

### Threshold

Overcaution > 65% of errors. r(ω, overcaution ratio) > 0.30, p < .01.

### Exploratory benchmarks

- 79% of errors are overcautious
- r(ω, overcaution ratio) = +0.810 (p < .0001)

---

## H4c: κ predicts pressing intensity

### Prediction

κ governs motor output — the effort cost parameter determines how hard people press. This is the activation side of the avoid-activate decomposition: ω drives avoidance (H4a-b), κ drives vigor.

### Test

r(κ, mean vigor).

### Threshold

r < -0.30, p < .01.

### Exploratory benchmarks

- r(κ, mean vigor) = -0.736 (p < .0001)
- r(ω, mean vigor) = +0.08 (null)

---

## H4d: Effort-driven avoidance is less optimal than threat-driven avoidance

### Prediction

The ω-κ balance predicts decision quality. People whose avoidance is primarily effort-driven (high κ relative to ω) are less optimal than those whose avoidance is primarily threat-driven (high ω relative to κ). Threat-sensitive avoidance is context-appropriate — you avoid when it's dangerous. Effort-sensitive avoidance is indiscriminate — you avoid heavy cookies regardless of threat level.

### Test

r(ω-κ angle, % optimal), where angle = atan2(κ_z, ω_z). Higher angle = more effort-focused.

### Threshold

r < -0.15, p < .01.

### Exploratory benchmarks

- r(angle, optimality) = -0.315 (p < .0001)
- r(angle, overcaution) = +0.436 (p < .0001)

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

| Test | Statistic | Threshold | Discovery value |
|------|-----------|-----------|-----------------|
| H4a: ω → escape | β | > 0, p < .01 | +0.060, p = .0002 |
| H4b: Overcaution % | % of errors | > 65% | 79% |
| H4b: ω → overcaution | r | > .30, p < .01 | +0.810 |
| H4c: κ → vigor | r | < -.30, p < .01 | -0.736 |
| H4d: angle → optimality | r | < -.15, p < .01 | -0.315 |
