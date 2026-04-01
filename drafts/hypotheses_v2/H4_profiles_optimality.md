# H4: Foraging Profiles and Optimality

## Overview

Individual differences in capture cost (ω) and effort cost (κ) define behaviorally distinct foraging profiles with consequential outcomes. The model parameters predict who survives, who earns, and what kind of errors people make.

---

## H4a: ω and κ define four behavioral profiles

### Prediction

Median-splitting on ω and κ produces four profiles with distinct earnings:

| | Low κ (effort-tolerant) | High κ (effort-averse) |
|---|---|---|
| **High ω** | Strategic — avoids danger, presses hard | Helpless — avoids AND presses weakly |
| **Low ω** | Resilient — engages everything, presses hard | Reckless — engages danger, presses weakly |

### Test

ANOVA on earnings across quadrants.

### Threshold

p < 0.01. Strategic should earn most.

### Exploratory benchmarks

| Profile | N | P(heavy) | Escape | Vigor | Earnings |
|---------|---|----------|--------|-------|----------|
| Strategic (Hi-ω Lo-κ) | 42 | 0.352 | 0.450 | 1.253 | +22.5 |
| Resilient (Lo-ω Lo-κ) | 103 | 0.608 | 0.337 | 1.026 | +16.1 |
| Helpless (Hi-ω Hi-κ) | 103 | 0.242 | 0.431 | 0.867 | -2.6 |
| Reckless (Lo-ω Hi-κ) | 42 | 0.529 | 0.317 | 0.796 | -10.8 |

---

## H4b: ω predicts survival on attack trials

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

## H4c: Overcaution is the dominant error

### Prediction

Among suboptimal choices, the majority (> 65%) are overcautious — choosing light when heavy has higher expected reward. ω predicts who makes overcautious errors.

### Test

1. Compute empirical expected reward per T×D cell. Define optimal choice. Classify errors.
2. Correlate ω with overcaution ratio.

### Threshold

Overcaution > 65% of errors. r(ω, overcaution ratio) > 0.30, p < .01.

### Exploratory benchmarks

- 79% of errors are overcautious
- r(ω, overcaution ratio) = +0.810 (p < .0001)
- Heavy optimal at T ≤ 0.5 (all distances), light optimal at T = 0.9

---

## Confirmation Plan

| Test | Statistic | Threshold | Discovery value |
|------|-----------|-----------|-----------------|
| H4a: Profile ANOVA | F(earnings) | p < .01 | Strategic: +22.5, Reckless: -10.8 |
| H4b: ω → escape | β | > 0, p < .01 | +0.060, p = .0002 |
| H4c: Overcaution % | % of errors | > 65% | 79% |
| H4c: ω → overcaution | r | > .30, p < .01 | +0.810 |
