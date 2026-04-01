# H4: Foraging Profiles and Optimality in ω-κ Space

## Overview

This hypothesis tests whether individual differences in capture cost (ω) and effort cost (κ) define behaviorally distinct foraging profiles with consequential outcomes. The key prediction: the two-parameter decomposition maps onto ecologically meaningful strategies, and the model parameters predict who forages successfully.

### Theoretical grounding

The starvation-predation tradeoff (Lima & Dill 1990; Brown 1988) predicts that foragers must balance energy acquisition against survival. Our joint model decomposes this into two independent cost parameters: ω (how costly capture is) and κ (how costly effort is). These define four behavioral strategies — not all equally viable.

---

## H4a: ω and κ define four behavioral profiles

### Prediction

Splitting subjects by median ω and median κ produces four quadrants with distinct behavioral signatures:

| | Low κ (effort-tolerant) | High κ (effort-averse) |
|---|---|---|
| **High ω (threat-sensitive)** | Strategic — avoids risky patches, presses hard when engaged | Helpless — avoids AND presses weakly |
| **Low ω (threat-tolerant)** | Resilient — engages everything, presses hard | Reckless — engages risky patches, presses weakly |

### Test

Fit CM2 to obtain per-subject ω and κ. Median-split into four quadrants. Report per-quadrant: P(heavy), mean vigor, escape rate, total earnings.

### Threshold

ANOVA on earnings across quadrants: p < 0.01. The "Strategic" quadrant (high ω, low κ) should have the highest earnings.

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

Higher ω (capture cost) predicts higher escape rate on attack trials. People who perceive capture as costly adopt strategies (avoidance + pressing above threshold) that increase survival when the predator appears.

### Test

OLS: escape_rate ~ ω_z + κ_z. ω should be significant, positive.

### Threshold

ω: β > 0, p < 0.01.

### Exploratory benchmarks

- ω → escape: β = +0.060, p = .0002
- κ → escape: β = -0.006, p = .72 (null)
- R² = 0.052

---

## H4c: The optimal strategy is selective avoidance

### Prediction

The empirically optimal strategy is to choose heavy cookies when expected reward is positive (T ≤ 0.5) and avoid when negative (T = 0.9). Most suboptimal decisions are overcautious — avoiding when engaging is optimal.

### Test

Compute expected reward for heavy vs light at each T×D cell from observed outcomes. Define optimal choice per cell. Classify errors as overcautious (chose light when heavy was optimal) vs reckless (chose heavy when light was optimal).

### Threshold

Heavy is optimal at T ≤ 0.5 for all distances. Light is optimal at T = 0.9 for all distances.

### Exploratory benchmarks

| T | D | E[R] Heavy | E[R] Light | Optimal |
|---|---|-----------|-----------|---------|
| 0.1 | 1 | +3.28 | +0.56 | Heavy |
| 0.1 | 3 | +2.60 | +0.56 | Heavy |
| 0.5 | 1 | +0.87 | -0.36 | Heavy |
| 0.5 | 3 | -0.63 | -0.66 | Heavy (marginal) |
| 0.9 | 1 | -2.25 | -1.42 | Light |
| 0.9 | 3 | -2.89 | -1.90 | Light |

---

## H4d: Discrepancy predicts escape beyond model parameters

### Prediction

Affective miscalibration (discrepancy) predicts escape rate even after controlling for ω and κ. People who are more anxious than the situation warrants escape less — their excessive caution leads to suboptimal avoidance without compensating survival benefit.

### Test

OLS: escape_rate ~ ω_z + κ_z + discrepancy_z. Discrepancy should be significant, negative.

### Threshold

Discrepancy: β < 0, p < 0.05.

### Exploratory benchmarks

- Full model: escape ~ ω + κ + disc + conf_disc (R² = 0.088)
- ω: β = +0.065, p = .0001
- Discrepancy: β = -0.043, p = .002
- κ and conf_disc: null

Interpretation: ω improves escape (strategic avoidance works), but excessive anxiety (discrepancy) impairs escape (overcaution backfires).

---

## Confirmation Plan

| Test | Statistic | Threshold | Discovery value |
|------|-----------|-----------|-----------------|
| H4a: Profile ANOVA | F(earnings) | p < 0.01 | Strategic: +22.5, Reckless: -10.8 |
| H4b: ω → escape | β | > 0, p < .01 | +0.060, p = .0002 |
| H4c: Optimal = selective | E[R] pattern | Heavy optimal at T ≤ 0.5 | Confirmed 6/9 cells |
| H4d: Disc → escape | β | < 0, p < .05 | -0.043, p = .002 |
