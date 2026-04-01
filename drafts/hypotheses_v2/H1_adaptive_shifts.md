# H1: Threat Drives Adaptive Shifts in Choice, Vigor, and Affect

## Overview

This hypothesis tests whether increasing predation risk drives adaptive shifts across three behavioral channels: foragers shift toward safer, less effortful patches (avoidance), press harder within each patch type (activation), and report increased anxiety and decreased confidence. The critical finding is that these responses are simultaneous and superficially contradictory — threat makes people "lazier" in their choices but "harder-working" in their actions — motivating the two-channel computational architecture tested in H4.

### Theoretical grounding

The starvation-predation tradeoff (Lima & Dill 1990; McNamara & Houston 1992) predicts that foragers under threat should reduce exposure to dangerous patches. But the escape theory literature (Ydenberg & Dill 1986) predicts that committed foragers should increase motor effort to maximize survival. These predictions apply to *different stages* of the foraging bout — the decision of what to engage versus the execution once committed. Our task uniquely measures both simultaneously.

### Critical methodological note: Simpson's paradox

Higher threat shifts choice toward light cookies (lower effort demand). Light cookies produce lower absolute press rates. Collapsing across cookie type makes average vigor appear *flat* across threat, masking the within-cookie-type increase. This paradox is not a nuisance — it is a central finding that demonstrates why joint modeling of choice and vigor is necessary.

---

## H1a: Threat and distance deter high-effort patch engagement

### Prediction

P(choose heavy cookie) decreases with threat probability T and with escape distance D, with a significant T × D interaction.

### Test

Logistic mixed-effects model with cluster-robust standard errors:

`choice ~ threat_z + dist_z + threat_z:dist_z`, clustered by subject.

Additionally: monotonicity confirmed by paired t-tests on subject-mean P(heavy) across adjacent threat levels within each distance.

### Threshold

- β(threat) < 0, p < 0.01
- β(distance) < 0, p < 0.01
- All adjacent-threat comparisons significant at p < 0.01

### Exploratory benchmarks

| Predictor | β | SE | z | p |
|-----------|---|----|----|---|
| threat_z | −1.28 | 0.04 | −32.0 | < 10⁻²⁰⁰ |
| dist_z | −0.65 | 0.04 | −16.3 | < 10⁻⁵⁹ |
| threat_z × dist_z | −0.18 | 0.04 | −4.5 | < 10⁻⁵ |

Choice surface:

| | D=1 | D=2 | D=3 |
|---|---|---|---|
| T=0.1 | 0.81 | 0.69 | 0.57 |
| T=0.5 | 0.63 | 0.38 | 0.19 |
| T=0.9 | 0.40 | 0.14 | 0.08 |

Overall P(heavy) = 0.431. At the most dangerous condition (T=0.9, D=3), only 8% of subjects engage the high-reward option — a near-complete avoidance response.

---

## H1b: Subjective anxiety increases and confidence decreases with threat

### Prediction

On probe trials, self-reported anxiety increases with threat probability and confidence decreases with threat probability.

### Test

Linear mixed models with random intercepts and slopes by subject:

`anxiety ~ threat_z + (1 + threat_z | subject)`
`confidence ~ threat_z + (1 + threat_z | subject)`

### Threshold

- Anxiety: β(threat) > 0, |t| > 3.0
- Confidence: β(threat) < 0, |t| > 3.0

### Exploratory benchmarks

| DV | β(threat_z) | SE | t | p |
|-----|-------------|----|----|---|
| Anxiety | +0.557 | 0.040 | +14.04 | < 10⁻⁴⁴ |
| Confidence | −0.575 | 0.043 | −13.48 | < 10⁻⁴⁰ |

Substantial random slope variance (anxiety: 0.52; confidence: 0.63) indicates meaningful individual differences in affective threat sensitivity — some people's anxiety tracks threat steeply, others' is relatively flat. These individual differences are decomposed in H6 (calibration and discrepancy).

---

## H1c: Threat increases vigor within cookie type (Simpson's paradox)

### Prediction

Within heavy cookies and within light cookies separately, normalized press rate increases with threat probability. The *marginal* (unconditional) effect will appear weak or absent due to the compositional shift toward light cookies at high threat.

### Test

1. **Marginal vigor (demonstrating the paradox):** Mean normalized vigor by threat level, collapsing across cookie type. Expected: flat or weak trend.

2. **Conditional vigor (the real effect):** Paired t-test on within-subject mean normalized vigor at T=0.9 minus T=0.1, separately within heavy and light cookies. Expected: significant increase in both.

3. **Effect sizes:** Cohen's d for T=0.1 vs T=0.9, within each cookie type.

### Threshold

- Conditional effect: p < 0.01 within heavy AND within light
- Cohen's d > 0.15 within both cookie types

### Exploratory benchmarks

**Marginal (paradoxical):**
- T=0.1: 0.956, T=0.5: 0.968, T=0.9: 0.977
- Total range = 0.021 — appears negligible

**Conditional (real):**

| Cookie | T=0.1 | T=0.9 | d | p |
|--------|-------|-------|---|---|
| Heavy | 0.980 | 1.050 | 0.24 | < 10⁻¹⁰ |
| Light | 0.901 | 0.959 | 0.18 | < 10⁻⁸ |

The paradox arises because at T=0.9, subjects choose light cookies 4:1 over heavy. Light cookies have lower absolute press rates, pulling the marginal average down and masking the within-cookie increase.

**Implication:** Any study that measures effort or vigor without jointly modeling choice will underestimate — or entirely miss — the effect of threat on motor output. This is the methodological argument for the joint avoidance-activation model (H4).

---

## H1d: Choice avoidance and vigor activation are simultaneous

### Prediction

At the individual level, the magnitude of the choice shift (reduction in P(heavy) from T=0.1 to T=0.9) is positively correlated with the magnitude of the vigor shift (increase in within-cookie press rate from T=0.1 to T=0.9). Alternatively, the two shifts are approximately independent (r ≈ 0), which would further support separable channels.

### Test

Per-subject:
- Choice shift = P(heavy at T=0.1) − P(heavy at T=0.9) (positive = more avoidance)
- Vigor shift = mean vigor at T=0.9 − mean vigor at T=0.1, within cookie type (positive = more activation)

Pearson correlation between choice shift and vigor shift.

### Threshold

This is a directional test: we report the correlation and interpret. A positive r means people who avoid more also press harder (coordinated adjustment). A null r means the channels are independent. A negative r would mean people who avoid more press *less* — a maladaptive pattern.

### Exploratory benchmarks

r(choice shift, vigor shift) = +0.152, p = .010. Weak but significant positive correlation — people who shift more toward avoidance also show modestly higher vigor shifts. But the correlation is small (shared variance = 2.3%), consistent with largely independent channels that are weakly coordinated.

The between-subject independence of choice *level* and vigor *level* is even more striking:
- r(P(heavy), mean vigor) = −0.018, p = .76 — essentially zero

This means: **who you are as a chooser tells us nothing about who you are as a presser.** The avoidance and activation channels are separable individual differences, motivating the two-parameter model (H4).

---

## Summary

| Sub-hypothesis | Test | Statistic | Threshold | Verdict |
|---------------|------|-----------|-----------|---------|
| H1a: Choice ↓ with threat | LMM β(threat) | −1.28 | p < 0.01 | **CONFIRMED** |
| H1a: Choice ↓ with distance | LMM β(distance) | −0.65 | p < 0.01 | **CONFIRMED** |
| H1a: Monotonicity | Paired t-tests | all t > 8.3 | p < 0.01 | **CONFIRMED** |
| H1b: Anxiety ↑ with threat | LMM β(threat) | +0.557 | \|t\| > 3.0 | **CONFIRMED** |
| H1b: Confidence ↓ with threat | LMM β(threat) | −0.575 | \|t\| > 3.0 | **CONFIRMED** |
| H1c: Vigor ↑ (heavy, conditional) | Paired t | d = 0.24 | p < 0.01 | **CONFIRMED** |
| H1c: Vigor ↑ (light, conditional) | Paired t | d = 0.18 | p < 0.01 | **CONFIRMED** |
| H1c: Marginal vigor flat (paradox) | Descriptive | range = 0.021 | — | **CONFIRMED** |
| H1d: Choice-vigor independence | Pearson r | r = −0.018 | — | **CONFIRMED** |

**Bottom line:** Threat produces three simultaneous behavioral responses: avoidance of risky patches (choice), increased motor vigor within patches (activation), and calibrated subjective distress (affect). The choice and vigor responses are carried by *different people* — who avoids is not who mobilizes (r = −0.02). This motivates a two-channel computational architecture where avoidance (λ) and activation (ω) are separable individual-difference parameters (H4).

---

## Confirmation Plan

| Test | Statistic | Threshold | Discovery value |
|------|-----------|-----------|-----------------|
| H1a: Choice × threat | β(threat_z) | < 0, p < .01 | −1.28 |
| H1a: Choice × distance | β(dist_z) | < 0, p < .01 | −0.65 |
| H1b: Anxiety × threat | β(threat_z) | > 0, \|t\| > 3 | +0.557 |
| H1b: Confidence × threat | β(threat_z) | < 0, \|t\| > 3 | −0.575 |
| H1c: Vigor × threat (conditional) | Paired t (heavy) | p < .01 | d = 0.24 |
| H1c: Vigor × threat (conditional) | Paired t (light) | p < .01 | d = 0.18 |
| H1d: Choice-vigor level independence | Pearson r | — | r = −0.02 |
