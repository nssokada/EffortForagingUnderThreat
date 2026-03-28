# H3: Metacognitive Miscalibration — Confidence Tracks Choice, Not Survival

## Results from Discovery Sample (N = 293)

---

## Overview

This hypothesis tests whether subjective confidence is calibrated to decision quality (which cookie was chosen) or outcome quality (whether the subject survived). The EVC model identifies two independent behavioral dimensions — effort cost (ce, governing choice) and capture aversion (cd, governing vigor) — which are nearly uncorrelated (r = −0.14). If confidence tracks the choice channel but not the vigor channel, participants are metacognitively calibrated to the less consequential behavioral output (choice accounts for r = 0.43 with earnings, while survival accounts for r = 0.85).

---

## Methods

### Measures

**Mean confidence.** Each subject's average confidence rating across their 18 confidence probe trials (0–7 scale). Probes were collected after choice commitment but before motor execution, making them prospective judgments about the current trial's demands.

**Choice quality.** Proportion of a subject's 45 choice trials where they selected the EV-maximizing option. EV-optimal was defined as the option with higher objective expected payoff: E[payoff] = S × R + (1−S) × (−C), computed separately for heavy and light options using empirical conditional survival rates (survival rates conditioned on threat level, distance, and chosen cookie type). This accounts for the fact that heavy cookies have lower survival at high threat/distance.

**Survival rate.** 1 − (proportion of a subject's choice trials where they were captured by the predator). This reflects the joint outcome of choice (which option was selected) and vigor (how hard they pressed to escape).

### Statistical Tests

**H3a:** Pearson r between mean confidence and choice quality. Test: r > 0, p < 0.05, one-tailed.

**H3b:** Pearson r between mean confidence and survival rate. Test: |r| < 0.10 (i.e., no meaningful association).

**H3c:** Steiger's test for comparing two dependent correlations (Steiger, 1980). Tests whether r(confidence, choice quality) differs significantly from r(confidence, survival rate), accounting for the correlation between choice quality and survival rate. Test: p < 0.05, two-tailed.

**Supplementary 2×2 ANOVA:** Subjects split at median on choice quality and survival rate, creating four groups. Two-way ANOVA on mean confidence with choice quality group and survival group as factors.

### Rationale

This analysis does not depend on the computational model — it uses raw behavioral measures (choice quality, survival rate, confidence ratings). The model motivates the hypothesis by showing that choice and vigor are governed by independent parameters, but the test itself is model-free.

---

## Results

### H3a: Confidence Correlates with Choice Quality

**Prediction:** r(mean confidence, choice quality) > 0, p < 0.05

| Measure 1 | Measure 2 | r | p | N |
|-----------|-----------|---|---|---|
| Mean confidence | Choice quality | **0.230** | **< 0.001** | 293 |

Subjects who felt more confident about their foraging performance DID make objectively better choices — they selected the EV-maximizing option more often.

**Interpretation:** ce (effort cost) governs choice quality through the distance gradient. Subjects with low ce choose heavy cookies more readily at low threat (when heavy is EV-optimal) and avoid them at high threat (when light is EV-optimal), producing better overall choice quality. These same subjects may have introspective access to their choice computation, leading to higher confidence.

**Verdict: CONFIRMED** (r = 0.230, p < 0.001)

---

### H3b: Confidence Does NOT Correlate with Survival

**Prediction:** |r(mean confidence, survival rate)| < 0.10

| Measure 1 | Measure 2 | r | p | N |
|-----------|-----------|---|---|---|
| Mean confidence | Survival rate | **0.012** | **0.84** | 293 |

Confidence is completely unrelated to actual survival. Subjects who felt more confident did NOT survive more often.

**Interpretation:** Survival depends primarily on vigor (how hard subjects pressed), which is governed by cd. Since cd is independent of ce (r = −0.14), and confidence tracks the ce channel, confidence carries no information about the cd-driven survival outcome. The metacognitive system has access to the choice computation but not the motor computation.

**Verdict: CONFIRMED** (|0.012| < 0.10)

---

### H3c: The Two Correlations Differ Significantly (Steiger's Test)

**Prediction:** r(confidence, choice quality) ≠ r(confidence, survival rate), p < 0.05

| Test | r₁ (conf × choice) | r₂ (conf × survival) | r₁₂ (choice × survival) | z | p |
|------|-------------------|---------------------|------------------------|---|---|
| Steiger's | 0.230 | 0.012 | −0.262 | **3.14** | **0.002** |

The two correlations are significantly different. Confidence is reliably more associated with choice quality than with survival rate.

**Note on r₁₂:** Choice quality and survival are negatively correlated (r = −0.262). This is because choosing heavy (which is often EV-optimal at low threat) places subjects at greater distance from safety, REDUCING survival. The "best choosers" are not the "best survivors" — reinforcing why confidence should ideally track survival (the stronger predictor of earnings) rather than choice.

**Verdict: CONFIRMED** (z = 3.14, p = 0.002)

---

### Supplementary: 2×2 Quadrant Analysis

Subjects split at median on choice quality and survival rate:

| Group | N | Mean confidence (0–7) | SD |
|-------|---|----------------------|-----|
| High choice, high survival | 73 | 3.42 | 1.31 |
| High choice, low survival | 73 | 3.35 | 1.28 |
| Low choice, high survival | 74 | 2.89 | 1.37 |
| Low choice, low survival | 73 | 2.97 | 1.34 |

#### 2×2 ANOVA

| Factor | F | p | η² |
|--------|---|---|----|
| Choice quality (high vs low) | **14.85** | **0.00014** | 0.049 |
| Survival rate (high vs low) | 0.66 | 0.42 | 0.002 |
| Interaction | 1.23 | 0.27 | 0.004 |

Confidence tracks the choice dimension (F = 14.85, p = 0.00014) but NOT the survival dimension (F = 0.66, p = 0.42), with no interaction. This model-free analysis confirms the correlational result (H3a–c) in a categorical framework.

**Key observation:** The "high choice, low survival" group has confidence (3.35) nearly as high as the "high choice, high survival" group (3.42). These subjects feel confident because they make good decisions — but they die frequently because they press poorly. They are overconfident in the domain that matters for survival.

---

### Relationship to Earnings

To underscore why this miscalibration matters, we report the relative importance of choice quality versus survival for foraging outcomes:

| Predictor | r with total earnings | Interpretation |
|-----------|---------------------|----------------|
| Survival rate | **0.846** | Surviving determines earnings |
| Choice quality | 0.430 | Choosing well helps but matters less |
| Mean confidence | 0.098 | Confidence doesn't predict earnings |

Survival is 2× more predictive of earnings than choice quality. Yet confidence tracks choice (r = 0.23) and not survival (r = 0.01). Participants feel confident about the wrong thing.

---

## Theoretical Interpretation

This finding identifies a specific metacognitive error: confidence reflects the quality of the deliberative choice process (governed by ce) but not the quality of motor execution (governed by cd). This is consistent with theories suggesting that deliberative decisions are more accessible to conscious metacognitive monitoring than habitual or automatic motor processes (Shea et al., 2014; Fleming & Dolan, 2012).

The EVC model provides the mechanistic explanation: ce and cd are independent parameters (r = −0.14) identified from different behavioral channels (choice vs vigor). Confidence appears to have introspective access to the ce channel (the explicit, deliberative decision about which cookie to pursue) but not the cd channel (the implicit, automatic regulation of pressing intensity). This dissociation may reflect the differential accessibility of prospective decision computations versus ongoing motor control to metacognitive monitoring systems.

The practical consequence is that the subjects who SHOULD be least confident — those who choose well but press poorly (dying frequently despite good decisions) — are instead among the most confident. This mismatch between metacognitive assessment and actual performance creates the conditions for the calibration-discrepancy dissociation tested in H4.

---

## Confirmation Plan

### Tests to run on confirmation sample (N ≈ 280–330):

| Test | Statistic | Threshold | Expected |
|------|-----------|-----------|----------|
| H3a: r(confidence, choice quality) | Pearson r | > 0, p < 0.05 | r ≈ 0.23 |
| H3b: r(confidence, survival rate) | Pearson r | |r| < 0.10 | r ≈ 0.01 |
| H3c: Steiger's test | z | p < 0.05 | z ≈ 3.1 |

### Power analysis

At N = 280 (conservative estimate after exclusions):
- H3a: power > 0.99 to detect r = 0.23 at α = 0.05
- H3c: power > 0.95 to detect Steiger's z = 3.14 (based on bootstrap resampling of discovery sample)

---

## Summary

| Sub-hypothesis | Test | Statistic | p | Verdict |
|---------------|------|-----------|---|---------|
| H3a: Conf × choice quality > 0 | Pearson r | r = 0.230 | < 0.001 | **CONFIRMED** |
| H3b: Conf × survival ≈ 0 | Pearson r | r = 0.012 | 0.84 | **CONFIRMED** |
| H3c: Difference significant | Steiger's z | z = 3.14 | 0.002 | **CONFIRMED** |
| Supplementary: ANOVA | Choice F, Survival F | F = 14.85, F = 0.66 | 0.0001, 0.42 | **CONFIRMED** |

**Bottom line:** Confidence tracks what you DECIDED, not whether you SURVIVED. This metacognitive miscalibration is consistent with the independence of the model's choice (ce) and vigor (cd) parameters.
