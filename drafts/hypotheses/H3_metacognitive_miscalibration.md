# H3: Metacognitive Miscalibration

## Hypothesis

Subjective confidence will track choice quality (whether the subject chose the EV-maximizing option) but NOT motor execution quality (whether the subject survived). This represents a systematic metacognitive miscalibration: people feel confident about their decisions rather than their outcomes.

---

## Motivation

The EVC model predicts both choice (via ce) and vigor (via cd) from a shared survival computation. However, these two parameters are nearly independent (r = −0.14), meaning knowing how someone decides tells you almost nothing about how hard they'll press. If confidence tracks the choice channel but not the vigor channel, participants are metacognitively calibrated to the wrong behavioral output — the one that matters less for survival.

This matters because:
- Survival → earnings: r = 0.846 (survival is the primary determinant of outcomes)
- Choice optimality → earnings: r = 0.430 (choice matters, but 2× less)

People SHOULD calibrate confidence to vigor (which determines survival), but they calibrate it to choice (which determines less).

---

## Definitions

**Choice quality:** Proportion of a subject's 45 choice trials where they selected the EV-maximizing option. Computed by comparing observed choice to the option with higher objective expected payoff (S×R − (1−S)×C for each option, using empirical conditional survival rates).

**Survival rate:** 1 − (proportion of trials where captured). Reflects the combined effect of choice (which option was selected) and vigor (how hard they pressed to escape).

**Mean confidence:** Average of a subject's 18 confidence probe ratings (0–7 scale).

---

## H3a: Confidence Correlates with Choice Quality

**Test:** Pearson r(mean confidence, choice quality) > 0, p < 0.05

### Results

- r = 0.230, p < .001
- Subjects who felt more confident DID make better choices

This makes sense: ce (effort cost) drives both confidence and choice quality. Low-ce subjects choose heavy cookies more readily (appropriate at T=0.1) and presumably feel confident about those choices.

**Verdict: CONFIRMED**

---

## H3b: Confidence Does NOT Correlate with Survival

**Test:** |r(mean confidence, survival rate)| < 0.10

### Results

- r = 0.012, p = 0.84
- Confidence is completely unrelated to actual survival

This is the key finding. Despite survival being the primary determinant of foraging success, participants who felt more confident did not actually survive more often. Their confidence reflected their CHOICES (governed by ce), not their EXECUTION (governed by cd).

**Verdict: CONFIRMED** (|0.012| < 0.10)

---

## H3c: Steiger's Test

**Test:** The difference between r(confidence, choice quality) and r(confidence, survival) is statistically significant, p < 0.05.

### Results

- r₁ = 0.230 (confidence × choice quality)
- r₂ = 0.012 (confidence × survival)
- r₁₂ = −0.262 (choice quality × survival — negative because choosing heavy HURTS survival at high threat)
- Steiger's z = 3.14, p = 0.002

The two correlations are significantly different. Confidence is reliably more associated with choice quality than with survival.

**Verdict: CONFIRMED**

---

## 2×2 Quadrant Analysis

To visualize the miscalibration, we split subjects at the median on choice quality and survival rate, creating four groups:

| Group | Choice | Survival | N | Mean confidence |
|-------|--------|----------|---|-----------------|
| Good chooser, good survivor | High | High | ~73 | Highest |
| Good chooser, bad survivor | High | Low | ~73 | High |
| Bad chooser, good survivor | Low | High | ~73 | Low |
| Bad chooser, bad survivor | Low | Low | ~73 | Lowest |

2×2 ANOVA:
- Choice quality main effect: F = 14.85, p = 0.00014
- Survival rate main effect: F = 0.66, p = 0.42
- Interaction: not significant

Confidence tracks the choice dimension (F = 14.85) but NOT the survival dimension (F = 0.66). This confirms the Steiger test result in a model-free way.

---

## Why This Happens

The model provides a mechanistic explanation. Choice and vigor are governed by independent parameters (ce and cd, r = −0.14). Confidence appears to reflect introspective access to the choice computation (ce channel) but not the motor computation (cd channel). This is consistent with metacognitive theories suggesting that deliberative decisions are more accessible to conscious awareness than motor execution (Shea et al., 2014).

The practical consequence: the "overconfident" subjects (high confidence, low survival) are those with low ce (they choose ambitiously) but high cd is uncorrelated with their confidence. They feel good about their choices without knowing they press poorly. Conversely, "underconfident" subjects (low confidence, high survival) are cautious choosers who happen to be vigorous pressers — the most successful foragers who don't know it.

---

## Connection to Clinical Findings

The metacognitive miscalibration identified here is domain-specific (choice vs vigor), but it parallels the broader calibration-discrepancy decomposition in H4. If confidence tracks the "wrong" channel, then the relationship between subjective affect and objective threat is imperfect — creating the space for the discrepancy that predicts clinical symptoms.

---

## Summary

| Sub-hypothesis | Test | Result | Statistic |
|---------------|------|--------|-----------|
| H3a: Conf × choice quality | r > 0, p < .05 | **CONFIRMED** | r = 0.230, p < .001 |
| H3b: Conf × survival | |r| < 0.10 | **CONFIRMED** | r = 0.012, p = .84 |
| H3c: Steiger's test | p < .05 | **CONFIRMED** | z = 3.14, p = .002 |

Confidence tracks what you DECIDED, not whether you SURVIVED. This metacognitive miscalibration is a central finding of the paper.
