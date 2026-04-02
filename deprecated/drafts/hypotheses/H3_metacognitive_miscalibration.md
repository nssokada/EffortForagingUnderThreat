# H3: Confidence Reflects Foraging Ambition, Not Task Performance

## Results from Discovery Sample (N = 293)

---

## Overview

This hypothesis tests whether between-subject confidence is associated with foraging ambition (willingness to pursue heavy cookies) or with task performance (EV-optimal choice quality, survival rate). The EVC model identifies two independent behavioral dimensions — effort cost (ce, governing choice) and capture aversion (cd, governing vigor) — which are nearly uncorrelated (r = −0.14). If confidence tracks the tendency to choose ambitiously but not the quality of those choices or their survival outcomes, this suggests confidence reflects a subjective sense of capability rather than objective accuracy.

---

## Methods

### Measures

**Mean confidence.** Each subject's average confidence rating across their 18 confidence probe trials (0–7 scale). Probes were collected after choice commitment but before motor execution, making them prospective judgments about the current trial's demands.

**P(choose heavy).** Proportion of a subject's 45 choice trials where they selected the heavy (high-effort, high-reward) cookie. This reflects foraging ambition — willingness to pursue costly, distant options.

**Choice quality.** Proportion of a subject's 45 choice trials where they selected the EV-maximizing option. EV-optimal was defined as the option with higher objective expected payoff: E[payoff] = S × R + (1−S) × (−C), computed separately for heavy and light options using empirical conditional survival rates.

**Survival rate.** 1 − (proportion of a subject's choice trials where they were captured by the predator). This reflects the joint outcome of choice (which option was selected) and vigor (how hard they pressed to escape).

### Statistical Tests

**H3a:** Pearson r between mean confidence and P(choose heavy). Test: r > 0, p < 0.05, one-tailed.

**H3b:** Pearson r between mean confidence and choice quality (EV-optimal). Test: |r| < 0.10 (i.e., no meaningful association).

**H3c:** Pearson r between mean confidence and survival rate. Test: |r| < 0.10 (i.e., no meaningful association).

### Rationale

This analysis does not depend on the computational model — it uses raw behavioral measures. The hypothesis was motivated by the discovery-sample finding that between-subject confidence does not predict task performance (confidence × choice quality: r = −0.08; confidence × survival: r = −0.05), despite strong within-subject tracking of survival by affect.

---

## Results

### H3a: Confidence Correlates with P(Choose Heavy)

**Prediction:** r(mean confidence, P(choose heavy)) > 0, p < 0.05

| Measure 1 | Measure 2 | r | p | N |
|-----------|-----------|---|---|---|
| Mean confidence | P(choose heavy) | **0.25** | **< 0.001** | 293 |

Subjects who felt more confident about their foraging performance chose the heavy (high-effort, high-reward) cookie more often.

**Interpretation:** Confident participants are more willing to commit to effortful foraging bouts. This reflects ambition — a willingness to engage with costly options — rather than calibrated accuracy.

**Verdict: CONFIRMED** (r = 0.25, p < 0.001)

---

### H3b: Confidence Does NOT Correlate with Choice Quality

**Prediction:** |r(mean confidence, choice quality)| < 0.10

| Measure 1 | Measure 2 | r | p | N |
|-----------|-----------|---|---|---|
| Mean confidence | Choice quality (EV-optimal) | **−0.08** | **0.16** | 293 |

Confidence is unrelated to the quality of foraging decisions. Subjects who felt more confident did NOT make more EV-optimal choices.

**Interpretation:** Choosing heavy is not the same as choosing optimally. At high threat/distance, the light cookie is often EV-optimal, but confident subjects choose heavy anyway. Confidence tracks the tendency to commit to effortful options, not the accuracy of those commitments.

**Verdict: CONFIRMED** (|−0.08| < 0.10)

---

### H3c: Confidence Does NOT Correlate with Survival

**Prediction:** |r(mean confidence, survival rate)| < 0.10

| Measure 1 | Measure 2 | r | p | N |
|-----------|-----------|---|---|---|
| Mean confidence | Survival rate | **−0.05** | **0.41** | 293 |

Confidence is unrelated to actual survival. Subjects who felt more confident did NOT survive more often.

**Interpretation:** Survival depends primarily on vigor (how hard subjects pressed), which is governed by cd. Since cd is independent of ce (r = −0.14), and confidence tracks the ce-related tendency to choose ambitiously, confidence carries no information about the cd-driven survival outcome.

**Verdict: CONFIRMED** (|−0.05| < 0.10)

---

### Relationship to Earnings

To underscore why this pattern matters, we report the relative importance of choice versus survival for foraging outcomes:

| Predictor | r with total earnings | Interpretation |
|-----------|---------------------|----------------|
| Survival rate | **0.846** | Surviving determines earnings |
| Choice quality | 0.430 | Choosing well helps but matters less |
| P(choose heavy) | 0.098 | Foraging ambition doesn't predict earnings |
| Mean confidence | 0.098 | Confidence doesn't predict earnings |

Survival is 2× more predictive of earnings than choice quality. Yet confidence tracks ambition (r = 0.25), not quality (r = −0.08) or survival (r = −0.05). Participants feel confident about how ambitiously they forage, not how well they perform.

---

## Theoretical Interpretation

This finding identifies a specific pattern: confidence reflects foraging ambition rather than foraging accuracy. Confident participants are more willing to pursue costly, high-reward options — but this ambition does not translate into better decisions or better survival outcomes.

The EVC model provides the mechanistic explanation: ce and cd are independent parameters (r = −0.14) identified from different behavioral channels (choice vs vigor). Confidence appears to reflect the subjective experience of low effort cost (willingness to engage with demanding options) rather than high capture aversion (investment in motor execution). This is consistent with the idea that prospective confidence judgments — collected before motor execution — are most sensitive to the decision stage of the task, which is governed by ce.

The practical consequence is that confidence is an unreliable signal of task competence. The association between confidence and ambition without a corresponding association with performance is consistent with the broader metacognitive literature showing that confidence is often better calibrated to decision processes than to outcome processes.

---

## Confirmation Plan

### Tests to run on confirmation sample (N ≈ 280–330):

| Test | Statistic | Threshold | Expected |
|------|-----------|-----------|----------|
| H3a: r(confidence, P(heavy)) | Pearson r | > 0, p < 0.05 | r ≈ 0.25 |
| H3b: r(confidence, choice quality) | Pearson r | |r| < 0.10 | r ≈ −0.08 |
| H3c: r(confidence, survival rate) | Pearson r | |r| < 0.10 | r ≈ −0.05 |

### Power analysis

At N = 280 (conservative estimate after exclusions):
- H3a: power > 0.99 to detect r = 0.25 at α = 0.05

---

## Summary

| Sub-hypothesis | Test | Statistic | p | Verdict |
|---------------|------|-----------|---|---------|
| H3a: Conf × P(heavy) > 0 | Pearson r | r = 0.25 | < 0.001 | **CONFIRMED** |
| H3b: Conf × choice quality ≈ 0 | Pearson r | r = −0.08 | 0.16 | **CONFIRMED** |
| H3c: Conf × survival ≈ 0 | Pearson r | r = −0.05 | 0.41 | **CONFIRMED** |

**Bottom line:** Confidence tracks how ambitiously you forage (P(choose heavy)), not how well you forage (choice quality, survival). Confident people choose heavy — but choosing heavy is not the same as choosing well.
