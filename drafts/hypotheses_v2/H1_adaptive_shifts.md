# H1: Threat Drives Adaptive Shifts in Choice, Vigor, and Affect

## Overview

Increasing predation risk drives adaptive shifts across three behavioral channels: foragers shift toward safer patches (avoidance), press harder within each patch type (activation), and report increased anxiety and decreased confidence. These responses are simultaneous and superficially contradictory — threat makes people "lazier" in their choices but "harder-working" in their actions — motivating the two-channel computational architecture tested in H3.

### Theoretical grounding

The starvation-predation tradeoff (Lima & Dill 1990) predicts that foragers under threat should reduce exposure to dangerous patches. Escape theory (Ydenberg & Dill 1986) predicts that committed foragers should increase motor effort to maximize survival. These apply to different stages of the foraging bout — the decision of what to engage versus the execution once committed.

---

## H1a: Threat and distance deter high-effort patch engagement

### Prediction

P(choose heavy cookie) decreases with threat probability T and escape distance D.

### Test

Logistic model with cluster-robust SE: `choice ~ threat_z + dist_z + threat_z:dist_z`, clustered by subject.

### Threshold

β(threat) < 0 AND β(distance) < 0, both p < 0.01.

### Exploratory benchmarks

- threat_z: β = −1.02, p < 10⁻¹¹⁰
- dist_z: β = −0.75, p < 10⁻¹²⁴
- threat_z × dist_z: β = −0.20, p < 10⁻¹⁵

---

## H1b: Anxiety increases and confidence decreases with threat

### Prediction

Self-reported anxiety increases and confidence decreases with threat probability.

### Test

Linear mixed models with random intercepts and slopes by subject:
- `anxiety ~ threat_z + (1 + threat_z | subject)`
- `confidence ~ threat_z + (1 + threat_z | subject)`

### Threshold

Anxiety: β > 0, |t| > 3.0. Confidence: β < 0, |t| > 3.0.

### Exploratory benchmarks

- Anxiety: β = +0.580, z = 14.7, p < 10⁻⁴⁸
- Confidence: β = −0.582, z = −13.7, p < 10⁻⁴³

---

## H1c: Threat increases vigor within cookie type

### Prediction

Within heavy and within light cookies separately, normalized press rate increases with threat. All vigor analyses condition on cookie type.

### Test

Paired t-test: within-subject mean normalized press rate at T=0.9 minus T=0.1, separately within heavy and light cookies.

### Threshold

p < 0.01 within both cookie types, d > 0.15.

### Exploratory benchmarks

| Cookie | T=0.1 | T=0.9 | d | p |
|--------|-------|-------|---|---|
| Heavy | 0.980 | 1.050 | 0.24 | < 10⁻¹⁰ |
| Light | 0.901 | 0.959 | 0.18 | < 10⁻⁸ |

---

## Confirmation Plan

| Test | Statistic | Threshold | Discovery value |
|------|-----------|-----------|-----------------|
| H1a: Choice × threat | β(threat_z) | < 0, p < .01 | −1.02 |
| H1a: Choice × distance | β(dist_z) | < 0, p < .01 | −0.75 |
| H1b: Anxiety × threat | β(threat_z) | > 0, |t| > 3 | +0.580 |
| H1b: Confidence × threat | β(threat_z) | < 0, |t| > 3 | −0.582 |
| H1c: Vigor × threat (heavy) | Paired t | p < .01, d > .15 | d = 0.24 |
| H1c: Vigor × threat (light) | Paired t | p < .01, d > .15 | d = 0.18 |
