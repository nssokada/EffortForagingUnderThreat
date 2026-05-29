# H1: Threat Drives Adaptive Shifts in Choice, Vigor, and Affect

## Preregistered prediction

Threat probability and distance will simultaneously reduce high-effort choices, increase motor vigor (within cookie type), increase anxiety, and decrease confidence.

## Tests and thresholds

| Test | Model | Threshold |
|------|-------|-----------|
| H1a: Threat → choice | Logistic, cluster-robust SE | β(threat) < 0, β(distance) < 0, both P < .01 |
| H1b: Threat/distance → anxiety | LMM: response ~ threat_z + dist_z + (1 + threat_z \| subject) | β(threat) > 0, β(dist) > 0, both \|t\| > 3 |
| H1b: Threat/distance → confidence | LMM: response ~ threat_z + dist_z + (1 + threat_z \| subject) | β(threat) < 0, β(dist) < 0, both \|t\| > 3 |
| H1c: Threat → vigor | Paired t: T=0.9 vs T=0.1, within heavy and light | Both P < .01 |

## Results

### H1a: Choice

Logistic regression with cluster-robust standard errors: choice ~ threat_z + dist_z + threat_z:dist_z.

|  | Exploratory (N = 290) | Confirmatory (N = 281) |
|--|----------------------|----------------------|
| β(threat) | −1.02 (z = −22.3, P < 0.001) | −0.91 (z = −19.8, P < 0.001) |
| β(distance) | −0.75 (z = −23.7, P < 0.001) | −0.67 (z = −22.1, P < 0.001) |
| β(threat × distance) | −0.20 (z = −8.0, P < 0.001) | −0.12 (z = −4.7, P < 0.001) |

Both threat and distance reduce P(heavy). The interaction indicates that at high threat, distance has an even stronger deterrent effect. **Confirmed in both samples.**

### H1b: Affect

Linear mixed models: response ~ threat_z + dist_z + (1 + threat_z | subject), fit separately for anxiety and confidence.

|  | Exploratory (N = 290) | Confirmatory (N = 281) |
|--|----------------------|----------------------|
| Anxiety: β(threat) | +0.580 (z = +14.67, P < 0.001) | +0.534 (z = +12.51, P < 0.001) |
| Anxiety: β(distance) | +0.230 (z = +9.93, P < 0.001) | +0.276 (z = +12.22, P < 0.001) |
| Confidence: β(threat) | −0.582 (z = −13.72, P < 0.001) | −0.671 (z = −15.25, P < 0.001) |
| Confidence: β(distance) | −0.295 (z = −12.84, P < 0.001) | −0.260 (z = −11.25, P < 0.001) |

All |z| > 3 with signs in the predicted direction for both threat and distance. Anxiety and confidence respond in opposite directions to each manipulation — threat and distance both raise subjective danger appraisal while lowering coping appraisal. **Confirmed in both samples.**

### H1c: Vigor

Within each cookie type, normalized press rate at T = 0.9 minus T = 0.1:

| Cookie | Exploratory d | Confirmatory d | Confirmatory P |
|--------|--------------|----------------|----------------|
| Heavy | +0.24 | +0.45 | < 0.001 |
| Light | +0.44 | +0.76 | < 0.001 |

The vigor effect is larger in the confirmatory sample. The within-cookie analysis controls for the compositional shift (threat changes which cookie is chosen, which changes average vigor). **Confirmed in both samples.**

## Summary

| Test | Exploratory | Confirmatory |
|------|-------------|--------------|
| H1a: threat → choice | PASS | PASS |
| H1a: distance → choice | PASS | PASS |
| H1b: threat → anxiety | PASS | PASS |
| H1b: distance → anxiety | PASS | PASS |
| H1b: threat → confidence | PASS | PASS |
| H1b: distance → confidence | PASS | PASS |
| H1c: threat → vigor | PASS | PASS |
| **Total** | **7/7** | **7/7** |

## Interpretation

Threat simultaneously drives three adaptive responses: avoidance (reduce risky choices), activation (press harder to escape faster), and calibrated affect (feel more anxious, less confident). These three channels are predicted by the same fitness function W(u) but emerge through different mechanisms — choice through the value comparison V_H vs V_L, vigor through the optimal pressing rate u*, and affect through the metacognitive monitoring system tested in H5.
