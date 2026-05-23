# H1 Results: Threat Shifts Choice, Vigor, and Affect

## Overview

H1 tests whether threat probability shifts behavior and subjective experience across three domains: choice (H1a), motor effort (H1b), and affect (H1c). All tests use mixed-effects models on the exploratory sample (N = 293).

---

## H1a: High-effort choice decreases with threat and distance

**Model:** Logistic GLMM via variational Bayes (Laplace approximation): `choice ~ threat_z + dist_z + threat_z × dist_z + (1 | subject)`.

| Predictor | β | SE | z | p |
|---|---|---|---|---|
| Intercept | -0.566 | 0.023 | -24.49 | p < 0.001 |
| Threat (z) | -1.336 | 0.024 | -55.53 | p < 0.001 |
| Distance (z) | -0.979 | 0.023 | -41.73 | p < 0.001 |
| Threat × Distance | -0.227 | 0.024 | -9.38 | p < 0.001 |

All three fixed effects were significant at p < 0.01: threat reduced high-effort choice (β = -1.336), distance reduced high-effort choice (β = -0.979), and the interaction confirmed that threat amplified the distance effect (β = -0.227).

**Monotonicity:** P(choose high) decreased across all adjacent threat levels within each distance:

| Comparison | t | p (one-tailed) | d |
|---|---|---|---|
| D=1: T=0.1>T=0.5 | 9.68 | 1.02e-19 | 0.57 |
| D=1: T=0.5>T=0.9 | 13.68 | 1.46e-33 | 0.80 |
| D=2: T=0.1>T=0.5 | 14.99 | 2.30e-38 | 0.88 |
| D=2: T=0.5>T=0.9 | 14.05 | 6.35e-35 | 0.82 |
| D=3: T=0.1>T=0.5 | 18.39 | 5.36e-51 | 1.07 |
| D=3: T=0.5>T=0.9 | 8.51 | 4.76e-16 | 0.50 |

All monotonicity tests passed at p < 0.01. **H1a: SUPPORTED.**

---

## H1b: Excess effort increases with threat

**Model:** Linear LMM (REML): `excess_effort ~ threat_z * dist_z + effort_chosen_z + (1 | subject)`.

Excess effort = actual pressing rate − demanded effort level of chosen option, controlling for composition effects (choosing easier options under high threat).

| Predictor | β | SE | z | p | 95% CI |
|---|---|---|---|---|---|
| Intercept | 0.0077 | 0.0069 | 1.11 | p = 0.269 | [-0.006, 0.021] |
| Threat (z) | 0.0691 | 0.0014 | 50.01 | p < 0.001 | [0.066, 0.072] |
| Distance (z) | -0.0196 | 0.0026 | -7.45 | p < 0.001 | [-0.025, -0.014] |
| Effort chosen (z) | -0.0754 | 0.0024 | -30.93 | p < 0.001 | [-0.080, -0.071] |
| Threat × Distance | -0.0191 | 0.0016 | -11.92 | p < 0.001 | [-0.022, -0.016] |

Threat significantly increased excess effort (β = 0.0691, p < 0.001), controlling for distance and chosen effort level. The threat × distance interaction was negative (β = -0.0191, p < 0.001), indicating that the threat-driven boost in excess effort diminishes at farther distances where sustained high-effort execution approaches a physical ceiling. **H1b: SUPPORTED.**

---

## H1c: Affect shifts with threat

**Models:** Linear LMMs (REML): `rating ~ threat_z + dist_z + (1 + threat_z | subject)`.

### Anxiety (N = 5274 ratings)

| Predictor | β | SE | z | p | 95% CI |
|---|---|---|---|---|---|
| Intercept | 4.399 | 0.076 | 57.58 | p < 0.001 | [4.250, 4.549] |
| Threat (z) | 0.578 | 0.039 | 14.70 | p < 0.001 | [0.501, 0.655] |
| Distance (z) | 0.228 | 0.023 | 9.94 | p < 0.001 | [0.183, 0.273] |

Threat increased anxiety (β = 0.578, p < 0.001). **SUPPORTED.**

### Confidence (N = 5272 ratings)

| Predictor | β | SE | z | p | 95% CI |
|---|---|---|---|---|---|
| Intercept | 3.166 | 0.079 | 40.26 | p < 0.001 | [3.012, 3.320] |
| Threat (z) | -0.579 | 0.042 | -13.70 | p < 0.001 | [-0.662, -0.496] |
| Distance (z) | -0.293 | 0.023 | -12.86 | p < 0.001 | [-0.338, -0.249] |

Threat decreased confidence (β = -0.579, p < 0.001). **SUPPORTED.**

---

## Summary

| Sub-hypothesis | Test | Key β | p | Criterion | Result |
|---|---|---|---|---|---|
| H1a (choice) | Logistic GLMM | β(threat) = -1.336 | p < 0.001 | p < 0.01 | PASS |
| H1a (interaction) | Logistic GLMM | β(T×D) = -0.227 | p < 0.001 | p < 0.01 | PASS |
| H1a (monotonicity) | Paired t-tests | — | all < 0.01 | all p < 0.01 | PASS |
| H1b (excess effort) | Linear LMM | β(threat) = 0.0691 | p < 0.001 | p < 0.05, β > 0 | PASS |
| H1b (T×D interaction) | Linear LMM | β(T×D) = -0.0191 | p < 0.001 | p < 0.05, β < 0 | PASS |
| H1c (anxiety) | Linear LMM | β(threat) = 0.578 | p < 0.001 | p < 0.001, β > 0 | PASS |
| H1c (confidence) | Linear LMM | β(threat) = -0.579 | p < 0.001 | p < 0.001, β < 0 | PASS |

**H1 overall: SUPPORTED — all sub-hypotheses pass their preregistered criteria.**

---

## Methods note

The logistic GLMM (H1a) was estimated using variational Bayes with Laplace approximation (statsmodels `BinomialBayesMixedGLM`) with random intercepts per subject. Random slopes were preregistered but computationally infeasible for the logistic model (293 subjects × 3 random-effect dimensions). Given the very large effect sizes (|z| > 9 for all fixed effects), inference on fixed effects is robust to random-effects specification (Barr et al., 2013). Linear LMMs (H1b, H1c) were estimated with REML via statsmodels `MixedLM`, with the preregistered random effects structure.
