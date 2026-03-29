# H1: Behavioral Effects of Threat on Choice, Vigor, and Affect

## Results from Discovery Sample (N = 293)

---

## Overview

This hypothesis tests whether threat probability and distance modulate three behavioral channels — foraging choice, motor vigor, and subjective affect — in a virtual effort-foraging task under parametric predation risk.

---

## Methods

### Participants

We recruited 350 participants via Prolific.co. After a 5-stage quality control pipeline (task completion, comprehension, behavioral consistency, effort calibration validation, outlier removal), the final discovery sample comprised N = 293 participants (83.7% retention). See Methods document for full exclusion criteria.

### Task

On each of 45 choice trials, participants chose between a heavy cookie (5 pts, requiring pressing at 60–100% of calibrated maximum, distance D ∈ {1, 2, 3}) and a light cookie (1 pt, 40% of maximum, always at D = 1) under threat probability T ∈ {0.1, 0.5, 0.9}. On 36 additional probe trials (forced identical options), participants rated either anxiety or confidence (0–7 scale) before pressing. Trial conditions were fully crossed: 3 threat × 3 distance × 5 repetitions = 45 choice trials; 3 threat × 3 distance × 2 cookie types × 2 rating types = 36 probe trials.

### Behavioral Measures

| Measure | Definition | Type |
|---------|-----------|------|
| Choice | Binary: heavy (1) or light (0) | Trial-level, Bernoulli |
| Excess effort | Median normalized press rate − required rate for chosen cookie, centered by cookie type | Trial-level, continuous |
| Anxiety rating | Self-report (0–7) on anxiety probe trials | Trial-level, ordinal treated as continuous |
| Confidence rating | Self-report (0–7) on confidence probe trials | Trial-level, ordinal treated as continuous |

Cookie-type centering of excess effort: We subtracted the population mean excess for each cookie type (heavy mean = 0.104, light mean = 0.543) to remove the demand confound (light cookies mechanically produce higher raw excess because the demand threshold is lower).

### Statistical Tests

**H1a (Choice):** Logistic mixed-effects model: `choice ~ threat_z + dist_z + threat_z:dist_z + (1 | subject)`, estimated via Laplace approximation in statsmodels. Monotonicity confirmed by paired t-tests on subject-mean P(choose heavy) across adjacent threat levels within each distance (all one-tailed, p < 0.01 threshold).

**H1b (Vigor):** Linear mixed-effects model on cookie-centered excess effort, conditioned on chosen cookie type: `excess_cc ~ threat_z + (1 | subject)`, estimated with REML. The conditioning on cookie type is necessary to resolve a Simpson's paradox (see results below).

**H1c (Affect):** Linear mixed models with random intercepts and slopes by subject: `anxiety ~ S_z + (1 + S_z | subject)` and `confidence ~ S_z + (1 + S_z | subject)`, where S is the EVC model's survival probability: S = (1 − T^0.210) + 0.098 × T^0.210 × p_esc. Estimated with REML in statsmodels.

---

## Results

### H1a: Threat and Distance Reduce High-Effort Choice

**Prediction:** P(choose heavy) will decrease with threat probability and with escape distance.

**Result: CONFIRMED**

#### Choice rates by condition (proportion choosing heavy)

| | D = 1 | D = 2 | D = 3 | Drop D1→D3 |
|---|---|---|---|---|
| **T = 0.1** | 0.808 | 0.692 | 0.565 | −0.243 |
| **T = 0.5** | 0.633 | 0.381 | 0.188 | −0.446 |
| **T = 0.9** | 0.397 | 0.138 | 0.078 | −0.319 |
| **Drop T=0.1→0.9** | −0.411 | −0.554 | −0.487 | |

Overall P(choose heavy) = 0.431 (SD = 0.203 across subjects), N = 13,185 choice trials.

#### Logistic mixed model

| Predictor | β | SE | z | p |
|-----------|---|----|----|---|
| threat_z | −1.28 | 0.04 | −32.0 | < 10⁻²⁰⁰ |
| dist_z | −0.65 | 0.04 | −16.3 | < 10⁻⁵⁹ |
| threat_z × dist_z | −0.18 | 0.04 | −4.5 | < 10⁻⁵ |

All three effects are significant at p < 0.001. Threat is the dominant effect (β = −1.28), distance is secondary (β = −0.65), and the interaction is significant but smaller (β = −0.18), indicating that the distance effect is amplified under moderate threat.

#### Monotonicity tests

All 18 adjacent-threat comparisons (3 distances × 6 ordered pairs) are significant by paired t-test on subject means:

| Comparison | D = 1 | D = 2 | D = 3 |
|-----------|-------|-------|-------|
| T=0.1 > T=0.5 | t = 8.4, p < 10⁻¹⁵ | t = 14.2, p < 10⁻³⁵ | t = 17.0, p < 10⁻⁴⁶ |
| T=0.5 > T=0.9 | t = 11.2, p < 10⁻²⁴ | t = 13.0, p < 10⁻³⁰ | t = 8.3, p < 10⁻¹⁴ |

**Interpretation:** Both threat and distance deter high-effort foraging, with the largest combined effect at T = 0.9, D = 3 (P = 0.078 — only 8% of subjects attempt the high-reward option). The interaction reflects that at low threat (T = 0.1), subjects tolerate distance (still 57% heavy at D = 3), but at medium threat (T = 0.5), distance has a much stronger deterrent effect (63% → 19%).

---

### H1b: Threat Increases Excess Vigor (Within Cookie Type)

**Prediction:** Excess effort, conditioned on chosen cookie type, will increase with threat probability.

**Result: CONFIRMED**

#### Simpson's paradox

Unconditional (marginal) mean excess effort shows a weak threat effect:
- T = 0.1: −0.016, T = 0.5: +0.001, T = 0.9: +0.015
- Marginal effect size: d = 0.28

This weak marginal effect is an artifact of choice reallocation. Under high threat, subjects shift from heavy to light cookies. Light cookies have lower required rates (0.4 vs 0.9 calMax), which mechanically lowers the denominator of the excess computation. Collapsing across choice masks the within-choice vigor increase. This is a textbook Simpson's paradox: the marginal trend is attenuated or reversed by a confounding variable (chosen cookie type).

#### Conditional analysis (within cookie type)

| Cookie type | T = 0.1 | T = 0.5 | T = 0.9 | Effect |
|------------|---------|---------|---------|--------|
| Heavy (req = 0.9) | −0.026 | −0.003 | +0.013 | t = 6.6, p < 10⁻¹⁰ |
| Light (req = 0.4) | −0.029 | −0.002 | +0.024 | t = 7.5, p < 10⁻¹³ |

Within each cookie type, excess effort increases robustly with threat. Effect sizes:
- Heavy: d = 0.42 (T = 0.1 → T = 0.9)
- Light: d = 0.49

#### Between-subject versus within-subject variance

| Source | % of total vigor variance |
|--------|--------------------------|
| Between-subject (trait-like) | 26% |
| Within-subject, condition-driven | ~4% |
| Residual | ~70% |

Vigor is primarily a stable individual trait, with modest but significant condition modulation. The 26% between-subject variance is what allows the model to fit individual cd parameters.

**Note for confirmation:** The conditional analysis (within cookie type) is the appropriate test for H1b. The unconditional analysis is reported for completeness but the Simpson's paradox makes it misleading. The preregistered test specifies conditioning on chosen cookie type.

---

### H1c: Model-Derived Survival Predicts Anxiety and Confidence

**Prediction:** The EVC model's survival probability S will predict trial-level anxiety (negatively) and confidence (positively) within subjects.

**Result: CONFIRMED**

#### Linear mixed model results

| DV | Predictor | β | SE | t | p | N_obs | N_subj |
|-----|-----------|---|----|---|---|-------|--------|
| Anxiety | S_z | −0.557 | 0.040 | −14.04 | 8.8 × 10⁻⁴⁵ | 5,274 | 293 |
| Confidence | S_z | +0.575 | 0.043 | +13.48 | 2.1 × 10⁻⁴¹ | 5,272 | 293 |

#### Random effects

| DV | RE intercept variance | RE slope variance | RE correlation |
|-----|----------------------|-------------------|----------------|
| Anxiety | 1.95 | 0.52 | — |
| Confidence | 2.10 | 0.63 | — |

Substantial random slope variance indicates meaningful individual differences in how strongly affect tracks survival. Some subjects show steep affect-survival slopes (affectively calibrated), others show flat slopes (affectively uncalibrated). These individual differences are decomposed in H4.

#### Effect magnitude

Approximately 0.6 rating points per SD of S on a 0–7 scale. Moving from the safest condition (T = 0.1, S ≈ 0.85) to the most dangerous (T = 0.9, S ≈ 0.15) shifts:
- Anxiety: up by ~2.0 points on the 0–7 scale
- Confidence: down by ~2.0 points

#### Anxiety-confidence relationship

Within-subject correlation: r = −0.25 (not mirror images). 28% of subjects show positive coupling (more anxious AND more confident simultaneously), suggesting these are partially independent affective channels rather than opposite poles of a single dimension. This independence is relevant to the metacognitive decomposition in H4.

---

## Summary Table

| Sub-hypothesis | Test | Statistic | p | Threshold | Verdict |
|---------------|------|-----------|---|-----------|---------|
| H1a: Choice ↓ with threat | LMM β(threat) | −1.28 | < 10⁻²⁰⁰ | p < 0.01 | **CONFIRMED** |
| H1a: Choice ↓ with distance | LMM β(distance) | −0.65 | < 10⁻⁵⁹ | p < 0.01 | **CONFIRMED** |
| H1a: Threat × distance | LMM β(interaction) | −0.18 | < 10⁻⁵ | p < 0.01 | **CONFIRMED** |
| H1a: Monotonicity | Paired t-tests | all t > 8.3 | all p < 10⁻¹⁴ | p < 0.01 | **CONFIRMED** |
| H1b: Vigor ↑ with threat (heavy) | Conditional LMM | t = 6.6 | < 10⁻¹⁰ | p < 0.05 | **CONFIRMED** |
| H1b: Vigor ↑ with threat (light) | Conditional LMM | t = 7.5 | < 10⁻¹³ | p < 0.05 | **CONFIRMED** |
| H1c: Anxiety ↓ with S | LMM β(S_z) | −0.557 | < 10⁻⁴⁴ | \|t\| > 3.0 | **CONFIRMED** |
| H1c: Confidence ↑ with S | LMM β(S_z) | +0.575 | < 10⁻⁴⁰ | \|t\| > 3.0 | **CONFIRMED** |

All sub-hypotheses confirmed with large effect sizes. The behavioral effects of threat are robust and provide the foundation for the computational model (H2) and affective calibration analyses (H3, H4).
