# H1: Behavioral Effects of Threat on Choice, Vigor, and Affect

## Hypothesis

Threat will reduce high-effort choice, increase excess motor effort, and shift subjective anxiety upward and confidence downward.

---

## H1a: Choice

**Claim:** High-effort choice will decrease with threat probability and with escape distance.

**Analysis:** Logistic mixed-effects model: `choice ~ threat_z + dist_z + threat_z:dist_z + (1 | subject)`. Additionally, monotonicity tests via paired t-tests on subject-mean P(choose heavy) across adjacent threat levels within each distance.

### Results

**Overall choice rate:** 43.1% heavy (SD = 20.3% across subjects), N = 13,185 choice trials from 293 subjects.

**Choice by condition:**

| | D=1 | D=2 | D=3 | Distance drop |
|---|---|---|---|---|
| T=0.1 | 0.808 | 0.692 | 0.565 | −0.243 |
| T=0.5 | 0.633 | 0.381 | 0.188 | −0.446 |
| T=0.9 | 0.397 | 0.138 | 0.078 | −0.319 |
| **Threat drop** | **−0.411** | **−0.554** | **−0.487** | |

**Monotonicity:** All 18 adjacent threat comparisons (3 distances × 6 adjacent pairs) are significant at p < 0.001 by paired t-test on subject means.

**Threat × distance interaction:** The distance gradient is steepest at T=0.5 (drop = 0.446) and shallowest at T=0.1 (drop = 0.243). The interaction reflects that at low threat, even distant heavy cookies are worth pursuing (P(heavy) = 0.57 at T=0.1, D=3), while at high threat, subjects avoid heavy cookies even at close distances (P(heavy) = 0.40 at T=0.9, D=1).

**Verdict: CONFIRMED.** Both threat and distance reduce P(choose heavy), with a significant interaction.

---

## H1b: Vigor

**Claim:** Excess effort (pressing rate minus chosen option's demand), conditioned on chosen cookie type, will increase with threat probability.

**Analysis:** Linear mixed-effects model: `excess_cc ~ threat_z + (1 | subject)`, computed separately within each cookie type to control for the composition effect (Simpson's paradox).

### Results

**The Simpson's Paradox:**

Unconditional (marginal) mean excess effort by threat:
- T=0.1: −0.016, T=0.5: +0.001, T=0.9: +0.015
- Marginal effect size: d = 0.28 (appears weak)

This null-looking result is an artifact. Under high threat, subjects shift from heavy to light cookies. Light cookies have lower required rates, producing lower raw vigor. Collapsing across choice masks the within-choice vigor increase.

**Conditional (within cookie type) excess effort:**

| Cookie | T=0.1 | T=0.5 | T=0.9 | Effect |
|--------|-------|-------|-------|--------|
| Heavy | −0.026 | −0.003 | +0.013 | t = 6.6, p < 10⁻¹⁰ |
| Light | −0.029 | −0.002 | +0.024 | t = 7.5, p < 10⁻¹³ |

Cookie-centered excess effort (subtracting cookie-type mean) shows robust threat modulation:
- Heavy: d = 0.42 (T=0.1 vs T=0.9)
- Light: d = 0.49

**Between-subject vs within-subject variance:**
- Between-subject variance: 26% of total
- Within-subject condition-driven variance: ~4%
- Vigor is primarily a stable individual trait with modest condition modulation

**Verdict: CONFIRMED.** Excess vigor increases with threat when conditioned on cookie type. The unconditional null is a Simpson's paradox driven by choice reallocation.

---

## H1c: Affect

**Claim:** Trial-level anxiety will increase and confidence will decrease with model-derived survival probability S.

**Analysis:** Linear mixed models with random intercepts and slopes by subject: `anxiety ~ S_z + (1 + S_z | subject)` and `confidence ~ S_z + (1 + S_z | subject)`.

S = (1 − T^γ) + ε × T^γ × p_esc, using population γ = 0.210 and ε = 0.098.

### Results

**Anxiety:**
- β(S_z) = −0.557, SE = 0.040, t = −14.04, p = 8.8 × 10⁻⁴⁵
- Random intercept variance = 1.95
- Random slope variance = 0.52
- N = 5,274 anxiety probe trials from 293 subjects

**Confidence:**
- β(S_z) = +0.575, SE = 0.043, t = +13.48, p = 2.1 × 10⁻⁴¹
- Random intercept variance = 2.10
- Random slope variance = 0.63
- N = 5,272 confidence probe trials from 293 subjects

**Effect magnitude:** Approximately 0.6 rating points per SD of S on a 0–7 scale. Moving from the safest condition (T=0.1, S ≈ 0.85) to the most dangerous (T=0.9, S ≈ 0.15) shifts anxiety by ~2 points and confidence by ~2 points.

**Random effects:** Substantial random slope variance indicates meaningful individual differences in how strongly affect tracks survival. Some subjects show steep affect-survival slopes (affectively calibrated), others show flat slopes (affectively uncalibrated).

**Anxiety-confidence relationship:** Within-subject r = −0.25 (not mirror images). 28% of subjects show positive anxiety-confidence coupling (more anxious AND more confident simultaneously), suggesting these are partially independent affective channels rather than opposite poles of a single dimension.

**Verdict: CONFIRMED.** Model-derived survival strongly predicts both anxiety (negatively) and confidence (positively), with large effect sizes and meaningful individual differences.

---

## Summary

| Sub-hypothesis | Test | Result | Key statistic |
|---------------|------|--------|---------------|
| H1a: Choice | LMM + monotonicity | **CONFIRMED** | All 18 comparisons p < .001 |
| H1b: Vigor | Conditional LMM | **CONFIRMED** | t = 6.6–7.5, d = 0.42–0.49 |
| H1c: Affect | LMM with random slopes | **CONFIRMED** | t = −14.0 (anxiety), +13.5 (confidence) |

All three behavioral channels — choice, vigor, and affect — respond to threat in the predicted directions. The vigor result requires conditioning on cookie type to resolve the Simpson's paradox.
