# H2 Results: Survival Computation Predicts Subjective Affect

**Date:** 2026-03-20
**Model:** S = (1-T) + T / (1 + λD), λ=2.0, D = distance + 1 (game units 1–3)
**Participants:** N=293 subjects (5,274 anxiety trials, 5,272 confidence trials)

---

## H2a: S Predicts Anxiety and Confidence

Within-person z-scores of S and response were computed for each participant separately.
Linear mixed models fit with random subject intercepts using statsmodels MixedLM.

### Anxiety (questionType=5)

Higher survival probability predicted lower anxiety:
β = −0.2812, SE = 0.0116, t = −24.14, p < 0.001
95% CI: [−0.3041, −0.2584]

**Supported.** Predicted direction: negative. Observed direction: negative.

### Confidence (questionType=6)

Higher survival probability predicted higher confidence:
β = +0.2803, SE = 0.0118, t = +23.67, p < 0.001
95% CI: [+0.2571, +0.3036]

**Supported.** Predicted direction: positive. Observed direction: positive.

---

## H2b: Effort Discounting (k) Moderates S → Affect

LMMs with S_z + S_z:k_z interaction (k log-transformed and population z-scored, random subject intercepts).
k_z is a between-subjects predictor and its main effect is not separately estimable from the random intercept
(perfect collinearity at the subject level); only the interaction term S_z:k_z is reported for inference.
A significant S_z:k_z interaction indicates that individuals with stronger effort discounting show
different S-to-affect coupling than individuals with weak effort discounting.

### Anxiety Moderation

S_z:k_z interaction: β = +0.0503, SE = 0.0117, t = +4.32, p < 0.001

**Significant.** Subjects with higher k (stronger effort discounting) show *weaker* S → anxiety coupling.
That is, their anxiety is less tightly coupled to trial-level survival probability changes —
suggesting that effort cost dominates their affective response rather than survival probability per se.

### Confidence Moderation

S_z:k_z interaction: β = −0.0403, SE = 0.0119, t = −3.40, p < 0.001

**Significant.** Subjects with higher k show *weaker* S → confidence coupling (less positive coupling).
Same interpretation as anxiety: high effort-discounters are less sensitive to survival-probability-driven
changes in confidence.

---

## H2c: Cross-Domain Threat Sensitivity

Per-subject threat sensitivity estimated as OLS slope of outcome on threat level (T ∈ {0.1, 0.5, 0.9}):
- **Choice threat sensitivity:** slope of choice (1=risky high option, 0=safe low option) on threat
- **Anxiety threat sensitivity:** slope of anxiety rating on threat
- **Confidence threat sensitivity:** slope of confidence rating on threat

Population-level mean slopes (expected directions):
- Choice: mean = −0.605 (higher threat → less risky choice, as predicted)
- Anxiety: mean = +1.768 (higher threat → more anxiety, as predicted)
- Confidence: mean = −1.772 (higher threat → less confidence, as predicted)

### Choice-threat slope vs Anxiety-threat slope

Pearson r = −0.386, p < 0.001, n = 293

**Significant coherent coupling.** The negative correlation is expected: subjects with a more negative
choice-threat slope (i.e., who reduce risky choices more under threat) also have a more positive
anxiety-threat slope (i.e., show larger anxiety increases under threat). Both effects reflect
greater threat sensitivity — they are the same construct expressed in two different domains.

### Choice-threat slope vs Confidence-threat slope

Pearson r = +0.372, p < 0.001, n = 293

**Significant coherent coupling.** The positive correlation is expected: subjects with a more negative
choice-threat slope also have a more negative confidence-threat slope (i.e., show larger confidence
drops under threat). Again both effects reflect greater threat sensitivity, now linking choice
avoidance with reduced subjective confidence under threat.

---

## Summary

| Test | Outcome | Key statistic |
|------|---------|---------------|
| H2a: S → Anxiety | Supported | β = −0.281, t = −24.14, p < 0.001 |
| H2a: S → Confidence | Supported | β = +0.280, t = +23.67, p < 0.001 |
| H2b: k × S → Anxiety (interaction) | Significant | β = +0.050, t = +4.32, p < 0.001 |
| H2b: k × S → Confidence (interaction) | Significant | β = −0.040, t = −3.40, p < 0.001 |
| H2c: Choice vs Anxiety threat sensitivity | Supported | r = −0.386, p < 0.001, n = 293 |
| H2c: Choice vs Confidence threat sensitivity | Supported | r = +0.372, p < 0.001, n = 293 |

All pre-registered H2 tests are supported. The survival computation S = (1-T) + T/(1+λD) with λ=2.0
predicts trial-level anxiety and confidence (H2a), individual differences in k moderate this coupling
(H2b), and individuals who are more threat-sensitive in choice are correspondingly more
threat-sensitive in subjective affect across both anxiety and confidence (H2c).
