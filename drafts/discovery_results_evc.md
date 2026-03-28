# Discovery Results: EVC-LQR Model

## H1: Behavioral Effects of Threat on Choice, Vigor, and Affect

### H1a: Threat reduces high-effort choice and distance increases the effect

**Result: CONFIRMED**

P(choose heavy) decreases monotonically with threat at every distance level:

| Distance | T=0.1 | T=0.5 | T=0.9 | Drop (T=0.1→0.9) |
|----------|-------|-------|-------|-------------------|
| D=1 | 0.808 | 0.633 | 0.397 | −0.411 |
| D=2 | 0.692 | 0.381 | 0.138 | −0.554 |
| D=3 | 0.565 | 0.188 | 0.078 | −0.487 |

All adjacent threat comparisons within each distance are significant (paired t-tests, all p < 0.001).

Distance also reduces P(heavy) within each threat level:
- T=0.1: D1=0.808 → D3=0.565 (drop = 0.243)
- T=0.5: D1=0.633 → D3=0.188 (drop = 0.446)
- T=0.9: D1=0.397 → D3=0.078 (drop = 0.319)

The threat × distance interaction is present: the distance gradient is steepest at T=0.5.

### H1b: Excess vigor increases with threat (conditioned on choice)

**Result: CONFIRMED**

Unconditional (marginal) vigor appears flat across threat — a Simpson's paradox. But conditioning on chosen cookie type reveals robust threat modulation:

| Cookie | T=0.1 | T=0.5 | T=0.9 | Effect |
|--------|-------|-------|-------|--------|
| Heavy | −0.026 | −0.003 | +0.013 | t = 6.6, p < 10⁻¹⁰ |
| Light | −0.029 | −0.002 | +0.024 | t = 7.5, p < 10⁻¹³ |

The marginal null is an artifact of choice reallocation: under high threat, subjects shift from heavy to light cookies, which have lower required rates and thus lower raw vigor, masking the within-choice vigor increase.

### H1c: Model-derived survival predicts anxiety and confidence

**Result: CONFIRMED**

Linear mixed models with random intercepts and slopes by subject:
- Anxiety ~ S_z: β = −0.557, SE = 0.040, t = −14.04, p = 8.8 × 10⁻⁴⁵
- Confidence ~ S_z: β = +0.575, SE = 0.043, t = +13.48, p = 2.1 × 10⁻⁴¹

Higher model-derived survival probability predicts lower anxiety and higher confidence within subjects. Effects are large (~0.6 rating points per SD of S on a 0–7 scale). Substantial random slope variance indicates meaningful individual differences in affect-survival coupling.

---

## H2: EVC-LQR Model Captures Choice and Vigor Jointly

### H2a: Per-subject choice fit

**Result: CONFIRMED**

- Per-subject choice r = 0.975, **r² = 0.951**
- Trial-level choice accuracy = 79.3%
- Choice AUC = 0.876
- BIC = 17,768

The model reproduces the full threat × distance choice surface. At the condition level, the maximum absolute prediction error is 0.13 (T=0.1 D=1: pred=0.87, obs=0.81).

Threshold: r² > 0.85. **PASSED** (0.951 > 0.85).

### H2b: Vigor fit

**Result: CONFIRMED**

- Trial-level vigor r = 0.715, **r² = 0.511**
- Per-subject vigor r = 0.829

The model correctly predicts vigor increases with threat and captures the direction of heavy/light vigor differences.

Threshold: r² > 0.30. **PASSED** (0.511 > 0.30).

### H2c: Distance gradient in choice

**Result: CONFIRMED**

Model predictions show clear distance gradient within each threat level:

| Condition | Predicted | Observed |
|-----------|-----------|----------|
| T=0.1 D=1 | 0.87 | 0.81 |
| T=0.1 D=3 | 0.53 | 0.57 |
| T=0.5 D=1 | 0.58 | 0.63 |
| T=0.5 D=3 | 0.18 | 0.19 |
| T=0.9 D=1 | 0.33 | 0.40 |
| T=0.9 D=3 | 0.10 | 0.08 |

The model captures the qualitative pattern: P(heavy) declines with distance at every threat level. The distance gradient was absent in all previous models with population-level effort cost.

### H2d: Parameter recovery

**Result: CONFIRMED**

Simulated 3 datasets × 50 subjects × 81 trials. Re-fitted with identical SVI procedure.

| Parameter | Recovery r (log space) | Threshold | Result |
|-----------|----------------------|-----------|--------|
| log(ce) | 0.916 | > 0.70 | PASS |
| log(cd) | 0.943 | > 0.70 | PASS |
| γ | recovered 0.255 vs true 0.210 | — | Slight positive bias |

### H2e: Parameter independence

**Result: CONFIRMED**

log(ce) × log(cd): r = −0.135, p = .003

Threshold: |r| < 0.25. **PASSED** (0.135 < 0.25).

The two parameters capture largely independent dimensions of individual variation.

### Model comparison (6 ablations)

| Model | Tests | BIC | ΔBIC vs FINAL |
|-------|-------|-----|---------------|
| M1: Effort only | Is threat needed? | TBD | TBD |
| M2: Threat only | Is effort needed? | TBD | TBD |
| M3: Separate models | Is joint needed? | TBD | TBD |
| M4: Population ce | Are effort diffs needed? | ~20,000 | ~+2,200 |
| M5: No γ | Is prob weighting needed? | ~18,600 | ~+800 |
| M6: Standard u² | Is LQR needed? | ~17,920 | ~+150 |
| **FINAL: EVC 2+2** | **Full model** | **17,768** | **0** |

M1–M3 being fitted. M4–M6 approximate from earlier runs.

### Population parameters

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| γ | 0.210 | Probability weighting: T=0.5 experienced as T^0.21 = 0.86 |
| ε | 0.098 | Efficacy: people underweight effort's survival benefit by ~90% |
| ce_vigor | 0.003 | LQR motor deviation cost (population) |
| τ | 0.476 | Choice temperature |

### Per-subject parameters (log space)

| Parameter | Log mean | Log SD | Geometric mean | Interpretation |
|-----------|----------|--------|----------------|----------------|
| ce | −0.47 | 0.78 | 0.62 | Effort cost per unit req²×D |
| cd | 3.44 | 1.57 | 31.3 | Capture aversion (inflated by low ε; effective incentive cd×ε ≈ 3.1) |

---

## H3: Metacognitive Miscalibration

### H3a: Confidence correlates with choice quality

**Result: CONFIRMED**

- r(mean confidence, choice quality) = 0.230, p < .001

Choice quality = proportion of trials where subject chose the EV-maximizing option.

### H3b: Confidence does NOT correlate with survival

**Result: CONFIRMED**

- r(mean confidence, survival rate) ≈ 0.012, p = 0.84

Confidence is unrelated to actual escape performance.

### H3c: Steiger's test

**Result: CONFIRMED**

Steiger's z comparing r(conf, choice quality) vs r(conf, survival):
- z = 3.14, p = 0.002 (in earlier analysis)

The two correlations are significantly different. Confidence tracks choice, not survival.

### Interpretation

Participants' confidence reflects the quality of their foraging DECISIONS (governed by ce) but not their actual SURVIVAL (governed by cd and vigor). Since survival determines outcomes more than choice does (survival→earnings r=0.85 vs choice→earnings r=0.43), this represents a systematic metacognitive miscalibration: people feel confident about the wrong thing.

---

## H4: Calibration-Discrepancy Double Dissociation

### Definitions

- **Calibration:** Within-subject Pearson r(anxiety rating, 1−S). How accurately anxiety tracks model-derived danger.
- **Discrepancy:** Mean residual of anxiety ratings after removing the population-level S-anxiety relationship. How much anxiety exceeds what danger warrants.

### H4a: Orthogonality

**Result: CONFIRMED**

r(calibration, discrepancy) = 0.019, p = .75

The two dimensions are nearly perfectly orthogonal — knowing how well someone tracks danger tells you nothing about whether they overestimate it.

### H4b: Calibration predicts performance

**Result: CONFIRMED**

- Calibration → choice quality: r = 0.230, p < .001
- Calibration → survival rate: r = 0.185, p = .002

Subjects whose anxiety more accurately tracks danger make better choices and survive more.

### H4c: Discrepancy predicts clinical symptoms

**Result: CONFIRMED**

Bayesian regression (bambi, weakly informative priors, 94% HDI):

| Clinical measure | β | 94% HDI | Direction |
|-----------------|---|---------|-----------|
| STAI-State | 0.338 | [0.22, 0.45] | Overanxious → higher state anxiety |
| STICSA | 0.285 | [0.17, 0.40] | Overanxious → higher somatic/cognitive anxiety |
| DASS-Anxiety | 0.275 | [0.16, 0.39] | Overanxious → higher anxiety |
| DASS-Stress | 0.217 | [0.10, 0.33] | Overanxious → higher stress |
| DASS-Depression | 0.206 | [0.09, 0.32] | Overanxious → higher depression |
| PHQ-9 | 0.212 | [0.10, 0.33] | Overanxious → higher depression |
| OASIS | 0.180 | [0.07, 0.30] | Overanxious → higher anxiety severity |
| AMI-Emotional | −0.222 | [−0.34, −0.11] | Overanxious → LESS emotional apathy |

8 of 9 clinical measures show credible associations (94% HDI excludes zero). The AMI-Emotional result is notable: overanxious individuals are the OPPOSITE of apathetic — they are motivationally engaged but affectively miscalibrated.

### H4d: Partial double dissociation

**Result: PARTIALLY CONFIRMED**

The dissociation is predominant but not perfectly clean:
- Calibration → STAI-State: r = 0.138, p = .019 (leakage)
- Discrepancy → survival: r = −0.153, p = .009 (leakage)

The dominant pattern holds: calibration primarily predicts performance, discrepancy primarily predicts clinical symptoms. But there is some cross-contamination, particularly for STAI-State.

---

## Exploratory: Model Parameters and Clinical Outcomes

### Direct parameter-clinical associations

**Result: NULL after FDR correction**

No individual correlation between log(ce) or log(cd) and any psychiatric subscale survives Benjamini-Hochberg FDR correction.

Best uncorrected: log(cd) → AMI-Emotional r = +0.120 (p = .039).

### Bayesian ROPE analysis

**Result: EVIDENCE FOR NULL**

Approximately 77% of posterior mass falls within ROPE [−0.1, 0.1] for both log(ce) and log(cd) across all clinical measures. This provides positive Bayesian evidence that model parameters do not predict psychiatric symptoms.

### Machine learning prediction

**Result: NULL**

All cross-validated R² values (elastic net, ridge) are negative for predicting clinical outcomes from log(ce) + log(cd) + discrepancy + calibration + interactions. The computational-metacognitive features do not predict individual symptom severity out of sample.

### Interpretation

The bridge from computation to psychopathology runs through METACOGNITION (discrepancy), not through the decision parameters (ce, cd) themselves. How people COMPUTE danger does not predict clinical symptoms. How they FEEL about danger — specifically, the systematic excess of anxiety relative to computed danger — does.

---

## Exploratory: Behavioral Profiles

### ce × cd quadrant analysis

Median split on log(ce) and log(cd) creates four profiles:

| Profile | ce | cd | P(heavy) | Survival | Earnings | N |
|---------|----|----|----------|----------|----------|---|
| Cautious | high | high | low | high | moderate | ~73 |
| Lazy | high | low | low | low | low | ~73 |
| Vigilant | low | high | moderate | high | **29.9** (highest) | ~73 |
| Bold | low | low | high | low | moderate | ~73 |

- P(heavy) R² = 0.953 from log(ce) + log(cd) + interaction
- AMI differs across quadrants: F(3,289) = 2.79, p = .041
- The Vigilant profile (low effort cost, high capture aversion) is the most adaptive strategy

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| N (exploratory) | 293 |
| Choice trials per subject | 45 |
| Probe trials per subject | 36 |
| Total trials per subject | 81 |
| Overall P(choose heavy) | 43.1% |
| Overall survival rate | 68.3% |
| Mean earnings | 12.2 ± 47.1 |
| Model BIC | 17,768 |
| Per-subject choice r² | 0.951 |
| Trial-level vigor r² | 0.511 |
| ce recovery r | 0.916 |
| cd recovery r | 0.943 |
| Discrepancy → STAI β | 0.338 |
| Params in ROPE | ~77% |
