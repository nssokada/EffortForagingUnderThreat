# H2: Threat Modulates Motor Vigor Independently of Choice, With Distinct Dynamics Across the Predatory Imminence Continuum

## Overview

This hypothesis tests whether threat probability modulates the vigor of motor execution — how hard participants press keys during the foraging bout — independently of which cookie they chose. It further tests whether the temporal dynamics of vigor follow the predatory imminence continuum: anticipatory mobilization driven by stated probability, followed by a reactive motor spike triggered by predator appearance that is independent of probability.

### Theoretical grounding

The predatory imminence continuum (Fanselow 1994; Mobbs et al. 2020) predicts qualitatively different defensive responses at different stages of a predatory encounter:

1. **Pre-encounter (anticipatory):** The organism knows threat exists but has not yet detected the predator. Defensive behavior is strategic, modulated by the probability and severity of threat. Motor preparation should scale with stated threat probability.

2. **Post-encounter (reactive):** The predator has been detected. Defensive behavior shifts from probability-based to presence-based — the encounter itself triggers a motor response regardless of the prior probability of attack. This response should be fast (hundreds of milliseconds) and stereotyped across individuals.

3. **Circa-strike (terminal):** The predator is closing in. Motor responses become maximal and may decouple from earlier strategic modulation.

### Critical confound: the Simpson's paradox in vigor

Higher threat shifts choice toward light cookies (lower effort demand). Light cookies require lower pressing rates. Collapsing across cookie type makes average vigor appear flat or declining with threat, masking the within-cookie-type threat modulation. ALL vigor analyses must control for cookie type — either by conditioning on cookie type or by including cookie as a covariate.

### Vigor metric

Normalized press rate = median(1/IPI) / calibrationMax, where IPI is the inter-keypress interval and calibrationMax is the participant's calibrated maximum pressing speed. This metric is on a common scale across participants and cookie types, ranges from 0 to ~1.5, and reflects the typical speed of individual keypresses within a trial or epoch.

For timecourse analyses: press rates computed in 200ms bins from raw keypress timestamps (matching the ~5Hz native keypress frequency), smoothed with a 3-point centered moving average (600ms effective window) for display.

---

## H2a: Threat increases pressing rate within cookie type

### Prediction

Within heavy cookies (req = 0.9) and within light cookies (req = 0.4) separately, higher threat probability produces higher normalized press rate.

### Test

For each participant, compute mean normalized press rate at T=0.1 and T=0.9, separately within heavy and light cookies. Paired t-test on the within-subject difference (T=0.9 minus T=0.1).

### Threshold

p < .01 within heavy cookies AND p < .01 within light cookies.

### Exploratory benchmarks

- Heavy cookies: Δ = +0.039, t = 7.40, p < 10⁻¹², d = +0.44
- Light cookies: Δ = +0.056, t = 13.53, p < 10⁻³², d = +0.80

---

## H2b: Predator encounter triggers a threat-independent motor spike

### Prediction

The appearance of the predator (on attack trials) produces a rapid increase in pressing rate relative to trials where the predator does not appear (non-attack trials), even though participants do not know the trial type in advance. This encounter spike does NOT scale with stated threat probability — it is the same magnitude at T=0.1, T=0.5, and T=0.9.

### Test

1. **Encounter spike:** For each participant, compute mean normalized press rate in the reactive epoch (encounterTime to encounterTime + 2s) on attack trials minus non-attack trials. One-sample t-test against zero.

2. **Threat independence:** For each participant, compute the encounter spike separately at each threat level. Paired t-test comparing spike at T=0.9 vs T=0.1.

### Threshold

1. Encounter spike: p < .001
2. Threat independence: p > .05 (the spike does NOT differ by threat level)

### Exploratory benchmarks

- Encounter spike: Δ = +0.037, t = 9.59, p < 10⁻¹⁹, d = +0.56
- 83% of participants show a positive spike
- Threat modulation of spike: t = -1.05, p = .29 (threat-independent)
- First significant 200ms bin: 300ms post-encounter

---

## H2c: Threat and encounter have distinct temporal signatures

### Prediction

The temporal shape of the vigor timecourse differs by threat level and by whether the predator appeared, as assessed by generalized additive mixed models (GAMs) with penalized smooth functions of time.

### Test

Fit two GAM models using natural cubic regression splines (K=10 basis functions) with cookie as a parametric covariate and random intercepts by participant:

1. **Encounter shape:** Compare a model with a smooth-by-attack-trial interaction (separate temporal shapes for attack vs non-attack) to a model with a single shared smooth. Likelihood ratio test.

2. **Threat shape:** Compare a model with smooth-by-threat-level interactions (separate shapes for T=0.1, T=0.5, T=0.9) to a model with a single shared smooth. Likelihood ratio test.

### Threshold

p < .01 for both LRTs.

### Exploratory benchmarks

- Encounter LRT: χ² = 359, df = 10, p ≈ 0
- Threat LRT (3 levels, joint): χ² = 292, df = 20, p ≈ 0
- All pairwise threat comparisons significant (T=0.5 vs T=0.1: p = .001; T=0.9 vs T=0.1: p ≈ 0; T=0.9 vs T=0.5: p ≈ 0)

---

## H2d: Stable individual differences in how threat modulates pre-encounter vigor

### Prediction

Participants differ in the degree to which threat probability increases their pre-encounter pressing rate, and this individual difference is stable across task blocks.

### Test

1. **Individual threat-vigor slopes:** For each participant, regress normalized press rate on threat probability across anticipatory-epoch observations (controlling for cookie type). Extract the per-subject slope.

2. **Between-subject variation:** Test whether the SD of slopes is greater than zero (which it trivially will be) — the substantive test is the cross-block stability below.

3. **Cross-block stability:** Compute the per-subject anticipatory threat-vigor slope separately within each of the three task blocks. Correlate Block 1 slopes with Block 2 slopes, Block 1 with Block 3, Block 2 with Block 3. Report the mean cross-block correlation.

### Threshold

Mean cross-block correlation > 0.20 (moderate stability).

### Exploratory benchmarks

- To be computed in confirmatory sample. The anticipatory threat × β interaction (z = 2.46, p = .014) from the epoch analysis demonstrates that individual differences in threat-vigor slopes exist and are moderated by the model's threat sensitivity parameter.

---

## H2e: Substantial individual differences in the encounter motor response

### Prediction

Participants vary widely in the magnitude of their encounter motor response (the press rate increase triggered by predator appearance). This variation is trait-like — stable across task blocks — indicating a reliable individual difference in reactive defensive motor mobilization.

### Test

1. **Individual variation:** Compute the per-subject encounter spike (attack minus non-attack normalized press rate in the reactive epoch). The coefficient of variation (SD/|mean|) exceeds 1, indicating that between-subject variation is larger than the population mean effect.

2. **Cross-block stability:** Compute the encounter spike separately within each of the three task blocks. Correlate Block 1 spikes with Block 2 spikes, etc. Report the mean cross-block correlation.

### Threshold

1. CV > 1
2. Mean cross-block r > 0.20

### Exploratory benchmarks

- Encounter spike: mean = +0.037, SD = 0.059, CV = 1.59
- Range: [-0.13, +0.69] — some participants show a large positive spike, others show no response or a decrease
- 83% of participants show a positive spike
- Cross-block stability: to be computed in confirmatory sample

---

## Analysis notes

### Cookie control

All analyses control for cookie type by one of:
- Running within cookie type separately (H2a)
- Using attack vs non-attack contrast which is balanced on cookie within subject (H2b)
- Including cookie as a parametric covariate in the GAM (H2c)
- Including cookie as a covariate in the per-subject slope regression (H2d)

### Temporal resolution

Epoch-level tests (H2a, H2b) use per-trial median values. Timecourse analyses (H2c) use 200ms bins. GAMs use the 200ms-binned data with spline smoothing. All raw data comes from keypress timestamps at the native ~5Hz resolution.

### Exclusions

Same as all analyses: subjects 154, 197, 208 excluded for calibration outliers (>2.5 SD on mean IPI). Terminal epoch analyses (if reported) restricted to attack trials only.
