# H5: Three Dimensions of Anxiety — Quality, Reactivity, and Level

## Overview

This hypothesis tests a triple dissociation in what anxiety does during foraging under threat. We decompose individual differences in the anxiety response into three orthogonal dimensions — **calibration** (signal quality), **slope** (reactivity to threat), and **mean level** (tonic intensity) — and show that each predicts a different domain of outcomes.

### Theoretical grounding

Metacognitive models of anxiety (Wells 2009) distinguish between the content of threat cognition and the accuracy of threat monitoring. Interoceptive prediction error frameworks (Paulus & Stein 2010) formalize anxiety as a mismatch between internal signals and environmental contingencies. Our decomposition operationalizes this: calibration measures how well anxiety tracks threat (signal quality), slope measures how strongly anxiety reacts to threat changes (reactivity), and mean level measures tonic anxiety intensity (the "baseline" that clinical measures capture).

### Definitions

- **Calibration** = within-subject Pearson r between anxiety rating and threat probability. Range [-1, 1]. Higher = anxiety is a better thermometer of danger.
- **Anxiety slope** = within-subject regression slope of anxiety on threat. Higher = more anxiety increase per unit threat.
- **Mean anxiety** = average anxiety rating across all probes. Higher = more anxious overall.

These are conceptually distinct: a person can have perfect calibration (r=1.0) with either a steep slope (reactive) or shallow slope (dampened), and at either a high or low mean level.

---

## H5a: The three dimensions are approximately orthogonal

### Prediction

Calibration, slope, and mean anxiety are not strongly correlated. Each captures a different aspect of the anxiety response.

### Test

Pairwise Pearson r among the three dimensions.

### Exploratory benchmarks

- r(calibration, mean anxiety): ~0.04 (orthogonal)
- r(slope, mean anxiety): moderate positive (steeper slope → higher mean, partially confounded)
- r(calibration, slope): related but distinct (both track threat, but calibration is about fit quality, slope about magnitude)

---

## H5b: Calibration predicts foraging optimality beyond the computational model

### Prediction

After controlling for ω and κ, calibration explains additional variance in foraging optimality, escape rate, and earnings. People whose anxiety is a better signal of danger make better decisions.

### Test

Hierarchical regression:
- Step 1: outcome ~ ω_z + κ_z
- Step 2: outcome ~ ω_z + κ_z + calibration_z
- Report ΔR² and calibration p-value.

### Threshold

Calibration ΔR² > 0.03 and p < .01 for at least two of: optimality, escape, earnings.

### Exploratory benchmarks

| Outcome | R²(ω+κ) | R²(+calibration) | ΔR² | cal p |
|---------|---------|-------------------|-----|-------|
| % Optimal | 0.476 | 0.544 | **+0.068** | <.0001 |
| Earnings | 0.009 | 0.067 | **+0.058** | <.0001 |
| Escape | 0.056 | 0.094 | **+0.038** | .0006 |

Interpretation: The model parameters capture WHO avoids and WHO presses hard. Calibration captures WHO makes GOOD decisions on top of that. Affect quality adds information that the rational computation doesn't contain.

---

## H5c: Anxiety slope predicts behavioral adaptation

### Prediction

People whose anxiety reacts more strongly to threat (steeper slope) also shift their behavior more across threat levels. Anxiety reactivity drives behavioral reactivity.

### Test

Correlate anxiety slope with:
- Choice shift = P(heavy at T=0.1) − P(heavy at T=0.9)
- Vigor shift = mean vigor at T=0.9 − mean vigor at T=0.1

### Threshold

r(anxiety slope, choice shift) > 0.20, p < .01.

### Exploratory benchmarks

- r(anxiety slope, choice shift) = **+0.389** (p < .0001)
- r(anxiety slope, vigor shift) = +0.004 (null)
- Anxiety slope → % optimal: r = +0.264
- Anxiety slope → earnings: r = +0.228

Interpretation: Anxiety slope predicts adaptive CHOICE shifting but not vigor shifting. People who feel more anxious at high threat also avoid more at high threat — their anxiety is driving appropriate behavioral adjustment.

---

## H5d: Mean anxiety predicts clinical symptoms but not foraging success

### Prediction

Mean task anxiety correlates with clinical measures (STAI, OASIS, DASS). Calibration and slope do not.

### Test

Correlate each anxiety dimension with clinical measures.

### Threshold

r(mean anxiety, STAI-State) > 0.15, p < .01.
|r(calibration, clinical)| < 0.15 for all measures.

### Exploratory benchmarks

| Clinical measure | r(mean anxiety) | r(calibration) | r(slope) |
|------------------|-----------------|----------------|----------|
| STAI-State | **+0.326** | +0.117 | +0.041 |
| STAI-Trait | -0.219 | -0.057 | +0.013 |
| OASIS | **+0.204** | -0.012 | -0.051 |
| DASS-Anxiety | **+0.251** | +0.012 | -0.011 |

Mean anxiety → clinical: significant. Calibration → clinical: null. Slope → clinical: null. Clinical distress tracks how anxious you ARE, not how well your anxiety tracks reality.

---

## H5e: Computational parameters are psychiatrically silent

### Prediction

ω and κ do not predict clinical symptom measures. The foraging computation is rational and unrelated to psychopathology.

### Test

Hierarchical regression:
- Step 1: clinical ~ ω_z + κ_z
- Step 2: clinical ~ ω_z + κ_z + mean_anxiety_z

### Threshold

Step 1 R² < 0.02 for all clinical measures.

### Exploratory benchmarks

| Measure | R²(ω+κ) | R²(+mean anxiety) | ΔR² |
|---------|---------|-------------------|-----|
| STAI-State | 0.001 | 0.107 | +0.106 |
| DASS-Total | 0.009 | 0.074 | +0.065 |
| STICSA | 0.002 | 0.078 | +0.076 |
| PHQ-9 | 0.008 | 0.058 | +0.050 |
| AMI-Total | 0.009 | 0.009 | +0.000 (null) |

ω, κ: silent. Mean anxiety: significant for all anxiety/distress measures. AMI (apathy): null for both — anxiety is specific to internalizing symptoms.

---

## H5f: ω maps to confidence, not anxiety

### Prediction

The computational capture cost parameter (ω) predicts subjective confidence (a prospective evaluation of one's situation) but not anxiety (an affective state). The computation informs judgment, not feeling.

### Test

Correlate ω with mean confidence and mean anxiety.

### Threshold

r(ω, confidence) < 0, p < .01. |r(ω, anxiety)| < 0.10.

### Exploratory benchmarks

- r(ω, confidence) = **-0.216** (p = .0002)
- r(κ, confidence) = -0.153 (p = .009)
- r(ω, anxiety) = +0.071 (null)
- r(κ, anxiety) = +0.026 (null)

---

## H5g: Confidence predicts error type, not error rate

### Prediction

Confident people don't make fewer errors — they make DIFFERENT errors. Confidence reduces overcautious errors but increases reckless errors.

### Test

Correlate mean confidence with number of overcautious and reckless errors separately.

### Threshold

r(confidence, overcautious) < 0 AND r(confidence, reckless) > 0, both p < .01.

### Exploratory benchmarks

- r(confidence, n_overcautious) = **-0.224** (p = .0001)
- r(confidence, n_reckless) = **+0.200** (p = .0006)

Confidence tilts the error distribution without improving accuracy overall.

---

## H5h: Anxiety at low threat predicts unnecessary avoidance

### Prediction

Anxiety at T=0.1 (low threat, where heavy is always optimal) predicts choosing light — unnecessary avoidance driven by anxiety that isn't warranted by the environment.

### Test

r(anxiety at T=0.1, P(heavy at T=0.1)). Expected negative.

### Threshold

r < -0.15, p < .01.

### Exploratory benchmarks

- T=0.1: r(anxiety, P(heavy)) = **-0.271** (p < .0001) — anxious at low threat → avoid unnecessarily
- T=0.9: r(anxiety, P(heavy)) = -0.040 (null) — at high threat, everyone avoids regardless

The overcaution problem is driven by people who are anxious when they shouldn't be.

---

## Summary: The Triple Dissociation

| Anxiety dimension | Predicts | Doesn't predict |
|-------------------|----------|-----------------|
| **Calibration** (quality) | Optimality, earnings, escape | Clinical symptoms |
| **Slope** (reactivity) | Choice adaptation, optimality | Vigor, clinical symptoms |
| **Mean level** (intensity) | Clinical symptoms | Foraging success |

The computation (ω, κ) governs foraging STRATEGY. Affect QUALITY (calibration) governs foraging WISDOM. Affect LEVEL (mean anxiety) governs clinical DISTRESS. Three parallel systems, three independent contributions.

---

## Confirmation Plan

| Test | Statistic | Threshold | Discovery value |
|------|-----------|-----------|-----------------|
| H5b: Cal ΔR² optimality | ΔR² | > .03, p < .01 | +0.068 |
| H5b: Cal ΔR² earnings | ΔR² | > .03, p < .01 | +0.058 |
| H5c: Slope → choice shift | r | > .20, p < .01 | +0.389 |
| H5d: Mean anx → STAI | r | > .15, p < .01 | +0.326 |
| H5d: Cal → STAI | r | |r| < .15 | +0.117 |
| H5e: ω+κ → clinical R² | R² | < .02 | 0.001 |
| H5f: ω → confidence | r | < 0, p < .01 | -0.216 |
| H5g: Conf → overcautious | r | < 0, p < .01 | -0.224 |
| H5g: Conf → reckless | r | > 0, p < .01 | +0.200 |
| H5h: Anx(T=0.1) → P(H) | r | < -.15, p < .01 | -0.271 |
