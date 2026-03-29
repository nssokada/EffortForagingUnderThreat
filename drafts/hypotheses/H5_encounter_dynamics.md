# H5: Within-Trial Encounter Dynamics Reveal a Trait-Like Defensive Reflex

## Results from Discovery Sample (N = 293)

---

## Overview

This hypothesis tests whether the moment-to-moment vigor response to predator encounter — measured from the 20Hz smoothed pressing timeseries — constitutes a stable individual difference that relates to the static model's capture aversion parameter (cd), is modulated by threat probability, and predicts clinical apathy. The analysis extends the static EVC model (which predicts trial-level average vigor) into the within-trial temporal domain, connecting to Mobbs et al.'s (2020) distinction between strategic and reactive defensive modes.

---

## Methods

### Data

Smoothed vigor timeseries at 20Hz from the kernel-smoothing pipeline (smoothed_vigor_ts.parquet, 3.9M rows). Each trial has ~148 timepoints spanning from trial start to trial end.

### Demand residualization

Raw vigor (vigor_norm, normalized by calibrated maximum) was residualized by subtracting the required press rate for the chosen cookie (0.9 for heavy, 0.4 for light), then cookie-type centered by subtracting the population mean excess for each cookie type (heavy mean = −0.206, light mean = 0.249). This produces a demand-free signal where zero = pressing at exactly the population-average rate for that cookie type.

### Encounter alignment

For attack trials (isAttackTrial = 1), the timeseries was aligned to predator encounter time (encounterTime). Time-relative-to-encounter (t_rel) was computed as t − encounterTime. Pre-encounter: t_rel < 0. Post-encounter: t_rel ≥ 0.

### Encounter reactivity

Per trial: reactivity = mean(excess_cc in post-encounter window) − mean(excess_cc in pre-encounter window).
Per subject: mean reactivity across all attack trials.

### Statistical tests

- **Trait stability:** Cross-block Pearson correlations (3 blocks × 3 pairs).
- **Threat modulation:** One-way ANOVA on per-subject mean reactivity by threat level.
- **Connection to static model:** Pearson r between reactivity and log(cd), log(ce).
- **Clinical associations:** Pearson r between reactivity and all psychiatric subscales.
- **Incremental prediction:** Hierarchical regression: Step 1 = log(cd) + log(ce), Step 2 = + reactivity. Test ΔR² for each clinical measure.
- **Piecewise slopes:** Per-subject OLS of excess_cc on t, computed separately for pre-encounter and post-encounter windows. Test whether slope changes at encounter.

---

## Results

### Encounter reactivity is NOT significantly different from zero on average

- Mean reactivity: −0.019 (SD = 0.28)
- t = −1.15, p = .25

On average, vigor does not reliably increase at predator encounter after demand residualization. The individual variation is massive (SD = 0.28) relative to the near-zero mean.

**Interpretation:** There is no universal encounter mobilization effect in the demand-residualized signal. The population-average encounter response is masked by enormous individual differences — some people ramp up dramatically, others show no change or even decrease.

### Encounter reactivity is highly trait-like

Cross-block stability correlations:

| Block pair | r |
|-----------|---|
| Block 0 ↔ Block 1 | 0.73 |
| Block 1 ↔ Block 2 | 0.82 |
| Block 0 ↔ Block 2 | 0.78 |
| **Mean** | **0.78** |

These are high stability coefficients, comparable to trait questionnaire measures. The encounter response is a stable individual difference that persists across ~20 minutes of task performance.

### Encounter reactivity correlates with static capture aversion (cd)

| Predictor | r | p |
|-----------|---|---|
| Reactivity × log(cd) | **+0.47** | **< 0.0001** |
| Reactivity × log(ce) | −0.04 | 0.49 |

The encounter reflex tracks cd, not ce. Individuals with high capture aversion (who press hard on average) also show larger encounter responses. This means cd captures both:
- **Tonic vigor:** how hard you press throughout the trial (trial-level)
- **Phasic vigor:** how strongly you mobilize when the predator appears (within-trial)

### Threat does NOT modulate the encounter response

| Threat | Mean reactivity | SD |
|--------|----------------|-----|
| T = 0.1 | −0.022 | 0.31 |
| T = 0.5 | −0.018 | 0.33 |
| T = 0.9 | −0.017 | 0.30 |

One-way ANOVA: F(2, 876) = 0.04, **p = .96**

The encounter response is identical across threat levels. Whether you knew there was a 10% or 90% chance of attack, when the predator appears, you respond by the same amount. This is fundamentally different from CHOICE, which is strongly threat-modulated (β = −1.28).

**Interpretation:** The encounter response operates at the reactive end of Mobbs' defensive cascade — a threat-independent motor mobilization triggered by predator detection, not a probability-weighted strategic adjustment. Choice reflects strategic defense (evaluating probabilities), while the encounter reflex reflects reactive defense (responding to presence).

### Piecewise slope change at encounter

| Phase | Mean slope (excess_cc / second) |
|-------|-------------------------------|
| Pre-encounter | −0.021 |
| Post-encounter | +0.029 |
| **Change** | **+0.050** |

Paired t-test on slope change: t = 5.91, **p < 10⁻⁸**

The vigor trajectory reliably shifts from slightly declining (pre-encounter) to increasing (post-encounter) at the moment of predator detection.

### Clinical associations

| Clinical measure | r | p | Significant? |
|-----------------|---|---|-------------|
| AMI_Behavioural | **−0.191** | **0.001** | *** |
| AMI_Total | **−0.188** | **0.001** | ** |
| PHQ9_Total | −0.060 | 0.30 | |
| DASS21_Depression | −0.037 | 0.53 | |
| STAI_State | −0.030 | 0.61 | |
| STICSA_Total | +0.032 | 0.58 | |
| OASIS_Total | −0.003 | 0.96 | |
| DASS21_Anxiety | +0.030 | 0.61 | |
| DASS21_Stress | −0.003 | 0.96 | |
| AMI_Emotional | −0.019 | 0.75 | |

Only apathy measures (AMI_Behavioural, AMI_Total) show significant associations. All anxiety, depression, and stress measures are null.

**Interpretation:** Encounter reactivity relates to MOTIVATIONAL engagement (apathy), not AFFECTIVE distress (anxiety). People with higher apathy show blunted encounter responses — a general motivational deficit affecting the phasic defensive mobilization.

### Incremental prediction: reactivity adds to static model for apathy

Hierarchical regression predicting AMI_Total:
- Step 1 (log(cd) + log(ce)): R² = 0.018, F = 2.62, p = .074
- Step 2 (+ reactivity): R² = 0.066, **ΔR² = 0.048, F_change = 14.4, p = .0002**

Reactivity explains 4.8% additional variance in apathy BEYOND what the static model parameters capture. This confirms that within-trial dynamics contain information not present in trial-level averages.

### Connection to metacognitive discrepancy

Reactivity × discrepancy: r = +0.096, p = .10 (trending but not significant)

The encounter reflex is largely independent of the affective discrepancy that predicts anxiety symptoms. This supports two dissociable clinical pathways:
- **Affective pathway:** discrepancy → anxiety symptoms (r = 0.18–0.34)
- **Motor pathway:** encounter reactivity → apathy (r = −0.19)

---

## Theoretical Interpretation

### Strategic vs reactive defense

The dissociation between threat-modulated choice and threat-independent encounter reactivity maps directly onto Mobbs et al.'s (2020) defensive behavior framework:

- **Strategic defense** (distant threat): Prefrontal evaluation of costs and benefits. Captured by ce (effort cost) in the choice equation. Threat-modulated (T enters through S).
- **Reactive defense** (proximal threat): Subcortical motor mobilization at predator detection. Captured by cd (capture aversion) in the vigor equation AND the encounter reflex. Threat-independent (same response regardless of T).

The static EVC model captures both strategic (ce, choice) and reactive (cd, tonic vigor) components. The within-trial dynamics add temporal resolution: cd governs not just how hard you press on average, but how quickly and strongly your motor system mobilizes when danger materializes.

### Two clinical pathways

| Pathway | Predictor | Clinical outcome | Mechanism |
|---------|-----------|-----------------|-----------|
| Affective | Discrepancy (anxiety > danger) | Anxiety symptoms (r = 0.18–0.34) | Interoceptive prediction error |
| Motor | Encounter reactivity | Apathy (r = −0.19) | Blunted defensive mobilization |

These pathways are independent (reactivity × discrepancy r = 0.10, ns), suggesting dissociable neural substrates: the affective pathway likely involves anterior insula and dACC (interoceptive monitoring), while the motor pathway involves PAG and basal ganglia (defensive motor circuits).

---

## Confirmation Plan

### Tests to run on confirmation sample (N ≈ 280–330):

| Test | Statistic | Threshold | Discovery value |
|------|-----------|-----------|-----------------|
| Trait stability | Cross-block r | > 0.50 | 0.78 |
| Reactivity × log(cd) | Pearson r | > 0, p < .05 | 0.47 |
| Threat non-modulation | ANOVA F | p > .10 | F=0.04, p=.96 |
| Reactivity → AMI | Pearson r | < 0, p < .05 | −0.19 |
| Incremental ΔR² | F-test | p < .05 | ΔR²=0.048, p=.0002 |

Note: H5 is EXPLORATORY in the discovery sample. If included in the preregistration, the thresholds above would be used. If not preregistered, these results would be reported as exploratory findings in the confirmatory analysis.

---

## Summary

| Finding | Statistic | p | Verdict |
|---------|-----------|---|---------|
| Trait-like stability | r = 0.78 cross-block | — | **CONFIRMED** |
| Correlates with cd | r = 0.47 | < .0001 | **CONFIRMED** |
| Independent of ce | r = −0.04 | .49 | **CONFIRMED** |
| NOT threat-modulated | F = 0.04 | .96 | **CONFIRMED** |
| Slope change at encounter | t = 5.91 | < 10⁻⁸ | **CONFIRMED** |
| Predicts apathy (AMI) | r = −0.19 | .001 | **CONFIRMED** |
| Anxiety measures | all |r| < 0.06 | all p > .30 | **NULL** |
| Incremental over static | ΔR² = 0.048 | .0002 | **CONFIRMED** |

**Bottom line:** The encounter reflex is a stable, threat-independent defensive mobilization that co-varies with capture aversion (cd) and predicts apathy but not anxiety. Together with the discrepancy→anxiety pathway, this establishes two dissociable clinical dimensions: affect predicts affect disorders, motor predicts motor disorders.
