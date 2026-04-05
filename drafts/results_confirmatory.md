# Results (Confirmatory Sample, N = 281)

All hypotheses were preregistered prior to analysis of the confirmatory sample (OSF: [URL]). The exploratory sample (N = 290) was used to develop all hypotheses, model specifications, and statistical thresholds. The confirmatory sample (N = 281, after exclusions from 350 recruited) was collected from an independent pool of Prolific participants using an identical task. Of 24 preregistered tests, 22 (92%) were confirmed.

---

## H1: Threat Drives Adaptive Shifts in Choice, Vigor, and Affect

**H1a.** Threat probability and distance both reduced high-effort choices, as predicted. In a logistic model with cluster-robust standard errors, threat had a strong negative effect on P(heavy) (β = -0.91, p < 10⁻⁸⁷) as did distance (β = -0.67, p < 10⁻¹⁰⁸), with a significant interaction (β = -0.12, p < 10⁻⁶). Both tests passed at p < .01.

**H1b.** Anxiety increased with threat (LMM: β = +0.53, z = 12.5, p < 10⁻³⁶) and confidence decreased with threat (β = -0.67, z = -15.3, p < 10⁻⁵²). Both |t| > 3, as preregistered.

**H1c.** Within each cookie type, normalized press rate increased with threat (LMM controlling for cookie type: β = +0.017, z = 17.4, p < 10⁻⁶⁷). The within-cookie analysis confirms that the threat-vigor relationship is not an artifact of Simpson's paradox (threat shifts cookie composition, which changes average vigor). Both heavy (d = 0.45) and light (d = 0.76) cookies showed significant increases from T=0.1 to T=0.9.

**All 5 H1 tests passed.**

---

## H2: Vigor Dynamics Follow the Predatory Imminence Continuum

**H2a.** Predator encounter triggered a rapid motor spike in pressing rate. The per-subject mean reactive-epoch pressing rate was significantly higher on attack versus non-attack trials (d = 0.65, p < 10⁻²⁴), with 83% of participants showing a positive spike. The spike did not scale with threat probability (T=0.9 vs T=0.1: p = 0.16), consistent with a reflexive post-encounter response that is threat-independent.

**H2b.** GAM likelihood ratio tests confirmed distinct temporal signatures for both encounter status (χ² = 1024.8, p < 10⁻³⁰⁰) and threat level (χ² = 114.8, p < 10⁻¹⁵). The vigor timecourse differed qualitatively between attack and non-attack trials (encounter-aligned) and between threat levels (trial-aligned), consistent with the predatory imminence continuum's distinction between anticipatory (threat-modulated) and reactive (encounter-triggered) defense.

**All 3 H2 tests passed.**

---

## H3: The Joint Fitness Model Outperforms All Alternatives

Four models were fitted via NumPyro HMC/NUTS (4 chains × 2,000 warmup + 4,000 samples, target_accept = 0.95) and compared on joint (choice + vigor) WAIC and PSIS-LOO. All models were evaluated on the same data to ensure fair comparison.

| Model | Description | WAIC | LOO | ΔWAIC vs M4 | Converged |
|-------|-------------|------|-----|-------------|-----------|
| M4 | Joint W(u) (ω + κ) | 12,252 | 12,263 | 0 (best) | Yes |
| M2 | Threat-only (ω, pop κ) | 13,873 | 13,881 | +1,621 | Yes |
| M3 | Single-param (θ = ω = κ) | 15,727 | 15,737 | +3,474 | No* |
| M1 | Effort-only (κ, null vigor) | 16,037 | 16,042 | +3,785 | Yes |

*M3 did not converge after doubled iterations (4,000 warmup + 8,000 samples), as preregistered for non-convergent models. The ΔWAIC is nevertheless decisive.

**H3a.** M4 outperformed M1 on both WAIC (Δ = +3,785) and LOO (Δ = +3,779). Threat matters beyond effort cost. **Confirmed.**

**H3b.** M4 outperformed M2 on both WAIC (Δ = +1,621) and LOO (Δ = +1,618). Individual differences in effort sensitivity matter beyond threat sensitivity. **Confirmed.**

**H3c.** M4 outperformed M3 on both WAIC (Δ = +3,474) and LOO (Δ = +3,474). Avoidance sensitivity and activation intensity are separable traits — a single parameter cannot serve both roles. **Confirmed.**

M4 achieved choice accuracy = 0.76 and vigor r² = 0.41 in the confirmatory sample. The supplementary scaled single-parameter model (M3b, exploratory) also lost decisively to M4 (ΔWAIC = +1,597), ruling out a simple scale mismatch as the explanation for M3's failure.

**All 3 H3 tests passed. WAIC and LOO agreed on all comparisons.**

---

## H4: Foraging Profiles and Optimality

All H4 regressions used Bayesian linear models (bambi, 4 chains × 2,000 draws + 1,000 tuning). Inference criterion: 95% HDI excludes zero.

**H4a.** Avoidance sensitivity predicted escape rate on attack trials (β = +0.046, 95% HDI [+0.017, +0.075]). People who perceive capture as costly adopt strategies that increase survival. **Confirmed.**

**H4b.** Among suboptimal choices, 90% were overcautious (choosing light when heavy had higher expected reward). Avoidance sensitivity predicted the overcaution ratio (β = +0.123, HDI [+0.109, +0.137]). High-ω individuals systematically avoid the risky option even when it is optimal. **Confirmed.**

**H4c.** Activation intensity predicted pressing intensity (β = -0.196, HDI [-0.217, -0.176]). κ governs motor output — the activation side of the avoid-activate decomposition. **Confirmed.**

**H4d.** The ω-κ angle predicted decision quality (β = -0.054, HDI [-0.072, -0.036]). Effort-driven avoidance (high κ relative to ω) is less optimal than threat-driven avoidance (high ω relative to κ) because it is indiscriminate across threat levels. **Confirmed.**

**H4e.** Neither choice consistency (β = +8.4, HDI [-2.3, +19.0]) nor intensity deviation (β = -4.1, HDI [-14.6, +7.4]) predicted earnings in the confirmatory sample. The exploratory effects (choice r = +0.25, intensity r = -0.43) did not replicate. **Not confirmed.**

**5/7 H4 tests passed.** The core predictions about ω, κ, and their balance all confirmed. The model consistency → earnings prediction (H4e) did not replicate, suggesting the indirect link between computational consistency and task performance is weaker than the direct parameter-outcome relationships.

---

## H5: Metacognitive Monitoring of the Foraging Computation

**H5a.** Anxiety calibration predicted foraging optimality beyond ω and κ. LOO comparison of base (pct_optimal ~ ω + κ) versus full (+ calibration) models favored the full model for all three outcomes: optimality, escape rate, and earnings (ΔELPD > 0, SE excluding zero for all three). The metacognitive monitor adds information the first-order computation does not contain. **Confirmed.**

**H5b.** Anxiety slope predicted adaptive choice shifting (β = +0.099, HDI [+0.065, +0.134]). People whose anxiety reacts more strongly to threat shift their choices more across threat levels. **Confirmed.**

**H5c.** Avoidance sensitivity predicted subjective confidence (β = -0.181, HDI [-0.340, -0.037]) but not anxiety (β = -0.067, HDI [-0.221, +0.078], within ROPE [-0.10, +0.10]). The computational capture-cost parameter maps onto a coping appraisal (confidence: "can I handle this?"), not an affective threat response (anxiety: "is this dangerous?"). This dissociation is consistent with Lazarus's (1991) distinction between secondary appraisal (coping evaluation) and primary appraisal (threat evaluation). **Confirmed.**

**H5d.** Confidence predicted error type but not error rate. Higher confidence predicted fewer overcautious errors (β = -1.48, HDI [-2.39, -0.54]) and more reckless errors (β = +0.29, HDI [+0.07, +0.52]). Confidence determines what people commit to, not whether they succeed. **Confirmed.**

**All 7 H5 tests passed.**

---

## Summary of Preregistered Tests

| Hypothesis family | Tests | Passed | Rate |
|-------------------|-------|--------|------|
| H1: Adaptive shifts | 5 | 5 | 100% |
| H2: Vigor dynamics | 3 | 3 | 100% |
| H3: Model comparison | 3 | 3 | 100% |
| H4: Profiles & optimality | 7 | 5 | 71% |
| H5: Metacognitive monitoring | 6 | 6 | 100% |
| **Total** | **24** | **22** | **92%** |

---

## Exploratory: Task Affect Dissociates Clinical Dimensions

In pooled analyses (N = 563, both samples), we examined whether the three task-elicited affect signals — anxiety level, confidence level, and anxiety calibration — predicted clinical symptom dimensions. All regressions included ω and κ as covariates.

Task-elicited anxiety level predicted clinical anxiety (DASS-Anxiety: β = +0.24, HDI excludes zero; OASIS: β = +0.20) and general distress (DASS-Depression: β = +0.19) but not apathy (AMI: null). Task-elicited confidence predicted depression and apathy specifically (DASS-Depression: β = -0.16; AMI: β = -0.22) but not clinical anxiety (DASS-Anxiety: null). Anxiety calibration predicted apathy (AMI: β = +0.11) — better calibrated individuals scored higher on apathy, consistent with emotional disengagement facilitating accurate threat monitoring.

These results suggest a triple dissociation: anxiety level indexes general distress, confidence indexes motivational deficit (depression/apathy), and calibration indexes the cost of accurate monitoring (disengagement). The computational parameters (ω, κ) did not predict clinical symptoms directly, consistent with the view that psychopathology arises from how people appraise their computations, not from the computations themselves.

The apathy factor (AMI) was the only clinical dimension that predicted foraging performance in Bayesian multiple regressions: AMI → escape rate (β = +0.36), AMI → earnings (β = +0.35), AMI → choice shift (β = +0.20), while all other clinical measures (DASS, PHQ-9, OASIS, MFIS, STICSA) were null after controlling for AMI. More apathetic individuals adopted cautious foraging strategies that maximized survival.
