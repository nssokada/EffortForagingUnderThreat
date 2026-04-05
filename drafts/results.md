# Results

All hypotheses were preregistered prior to analysis of the confirmatory sample (OSF: [URL]). The exploratory sample (N = 290) was used to develop all hypotheses, model specifications, and statistical thresholds. An independent confirmatory sample (N = 281) was collected using an identical task. We report confirmatory results as the primary analysis with exploratory results shown for comparison. Of 24 preregistered tests, 22 (92%) were confirmed.

---

## H1: Threat Drives Adaptive Shifts in Choice, Vigor, and Affect

**H1a.** Threat probability and distance both reduced high-effort patch choices (Fig. Xa). In a logistic model with cluster-robust standard errors, threat (confirmatory: β = −0.91, z = −19.8, P < 0.001; exploratory: β = −1.02, z = −22.3, P < 0.001) and distance (confirmatory: β = −0.67, z = −22.1, P < 0.001; exploratory: β = −0.75, z = −23.7, P < 0.001) both had strong negative effects, with a significant interaction (confirmatory: β = −0.12, P < 0.001). Both tests passed in both samples.

**H1b.** Anxiety increased with threat (confirmatory: β = 0.53, z = 12.5, P < 0.001; exploratory: β = 0.58, z = 14.7, P < 0.001) and confidence decreased (confirmatory: β = −0.67, z = −15.3, P < 0.001; exploratory: β = −0.58, z = −13.7, P < 0.001). All |z| > 3, as preregistered.

**H1c.** Within each cookie type, normalized press rate increased with threat (Fig. Xb). Controlling for cookie type in a linear mixed model, threat increased vigor in both samples (confirmatory: β = 0.017, z = 17.4, P < 0.001; exploratory: β = 0.020, z = 18.4, P < 0.001). Effect sizes were moderate for heavy cookies (confirmatory: d = 0.45; exploratory: d = 0.24) and large for light cookies (confirmatory: d = 0.76; exploratory: d = 0.44). The within-cookie analysis confirms that the threat–vigor relationship holds when controlling for cookie type.

All 5 H1 tests passed in both samples.

---

## H2: Vigor Dynamics Follow the Predatory Imminence Continuum

**H2a.** Predator encounter triggered a rapid motor spike in pressing rate (Fig. Xc). The per-subject encounter spike (attack minus non-attack reactive-epoch pressing rate) was large and reliable (confirmatory: d = 0.65, t(280) = 11.0, P < 0.001; exploratory: d = 0.56, P < 0.001), with 83% of participants showing a positive spike in both samples. The spike magnitude did not differ between high and low threat (confirmatory: t = −1.42, P = 0.16; exploratory: P = 0.21), consistent with a reflexive post-encounter defense response.

**H2b.** Generalized additive model (GAM) likelihood ratio tests confirmed distinct temporal signatures for encounter status (confirmatory: χ² = 1,025, P < 0.001; exploratory: χ² = 760, P < 0.001) and threat level (confirmatory: χ² = 115, P < 0.001; exploratory: χ² = 292, P < 0.001), consistent with the predatory imminence continuum's distinction between anticipatory (threat-modulated) and reactive (encounter-triggered) defense.

All 3 H2 tests passed in both samples.

---

## H3: The Joint Fitness Model Outperforms All Alternatives

We compared four models using identical HMC/NUTS inference (4 chains × 2,000 warmup + 4,000 samples, target acceptance = 0.95). All models were evaluated on the same joint likelihood (choice + vigor) using WAIC (primary) and PSIS-LOO (robustness).

| Model | Description | WAIC (conf.) | ΔWAIC | WAIC (expl.) | ΔWAIC |
|-------|-------------|-------------|-------|-------------|-------|
| M4 | Joint W(u): ω + κ | 12,252 | — | 12,776 | — |
| M2 | Threat-only: ω, pop. κ | 13,873 | +1,621 | 14,742 | +1,966 |
| M3 | Single-param: θ = ω = κ | 15,727 | +3,474 | 15,374 | +2,599 |
| M1 | Effort-only: κ, null vigor | 16,037 | +3,785 | 17,505 | +4,729 |

M4 outperformed all alternatives decisively. WAIC and PSIS-LOO agreed on every comparison in both samples. M4 achieved choice accuracy of 0.76 (confirmatory) / 0.77 (exploratory) and vigor r² = 0.41 / 0.37. A supplementary scaled single-parameter model (M3b: θ as ω, αθ as κ) also lost to M4 (ΔWAIC = 1,597 confirmatory, 1,959 exploratory), ruling out a simple scale mismatch. M3 did not converge in the confirmatory sample even after doubled iterations (4,000 warmup + 8,000 samples), as prespecified; the ΔWAIC was nevertheless decisive.

**H3a** (M4 vs M1): confirmed. **H3b** (M4 vs M2): confirmed. **H3c** (M4 vs M3): confirmed.

---

## H4: Foraging Profiles and Optimality

All regressions used Bayesian linear models (bambi; 4 chains × 2,000 draws + 1,000 tuning). We report posterior means with 95% highest density intervals (HDI).

**H4a.** Avoidance sensitivity (ω) predicted escape rate on attack trials (confirmatory: β = 0.046, 95% HDI [0.017, 0.075]; exploratory: β = 0.060, 95% HDI [0.029, 0.093]). Participants who perceived capture as more costly adopted strategies that increased survival. Confirmed.

**H4b.** Among suboptimal choices, the majority were overcautious — choosing the low-reward patch when the high-reward patch had higher expected value (confirmatory: 90%; exploratory: 79%). ω predicted the overcaution ratio (confirmatory: β = 0.123, 95% HDI [0.109, 0.137]; exploratory: β = 0.177, 95% HDI [0.163, 0.193]). Confirmed.

**H4c.** Activation intensity (κ) predicted pressing intensity (confirmatory: β = −0.196, 95% HDI [−0.217, −0.176]; exploratory: β = −0.194, 95% HDI [−0.215, −0.173]). The effort cost parameter governs motor output — the activation side of the avoid–activate decomposition. Confirmed.

**H4d.** The ω–κ angle predicted decision quality (confirmatory: β = −0.054, 95% HDI [−0.072, −0.036]; exploratory: β = −0.041, 95% HDI [−0.055, −0.026]). Effort-driven avoidance (high κ relative to ω) was less optimal than threat-driven avoidance because it is indiscriminate across threat levels. Confirmed.

**H4e.** Neither choice consistency (confirmatory: β = 8.4, 95% HDI [−2.3, 19.0]) nor intensity deviation (confirmatory: β = −4.1, 95% HDI [−14.6, 7.4]) significantly predicted earnings. Both effects were significant in the exploratory sample (choice: β = 14.3, 95% HDI [5.0, 23.2]; intensity: β = −19.3, 95% HDI [−28.8, −9.4]) but did not replicate. Not confirmed.

5 of 7 H4 tests passed in the confirmatory sample (7 of 7 in exploratory).

---

## H5: Metacognitive Monitoring of the Foraging Computation

**H5a.** Anxiety calibration (within-subject r between anxiety and threat) predicted foraging optimality beyond ω and κ. LOO-CV comparison of base (optimality ~ ω + κ) versus full (+ calibration) models favoured the full model for all three outcomes (optimality, escape rate, and earnings; all ΔELPD > 0 with standard errors excluding zero) in both samples. Confirmed.

**H5b.** Anxiety slope (reactivity to threat) predicted adaptive choice shifting (confirmatory: β = 0.099, 95% HDI [0.065, 0.134]; exploratory: β = 0.123, 95% HDI excluding zero). Participants whose anxiety responded more strongly to threat shifted their choices more across threat levels. Confirmed.

**H5c.** ω predicted subjective confidence (confirmatory: β = −0.181, 95% HDI [−0.340, −0.037]) but not anxiety (confirmatory: β = −0.067, 95% HDI [−0.221, 0.078], within the prespecified ROPE of [−0.10, 0.10]). The computational capture-cost parameter maps onto a coping appraisal ("can I handle this?") rather than an affective threat response ("is this dangerous?"), consistent with Lazarus's distinction between secondary and primary appraisal. Confirmed.

**H5d.** Confidence predicted error type but not error rate. Higher confidence predicted fewer overcautious errors (confirmatory: β = −1.48, 95% HDI [−2.39, −0.54]) and more reckless errors (confirmatory: β = 0.29, 95% HDI [0.07, 0.52]). Confidence determines what foragers commit to, not whether they succeed. Confirmed.

All 7 H5 tests passed in both samples.

---

## Summary of Preregistered Tests

| Hypothesis family | Tests | Exploratory | Confirmatory |
|-------------------|-------|-------------|--------------|
| H1: Adaptive shifts | 5 | 5/5 | 5/5 |
| H2: Vigor dynamics | 3 | 3/3 | 3/3 |
| H3: Model comparison | 3 | 3/3 | 3/3 |
| H4: Profiles and optimality | 7 | 7/7 | 5/7 |
| H5: Metacognitive monitoring | 6 | 6/6 | 6/6 |
| **Total** | **24** | **24/24** | **22/24 (92%)** |

---

## Exploratory: Task Affect Dissociates Clinical Symptom Dimensions

In pooled analyses (N = 563), we examined whether the three task-elicited affect signals predicted clinical symptom dimensions, controlling for ω and κ (Table X). Task-elicited anxiety level predicted clinical anxiety (DASS-Anxiety: β = 0.24, 95% HDI excluding zero) and general distress but not apathy. Confidence predicted depression (DASS-Depression: β = −0.16, 95% HDI excluding zero) and apathy (AMI: β = −0.22, 95% HDI excluding zero) but not clinical anxiety. Calibration predicted apathy (AMI: β = 0.11, 95% HDI excluding zero), suggesting that accurate threat monitoring is associated with motivational disengagement. The computational parameters (ω, κ) did not directly predict clinical symptoms, consistent with the view that psychopathology relates to how individuals appraise their survival computations rather than to the computations themselves.

In Bayesian multiple regressions with all clinical scales entered simultaneously, only apathy (AMI) predicted foraging outcomes: escape rate (β = 0.36, 95% HDI excluding zero), earnings (β = 0.35), and choice shift (β = 0.20). All other clinical measures were non-significant after controlling for apathy.
