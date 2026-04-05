# Results

All hypotheses were preregistered prior to analysis of the confirmatory sample (OSF: [URL]). The exploratory sample (N = 290, after exclusions from 350 recruited) was used to develop all hypotheses, model specifications, and statistical thresholds. The confirmatory sample (N = 281, after exclusions from 350 recruited) was collected independently using an identical task. We report confirmatory results as the primary analysis; exploratory results are shown alongside for comparison. Of 24 preregistered tests, 22 (92%) were confirmed in the independent sample.

---

## H1: Threat Drives Adaptive Shifts in Choice, Vigor, and Affect

**H1a.** Threat probability and distance both reduced high-effort choices in both samples. Logistic regression with cluster-robust standard errors:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(threat) | -1.02 (p < 10⁻¹¹⁰) | -0.91 (p < 10⁻⁸⁷) |
| β(distance) | -0.75 (p < 10⁻¹²⁴) | -0.67 (p < 10⁻¹⁰⁸) |
| Interaction | -0.20 (p < 10⁻¹⁵) | -0.12 (p < 10⁻⁶) |

Both tests passed (p < .01) in both samples. **Confirmed.**

**H1b.** Anxiety increased with threat and confidence decreased with threat in both samples. Linear mixed models (response ~ threat_z + (1 + threat_z | subj)):

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| Anxiety: β(threat) | +0.58 (z = 14.7) | +0.53 (z = 12.5) |
| Confidence: β(threat) | -0.58 (z = -13.7) | -0.67 (z = -15.3) |

Both |t| > 3 in both samples. **Confirmed.**

**H1c.** Within each cookie type, normalized press rate increased with threat. LMM controlling for cookie type:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(threat) | +0.020 (z = 18.4) | +0.017 (z = 17.4) |
| d (heavy, T=0.1→0.9) | +0.24 | +0.45 |
| d (light, T=0.1→0.9) | +0.44 | +0.76 |

The within-cookie analysis confirms that the threat-vigor relationship is not an artifact of Simpson's paradox. **Confirmed.**

**All 5 H1 tests passed in both samples.**

---

## H2: Vigor Dynamics Follow the Predatory Imminence Continuum

**H2a.** Predator encounter triggered a rapid motor spike in pressing rate in both samples:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| d (spike) | +0.56 | +0.65 |
| p | < 10⁻²⁰ | < 10⁻²⁴ |
| % subjects positive | 83% | 83% |

The spike did not scale with threat probability in either sample (both p > .05), consistent with a reflexive post-encounter response. **Confirmed.**

**H2b.** GAM likelihood ratio tests confirmed distinct temporal signatures in both samples:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| Encounter LRT χ² | 760 (p ≈ 0) | 1025 (p ≈ 0) |
| Threat LRT χ² | 292 (p < 10⁻¹⁵) | 115 (p < 10⁻¹⁵) |

Both p < .01 in both samples. **Confirmed.**

**All 3 H2 tests passed in both samples.**

---

## H3: The Joint Fitness Model Outperforms All Alternatives

Four models were fitted via NumPyro HMC/NUTS (4 chains, target_accept = 0.95) and compared on joint (choice + vigor) WAIC and PSIS-LOO. All models were evaluated on the same data.

### Exploratory Sample

| Model | WAIC | ΔWAIC | LOO | Converged |
|-------|------|-------|-----|-----------|
| M4 (Joint) | 12,776 | 0 | 12,779 | Yes |
| M2 (Threat-only) | 14,742 | +1,966 | 14,745 | Yes |
| M3 (Single-param) | 15,374 | +2,599 | 15,404 | Yes* |
| M1 (Effort-only) | 17,505 | +4,729 | 17,509 | Yes |

### Confirmatory Sample

| Model | WAIC | ΔWAIC | LOO | Converged |
|-------|------|-------|-----|-----------|
| M4 (Joint) | 12,252 | 0 | 12,263 | Yes |
| M2 (Threat-only) | 13,873 | +1,621 | 13,881 | Yes |
| M3 (Single-param) | 15,727 | +3,474 | 15,737 | No** |
| M1 (Effort-only) | 16,037 | +3,785 | 16,042 | Yes |

*Converged after doubled iterations (4,000 warmup + 8,000 samples).
**Did not converge even after doubled iterations; ΔWAIC is nevertheless decisive.

**H3a.** M4 outperformed M1 (ΔWAIC = +3,785 / +4,729; ΔLOO agrees). **Confirmed.**
**H3b.** M4 outperformed M2 (ΔWAIC = +1,621 / +1,966; ΔLOO agrees). **Confirmed.**
**H3c.** M4 outperformed M3 (ΔWAIC = +3,474 / +2,599; ΔLOO agrees). **Confirmed.**

WAIC and LOO agreed on all comparisons in both samples. M4 achieved choice accuracy = 0.76–0.77 and vigor r² = 0.37–0.41. The supplementary scaled single-parameter model (M3b) also lost decisively (ΔWAIC = +1,597–1,959), ruling out a scale mismatch.

**All 3 H3 tests passed in both samples.**

---

## H4: Foraging Profiles and Optimality

All regressions used Bayesian linear models (bambi, 4 chains × 2,000 draws + 1,000 tuning, 95% HDI).

**H4a.** Avoidance sensitivity predicted escape rate:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(ω) | +0.060 [+0.029, +0.093] | +0.046 [+0.017, +0.075] |
| Pass | Yes | Yes |

**H4b.** Overcaution was the dominant error (79% exploratory, 90% confirmatory). ω predicted overcaution ratio:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(ω → OC) | +0.177 [+0.163, +0.193] | +0.123 [+0.109, +0.137] |
| Pass | Yes | Yes |

**H4c.** κ predicted pressing intensity:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(κ) | -0.194 [-0.215, -0.173] | -0.196 [-0.217, -0.176] |
| Pass | Yes | Yes |

**H4d.** The ω-κ angle predicted decision quality:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(angle) | -0.041 [-0.055, -0.026] | -0.054 [-0.072, -0.036] |
| Pass | Yes | Yes |

**H4e.** Model consistency did not predict earnings in the confirmatory sample:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(choice cons) | +14.3 [+5.0, +23.2] ✓ | +8.4 [-2.3, +19.0] ✗ |
| β(intensity dev) | -19.3 [-28.8, -9.4] ✓ | -4.1 [-14.6, +7.4] ✗ |

The indirect link between computational consistency and earnings did not replicate. **Not confirmed.**

**5/7 H4 tests passed in the confirmatory sample (7/7 in exploratory).**

---

## H5: Metacognitive Monitoring of the Foraging Computation

**H5a.** Anxiety calibration predicted foraging optimality beyond ω and κ. LOO comparison of base vs full model:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| Optimality ΔELPD | > 0 (SE excl 0) | > 0 (SE excl 0) |
| Escape ΔELPD | > 0 (SE excl 0) | > 0 (SE excl 0) |
| Earnings ΔELPD | > 0 (SE excl 0) | > 0 (SE excl 0) |
| Outcomes improved | 3/3 | 3/3 |

**Confirmed.**

**H5b.** Anxiety slope predicted adaptive choice shifting:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(slope → shift) | +0.123 [HDI excl 0] | +0.099 [+0.065, +0.134] |
| Pass | Yes | Yes |

**Confirmed.**

**H5c.** ω predicted confidence but not anxiety — the appraisal dissociation:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| ω → confidence | β < 0, HDI excl 0 ✓ | -0.181 [-0.340, -0.037] ✓ |
| ω → anxiety | HDI incl 0 ✓ | -0.067 [-0.221, +0.078] ✓ (ROPE) |

The computational capture-cost parameter maps onto a coping appraisal (confidence), not an affective threat response (anxiety). **Confirmed.**

**H5d.** Confidence predicted error type, not error rate:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| conf → overcautious | β < 0, HDI excl 0 ✓ | -1.48 [-2.39, -0.54] ✓ |
| conf → reckless | β > 0, HDI excl 0 ✓ | +0.29 [+0.07, +0.52] ✓ |

**Confirmed.**

**All 7 H5 tests passed in both samples.**

---

## Summary

| Hypothesis | Tests | Exploratory | Confirmatory |
|------------|-------|-------------|--------------|
| H1: Adaptive shifts | 5 | 5/5 | 5/5 |
| H2: Vigor dynamics | 3 | 3/3 | 3/3 |
| H3: Model comparison | 3 | 3/3 | 3/3 |
| H4: Profiles & optimality | 7 | 7/7 | 5/7 |
| H5: Metacognition | 6 | 6/6 | 6/6 |
| **Total** | **24** | **24/24** | **22/24 (92%)** |

---

## Exploratory: Task Affect Dissociates Clinical Dimensions

In pooled analyses (N = 563), we examined whether task-elicited affect signals predicted clinical symptom dimensions, with ω and κ as covariates.

**Triple dissociation of affect → clinical mapping:**

| Task affect | DASS-Anxiety | DASS-Depression | AMI (apathy) | STAI Trait |
|-------------|-------------|-----------------|-------------|-----------|
| Mean anxiety level | **+0.24*** | +0.19* | null | -0.17* |
| Confidence level | null | **-0.16*** | **-0.22*** | +0.17* |
| Calibration | null | null | **+0.11*** | -0.09* |

Task anxiety indexes general distress. Low confidence specifically predicts depression and apathy. Good calibration predicts apathy — disengagement facilitates accurate threat monitoring but at a motivational cost.

The computational parameters (ω, κ) did not directly predict clinical symptoms, consistent with the view that psychopathology arises from how people appraise their computations (the affect layer), not from the computations themselves.

**Apathy as the behaviorally relevant clinical dimension:** In Bayesian multiple regressions with all clinical scales entered simultaneously, only AMI (apathy) predicted foraging outcomes — escape rate (β = +0.36), earnings (β = +0.35), and choice shift (β = +0.20). All other clinical measures (DASS, PHQ-9, OASIS, MFIS, STICSA, STAI) were null after controlling for AMI.
