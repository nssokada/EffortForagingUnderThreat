# Preregistration: Joint Optimization of Patch Selection and Motor Vigor Under Predation Risk

**Authors:** Noah Okada, Ketika Garg, Toby Wise, Dean Mobbs
**Target:** Nature Communications
**Registration:** Prior to confirmatory sample analysis (N ~ 350)
**Exploratory sample:** N = 290 (after exclusions from 350 collected)

---

## 1. Study Overview

### Scientific question

How do humans jointly determine which patches to forage and how intensely to work when foraging under predation risk? We test whether a single fitness function — grounded in optimal foraging theory (Bednekoff 2007; Brown 1999; Lima & Dill 1990) — can simultaneously explain patch selection (choice) and motor execution (vigor) through two individual-difference parameters: the subjective cost of capture (omega) and the subjective cost of effort (kappa).

### Core claims

1. Threat drives simultaneous but dissociable behavioral responses: avoidance in choice, activation in vigor, and calibrated affect (H1).
2. Vigor dynamics follow the predatory imminence continuum with distinct anticipatory and reactive signatures (H2).
3. A joint fitness function W(u) with two per-subject parameters outperforms simpler alternatives (H3).
4. The two parameters define ecologically meaningful foraging profiles that predict survival and optimality (H4).
5. Metacognitive signals — anxiety calibration and confidence — monitor the survival computation and independently predict foraging efficiency and decision style (H5).

### Task

Participants forage in a circular arena under predation risk. On each trial, they choose between a high-reward/high-effort cookie (R=5, req=0.9) and a low-reward/low-effort cookie (R=1, req=0.4), then press keys (S+D+F) to transport the cookie to the safe zone. A predator may appear based on the stated threat probability (T in {0.1, 0.5, 0.9}). The heavy cookie varies in distance (D in {1, 2, 3}); the light cookie is always at D=1. Capture costs 5 points plus the cookie reward. Probe trials (forced-choice, identical options) collect prospective anxiety and confidence ratings before pressing.

### Design

- 3 blocks x 27 trials = 81 events per participant (45 choice + 36 probe)
- Threat (T), distance (D), and cookie assignment fully crossed
- Calibration phase establishes maximum pressing speed per participant
- Questionnaires: DASS-21, PHQ-9, OASIS, STAI (State + Trait), AMI, MFIS, STICSA

---

## 2. Samples

### Exploratory sample

N = 350 recruited from Prolific. After exclusions (calibration outliers > 2.5 SD on mean IPI: subjects 154, 197, 208), N = 290 analyzed. All hypotheses developed and benchmarks established on this sample.

### Confirmatory sample

N ~ 350 from Prolific (independent from exploratory). Same task, same exclusion criteria. All hypotheses tested on this sample at preregistered thresholds. Data collected but not analyzed at time of registration.

---

## 3. Data and Metrics

### Choice

Binary: heavy (1) or light (0) on each choice trial. Per-subject P(heavy) as summary.

### Vigor

Normalized press rate = median(1/IPI) / calibrationMax, where IPI is inter-keypress interval. For timecourse analyses: 200ms bins from raw keypress timestamps at native ~5Hz, smoothed with 3-point centered moving average (600ms window). For model fitting: per-subject condition cell means (subject x threat x distance x cookie type, ~18 cells per subject, ~5,200 total), weighted by sqrt(n_trials) in the likelihood.

### Affect

Anxiety and confidence ratings on probe trials (1-10 scale). Per-subject decomposition: mean level, slope (regression on threat), and calibration (correlation with threat).

### Survival

Escape rate = proportion of attack trials where participant was not captured (trialEndState = 'escaped').

### Exclusions

Subjects with calibration mean IPI > 2.5 SD from the sample mean. No other exclusions.

---

## 4. The Joint Fitness Model

### Fitness function

W(u) = S(u, T, D) . R - (1 - S(u, T, D)) . omega . (R + C) - kappa . (u - req)^2 . D

where:
- S(u, T, D) = exp(-h . T^gamma . D / speed(u)) is survival probability
- speed(u) = sigmoid((u - 0.25 . req) / sigma_sp) is movement speed, saturating above req
- omega_i = per-subject cost of capture (how much being caught matters)
- kappa_i = per-subject cost of effort (how costly pressing is)
- R = cookie reward (5 or 1), C = 5 (capture penalty), req = required pressing rate (0.9 or 0.4)
- h, gamma, sigma_sp = population parameters (hazard scale, hazard exponent, speed saturation)

### Choice prediction

For each choice trial, compute V_H = max_u W_H(u) and V_L = max_u W_L(u) via soft-argmax over a grid of 40 pressing rates (0.1 to 1.5). P(heavy) = sigmoid((V_H - V_L) / tau).

### Vigor prediction

For each condition cell, compute u* = argmax_u W(u) for the chosen cookie. Observed cell-mean rate ~ Normal(u* + b_cookie . is_heavy, sigma_v / sqrt(n_trials)).

### Joint likelihood

L = Product over choice trials of Bernoulli(P_heavy) x Product over vigor cells of Normal(u*, sigma_cell)

Both omega and kappa enter both likelihoods through the same W function. Parameters estimated via NumPyro SVI (AutoNormal guide, ClippedAdam optimizer, 35,000 steps).

### Per-subject parameters

| Parameter | Symbol | Enters choice | Enters vigor | Recovery r |
|-----------|--------|--------------|-------------|-----------|
| Cost of capture | omega | Yes (via V_H, V_L) | Yes (via u*) | 0.90 |
| Cost of effort | kappa | Yes (via V_H, V_L) | Yes (via u*) | 0.78 |

### Population parameters

gamma (hazard exponent), h (hazard scale), tau (choice noise), sigma_v (vigor noise), sigma_sp (speed saturation width), b_cookie (residual cookie intercept).

---

## 5. Hypotheses

### H1: Threat Drives Adaptive Shifts in Choice, Vigor, and Affect

**H1a.** P(heavy) decreases with threat (logistic mixed model, beta(threat) < 0, p < .01) and with distance (beta(dist) < 0, p < .01).

**H1b.** Anxiety increases with threat (LMM beta > 0, |t| > 3.0) and confidence decreases (LMM beta < 0, |t| > 3.0).

**H1c.** Within heavy and within light cookies separately, normalized press rate increases from T=0.1 to T=0.9 (paired t, p < .01 both, d > 0.15). The marginal (unconditional) effect appears flat due to Simpson's paradox.

**H1d.** Per-subject P(heavy) and mean vigor are approximately independent (|r| < 0.10). Who avoids is not who mobilizes.

### H2: Vigor Dynamics Across the Predatory Imminence Continuum

**H2a.** Threat increases pressing rate within cookie type. Paired t-test (T=0.9 vs T=0.1): p < .01 for both heavy and light cookies.

**H2b.** Predator encounter triggers a motor spike (attack vs non-attack reactive epoch, one-sample t vs 0, p < .001) that does NOT scale with threat probability (paired t comparing spike at T=0.9 vs T=0.1: p > .05).

**H2c.** GAM likelihood ratio tests confirm distinct temporal signatures for encounter (chi-squared, p < .01) and threat (chi-squared, p < .01). GAMs use natural cubic regression splines (K=10) with cookie covariate and random intercepts by subject.

### H3: The Joint Fitness Model Outperforms Alternatives

We compare four models, all using cell-mean vigor data with saturating survival function:

- **M1 (Effort-only):** kappa per-subject, no threat in choice, no vigor model.
- **M2 (Threat-only):** omega per-subject, population kappa. No per-subject effort term in choice.
- **M3 (Single-parameter):** theta = omega = kappa. One parameter enters both channels.
- **M4 (Joint W):** omega + kappa per-subject, both enter W(u), joint likelihood.

All four models are fitted with identical inference: NumPyro HMC/NUTS, 4 chains × 2,000 warmup + 4,000 sampling iterations, target_accept = 0.95, max_tree_depth = 10. Models share the same population parameters (gamma, h, tau, sigma_v, sigma_sp, b_cookie) wherever structurally applicable; they differ only in the per-subject parameterization described above.

**Convergence requirements:** R-hat < 1.01 and bulk ESS > 400 for all parameters. If any model fails to converge, we double the sampling iterations (to 8,000) before declaring non-convergence.

**Primary criterion:** WAIC, computed from pointwise log-likelihoods across posterior samples (ArviZ). WAIC naturally penalizes model complexity through the effective number of parameters p_WAIC.

**Robustness criterion:** Approximate LOO-CV via Pareto-smoothed importance sampling (PSIS-LOO; ArviZ). If WAIC and LOO-CV agree on the winning model, we report the concordant result. If they disagree, we report both metrics, flag the discrepancy, and interpret the comparison as equivocal for the affected contrast — the hypothesis is supported only if both criteria agree.

**H3a.** M4 outperforms M1 (delta-WAIC > 0). Threat matters beyond effort.

**H3b.** M4 outperforms M2 (delta-WAIC > 0). Individual effort differences matter.

**H3c.** M4 outperforms M3 (delta-WAIC > 0). Capture cost and effort cost are separable — one parameter cannot serve both roles.

### H4: Foraging Profiles and Optimality

**H4a.** omega predicts escape rate on attack trials (OLS beta > 0, p < .01). People who perceive capture as costly adopt strategies that increase survival.

**H4b.** Among suboptimal foraging choices, the majority (> 65%) are overcautious (choosing light when heavy has higher expected reward). omega predicts the overcaution ratio (r > 0.30, p < .01).

**H4c.** kappa predicts pressing intensity (r(kappa, mean vigor) < -0.30, p < .01). The effort cost parameter governs motor output — the activation side of the avoid-activate decomposition.

**H4d.** The omega-kappa angle predicts decision quality: r(atan2(kappa_z, omega_z), % optimal) < -0.15, p < .01. Effort-driven avoidance is less optimal than threat-driven avoidance because it is indiscriminate across threat levels.

### H5: Metacognitive Monitoring of the Foraging Computation

We test how accurately metacognitive signals — anxiety and confidence — monitor the first-order survival computation, and whether monitoring accuracy predicts foraging efficiency beyond the model parameters. Following the two-stage metacognitive architecture (Fleming & Daw 2017), the computation (ω, κ) is the first-order process; anxiety monitors threat (primary appraisal, Lazarus 1991) and confidence monitors coping capacity (secondary appraisal).

**H5a.** Anxiety calibration predicts foraging optimality beyond omega and kappa. Hierarchical regression: delta-R-squared > 0.03 (p < .01) for at least two of: % optimal, escape rate, earnings. The metacognitive monitor adds information the first-order computation doesn't contain.

**H5b.** Anxiety slope (reactivity to threat) predicts choice adaptation: r(anxiety slope, choice shift from T=0.1 to T=0.9) > 0.20, p < .01. Anxiety reactivity drives the avoidance channel but not vigor (slope-vigor r expected null).

**H5c.** Omega predicts subjective confidence (r < 0, p < .01) but not anxiety (|r| < 0.10). The computational capture-cost parameter maps onto a coping appraisal, not an affective state.

**H5d.** Confidence predicts error type, not error rate: r(confidence, n_overcautious) < 0 AND r(confidence, n_reckless) > 0, both p < .01.

---

## 6. Analysis Pipeline

### Preprocessing

1. Raw JSON parsed to behavior.csv, feelings.csv, psych.csv
2. Calibration outlier exclusion (> 2.5 SD on mean IPI)
3. Vigor metrics computed: normalized press rate per trial, per epoch, per condition cell

### Model fitting

1. Cell means computed: per-subject mean normalized press rate at each threat x distance x cookie combination (~18 cells per subject)
2. All four models (M1–M4) fitted via NumPyro HMC/NUTS (4 chains × 2,000 warmup + 4,000 samples, target_accept = 0.95; see H3)
3. Parameter extraction: posterior mean omega_i and kappa_i per subject from M4
4. Model comparison: WAIC (primary) with PSIS-LOO robustness check, computed from posterior samples

### Statistical tests

- Mixed models: statsmodels MixedLM with REML
- GAMs: natural cubic regression splines via patsy + statsmodels MixedLM
- All t-tests: two-tailed unless directional prediction specified
- Multiple comparison correction: not applied to preregistered tests (each is a specific directional prediction)

---

## 7. Exploratory Benchmarks (Discovery Sample, N = 290)

These values are from the exploratory sample and serve as benchmarks, not thresholds. The confirmatory sample may produce different magnitudes.

| Hypothesis | Key statistic | Discovery value |
|-----------|--------------|-----------------|
| H1a: Threat → choice | beta(threat) | -1.02 |
| H1b: Threat → anxiety | beta | +0.580 |
| H1c: Vigor within heavy | d | +0.24 |
| H1d: Choice-vigor shift independence | r | +0.046 |
| H2a: Threat → vigor (heavy) | d | +0.44 |
| H2b: Encounter spike | d | +0.56 |
| H2b: Threat-independent | p | 0.206 |
| H2c: GAM encounter LRT | chi-sq | 760 |
| H3a: M4 vs M1 | delta-WAIC | TBD (MCMC) |
| H3c: M4 vs M3 | delta-WAIC | TBD (MCMC) |
| H4a: omega → escape | beta | +0.060, p = .0002 |
| H4b: Overcaution % | % | 79% |
| H4b: omega → OC ratio | r | +0.810 |
| H4c: kappa → vigor | r | -0.736 |
| H4d: angle → optimality | r | -0.315 |
| H5a: Calibration delta-R-sq | delta-R-sq | +0.068 (optimality) |
| H5b: Slope → choice shift | r | +0.389 |
| H5c: omega → confidence | r | -0.216 |
| H5d: Conf → overcautious | r | -0.224 |

---

## 8. Supplementary Analyses (not preregistered)

The following analyses will be reported but are not part of the confirmatory test:

1. **Separate-equations model (M_separate):** lambda (choice-only) + omega (vigor-only) with no shared W function. Tests whether the joint constraint hurts fit relative to unconstrained separate equations.
2. **Parameter recovery:** Simulation-based verification that omega and kappa are identifiable from the joint likelihood.
3. **Posterior predictive checks:** Figures showing model-predicted vs observed choice and vigor by condition.
4. **Encounter spike individual differences:** CV and split-half reliability of the encounter motor response. omega → spike correlation.
5. **Clinical regression tables:** Full omega + kappa + affect → clinical symptom regressions for all questionnaire measures.
6. **Trial-level anxiety → vigor:** LMM testing whether within-person anxiety fluctuations predict pressing intensity beyond threat level. Exploratory: small but significant effect (β = 0.004, p = .001), consistent with the affective gradient hypothesis (Shenhav 2024) but too small for individual-level prediction.
7. **Threat response angle × clinical symptoms:** Whether the direction of one's threat response (avoidance-dominant vs activation-dominant) predicts apathy measures (AMI). Exploratory: trending signal for AMI Emotional and AMI Total.

---

## 9. Code and Data Availability

All analysis code is available at [repository URL]. The preprocessing pipeline, model fitting code, and statistical tests are fully reproducible. Raw data will be shared upon publication.

---

## 10. Theoretical Framework

This study bridges four literatures:

1. **Optimal foraging under predation** (Bednekoff 2007; Brown 1999; Lima & Dill 1990): The fitness function W(u) directly implements Bednekoff's life-history framework where foraging effort is optimized against survival probability.

2. **Motor vigor and reward** (Shadmehr & Krakauer 2008; Yoon et al. 2018): Motor intensity is chosen to maximize reward rate. Our kappa parameter captures the metabolic cost that governs this optimization.

3. **Predatory imminence continuum** (Fanselow 1994; Mobbs et al. 2020): H2 tests whether vigor dynamics follow the pre-encounter (strategic, threat-modulated) to post-encounter (reactive, threat-independent) transition predicted by the defense cascade.

4. **Metacognitive models of anxiety** (Wells 2009; Paulus & Stein 2010): H5 tests whether clinical symptoms track affect LEVEL (intensity) rather than affect QUALITY (calibration) — supporting the view that psychopathology arises from miscalibrated threat monitoring, not from the threat computation itself.
