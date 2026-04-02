# OSF Preregistration: Joint Optimization of Patch Selection and Motor Vigor Under Predation Risk

## Description

How do humans jointly determine which patches to forage and how intensely to work when foraging under predation risk? We developed a foraging task where participants choose between high-reward/high-effort and low-reward/low-effort patches while facing probabilistic predator attacks, then execute their choice by pressing keys to transport the reward to safety. Through behavioral analyses and computational modeling, we test whether a single fitness function — grounded in optimal foraging theory — simultaneously explains patch selection and motor execution through two individual-difference parameters: avoidance sensitivity (omega, the subjective cost of capture) and activation intensity (kappa, the subjective cost of effort). We further examine how metacognitive signals — anxiety and confidence — monitor the foraging computation and independently predict foraging efficiency and decision style.

---

## Study Information

### Hypotheses

**H1: Threat will reduce high-effort choices, increase motor vigor, and shift anxiety upward and confidence downward.**

H1a. High-effort choices will decrease with threat probability and distance.
H1b. Anxiety will increase with threat and distance. Confidence will decrease with threat and distance.
H1c. Within each chosen effort level, pressing intensity will increase with threat.

**H2: Motor vigor will follow the predatory imminence continuum, with distinct anticipatory and reactive dynamics.**

H2a. Within each effort level, pressing rate will increase with threat probability.
H2b. Predator encounter will trigger a rapid motor spike in pressing rate.
H2c. The temporal shape of the vigor timecourse will differ by encounter status and by threat level.

**H3: A joint fitness model with two per-subject parameters will outperform simpler alternatives.**

H3a. The joint model will outperform an effort-only model that ignores threat.
H3b. The joint model will outperform a threat-only model that lacks individual effort sensitivity.
H3c. The joint model will outperform a single-parameter model, demonstrating that capture cost and effort cost are separable traits.

**H4: The model parameters will predict survival, error patterns, and decision quality.**

H4a. Higher capture cost will predict higher escape rates on attack trials.
H4b. The majority of suboptimal choices will be overcautious, and capture cost will predict who is overcautious.
H4c. Higher effort cost will predict lower pressing intensity.
H4d. The balance between capture cost and effort cost will predict decision quality: effort-driven avoidance will be less optimal than threat-driven avoidance.
H4e. Consistency with the joint fitness function — across both patch selection and motor intensity — will predict foraging earnings. Both choice consistency and intensity pattern match will independently contribute.

**H5: Anxiety and confidence will independently monitor the foraging computation and predict foraging efficiency beyond the model parameters.**

H5a. Anxiety calibration (how well anxiety tracks threat) will predict foraging optimality beyond the model parameters.
H5b. Anxiety reactivity (slope on threat) will predict adaptive choice shifting across threat levels.
H5c. Capture cost will predict subjective confidence but not anxiety.
H5d. Confidence will predict the type of errors people make — fewer overcautious errors but more reckless errors — without affecting overall error rate.

---

## Design Plan

### Study type

Experiment - Participants complete a computerized foraging task with randomized trial conditions.

### Blinding

No blinding is involved in this study. Participants are aware of the stated threat probability on each trial.

### Study design

Participants forage in a circular arena under predation risk. On each trial, they choose between a high-reward/high-effort cookie (R=5 points, required pressing rate 0.9) and a low-reward/low-effort cookie (R=1 point, required pressing rate 0.4), then press keys (S+D+F) repeatedly to transport the chosen cookie to the safe zone at the center. A predator may appear based on the stated threat probability (T in {0.1, 0.5, 0.9}). The heavy cookie varies in distance from the safe zone (D in {1, 2, 3}, corresponding to 5, 7, or 9 game units); the light cookie is always at D=1. Capture costs 5 points plus the current cookie reward. Participants earn a monetary bonus proportional to their total score.

The task includes two trial types: choice trials (45 per participant), where participants freely select between heavy and light cookies, and probe trials (36 per participant), where both options are identical (forced choice) and participants rate their prospective anxiety and confidence on 1-10 scales before pressing. Anxiety and confidence are measured on separate probe trials.

A calibration phase at the start establishes each participant's maximum pressing speed, used to normalize pressing rates across participants.

The task comprises 3 blocks of 27 trials each (81 total events per participant). Threat probability, distance, and cookie assignment are fully crossed within blocks.

### Randomization

Trial order within blocks is randomized. Threat and distance conditions are balanced across blocks.

---

## Sampling Plan

### Existing data

Registration prior to analysis of confirmatory data. An exploratory dataset has been collected and fully analyzed. The confirmatory dataset has been collected but not analyzed.

### Explanation of existing data

We collected an exploratory sample of 350 participants (290 after exclusions). All hypotheses, model specifications, thresholds, and analysis pipelines were developed on this exploratory sample. We then collected an independent confirmatory sample of approximately 350 participants. The confirmatory data have not been analyzed at the time of this registration.

### Data collection procedures

Participants are recruited through Prolific. They must be 18-65 years old, fluent in English, and have normal or corrected-to-normal vision. They are paid a base rate for participation plus a performance bonus proportional to their total score. After the foraging task, participants complete questionnaires: DASS-21, PHQ-9, OASIS, STAI (State and Trait), AMI, MFIS, and STICSA.

### Sample size

Target: approximately 350 participants for the confirmatory sample, matching the exploratory sample.

### Sample size rationale

The exploratory sample of 290 (after exclusions) provided sufficient power for all preregistered effects. We match this sample size to ensure comparable power.

### Stopping rule

Data collection stops when approximately 350 participants have completed the task.

---

## Variables

### Manipulated variables

- **Threat probability (T):** {0.1, 0.5, 0.9} — stated predator attack probability, displayed before each trial.
- **Distance (D):** {1, 2, 3} — distance of the heavy cookie from the safe zone. The light cookie is always at distance 1.
- **Trial type:** Choice (free selection) or probe (forced, with affect ratings).

### Measured variables

**Behavioral:**
- Choice: binary (heavy = 1, light = 0) on each choice trial.
- Pressing rate: inter-keypress intervals recorded at ~5Hz. Primary metric: normalized press rate = median(1/IPI) / calibrationMax.
- Trial outcome: escaped or captured. Reward earned per trial.
- Total earnings: sum of rewards across all trials (determines bonus payment).

**Affective (probe trials only):**
- Anxiety rating (1-10): "How anxious are you about this trial?"
- Confidence rating (1-10): "How confident are you that you will succeed?"

**Questionnaires:** DASS-21, PHQ-9, OASIS, STAI (State + Trait), AMI, MFIS, STICSA.

### Indices

**Model parameters (from the joint fitness model M4):**
- omega: per-subject avoidance sensitivity (subjective cost of capture).
- kappa: per-subject activation intensity (subjective cost of effort). Enters choice through a total demand cost (kappa * req * D) and vigor through a quadratic deviation cost (kappa * (u-req)^2 * D).

**Affect indices (from probe trial regressions):**
- Anxiety calibration: within-subject r(anxiety, threat). Higher = anxiety better tracks danger.
- Anxiety slope: within-subject regression slope of anxiety on threat.
- Mean confidence: average confidence rating across probes.

**Behavioral indices:**
- Escape rate: proportion of attack trials survived.
- Overcaution ratio: proportion of errors that are overcautious (chose light when heavy had higher expected reward).
- Omega-kappa angle: atan2(kappa_z, omega_z). Higher = more effort-driven avoidance.

---

## Analysis Plan

### Statistical models

**H1:**
- H1a: Logistic model with cluster-robust SE: choice ~ threat_z + dist_z + threat_z:dist_z, clustered by subject. Both beta(threat) and beta(distance) will be negative at p < .01.
- H1b: Linear mixed models: response ~ threat_z + dist_z + (1 + threat_z | subject), separately for anxiety and confidence. Anxiety: both beta(threat) > 0 and beta(dist) > 0, |t| > 3. Confidence: both beta(threat) < 0 and beta(dist) < 0, |t| > 3.
- H1c: Paired t-tests: within-subject mean normalized press rate at T=0.9 minus T=0.1, separately within heavy and light cookies. Both p < .01, d > 0.15.

**H2:**
- H2a: Same paired t-tests as H1c.
- H2b: Encounter spike = per-subject mean reactive-epoch pressing rate on attack minus non-attack trials. One-sample t vs 0, p < .001, d > 0.20.
- H2c: GAMs with natural cubic regression splines (K=10) fitted via MixedLM with cookie covariate and random intercepts. Likelihood ratio tests for smooth-by-condition interactions, both p < .01.

**H3: Computational model specification and comparison.**

We develop a joint fitness model grounded in optimal foraging theory (Bednekoff 2007; Brown 1999). The organism maximizes fitness W(u) to determine both which patch to select and how intensely to press:

W(u) = S(u) * R - (1 - S(u)) * omega * (R + C) - kappa * (u - req)^2 * D

where:
- u = pressing rate (normalized by calibration maximum)
- S(u, T, D) = exp(-h * T^gamma * D / speed(u)) is survival probability
- speed(u) = sigmoid((u - 0.25 * req) / sigma_sp) is movement speed, saturating above the required pressing rate
- R = cookie reward (5 for heavy, 1 for light)
- C = 5 (capture penalty)
- req = required pressing rate (0.9 for heavy, 0.4 for light)
- D = distance from safe zone (1-3 for heavy, always 1 for light)
- omega_i = per-subject avoidance sensitivity (subjective cost of capture)
- kappa_i = per-subject activation intensity (subjective cost of effort)
- h, gamma, sigma_sp = population parameters (hazard scale, hazard exponent, speed saturation width)

Choice prediction: For each cookie j, compute V_j = max_u W_j(u) - kappa * req_j * D_j. The first term (max_u W) is the optimized fitness given the pressing rate. The second term (kappa * req * D) is the total demand cost — the sustained metabolic cost of choosing that cookie, proportional to the required pressing rate times the distance. P(heavy) = sigmoid((V_H - V_L) / tau), where tau is a population noise parameter.

Vigor prediction: For the chosen cookie, u* = argmax_u W(u) determines the optimal pressing rate. The vigor likelihood uses per-subject condition cell means (subject x threat x distance x cookie, ~18 cells per subject, ~5,200 total): observed cell-mean rate ~ Normal(u* + b_cookie * is_heavy, sigma_v / sqrt(n_trials)).

The total demand cost (kappa * req * D) enters the choice equation but not the vigor optimization. This reflects the distinction between deciding how much effort to commit (total demand for the full trial) and optimizing moment-to-moment pressing intensity (marginal deviation cost). Both are governed by the same kappa — a person's effort sensitivity determines both whether they take the hard job and how hard they work on it.

Model fitting: All models fitted via NumPyro HMC/NUTS (4 chains x 2,000 warmup + 4,000 samples, target_accept = 0.95, max_tree_depth = 10). Convergence required: R-hat < 1.01 and bulk ESS > 400 for all parameters. If any model fails to converge, we double the sampling iterations before declaring non-convergence.

Parameter recovery: We simulate 100 synthetic subjects from known omega and kappa values, fit the model to the simulated data, and correlate recovered with true parameters. Recovery benchmarks from the exploratory sample: omega r = 0.94, kappa r = 0.92.

Model comparison: Primary criterion is WAIC computed from pointwise log-likelihoods via ArviZ. Robustness criterion is approximate LOO-CV via Pareto-smoothed importance sampling (PSIS-LOO). A hypothesis is supported only if WAIC and LOO agree. For H3a (joint vs effort-only), choice-only WAIC is used since the effort-only model has no vigor likelihood.

The four models compared:

M1 (Effort-only): kappa_i per-subject. Choice: delta_V = delta_R - kappa_i * delta_effort(D). No survival function, no threat term, no vigor model. Tests whether threat adds anything beyond effort cost.

M2 (Threat-only): omega_i per-subject, population kappa. Choice and vigor both from W(u), but kappa is shared across all subjects. Tests whether individual effort sensitivity matters or only threat sensitivity.

M3 (Single-parameter): theta_i per-subject, entering W(u) as both omega and kappa (theta = omega = kappa). Tests whether a single trait can serve both avoidance and activation roles.

M4 (Joint model): omega_i and kappa_i per-subject, both entering W(u) through the fitness function described above. This is the full model. Both parameters are identifiable (recovery r > 0.90) and approximately orthogonal (r = 0.21 in the exploratory sample).

**H4 and H5: Bayesian linear models.**

All H4 and H5 regressions fitted with Bayesian linear models (bambi; Capretto et al. 2022) using default weakly informative priors (bambi defaults: Normal(0, sigma) for coefficients scaled by data). Posterior sampling: 4 chains x 2,000 draws + 1,000 tuning. Inference criterion: 95% highest density interval (HDI) excludes zero for directional predictions.

**H4:**
- H4a: escape_rate ~ omega_z + kappa_z. omega posterior mean > 0, 95% HDI excludes zero.
- H4b: Classify errors from empirical expected reward per T x D cell. Overcaution > 65% of errors (descriptive). overcaution_ratio ~ omega_z: posterior mean > 0.30 (standardized), 95% HDI excludes zero.
- H4c: mean_vigor ~ kappa_z: posterior mean < 0, |standardized beta| > 0.30, 95% HDI excludes zero.
- H4d: pct_optimal ~ angle_z (where angle = atan2(kappa_z, omega_z)): posterior mean < 0, 95% HDI excludes zero.
- H4e: earnings ~ choice_consistency_z + intensity_pattern_z. Both posterior means > 0, both 95% HDIs exclude zero.

**H5:**
- H5a: Bayesian model comparison. Fit outcome ~ omega_z + kappa_z (base) and outcome ~ omega_z + kappa_z + calibration_z (full) for each of: pct_optimal, escape_rate, earnings. Compare via LOO (ArviZ). Calibration improves model fit (delta-ELPD > 0, SE excludes zero) for at least two of the three outcomes.
- H5b: choice_shift ~ anxiety_slope_z. Posterior mean > 0, standardized beta > 0.20, 95% HDI excludes zero.
- H5c: mean_confidence ~ omega_z: posterior mean < 0, 95% HDI excludes zero. mean_anxiety ~ omega_z: 95% HDI includes zero (null prediction).
- H5d: n_overcautious ~ confidence_z: posterior mean < 0, 95% HDI excludes zero. n_reckless ~ confidence_z: posterior mean > 0, 95% HDI excludes zero.

### Transformations

- omega and kappa log-transformed before z-scoring.
- Threat and distance z-scored for regressions.
- Vigor normalized by calibration maximum.

### Inference criteria

- H1, H2: Frequentist (p < 0.01 for directional tests, |t| > 3 for LMMs). These are standard behavioral tests where Bayesian methods add little.
- H3: WAIC (primary) + PSIS-LOO (robustness). Delta > 0 with both agreeing.
- H4, H5: Bayesian — 95% HDI excludes zero for directional predictions. Effect size thresholds specified per hypothesis.
- Multiple comparison correction not applied — each test is a specific directional prediction from the exploratory sample.

### Data exclusion

Subjects with calibration mean IPI > 2.5 SD from sample mean. No other exclusions.

### Missing data

Non-response trials excluded. Per-subject indices computed from available trials.

### Exploratory analysis

1. **Separate-equations model:** lambda (choice-only) + omega (vigor-only) with no shared W function. Tests whether the joint constraint hurts fit relative to unconstrained separate equations.
2. **Scaled single-parameter model (M3b):** theta as omega, alpha*theta as kappa (alpha = population scaling factor). Tests if M3's failure is merely a scale mismatch rather than genuine separability.
3. **Parameter recovery:** Simulate 100 subjects from known omega and kappa, refit, and verify recovery (r > 0.85 for both).
4. **Posterior predictive checks:** Model-predicted vs observed choice and vigor by condition.
5. **Encounter spike individual differences:** CV, split-half reliability, and model parameter correlations with the reactive motor response.
6. **Clinical regressions:** Full omega + kappa + affect measures → all questionnaire scores (DASS-21, PHQ-9, OASIS, STAI, AMI, MFIS, STICSA). We expect omega and kappa to be psychiatrically silent (R-squared < 0.02).
7. **Trial-level anxiety-vigor coupling:** LMM testing whether within-person anxiety fluctuations predict pressing intensity beyond threat level.
8. **Four foraging profiles:** Median split on omega x kappa producing Strategic (hi-omega, lo-kappa), Resilient (lo-omega, lo-kappa), Reckless (lo-omega, hi-kappa), and Helpless (hi-omega, hi-kappa) profiles with associated earnings and escape rates.
9. **Frequentist robustness:** Key H4 and H5 results replicated with OLS/Pearson r (p < .01) to confirm consistency across inference frameworks.
10. **Normative benchmark:** Calibrated agent analysis comparing participant behavior to a model-derived optimal strategy. Quantification of the overcaution cost in points and bonus payment.
