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

H2a. Predator encounter will trigger a rapid motor spike in pressing rate.
H2b. The temporal shape of the vigor timecourse will differ by encounter status and by threat level.

**H3: A joint fitness model with two per-subject parameters will outperform simpler alternatives.**

H3a. The joint model will outperform an effort-only model that ignores threat.
H3b. The joint model will outperform a threat-only model that lacks individual effort sensitivity.
H3c. The joint model will outperform a single-parameter model, demonstrating that capture cost and effort cost are separable traits.

**H4: The model parameters will predict survival, error patterns, and decision quality.**

H4a. Higher capture cost will predict higher escape rates on attack trials.
H4b. Capture cost will predict the proportion of overcautious errors: higher omega will predict a greater share of suboptimal choices that are overcautious (choosing light when heavy has higher expected reward).
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

The task includes two trial types: choice trials (45 per participant), where participants freely select between heavy and light cookies, and probe trials (36 per participant), where both options are identical (forced choice) and participants rate their prospective anxiety and confidence on 1-10 scales before pressing. On each probe trial, the cookie type (heavy or light), threat probability, and distance are assigned to fully cross all conditions: 3 threat × 3 distance × 2 cookie types = 18 unique conditions, each sampled once for anxiety and once for confidence. Anxiety and confidence are measured on separate probe trials (18 each).

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

Participants are recruited through Prolific. They must be 18-65 years old, fluent in English, and have normal or corrected-to-normal vision. They are paid a base rate for participation plus a performance bonus proportional to their total score. The study flow is: (1) instruction comprehension assessment in Qualtrics, (2) brief video game use questionnaire, (3) foraging task built in Unity and deployed as a WebGL application in the participant's browser, (4) post-task questionnaires: DASS-21, PHQ-9, OASIS, STAI (State and Trait), AMI, MFIS, and STICSA.

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
- Pressing rate: keypresses recorded at native ~5Hz (S+D+F keys). Inter-press intervals (IPI) computed as successive timestamp differences; IPIs < 10 ms removed as artifacts. Primary metric: normalized press rate = median(1/IPI) / calibrationMax. For timecourse analyses (H2): 200ms bins smoothed with 3-point centered moving average (600ms window). For model fitting (H3): per-subject condition cell means (subject × threat × distance × cookie type, ~18 cells per subject), each the median normalized rate across trials within that condition.
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
- Anxiety calibration: within-subject r(anxiety, threat), computed from ~18 anxiety probe trials per subject (6 per threat level). Higher = anxiety better tracks danger.
- Anxiety slope: within-subject regression slope of anxiety on threat.
- Mean confidence: average confidence rating across probes.

Note: With ~18 probes per subject, individual calibration and slope estimates will have substantial sampling error. We will report split-half reliability of these indices (see Exploratory analysis 4).

**Behavioral indices:**
- Escape rate: proportion of attack trials survived.
- Overcaution ratio: proportion of errors that are overcautious (chose light when heavy had higher expected reward).
- Omega-kappa angle: atan2(kappa_z, omega_z). Higher = more effort-driven avoidance.

**Model consistency indices (H4e):**
- Choice consistency: per-subject fraction of choice trials where actual choice matches model prediction (predict heavy if V_H > V_L given subject's omega and kappa, light otherwise).
- Intensity deviation: per-subject RMSE between model-predicted optimal pressing rate (u* = argmax_u W(u)) and observed cell-mean rate, computed across the subject's condition cells. Lower = vigor closer to model prediction.

---

## Analysis Plan

### Statistical models

**H1:**
- H1a: We will fit a logistic model with cluster-robust SE: choice ~ threat_z + dist_z + threat_z:dist_z, clustered by subject. We predict both beta(threat) and beta(distance) will be negative.
- H1b: We will fit linear mixed models: response ~ threat_z + dist_z + (1 + threat_z | subject), separately for anxiety and confidence. We predict anxiety increases with threat and distance, and confidence decreases with threat and distance.
- H1c: We will compute paired t-tests on within-subject mean normalized press rate at T=0.9 minus T=0.1, separately within heavy and light cookies. We predict both comparisons will be positive (vigor increases with threat).

**H2:**
- H2a: We will compute the encounter spike as per-subject mean reactive-epoch pressing rate on attack minus non-attack trials, tested with a one-sample t-test against zero. We predict a positive spike.
- H2b: We will fit GAMs with natural cubic regression splines (K=10) via MixedLM with cookie covariate and random intercepts. We will use likelihood ratio tests for smooth-by-condition interactions to test for distinct temporal signatures by encounter status and threat level.

**H3: Computational model specification and comparison.**

We will fit a joint fitness model grounded in optimal foraging theory (Bednekoff 2007; Brown 1999). The organism maximizes fitness W(u) to determine both which patch to select and how intensely to press:

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

Population parameter priors (all weakly informative):
- gamma: Normal(0, 0.5) on log-scale, clipped to [0.1, 3.0]
- h: Normal(0, 1) on log-scale
- sigma_sp: Normal(-1, 0.5) on log-scale, clipped to [0.01, 1.0]
- tau: Normal(0, 1) on log-scale, clipped to [0.01, 50.0]
- sigma_v: HalfNormal(0.3)
- b_cookie: Normal(0, 0.5)

Per-subject parameter priors (hierarchical, non-centered):
- omega_i = exp(m_omega + s_omega * z_i), where m_omega ~ Normal(0, 1), s_omega ~ HalfNormal(1.0), z_i ~ Normal(0, 1)
- kappa_i = exp(m_kappa + s_kappa * z_i), where m_kappa ~ Normal(-1, 1), s_kappa ~ HalfNormal(0.5), z_i ~ Normal(0, 1)

Choice prediction: For each cookie j, compute V_j = max_u W_j(u) - kappa * req_j * D_j. The first term (max_u W) is the optimized fitness given the pressing rate. The second term (kappa * req * D) is the total demand cost — the sustained metabolic cost of choosing that cookie, proportional to the required pressing rate times the distance. P(heavy) = sigmoid((V_H - V_L) / tau), where tau is a population noise parameter.

Vigor prediction: For the chosen cookie, u* = argmax_u W(u) determines the optimal pressing rate. The vigor likelihood uses per-subject condition cell means (subject x threat x distance x cookie, ~18 cells per subject, ~5,200 total): observed cell-mean rate ~ Normal(u* + b_cookie * is_heavy, sigma_v / sqrt(n_trials)). We use cell means rather than trial-level data because the fitness function predicts a single optimal rate per condition — trial-to-trial variance within a condition reflects motor noise, not parametric signal. The sqrt(n_trials) denominator ensures that cells with fewer observations receive proportionally less weight.

The total demand cost (kappa * req * D) enters the choice equation but not the vigor optimization. This reflects the distinction between deciding how much effort to commit (total demand for the full trial) and optimizing moment-to-moment pressing intensity (marginal deviation cost). Both are governed by the same kappa — a person's effort sensitivity determines both whether they take the hard job and how hard they work on it.

Model fitting: All models will be fitted via NumPyro HMC/NUTS (4 chains x 2,000 warmup + 4,000 samples, target_accept = 0.95, max_tree_depth = 10). We will require convergence: R-hat < 1.01 and bulk ESS > 400 for all parameters. If any model fails to converge, we will double the sampling iterations before declaring non-convergence.

Parameter recovery: We will simulate 500 synthetic subjects from known omega and kappa values, fit the model to the simulated data, and correlate recovered with true parameters to verify identifiability.

Model comparison: We will use WAIC computed from pointwise log-likelihoods via ArviZ as the primary criterion. The robustness criterion will be approximate LOO-CV via Pareto-smoothed importance sampling (PSIS-LOO). A hypothesis is supported only if WAIC and LOO agree. All four models will be evaluated on the same joint likelihood (choice + vigor) to ensure fair comparison.

The four models compared:

M1 (Effort-only): kappa_i per-subject. Choice: delta_V = delta_R - kappa_i * delta_effort(D). No survival function, no threat term. Vigor: intercept-only (cell mean ~ Normal(mu + b_cookie * is_heavy, sigma_v / sqrt(n_trials))) with no condition structure. Tests whether threat adds anything beyond effort cost.

M2 (Threat-only): omega_i per-subject, population kappa. Choice and vigor both from W(u), but kappa is shared across all subjects. Tests whether individual effort sensitivity matters or only threat sensitivity.

M3 (Single-parameter): theta_i per-subject, entering W(u) as both omega and kappa (theta = omega = kappa). Tests whether a single trait can serve both avoidance and activation roles.

M4 (Joint model): omega_i and kappa_i per-subject, both entering W(u) through the fitness function described above. This is the full model. Both parameters are identifiable and approximately orthogonal in the exploratory sample.

**H4 and H5: Bayesian linear models.**

All H4 and H5 regressions will be fitted with Bayesian linear models (bambi; Capretto et al. 2022) using default weakly informative priors (bambi defaults: Normal(0, sigma) for coefficients scaled by data). Posterior sampling: 4 chains x 2,000 draws + 1,000 tuning.

**H4:**
- H4a: We will fit escape_rate ~ omega_z + kappa_z. We predict omega will be positive.
- H4b: We will classify errors from empirical expected reward per T x D cell and fit overcaution_ratio ~ omega_z. We predict omega will be positive. We will report the overall overcaution percentage descriptively.
- H4c: We will fit mean_vigor ~ kappa_z. We predict kappa will be negative.
- H4d: We will fit pct_optimal ~ angle_z (where angle = atan2(kappa_z, omega_z)). We predict angle will be negative (effort-driven avoidance is less optimal).
- H4e: We will fit earnings ~ choice_consistency_z + intensity_deviation_z. We predict choice_consistency will be positive and intensity_deviation will be negative (less deviation = more earnings).

**H5:**
- H5a: We will compare base (pct_optimal ~ omega_z + kappa_z) and full (pct_optimal ~ omega_z + kappa_z + calibration_z) models via LOO-CV. We predict calibration will improve model fit. Escape rate and earnings will be tested as supporting outcomes.
- H5b: We will fit choice_shift ~ anxiety_slope_z. We predict anxiety slope will be positive.
- H5c: We will fit mean_confidence ~ omega_z and mean_anxiety ~ omega_z. We predict omega will be negative for confidence and practically zero for anxiety (ROPE test).
- H5d: We will fit n_overcautious ~ confidence_z and n_reckless ~ confidence_z. We predict confidence will be negative for overcautious errors and positive for reckless errors.

### Transformations

- omega and kappa will be log-transformed before z-scoring.
- Threat and distance will be z-scored for regressions.
- Vigor will be normalized by calibration maximum.

### Inference criteria

- H1, H2: We will use frequentist inference. Directional tests (t-tests, logistic regression coefficients): p < .01. LMM coefficients: |t| > 3. LRT for GAMs: p < .01.
- H3: We will compare models using WAIC (primary) and PSIS-LOO (robustness). A hypothesis is supported only if both criteria agree (delta-WAIC > 0 and delta-LOO > 0 favoring M4).
- H4, H5: We will use Bayesian inference. A directional prediction is supported if the 95% HDI excludes zero in the predicted direction. For H5a (LOO comparison), calibration improves fit if delta-ELPD > 0 with SE excluding zero. For H5c (null prediction for anxiety), we will use a ROPE of [-0.10, +0.10] on the standardized beta — the null is supported if the 95% HDI falls entirely within the ROPE.
- We will not apply multiple comparison correction — each test is a specific directional prediction from the exploratory sample.

### Data exclusion

**Subject-level:**
- Incomplete data: participants must complete all 81 trials and have data in all modalities (behavioral, probe ratings, questionnaires).
- Calibration outliers: mean inter-press interval during the calibration phase > 2.5 SD from the sample mean.
- Task engagement: escape rate < 35% across attack trials.

In the exploratory sample, these criteria excluded 60 of 350 participants (57 for incomplete data or task engagement, 3 calibration outliers), yielding N = 290 analyzed.

**Trial-level:**
- Non-response trials (no keypresses recorded) will be excluded from per-subject indices.
- Inter-press intervals < 10 ms will be treated as artifacts and removed before computing pressing rate.

### Missing data

Non-response trials will be excluded. Per-subject indices will be computed from available trials.

### Exploratory analysis

1. **Separate-equations model:** We will fit a model with lambda (choice-only) + omega (vigor-only) and no shared W function, to test whether the joint constraint hurts fit relative to unconstrained separate equations.
2. **Scaled single-parameter model (M3b):** We will fit theta as omega with alpha*theta as kappa (alpha = population scaling factor), to test whether M3's failure is merely a scale mismatch rather than genuine separability.
3. **Posterior predictive checks:** We will generate model-predicted vs observed choice and vigor by condition.
4. **Affect index reliability:** We will compute split-half reliability (odd/even probe trials) for anxiety calibration, anxiety slope, and mean confidence to assess the stability of these indices given the limited number of probes per subject (~18).
5. **Encounter spike individual differences:** We will compute CV, split-half reliability, and model parameter correlations with the reactive motor response.
6. **Clinical regressions:** We will regress all questionnaire scores (DASS-21, PHQ-9, OASIS, STAI, AMI, MFIS, STICSA) on omega + kappa + affect measures. All confirmatory hypotheses (H1-H5) will be tested on the confirmatory sample alone; we will then pool both samples (~580 subjects) for clinical regressions to maximize power for detecting small effects.
7. **Trial-level anxiety-vigor coupling:** We will fit an LMM testing whether within-person anxiety fluctuations predict pressing intensity beyond threat level.
8. **Frequentist robustness:** We will replicate key H4 and H5 results with OLS/Pearson r (p < .01) to confirm consistency across inference frameworks.
9. **Normative benchmark:** We will compare participant behavior to a model-derived optimal strategy and quantify the overcaution cost in points and bonus payment.
