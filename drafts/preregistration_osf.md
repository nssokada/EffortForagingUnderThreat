# OSF Preregistration: Joint Optimization of Patch Selection and Motor Vigor Under Predation Risk

## Description

How do humans jointly determine which patches to forage and how intensely to work when foraging under predation risk? We developed a foraging task where participants choose between high-reward/high-effort and low-reward/low-effort patches while facing probabilistic predator attacks, then execute their choice by pressing keys to transport the reward to safety. Through behavioral analyses and computational modeling, we test whether a single fitness function — grounded in optimal foraging theory — simultaneously explains patch selection (choice) and motor execution (vigor) through two individual-difference parameters: the subjective cost of capture (omega) and the subjective cost of effort (kappa). We further examine how metacognitive signals — anxiety and confidence — monitor the foraging computation and independently predict foraging efficiency and decision style.

---

## Study Information

### Hypotheses

We investigate three domains: foraging behavior (choice and vigor), computational architecture (model comparison), and metacognitive monitoring (anxiety and confidence).

**H1: Threat drives adaptive shifts in choice, vigor, and affect.**
H1a. P(choose heavy) decreases with threat probability and distance.
H1b. Anxiety increases and confidence decreases with threat.
H1c. Within each cookie type, pressing rate increases with threat. All vigor analyses condition on cookie type.

**H2: Vigor dynamics follow the predatory imminence continuum.**
H2a. Threat increases pressing rate within cookie type (paired t, p < .01 both cookies).
H2b. Predator encounter triggers a motor spike (d > 0, p < .001) that does not scale with threat probability (p > .05).
H2c. GAM likelihood ratio tests confirm distinct temporal signatures for encounter and threat (both p < .01).

**H3: A joint fitness model with two per-subject parameters outperforms simpler alternatives.**
H3a. The joint model (M4: omega + kappa) outperforms the effort-only model (M1: kappa only). Delta-WAIC > 0.
H3b. M4 outperforms the threat-only model (M2: omega only). Delta-WAIC > 0.
H3c. M4 outperforms the single-parameter model (M3: theta = omega = kappa). Delta-WAIC > 0.

**H4: The model parameters predict survival, error patterns, and decision quality.**
H4a. omega predicts escape rate on attack trials (OLS beta > 0, p < .01).
H4b. Among suboptimal choices, > 65% are overcautious (choosing safe when risky is optimal). omega predicts the overcaution ratio (r > 0.30, p < .01).
H4c. kappa predicts pressing intensity (r < -0.30, p < .01).
H4d. The omega-kappa angle predicts decision quality: effort-driven avoidance is less optimal than threat-driven avoidance (r < -0.15, p < .01).

**H5: Metacognitive signals monitor the foraging computation and independently predict efficiency.**
H5a. Anxiety calibration (within-subject r between anxiety and threat) predicts optimality beyond omega and kappa (hierarchical regression delta-R-squared > 0.03, p < .01, for at least two of: % optimal, escape, earnings).
H5b. Anxiety slope predicts adaptive choice shifting (r > 0.20, p < .01) but not vigor shifting.
H5c. omega predicts confidence (r < 0, p < .01) but not anxiety (|r| < 0.10).
H5d. Confidence predicts error type: r(confidence, overcautious errors) < 0 AND r(confidence, reckless errors) > 0, both p < .01.

---

## Design Plan

### Study type

Experiment - Participants complete a computerized foraging task with randomized trial conditions.

### Blinding

No blinding is involved in this study. Participants are aware of the stated threat probability on each trial.

### Study design

Participants forage in a circular arena under predation risk. On each trial, they choose between a high-reward/high-effort cookie (R=5 points, required pressing rate 0.9) and a low-reward/low-effort cookie (R=1 point, required pressing rate 0.4), then press keys (S+D+F) repeatedly to transport the chosen cookie to the safe zone at the center. A predator may appear based on the stated threat probability (T in {0.1, 0.5, 0.9}). The heavy cookie varies in distance from the safe zone (D in {1, 2, 3}, corresponding to 5, 7, or 9 game units); the light cookie is always at D=1. Capture costs 5 points plus the current cookie reward. Participants earn a monetary bonus proportional to their total score.

The task includes two trial types: choice trials (45 per participant), where participants freely select between heavy and light cookies, and probe trials (36 per participant), where both options are identical (forced choice) and participants rate their prospective anxiety ("How anxious are you about this trial?") and confidence ("How confident are you that you will succeed?") on 1-10 scales before pressing. Anxiety and confidence are measured on separate probe trials.

A calibration phase at the start of the task establishes each participant's maximum pressing speed, which is used to normalize pressing rates across participants.

The task comprises 3 blocks of 27 trials each (81 total events per participant). Threat probability (T), distance (D), and cookie assignment are fully crossed within blocks.

### Randomization

Trial order within blocks is randomized. Threat and distance conditions are balanced across blocks. Seven unique probe schedules are used across subjects.

---

## Sampling Plan

### Existing data

Registration prior to analysis of confirmatory data. An exploratory dataset has been collected and fully analyzed. The confirmatory dataset has been collected but not analyzed.

### Explanation of existing data

We collected an exploratory sample of 350 participants (290 after exclusions). All hypotheses, model specifications, thresholds, and analysis pipelines were developed on this exploratory sample. We then collected an independent confirmatory sample of approximately 350 participants. The confirmatory data have not been analyzed at the time of this registration. We aim to replicate all preregistered analyses on the confirmatory sample.

### Data collection procedures

Participants are recruited through Prolific. Participants must be 18-65 years old, fluent in English, and have normal or corrected-to-normal vision. They are paid a base rate for participation plus a performance bonus proportional to their total score in the task. After the foraging task, participants complete a battery of questionnaires: DASS-21, PHQ-9, OASIS, STAI (State and Trait), AMI, MFIS, and STICSA.

### Sample size

Target: approximately 350 participants for the confirmatory sample, matching the exploratory sample.

### Sample size rationale

The exploratory sample of 290 (after exclusions) provided sufficient power for all preregistered effects (smallest effect: H1d choice-vigor shift independence, where we test for a null; largest effects: H4b omega-overcaution r = +0.81, H5b slope-choice shift r = +0.39). We match this sample size to ensure comparable power.

### Stopping rule

Data collection stops when approximately 350 participants have completed the task on Prolific.

---

## Variables

### Manipulated variables

- **Threat probability (T):** {0.1, 0.5, 0.9} — stated predator attack probability, displayed to participant before each trial.
- **Distance (D):** {1, 2, 3} — distance of the heavy cookie from the safe zone (5, 7, or 9 game units). The light cookie is always at distance 1.
- **Trial type:** Choice (free selection) or probe (forced, with affect ratings).

### Measured variables

**Behavioral:**
- Choice: binary (heavy = 1, light = 0) on each choice trial.
- Pressing rate: inter-keypress intervals (IPI) recorded at ~5Hz native resolution. Primary metric: normalized press rate = median(1/IPI) / calibrationMax.
- Trial outcome: escaped or captured (trialEndState). Reward earned per trial (trialReward).
- Total earnings: sum of trialReward across all trials (determines bonus payment).

**Affective (probe trials only):**
- Anxiety rating (1-10 scale): "How anxious are you about this trial?"
- Confidence rating (1-10 scale): "How confident are you that you will succeed?"

**Questionnaires:**
- DASS-21 (Depression, Anxiety, Stress subscales)
- PHQ-9 (depression severity)
- OASIS (overall anxiety)
- STAI (State and Trait anxiety)
- AMI (Apathy Motivation Index: Behavioural, Social, Emotional subscales)
- MFIS (Modified Fatigue Impact Scale: Physical, Cognitive, Psychosocial)
- STICSA (State-Trait Inventory for Cognitive and Somatic Anxiety)

### Indices

**Per-subject affect indices (derived from probe trial regressions):**
- Anxiety calibration: within-subject Pearson r between anxiety rating and threat probability. Higher = anxiety better tracks danger.
- Anxiety slope: within-subject regression slope of anxiety on threat. Higher = more reactive.
- Mean anxiety: average anxiety rating across all probe trials.
- Mean confidence: average confidence rating across all probe trials.

**Per-subject model parameters (from computational model M4):**
- omega (cost of capture): how much being caught matters to this person.
- kappa (cost of effort): how costly pressing is for this person.
- Both estimated via joint likelihood on choice + vigor data.

**Derived behavioral indices:**
- Escape rate: proportion of attack trials where participant escaped.
- Choice shift: P(heavy at T=0.1) - P(heavy at T=0.9).
- Vigor shift: within-cookie mean pressing rate at T=0.9 minus T=0.1.
- Overcaution ratio: proportion of suboptimal choices that are overcautious (chose light when heavy had higher expected reward).
- omega-kappa angle: atan2(kappa_z, omega_z), capturing whether avoidance is primarily threat-driven (low angle) or effort-driven (high angle).

---

## Analysis Plan

### Statistical models

**H1 (Behavioral effects):**
- H1a: Logistic model with cluster-robust SE: choice ~ threat_z + dist_z + threat_z:dist_z, clustered by subject.
- H1b: Linear mixed models with random intercepts and slopes: response ~ threat_z + (1 + threat_z | subject), separately for anxiety and confidence.
- H1c: Paired t-tests: within-subject mean normalized press rate at T=0.9 minus T=0.1, separately within heavy and light cookies.

**H2 (Vigor dynamics):**
- H2a: Same as H1c (paired t within cookie type).
- H2b: Encounter spike = per-subject mean reactive-epoch norm_rate on attack minus non-attack trials. One-sample t vs 0. Threat modulation: paired t comparing spike at T=0.9 vs T=0.1.
- H2c: GAM models using natural cubic regression splines (K=10 basis functions) fitted via statsmodels MixedLM with cookie covariate and random intercepts by subject. Likelihood ratio tests comparing models with vs without smooth-by-condition interactions.

**H3 (Model comparison):**
All four models fitted via NumPyro HMC/NUTS (4 chains x 2,000 warmup + 4,000 samples, target_accept = 0.95, max_tree_depth = 10). Convergence required: R-hat < 1.01 and bulk ESS > 400 for all parameters. If any model fails to converge, sampling iterations are doubled before declaring non-convergence. WAIC computed from pointwise log-likelihoods via ArviZ. PSIS-LOO as robustness check. Hypothesis supported only if WAIC and LOO agree. For the H3a comparison (M4 vs M1), choice-only WAIC is used since M1 has no vigor likelihood.

**H4 (Optimality):**
- H4a: OLS regression escape_rate ~ omega_z + kappa_z.
- H4b: Per-trial optimality defined from empirical expected reward at each threat x distance cell. Overcaution ratio = overcautious errors / total errors. Pearson r(omega_z, overcaution ratio).
- H4c: Pearson r(kappa_z, mean vigor).
- H4d: Pearson r(atan2(kappa_z, omega_z), % optimal).

**H5 (Metacognitive monitoring):**
- H5a: Hierarchical regression. Step 1: outcome ~ omega_z + kappa_z. Step 2: + calibration_z. Report delta-R-squared.
- H5b: Pearson r(anxiety slope, choice shift).
- H5c: Pearson r(omega_z, mean confidence) and r(omega_z, mean anxiety).
- H5d: Pearson r(mean confidence, n_overcautious) and r(mean confidence, n_reckless).

### Transformations

- omega and kappa are log-transformed before z-scoring for all regressions and correlations (the parameters are estimated on an exponential scale).
- Threat and distance are z-scored for regression models.
- Vigor normalized by calibration maximum (median(1/IPI) / calibrationMax).

### Inference criteria

- p < 0.01 for all preregistered directional tests (specified per hypothesis).
- For WAIC/LOO model comparison: delta > 0 (positive = joint model wins).
- For null predictions (H1d independence, H2b threat-independence, H5c omega-anxiety null): |r| < threshold or p > .05 as specified.
- Multiple comparison correction is not applied to preregistered tests — each is a specific directional prediction derived from the exploratory sample.

### Data exclusion

Subjects with calibration mean IPI > 2.5 SD from the sample mean are excluded (calibration outliers indicating inability to perform the pressing task). No other exclusions are applied.

### Missing data

Trials where participants failed to respond are excluded from analysis. Per-subject indices are computed from available trials only.

### Exploratory analysis

The following analyses will be reported but are not part of the confirmatory tests:

1. **Separate-equations model:** lambda (choice-only) + omega (vigor-only) with no shared W function, testing whether the joint constraint hurts fit.
2. **Scaled single-parameter model (M3b):** theta enters as omega, alpha*theta enters as kappa (alpha = population scaling), testing whether M3's failure is merely a scale mismatch.
3. **Parameter recovery:** Simulation-based verification that omega and kappa are identifiable from the joint likelihood.
4. **Posterior predictive checks:** Model-predicted vs observed choice and vigor by condition.
5. **Encounter spike individual differences:** CV, split-half reliability, and model parameter correlations.
6. **Clinical regressions:** Full omega + kappa + affect measures predicting questionnaire scores.
7. **Trial-level anxiety-vigor coupling:** LMM testing within-person anxiety → pressing intensity beyond threat level.
8. **Four foraging profiles:** Descriptive median-split on omega and kappa producing Strategic, Resilient, Reckless, and Helpless profiles with associated earnings and escape rates.

### Other

All analysis code, preprocessing pipelines, model fitting scripts, and pre-built model input data are available in the project repository. The computational model, hypothesis tests, and notebooks are fully reproducible.
