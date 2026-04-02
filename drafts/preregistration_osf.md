# OSF Preregistration: Joint Optimization of Patch Selection and Motor Vigor Under Predation Risk

## Description

How do humans jointly determine which patches to forage and how intensely to work when foraging under predation risk? We developed a foraging task where participants choose between high-reward/high-effort and low-reward/low-effort patches while facing probabilistic predator attacks, then execute their choice by pressing keys to transport the reward to safety. Through behavioral analyses and computational modeling, we test whether a single fitness function — grounded in optimal foraging theory — simultaneously explains patch selection and motor execution through two individual-difference parameters: the subjective cost of capture (omega) and the subjective cost of effort (kappa). We further examine how metacognitive signals — anxiety and confidence — monitor the foraging computation and independently predict foraging efficiency and decision style.

---

## Study Information

### Hypotheses

**H1: Threat will reduce high-effort choices, increase motor vigor, and shift anxiety upward and confidence downward.**

H1a. High-effort choices will decrease with threat probability and distance.
H1b. Anxiety will increase and confidence will decrease with threat.
H1c. Within each chosen effort level, pressing intensity will increase with threat.

**H2: Motor vigor will follow the predatory imminence continuum, with distinct anticipatory and reactive dynamics.**

H2a. Within each effort level, pressing rate will increase with threat probability.
H2b. Predator encounter will trigger a rapid motor spike that does not scale with threat probability.
H2c. The temporal shape of the vigor timecourse will differ by encounter status and by threat level (GAM likelihood ratio tests).

**H3: A joint fitness model with two per-subject parameters will outperform simpler alternatives.**

H3a. The joint model (M4: omega + kappa) will outperform the effort-only model (M1), which ignores threat.
H3b. M4 will outperform the threat-only model (M2), which lacks individual effort sensitivity.
H3c. M4 will outperform the single-parameter model (M3), demonstrating that capture cost and effort cost are separable traits.

**H4: The model parameters will predict survival, error patterns, and decision quality.**

H4a. Higher omega will predict higher escape rates on attack trials.
H4b. The majority of suboptimal choices will be overcautious, and omega will predict who is overcautious.
H4c. Higher kappa will predict lower pressing intensity.
H4d. The omega-kappa balance will predict decision quality: effort-driven avoidance will be less optimal than threat-driven avoidance.

**H5: Anxiety and confidence will independently monitor the foraging computation and predict foraging efficiency beyond the model parameters.**

H5a. Anxiety calibration (how well anxiety tracks threat) will predict foraging optimality beyond omega and kappa.
H5b. Anxiety reactivity (slope on threat) will predict adaptive choice shifting across threat levels.
H5c. Omega will predict subjective confidence but not anxiety.
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
- omega: per-subject cost of capture.
- kappa: per-subject cost of effort.

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
- H1b: Linear mixed models with random intercepts and slopes: response ~ threat_z + (1 + threat_z | subject), separately for anxiety and confidence. Anxiety beta > 0, |t| > 3. Confidence beta < 0, |t| > 3.
- H1c: Paired t-tests: within-subject mean normalized press rate at T=0.9 minus T=0.1, separately within heavy and light cookies. Both p < .01, d > 0.15.

**H2:**
- H2a: Same paired t-tests as H1c.
- H2b: Encounter spike = per-subject mean reactive-epoch pressing rate on attack minus non-attack trials. One-sample t vs 0 (p < .001). Threat modulation: paired t comparing spike at T=0.9 vs T=0.1 (p > .05).
- H2c: GAMs with natural cubic regression splines (K=10) fitted via MixedLM with cookie covariate and random intercepts. Likelihood ratio tests for smooth-by-condition interactions, both p < .01.

**H3:**
All four models fitted via NumPyro HMC/NUTS (4 chains x 2,000 warmup + 4,000 samples, target_accept = 0.95, max_tree_depth = 10). Convergence: R-hat < 1.01, bulk ESS > 400. Primary criterion: WAIC. Robustness: PSIS-LOO. Hypothesis confirmed only if both agree. For H3a (M4 vs M1), choice-only WAIC is used since M1 has no vigor likelihood.

The four models:
- M1 (Effort-only): kappa per-subject, no threat, no vigor model.
- M2 (Threat-only): omega per-subject, population kappa.
- M3 (Single-parameter): theta = omega = kappa.
- M4 (Joint): omega + kappa per-subject, both entering W(u).

**H4:**
- H4a: OLS: escape_rate ~ omega_z + kappa_z. omega beta > 0, p < .01.
- H4b: Classify errors from empirical expected reward per T x D cell. Overcaution > 65%. r(omega, overcaution ratio) > 0.30, p < .01.
- H4c: r(kappa, mean vigor) < -0.30, p < .01.
- H4d: r(atan2(kappa_z, omega_z), % optimal) < -0.15, p < .01.

**H5:**
- H5a: Hierarchical regression. Step 1: outcome ~ omega_z + kappa_z. Step 2: + calibration_z. delta-R-squared > 0.03, p < .01, for at least two of: % optimal, escape, earnings.
- H5b: r(anxiety slope, choice shift) > 0.20, p < .01.
- H5c: r(omega, confidence) < 0, p < .01. |r(omega, anxiety)| < 0.10.
- H5d: r(confidence, n_overcautious) < 0 AND r(confidence, n_reckless) > 0, both p < .01.

### Transformations

- omega and kappa log-transformed before z-scoring.
- Threat and distance z-scored for regressions.
- Vigor normalized by calibration maximum.

### Inference criteria

p < 0.01 for all directional tests. delta-WAIC > 0 for model comparisons. Multiple comparison correction not applied — each test is a specific directional prediction from the exploratory sample.

### Data exclusion

Subjects with calibration mean IPI > 2.5 SD from sample mean. No other exclusions.

### Missing data

Non-response trials excluded. Per-subject indices computed from available trials.

### Exploratory analysis

1. Separate-equations model (lambda for choice, omega for vigor, no shared W) testing whether the joint constraint hurts fit.
2. Scaled single-parameter model (M3b): theta as omega, alpha*theta as kappa, testing if M3 failure is a scale artifact.
3. Parameter recovery via simulation.
4. Posterior predictive checks.
5. Encounter spike individual differences (CV, split-half reliability).
6. Clinical regressions (omega + kappa + affect → questionnaire scores).
7. Trial-level anxiety-vigor coupling.
8. Four foraging profiles (Strategic, Resilient, Reckless, Helpless) with earnings and escape rates.
