# Preregistration: Effort Foraging Under Threat

## Research Questions

The overarching questions being asked are:

- **(A)** How do humans integrate energetic cost and predation risk when foraging?
- **(B)** Does this integration manifest coherently across choice behavior, motor execution, and subjective experience?

We investigate these questions across three domains: **foraging behavior** (choice and vigor), **computational architecture** (model comparison), and **metacognitive monitoring** (anxiety and confidence).

---

## Hypotheses

### H1 — Threat reshapes choice, vigor, and affect

Threat will reduce high-effort choices, increase motor vigor, and shift anxiety upward and confidence downward.

- **H1a.** High-effort choices will decrease with threat probability and distance.
- **H1b.** Anxiety will increase with threat and distance. Confidence will decrease with threat and distance.
- **H1c.** Within each chosen effort level, pressing intensity will increase with threat.

### H2 — Vigor follows the predatory imminence continuum

Motor vigor will follow the predatory imminence continuum, with distinct anticipatory and reactive dynamics.

- **H2a.** Predator encounter will trigger a rapid motor spike in pressing rate.
- **H2b.** The temporal shape of the vigor timecourse will differ by encounter status and by threat level.

### H3 — A joint fitness model outperforms alternatives

A joint fitness model with two per-subject parameters (subjective cost of capture, subjective cost of effort) will outperform simpler alternatives.

- **H3a.** The joint model will outperform an **effort-only** model that ignores threat.
- **H3b.** The joint model will outperform a **threat-only** model that lacks individual effort sensitivity.
- **H3c.** The joint model will outperform a **single-parameter** model, demonstrating that capture cost and effort cost are separable traits.

### H4 — Model parameters predict survival, errors, and decision quality

- **H4a.** Higher capture cost (ω) will predict higher escape rates on attack trials.
- **H4b.** Capture cost will predict the proportion of overcautious errors: higher ω will predict a greater share of suboptimal choices that are overcautious (choosing light when heavy has higher expected reward).
- **H4c.** Higher effort cost (κ) will predict lower pressing intensity.
- **H4d.** The balance between capture cost and effort cost will predict decision quality: effort-driven avoidance will be less optimal than threat-driven avoidance.
- **H4e.** Consistency with the joint fitness function will predict foraging earnings. Both choice consistency and intensity pattern match will independently contribute.

### H5 — Anxiety and confidence monitor the foraging computation

Anxiety and confidence will independently monitor the foraging computation and predict foraging efficiency beyond the model parameters.

- **H5a.** Anxiety **calibration** (how well anxiety tracks threat) will predict foraging optimality beyond the model parameters.
- **H5b.** Anxiety **reactivity** (slope on threat) will predict adaptive choice shifting across threat levels.
- **H5c.** Capture cost will predict subjective **confidence** but not anxiety.
- **H5d.** Confidence will predict the *type* of errors people make — fewer overcautious errors but more reckless errors — without affecting overall error rate.

---

## Foreknowledge of Data

**Status:** Data exists but the authors have not observed it yet.

At least some of the data that will be used for this analysis plan exists and is possible for the authors to access. However, the authors certify that they have not accessed any of that data and will not do so until after this plan is registered.

### Managing unintended influences

We collected an **exploratory sample** of 350 participants (290 after exclusions). All hypotheses, model specifications, thresholds, and analysis pipelines were developed on this exploratory sample. We then collected an independent **confirmatory sample** of approximately 350 participants. The confirmatory data have not been analyzed at the time of this registration. We aim to replicate all preregistered analyses on the confirmatory sample.

---

## Research Design

**Study type:** Randomized Experiment (random assignment of subjects to treatments or conditions).

**Blinding:** No blinding is involved.

### Study design

Participants forage in a circular arena under predation risk. On each trial, they choose between:

| Cookie | Reward | Required pressing rate | Distance |
|---|---|---|---|
| **Heavy** | 5 points | 90% of calibrated max | D ∈ {1, 2, 3} = {5, 7, 9} game units |
| **Light** | 1 point | 40% of calibrated max | Always D = 1 |

Participants use the keyboard to exert effort by holding `S+D+F` while pressing `A` repeatedly to transport the chosen cookie to the safe zone at the center. A predator may appear based on the stated threat probability **T ∈ {0.1, 0.5, 0.9}**.

**Capture penalty:** −5 points plus loss of the current cookie reward.
**Reward:** Monetary bonus proportional to total score.

### Trial types

| Type | Count | Description |
|---|---|---|
| **Choice trials** | 45 | Participants freely select between heavy and light cookies |
| **Probe trials** | 36 | Both options are identical (forced choice); participants rate prospective anxiety and confidence on 1–10 scales before pressing |

On each probe trial, the cookie type (heavy/light), threat probability, and distance are assigned to fully cross all conditions: **3 threat × 3 distance × 2 cookie types = 18 unique conditions**, each sampled once for anxiety and once for confidence (18 anxiety + 18 confidence probes).

### Calibration and structure

- A **calibration phase** at the start of the task establishes each participant's maximum pressing speed, used to normalize pressing rates across participants.
- Calibration is repeated **before each block** to control for fatigue.
- **3 blocks × 27 trials = 81 total events** per participant.
- Threat (T), distance (D), and cookie assignment are fully crossed within blocks.
- Trial order within blocks is randomized; threat and distance conditions are balanced across blocks.
- **Seven unique probe schedules** are used across subjects.

### Randomization

Trial order within blocks is randomized. Threat and distance conditions are balanced across blocks. Seven unique probe schedules are used across subjects.

---

## Sampling

### Data collection procedures

Participants are recruited through **Prolific**. Eligibility:

- 18–65 years old
- Fluent in English
- Normal or corrected-to-normal vision

**Compensation:** $10/hr base + performance bonus ($0.10/pt).

**Study flow:**

1. Instruction comprehension assessment (Qualtrics)
2. Brief video game use questionnaire
3. Foraging task (Unity, WebGL in browser)
4. Post-task questionnaires: DASS-21, PHQ-9, OASIS, STAI (State + Trait), AMI, MFIS, STICSA

### Sample size

**Target:** ~350 participants for the confirmatory sample, matching the exploratory sample. Full combined sample ≈ 640 participants.

### Sample size rationale

The exploratory sample of 290 (after exclusions) provided sufficient power for all preregistered effects:

- **Smallest effect:** H1d choice-vigor shift independence (a null test)
- **Largest effects:** H4b ω–overcaution r = +0.81; H5b slope–choice shift r = +0.39

We match this sample size to ensure comparable power.

### Stopping rule

Data collection stops when approximately **700 participants** have completed the task on Prolific. This ensures both samples (exploratory and confirmatory) have sufficient power for all analyses.

---

## Variables

### Manipulated variables

| Variable | Levels | Description |
|---|---|---|
| **Threat probability (T)** | {0.1, 0.5, 0.9} | Stated predator attack probability, displayed before each trial |
| **Distance / Effort (D)** | {1, 2, 3} | Distance of the heavy cookie from safe zone (5, 7, 9 game units). Light cookie always at D=1 |
| **Effort level** | {0.4, 0.9} | Light: 40% of calibrated max press rate. Heavy: 90% |
| **Trial type** | Choice / Probe | Free selection vs. forced with affect ratings |

### Measured variables

**Behavioral:**

- **Choice:** binary (heavy = 1, light = 0) on each choice trial
- **Pressing rate:** inter-keypress intervals (IPI) at ~5 Hz native resolution. Primary metric: normalized press rate = `median(1/IPI) / calibrationMax`
- **Trial outcome:** escaped or captured (`trialEndState`). Reward earned per trial (`trialReward`)
- **Total earnings:** sum of `trialReward` across all trials (determines bonus)

**Affective (probe trials only):**

- **Anxiety rating** (1–10): *"How anxious are you about being captured in this trial?"*
- **Confidence rating** (1–10): *"How confident are you in your ability to reach safety in this trial?"*

**Questionnaires:**

- **DASS-21** — Depression, Anxiety, Stress subscales
- **PHQ-9** — depression severity
- **OASIS** — overall anxiety
- **STAI** — State and Trait anxiety
- **AMI** — Apathy Motivation Index (Behavioural, Social, Emotional)
- **MFIS** — Modified Fatigue Impact Scale (Physical, Cognitive, Psychosocial)
- **STICSA** — State-Trait Inventory for Cognitive and Somatic Anxiety

### Indices

**Model parameters (from joint fitness model M4):**

- **ω (omega):** per-subject avoidance sensitivity (subjective cost of capture)
- **κ (kappa):** per-subject activation intensity (subjective cost of effort). Enters choice through total demand cost (`κ * req * D`) and vigor through quadratic deviation cost (`κ * (u − req)² * D`)

> **Note on affect indices:** With ~18 probes per subject, individual calibration and slope estimates will have substantial sampling error. We will report split-half reliability of these indices (see Exploratory analysis 4).

**Affect indices (from probe trial regressions):**

- **Anxiety calibration:** within-subject `r(anxiety, threat)`. Higher = anxiety better tracks danger
- **Anxiety slope:** within-subject regression slope of anxiety on threat
- **Mean confidence:** average confidence rating across probes

**Behavioral indices:**

- **Escape rate:** proportion of attack trials survived
- **Overcaution ratio:** proportion of errors that are overcautious (chose light when heavy had higher expected reward)
- **ω–κ angle:** `atan2(κ_z, ω_z)`. Higher = more effort-driven avoidance

**Model consistency indices (H4e):**

- **Choice consistency:** per-subject fraction of choice trials where actual choice matches model prediction (predict heavy if V_H > V_L given subject's ω and κ; light otherwise)
- **Intensity deviation:** per-subject RMSE between model-predicted optimal pressing rate (`u* = argmax_u W(u)`) and observed cell-mean rate, computed across the subject's condition cells. Lower = vigor closer to model prediction

---

## Analysis Plan

### H1 — Statistical tests

- **H1a.** Logistic model with cluster-robust SE: `choice ~ threat_z + dist_z + threat_z:dist_z`, clustered by subject. *Prediction:* both β(threat) and β(distance) negative.
- **H1b.** Linear mixed models: `response ~ threat_z + dist_z + (1 + threat_z | subject)`, separately for anxiety and confidence. *Prediction:* anxiety increases with threat and distance; confidence decreases.
- **H1c.** Paired t-tests on within-subject mean normalized press rate at T=0.9 minus T=0.1, separately within heavy and light. *Prediction:* both comparisons positive.

### H2 — Vigor dynamics

- **H2a.** Compute encounter spike as per-subject mean reactive-epoch pressing rate on attack minus non-attack trials; one-sample t-test against zero. *Prediction:* positive spike.
- **H2b.** GAMs with natural cubic regression splines (K=10) via MixedLM with cookie covariate and random intercepts. Likelihood ratio tests for smooth-by-condition interactions to test for distinct temporal signatures by encounter status and threat level.

### H3 — Computational model specification and comparison

We will fit a joint fitness model grounded in optimal foraging theory (Bednekoff 2007; Brown 1999). The organism maximizes fitness `W(u)` to determine both which patch to select and how intensely to press:

$$
W(u) = S(u) \cdot R - (1 - S(u)) \cdot \omega \cdot (R + C) - \kappa \cdot (u - \text{req})^2 \cdot D
$$

**Where:**

| Symbol | Definition |
|---|---|
| `u` | Pressing rate (normalized by calibration maximum) |
| `S(u, T, D) = exp(-h · T^γ · D / speed(u))` | Survival probability |
| `speed(u) = sigmoid((u − 0.25·req) / σ_sp)` | Movement speed, saturating above required pressing rate |
| `R` | Cookie reward (5 for heavy, 1 for light) |
| `C = 5` | Capture penalty |
| `req` | Required pressing rate (0.9 heavy, 0.4 light) |
| `D` | Distance from safe zone (1–3 heavy; always 1 light) |
| `ω_i` | Per-subject avoidance sensitivity (subjective cost of capture) |
| `κ_i` | Per-subject activation intensity (subjective cost of effort) |
| `h, γ, σ_sp` | Population parameters (hazard scale, hazard exponent, speed saturation width) |

#### Priors

**Population parameters (all weakly informative):**

```
γ        ~ Normal(0, 0.5)   on log-scale, clipped to [0.1, 3.0]
h        ~ Normal(0, 1)     on log-scale
σ_sp     ~ Normal(-1, 0.5)  on log-scale, clipped to [0.01, 1.0]
τ        ~ Normal(0, 1)     on log-scale, clipped to [0.01, 50.0]
σ_v      ~ HalfNormal(0.3)
b_cookie ~ Normal(0, 0.5)
```

**Per-subject parameters (hierarchical, non-centered):**

```
ω_i = exp(m_ω + s_ω · z_i)    where m_ω ~ Normal(0, 1),  s_ω ~ HalfNormal(1.0), z_i ~ Normal(0, 1)
κ_i = exp(m_κ + s_κ · z_i)    where m_κ ~ Normal(-1, 1), s_κ ~ HalfNormal(0.5), z_i ~ Normal(0, 1)
```

#### Choice prediction

For each cookie j, compute:

$$
V_j = \max_u W_j(u) - \kappa \cdot \text{req}_j \cdot D_j
$$

The first term (`max_u W`) is the optimized fitness given the pressing rate. The second term (`κ · req · D`) is the **total demand cost** — the sustained metabolic cost of choosing that cookie, proportional to required pressing rate × distance.

$$
P(\text{heavy}) = \sigma\!\left(\frac{V_H - V_L}{\tau}\right)
$$

where τ is a population noise parameter.

#### Vigor prediction

For the chosen cookie, `u* = argmax_u W(u)` determines the optimal pressing rate. The vigor likelihood uses per-subject condition cell means (subject × threat × distance × cookie, ~18 cells per subject, ~5,200 total):

$$
\text{observed cell-mean rate} \sim \text{Normal}\!\left(u^* + b_{\text{cookie}} \cdot \text{is\_heavy},\ \sigma_v / \sqrt{n_{\text{trials}}}\right)
$$

We use cell means rather than trial-level data because the fitness function predicts a single optimal rate per condition — trial-to-trial variance within a condition reflects motor noise, not parametric signal. The `√n_trials` denominator ensures that cells with fewer observations receive proportionally less weight.

> The total demand cost (`κ · req · D`) enters the choice equation but not the vigor optimization. This reflects the distinction between deciding *how much effort to commit* (total demand for the full trial) and *optimizing moment-to-moment pressing intensity* (marginal deviation cost). Both are governed by the same κ — indicating that a person's effort sensitivity determines both whether they take the hard job and how hard they work on it.

#### Model fitting

- **Inference:** NumPyro HMC/NUTS
- **Chains:** 4 × 2,000 warmup + 4,000 samples
- **target_accept** = 0.95, **max_tree_depth** = 10
- **Convergence requirement:** R̂ < 1.01 and bulk ESS > 400 for all parameters
- If any model fails to converge, sampling iterations will be doubled before declaring non-convergence

#### Parameter recovery

We will simulate 500 synthetic subjects from known ω and κ values, fit the model to the simulated data, and correlate recovered with true parameters to verify identifiability.

#### Model comparison

- **Primary:** WAIC computed from pointwise log-likelihoods via ArviZ
- **Robustness:** approximate LOO-CV via Pareto-smoothed importance sampling (PSIS-LOO)
- A hypothesis is supported only if **WAIC and LOO agree**
- All four models evaluated on the same joint likelihood (choice + vigor) to ensure fair comparison

#### The four models compared

| Model | Per-subject parameters | Description |
|---|---|---|
| **M1 (Effort-only)** | κ_i | Choice: `ΔV = ΔR − κ_i · Δeffort(D)`. No survival function, no threat term. Vigor: intercept-only with no condition structure. **Tests whether threat adds anything beyond effort cost.** |
| **M2 (Threat-only)** | ω_i (population κ) | Choice and vigor both from `W(u)`, but κ shared across subjects. **Tests whether individual effort sensitivity matters or only threat sensitivity.** |
| **M3 (Single-parameter)** | θ_i | θ enters `W(u)` as both ω and κ (θ = ω = κ). **Tests whether a single trait can serve both avoidance and activation roles.** |
| **M4 (Joint model)** | ω_i and κ_i | Both enter `W(u)` through the fitness function. The full model. Both parameters are identifiable and approximately orthogonal in the exploratory sample. |

### H4 — Bayesian linear models

All H4 and H5 regressions will be fitted with Bayesian linear models (using `bambi`) using default weakly informative priors (Normal(0, σ) for coefficients scaled by data). Posterior sampling: **4 chains × 2,000 draws + 1,000 tuning**.

- **H4a.** `escape_rate ~ ω_z + κ_z`. *Prediction:* ω positive.
- **H4b.** Classify errors from empirical expected reward per T×D cell; fit `overcaution_ratio ~ ω_z`. *Prediction:* ω positive. Overall overcaution percentage reported descriptively.
- **H4c.** `mean_vigor ~ κ_z`. *Prediction:* κ negative.
- **H4d.** `pct_optimal ~ angle_z` (angle = `atan2(κ_z, ω_z)`). *Prediction:* angle negative (effort-driven avoidance is less optimal).
- **H4e.** `earnings ~ choice_consistency_z + intensity_deviation_z`. *Prediction:* choice_consistency positive; intensity_deviation negative.

### H5 — Anxiety and confidence

- **H5a.** Compare **base** (`pct_optimal ~ ω_z + κ_z`) and **full** (`pct_optimal ~ ω_z + κ_z + calibration_z`) models via LOO-CV. *Prediction:* calibration improves model fit. Escape rate and earnings tested as supporting outcomes.
- **H5b.** `choice_shift ~ anxiety_slope_z`. *Prediction:* anxiety slope positive.
- **H5c.** `mean_confidence ~ ω_z` and `mean_anxiety ~ ω_z`. *Prediction:* ω negative for confidence; practically zero for anxiety (ROPE test).
- **H5d.** `n_overcautious ~ confidence_z` and `n_reckless ~ confidence_z`. *Prediction:* confidence negative for overcautious errors; positive for reckless errors.

---

## Transformations

- ω and κ **log-transformed** before z-scoring
- Threat and distance **z-scored** for regressions
- Vigor **normalized** by calibration maximum

---

## Inference Criteria

| Hypothesis family | Framework | Criterion |
|---|---|---|
| **H1, H2** | Frequentist | Directional tests (t-tests, logistic): *p* < .01. LMM coefficients: \|t\| > 3. LRT for GAMs: *p* < .01 |
| **H3** | Model comparison | WAIC (primary) and PSIS-LOO (robustness). A hypothesis is supported only if both criteria agree (ΔWAIC > 0 **and** ΔLOO > 0 favoring M4) |
| **H4, H5** | Bayesian | Directional prediction supported if 95% HDI excludes zero in predicted direction |
| **H5a (LOO)** | Bayesian | Calibration improves fit if ΔELPD > 0 with SE excluding zero |
| **H5c (null)** | ROPE | [−0.10, +0.10] on standardized β; null supported if 95% HDI falls entirely within ROPE |

---

## Data Inclusion and Exclusion

### Participant-level

- **Incomplete data:** participants must complete all 81 trials and have data in all modalities (behavioral, probe ratings, questionnaires)
- **Calibration outliers:** mean inter-press interval during calibration > 2.5 SD from sample mean
- **Task engagement:** escape rate < 35% across attack trials

> In the exploratory sample, these criteria excluded **60 of 350** participants (57 for incomplete data or task engagement; 3 calibration outliers), yielding **N = 290** analyzed.

### Trial-level

- Non-response trials (no keypresses recorded) excluded from per-subject indices
- Inter-press intervals < 10 ms treated as artifacts and removed before computing pressing rate

### Missing data

Non-response trials excluded. Per-subject indices computed from available trials.

---

## Other Planned Analyses

1. **Separate-equations model.** Fit a model with λ (choice-only) + ω (vigor-only) and no shared W function, to test whether the joint constraint hurts fit relative to unconstrained separate equations.
2. **Scaled single-parameter model (M3b).** Fit θ as ω with α·θ as κ (α = population scaling factor), to test whether M3's failure is merely a scale mismatch rather than genuine separability.
3. **Posterior predictive checks.** Generate model-predicted vs. observed choice and vigor by condition.
4. **Affect index reliability.** Compute split-half reliability (odd/even probe trials) for anxiety calibration, anxiety slope, and mean confidence to assess stability given the limited number of probes per subject (~18).
5. **Encounter spike individual differences.** Compute CV, split-half reliability, and model parameter correlations with the reactive motor response.
6. **Clinical regressions.** Regress all questionnaire scores (DASS-21, PHQ-9, OASIS, STAI, AMI, MFIS, STICSA) on ω + κ + affect measures. All confirmatory hypotheses (H1–H5) tested on the confirmatory sample alone; both samples (~580 subjects) pooled for clinical regressions to maximize power for detecting small effects.
7. **Trial-level anxiety–vigor coupling.** Fit an LMM testing whether within-person anxiety fluctuations predict pressing intensity beyond threat level.
8. **Frequentist robustness.** Replicate key H4 and H5 results with OLS/Pearson r (*p* < .01) to confirm consistency across inference frameworks.
9. **Normative benchmark.** Compare participant behavior to a model-derived optimal strategy and quantify the over-caution cost in points and bonus payment.
