# Methods

## Results from Discovery Sample (N = 293)

---

## Participants

We recruited 350 participants from Prolific.co (https://prolific.co) for an online behavioral study. The study was approved by the Caltech Institutional Review Board, and all participants provided informed consent. After quality control (see Pre-processing below), the final discovery sample comprised N = 293 participants (83.7% retention). Demographic details are reported in the Participant Characteristics section below.

---

## Task Design

### Overview

Participants completed an effort-based foraging task implemented in Unity (WebGL) and presented in a desktop browser. The task featured a circular arena viewed from above, with a safe zone at the center and cookies appearing at radial distances.

### Effort Calibration

Before the main task, participants completed three 10-second trials of pressing the S, D, and F keys simultaneously as fast as possible. The maximum press count across trials defined each participant's calibrated maximum (f_max). All subsequent effort demands were expressed relative to this individual calibration.

### Trial Structure — Choice Trials (Type 1)

On each choice trial, two cookies appeared:
- **Heavy cookie:** 5 points reward, requiring sustained pressing at 60–100% of f_max (effort levels: 0.6, 0.8, 1.0 depending on distance), at distance D ∈ {1, 2, 3} (corresponding to 5, 7, 9 game units from center)
- **Light cookie:** 1 point reward, requiring pressing at 40% of f_max, always at distance 1 (5 game units)

Participants clicked to select a cookie (commitment was irrevocable), then pressed S+D+F repeatedly to transport it to the safe zone. Movement speed depended on press rate relative to the cookie's demand threshold:
- ≥ 100% of required rate → full speed
- ≥ 50% → half speed
- ≥ 25% → quarter speed
- < 25% → movement decays to zero

### Threat Manipulation

Each trial had an independent predation probability T ∈ {0.1, 0.5, 0.9}, displayed visually to the participant. On attack trials:
1. A predator spawned at the perimeter position closest to the participant at a pre-determined encounter time (calibrated to approximately half the expected return time at full speed)
2. The predator approached at 0.5 units/sec (slow approach phase)
3. The predator struck at a Gaussian-distributed time centered at twice the encounter time, accelerating to 4× the participant's maximum movement speed (unavoidable full-attack phase)

Being captured cost the participant 5 points (the capture penalty C) plus the value of the current cookie was not received.

**Key design feature:** Predator spawn timing is calibrated to half the player's return time at full speed. This means distance does NOT independently affect survival probability at full speed — only threat probability determines whether an attack occurs. Distance affects the cumulative effort required (more distance = more sustained pressing = higher effort cost).

### Trial Structure — Probe Trials (Types 5 and 6)

On probe trials, both cookie options were identical (same weight, same distance). After selecting a cookie, the game paused and participants rated either:
- **Anxiety:** "How anxious are you about being captured?" (0–7 scale)
- **Confidence:** "How confident are you about reaching safety?" (0–7 scale)

The trial then proceeded normally with pressing and potential predator attack. Probe ratings were prospective judgments about the current trial's demands, collected after choice commitment but before any motor execution.

Probe trials spanned all combinations of threat (3 levels) × distance (3 levels, from startDistance: 5, 7, 9 game units) × cookie type (heavy, light) × rating type (anxiety, confidence).

### Block Structure

Three blocks of 27 events each (81 total per participant):
- 15 choice trials per block (45 total)
- 12 probe trials per block (36 total: 6 anxiety + 6 confidence)

### Reward Structure

| Event | Payoff |
|-------|--------|
| Survive with heavy cookie | +5 points |
| Survive with light cookie | +1 point |
| Captured (any cookie) | −5 points (penalty), cookie reward not received |

---

## Psychiatric Assessment

Between task blocks, participants completed the following validated questionnaires:

| Instrument | Subscales | Items |
|-----------|-----------|-------|
| DASS-21 | Depression, Anxiety, Stress | 21 |
| PHQ-9 | Depression severity | 9 |
| OASIS | Overall anxiety severity | 5 |
| STAI | State anxiety | 20 |
| AMI | Behavioural, Social, Emotional apathy | 18 |
| MFIS | Physical, Cognitive, Psychosocial fatigue | 21 |
| STICSA | Cognitive and somatic anxiety | 21 |

All subscale scores were z-scored across participants before analysis.

---

## Pre-processing and Data Quality

### Five-stage pipeline

1. **Completion:** Participants who did not finish all 81 trials were excluded.
2. **Comprehension:** Participants failing comprehension checks were excluded.
3. **Behavioral consistency:** Participants with implausible keypresses (max single-trial press rate > 3 SD above sample mean, indicating automated input) or zero presses on > 50% of regular trials (disengagement) were excluded.
4. **Calibration validation:** Participants with fewer than 10 presses across calibration trials were excluded.
5. **Outlier removal:** Participants with overall escape rate < 35% across attack trials (low engagement) were excluded.

Additional exclusion for affect analyses: participants with < 80% probe completion (< 29/36 probes) were excluded from H1c, H3, and H4 analyses only.

No post-hoc exclusions based on model fit quality or statistical extremity.

### Retention

350 recruited → 293 retained (83.7%). Exclusion breakdown available in supplementary materials.

### Vigor computation

1. Parse `alignedEffortRate` column (string of press timestamps) for each trial
2. Compute inter-press intervals (IPIs)
3. Filter IPIs > 0.01s (remove double-presses/artifacts)
4. Require ≥ 5 valid IPIs per trial (trials with fewer excluded from vigor analyses)
5. Median rate = median(1 / IPI) / calibrationMax → normalized to calMax units
6. Excess effort = median rate − required rate for chosen cookie (0.9 for heavy, 0.4 for light)
7. Cookie-type centering: subtract population mean excess for each cookie type (heavy mean = 0.104, light mean = 0.543)

Cookie-type centering removes the demand confound (light cookies mechanically produce higher raw excess because the demand threshold is lower) while preserving between-subject variation in pressing intensity.

---

## EVC-LQR Model

### Architecture

The model has two per-subject parameters drawn from log-normal distributions with non-centered parameterization, plus population-level parameters.

**Per-subject parameters:**
- c_e (effort cost): governs choice through distance-dependent effort penalty
- c_d (capture aversion): governs vigor through the survival incentive

**Population parameters:**

| Parameter | Prior | Description |
|-----------|-------|-------------|
| γ (gamma) | exp(Normal(0, 0.5)), clipped [0.1, 3.0] | Probability weighting exponent |
| ε (epsilon) | exp(Normal(−1, 0.5)) | Effort efficacy |
| ce_vigor | exp(Normal(−3, 1)) | LQR deviation motor cost |
| τ (tau) | exp(Normal(−1, 1)), clipped [0.01, 20] | Choice temperature |
| p_esc | sigmoid(Normal(0, 1)) | Escape probability at full speed |
| σ_motor | exp(Normal(−1, 0.5)), clipped [0.01, 1] | Motor noise width |
| σ_v | HalfNormal(0.5) | Vigor observation noise |

### Choice Model

The differential expected utility of heavy versus light is:

```
ΔEU = S × (R_H − R_L) − c_e,i × (req_H² × D_H − req_L² × D_L)
     = S × 4 − c_e,i × (0.81 × D_H − 0.16)
```

where S = (1 − T^γ) + ε × T^γ × p_esc is the subjective survival probability. The capture aversion c_d does not appear in the choice equation because its contribution to the differential EU is collinear with the reward differential — both scale with the reward difference and are functions of S, making c_d unidentifiable from choice data.

P(choose heavy) = sigmoid(ΔEU / τ)

### Vigor Model

For each trial, the model computes an optimal press rate u* by maximizing:

```
EU(u) = S(u) × R − (1−S(u)) × c_d,i × (R+C) − ce_vigor × (u−req)² × D
```

where S(u) = (1 − T^γ) + ε × T^γ × p_esc × sigmoid((u − req) / σ_motor), and the effort cost is the LQR deviation cost (u − req)². Pressing at the required rate (u = req) incurs zero additional motor cost; the commitment cost was already paid at the choice stage. The optimal u* is computed via softmax-weighted grid search over u ∈ [0.1, 1.5] with 30 grid points and temperature β = 10.

### Data

- **Choice likelihood:** Bernoulli on observed choice. All 81 trials enter the choice likelihood, but probe trials contribute a constant (ΔEU = 0 → P(H) = 0.5) because both options are identical.
- **Vigor likelihood:** Normal(predicted excess, σ_v) on cookie-centered excess effort. All 81 trials contribute.
- Probe distances derived from `startDistance` column (5 → D=1, 7 → D=2, 9 → D=3).

### Joint Likelihood

Both likelihoods are evaluated simultaneously during fitting. The shared parameters (ε, γ) link the two behavioral channels.

---

## Model Fitting

The model was fitted using NumPyro's stochastic variational inference (SVI) with an AutoNormal guide (mean-field variational approximation). Optimization used Adam (learning rate = 0.002) for 40,000 steps. The final ELBO was used to compute BIC = 2 × loss + k × log(N), where k = 2 × N_subjects + number of population parameters.

---

## Parameter Recovery

We simulated 3 synthetic datasets of 50 subjects × 81 trials each, drawing subject parameters from the fitted population distribution and generating choices and vigor from the generative model with the same task design. Each dataset was re-fitted with identical SVI procedure (25,000 steps). Recovery was assessed as the Pearson correlation between true and recovered parameters in log space.

---

## Model Comparison

Six ablation models were fitted, each removing one component from the full model. All models were evaluated on the same data (81 trials per subject for both likelihoods) to ensure fair BIC comparison. See H2 document for details and results.

---

## Affect Analysis

Linear mixed models (statsmodels MixedLM, REML estimation, L-BFGS optimizer) predicted anxiety and confidence ratings from standardized survival probability (S_z) with random intercepts and slopes by subject. Random effects accounted for individual differences in both baseline affect and sensitivity to survival probability.

---

## Metacognitive Decomposition

**Calibration:** Per-subject Pearson correlation between anxiety ratings and model-derived danger (1 − S), computed across the subject's 18 anxiety probe trials. Danger = 1 − S, where S uses population-level γ and ε from the fitted model.

**Discrepancy:** Per-subject mean residual from the population-level regression of anxiety on S. The population-level regression establishes the "average" anxiety-danger relationship; each subject's residual indicates whether they are systematically more anxious (positive discrepancy) or less anxious (negative discrepancy) than average at the same danger level.

---

## Clinical Analysis

### Frequentist

Pearson correlations between log-transformed model parameters (log(ce), log(cd)) and z-scored psychiatric subscales. Multiple comparison correction via Benjamini-Hochberg FDR within each parameter.

### Bayesian

Bayesian linear regression (bambi 0.17 with PyMC 5.28 backend, weakly informative priors, 2,000 draws × 4 chains, target_accept = 0.9). Each clinical measure was predicted from log(ce) + log(cd) + discrepancy + calibration. Region of practical equivalence (ROPE): |β| < 0.10 (standardized). Posterior mass within ROPE was computed for each predictor to evaluate evidence for null effects.

### Machine Learning

Cross-validated prediction of each clinical measure from log(ce) + log(cd) + discrepancy + calibration + all pairwise interactions, using elastic net (ElasticNetCV, l1_ratio ∈ {0.1, 0.5, 0.7, 0.9, 0.95, 1.0}, 50 alphas) and ridge regression (RidgeCV), with repeated 10-fold cross-validation (5 repeats). CV R² reported as mean ± SD across folds.

---

## Statistical Analysis

All tests were two-tailed unless otherwise specified. Effect sizes are reported as Pearson r, standardized β, or R². ANOVA used type II sums of squares. Steiger's test (1980) was used to compare dependent correlations. All analyses were conducted in Python 3.11 using NumPyro 0.15, JAX 0.4, statsmodels 0.14, scipy 1.12, scikit-learn 1.4, bambi 0.17, and PyMC 5.28.

---

## Software and Reproducibility

- Model code: `/workspace/scripts/modeling/evc_final_2plus2.py`
- Analysis scripts: `/workspace/notebooks/07_evc_pipeline/`
- All code available at [repository URL]
- Data will be shared on OSF upon acceptance
