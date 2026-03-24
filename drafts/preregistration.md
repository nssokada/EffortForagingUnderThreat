# Preregistration: Confirmatory Replication of Effort-Threat Integration in Human Foraging

**Format:** AsPredicted (#XXXXXX)
**Title:** Humans Reallocate Effort Across Decision and Action When Foraging Under Threat — Confirmatory Sample
**Authors:** Noah Okada, Ketika Garg, Toby Wise, Dean Mobbs
**Affiliation:** Caltech
**Date of preregistration:** [Date to be filled at submission]
**Confirmatory sample:** N=350 (data collected, not yet analyzed)
**Exploratory sample:** N=293 (basis for all hypotheses and parameter specifications below)

---

## Have the data been collected already?

Yes. The confirmatory sample (N=350, recruited via Prolific) has been collected but **has not been analyzed in any form**. No preprocessing, model fitting, or statistical tests have been performed on these data. All hypotheses and analysis plans below are derived exclusively from the exploratory sample (N=293).

---

## 1. Hypotheses

We test six pre-specified hypotheses derived from the exploratory sample (N=293). Each hypothesis states an expected direction and minimum threshold. All tests are one-tailed where a directional prediction is made; two-tailed otherwise.

---

### H1 — The survival-weighted additive-effort choice model best explains foraging behavior

**Statement:** Choice behavior is best explained by a model in which (a) energetic effort costs are subtracted additively from expected reward, (b) escape probability follows a hyperbolic function of distance, and (c) residual threat aversion enters as a subject-specific additive bias β. The winning model has subjective value:

```
SV = R · S − k · E − β · (1 − S)
S  = (1 − T) + T / (1 + λ · D)
```

where R is reward magnitude (R_H = 5, R_L = 1), S is option-specific survival probability (computed separately for S_H and S_L using each option's distance), T is threat probability ∈ {0.1, 0.5, 0.9}, D is cookie distance ∈ {1, 2, 3}, E is normalized effort demand, k is subject-specific effort discounting, β is subject-specific threat bias (subjective capture cost), and λ is a population-level hazard parameter estimated from data (exploratory: λ ≈ 14.0).

Choices are modeled via softmax on the value difference:

```
p(choose H) = σ(τ · (SV_H − SV_L))
```

with τ a population-level inverse-temperature.

**Operationalization of "best":** This model (M5 in the comparison scheme) must achieve the highest ELBO among all five models in the comparison set.

**Models compared (5, each testing one structural question):**

| Model | Structure | Question |
|---|---|---|
| M1 | SV = R·exp(−kE) | Does threat matter? |
| M2 | SV = R·exp(−kE) − β·T·D | Mechanistic S or linear features? |
| M3 | SV = R·exp(−kE)·S − β·(1−S), S=exp(−λTD) | Which survival kernel? |
| M4 | SV = R·exp(−kE)·S − β·(1−S), S=(1−T)+T/(1+λD) | Which effort structure? |
| M5 | SV = R·S − k·E − β·(1−S), S=(1−T)+T/(1+λD) | **Winner** |

**Primary tests:**
1. Additive effort (M5) outperforms multiplicative effort (M4) by ΔELBO > 0 (exploratory: +158).
2. Hyperbolic survival (M4) outperforms exponential survival (M3) by ΔELBO > 0 (exploratory: +174).
3. The survival model (M5) substantially outperforms the effort-only baseline (M1) by ΔELBO > 100 (exploratory: +2,038).

**Support criterion:** All three tests must be directionally consistent. Tests 1 and 2 are the primary confirmatory tests.

**Rejection criterion:** Either multiplicative effort or exponential survival outperforms the winning specification.

---

### H2 — Model-derived survival predicts trial-level anxiety and confidence

**Statement:** The survival probability S, computed from each participant's fitted parameters using the winning model M5 (S = (1−T) + T/(1+λD), λ estimated from confirmatory choice model fit), negatively predicts trial-level anxiety ratings and positively predicts trial-level confidence ratings at the within-subject level. This constitutes evidence that subjective affect tracks the same latent survival computation that governs choice.

**Primary tests:**
1. Mixed-effects linear regression: `anxiety ~ S_probe_z + (1 + S_probe_z | subject)`. Test: β(S_probe_z) < 0, with |t| > 3.0 (exploratory: β = −0.605, t = −25.63).
2. Mixed-effects linear regression: `confidence ~ S_probe_z + (1 + S_probe_z | subject)`. Test: β(S_probe_z) > 0, with |t| > 3.0 (exploratory: β = +0.612, t = +25.65).

**Secondary test (parameter moderation):**
3. k (effort discounting) negatively predicts mean anxiety and positively predicts mean confidence between subjects (between-subjects OLS: mean affect ~ k_z). Directional, p < 0.05 one-tailed (exploratory: k → anxiety β = +0.127, p = 0.032; k → confidence β = −0.154, p = 0.009).

**S_probe computation:** For each probe trial, S_probe = (1 − T) + T / (1 + λ · D) with λ (estimated from confirmatory choice model fit) (population-level, fixed from exploratory fit). The subject-specific posterior mean k and β from the M5 SVI fit are used only for the parameter moderation secondary tests; S_probe itself uses only population-level λ.

**Support criterion:** Both primary tests (H2.1 and H2.2) must be significant in the predicted direction. The secondary test is confirmatory for the anxiety direction only (k → higher mean anxiety).

**Rejection criterion:** Either primary β is in the wrong direction, or both are non-significant after Bonferroni correction for the two primary tests (α = 0.025 per test).

---

### H3 — Danger drives excess motor vigor

**Statement:** Lower survival probability S causes participants to press harder than the task requires, expressed as a positive population-mean danger mobilization parameter δ in a hierarchical Bayesian model of excess effort.

**Vigor model:**
```
excess_ij = α_i + δ_i · (1 − S_ij) + ε_ij
ε_ij ~ Normal(0, σ)
```

where excess_ij is the difference between observed capacity-normalized vigor and the effort demand of the chosen option, S_ij = (1−T) + T/(1+λD_chosen) uses the distance of the chosen cookie and λ from the choice model, α_i is subject-specific baseline excess effort, and δ_i is subject-specific danger-responsive mobilization. All subject-level parameters have hierarchical normal priors estimated via MCMC (NumPyro NUTS) or SVI.

**Primary tests:**
1. Population mean μ_δ > 0 (one-tailed). The posterior 95% credible interval for μ_δ must exclude zero. (Exploratory: μ_δ = +0.211, P(μ_δ > 0) = 1.0.)
2. The proportion of subjects with posterior mean δ_i > 0 must exceed 80% (exploratory: 98.3%).

**Secondary test:**
3. σ_δ > 0.05, confirming that individual differences in danger mobilization are recoverable. (Exploratory: σ_δ = 0.146, shrinkage from OLS = 40.6%.)

**Support criterion:** Both primary tests (H3.1 and H3.2) must be met.

**Rejection criterion:** μ_δ posterior includes zero, or fewer than 65% of subjects show δ_i > 0.

---

### H4 — Choice shift and vigor shift under threat are coherently coupled and predict outcomes

**Statement:** The direction of behavioral adaptation under threat is coherent across behavioral channels: participants who shift more conservatively in their choices (choosing lower-effort, safer options under high threat) press harder during execution. This coherence constitutes evidence that a common survival computation governs both channels, with individual differences in computational parameters (k, β, α) determining how each participant expresses that adaptation.

This hypothesis encompasses four pre-specified tests of the choice–vigor dissociation architecture:

**H4.1 — Choice and vigor are independent at the population level**
P(choose high-effort cookie) and tonic vigor α_bayes (posterior mean from HBM) are not correlated across subjects. Pearson r(choice, vigor) is expected to be near zero: |r| < 0.10 (exploratory: r = +0.008, p = 0.894). This test is two-tailed; we predict no significant correlation.

**H4.2 — k is the primary choice parameter, not vigor**
Multiple regression of choice ~ k + β + α (standardized). k must be the dominant predictor (largest |β|), with adj.R² > 0.60 (exploratory: adj.R² = 0.823, k dominant with β = −0.685). Vigor α must not be significant (p > 0.05) in this regression.

**H4.3 — Threat reverses the choice–vigor coupling**
The per-threat correlation between P(choose high) and tonic vigor changes sign from positive at low threat (T = 0.1) to negative at high threat (T = 0.9). Formally: r(choice, vigor | T=0.1) > 0 and r(choice, vigor | T=0.9) < 0, and the Fisher z-test comparing these two correlations is significant (p < 0.01, two-tailed). Exploratory: r(T=0.1) = +0.196, p = 0.001; r(T=0.9) = −0.219, p < 0.001; Fisher z = 5.07, p < 0.0001.

**H4.4 — Vigor, not choice, predicts escape outcomes**
Trial-level mixed-effects logistic regression: `escaped ~ vigor_z + choice_z + threat_z + dist_z + (1|subject)` on attack trials only. Vigor must be the dominant predictor: β(vigor_z) > β(choice_z) in absolute value, and β(choice_z) ≤ 0 (directional; choosing the hard, far option uses more time exposed and hurts escape). Exploratory: vigor β = +0.091, p = 10⁻⁷⁷; choice β = −0.177, choosing high hurts escape.

**Support criterion:** At least three of the four sub-tests (H4.1–H4.4) must be directionally consistent with exploratory findings. H4.3 and H4.4 are the primary tests given their strongest prior evidence.

**Rejection criterion:** Vigor and choice are significantly positively correlated at the population level (H4.1 fails), or vigor fails to predict escape beyond conditions (H4.4 fails), or the threat-driven sign reversal is absent and the Fisher z-test is non-significant (H4.3 fails).

---

### H5 — Choice and vigor parameters are coupled across independently estimated Bayesian models

**Statement:** Individual differences in threat bias ($\beta$, from the choice model) and danger-responsive vigor mobilization ($\delta$, from the vigor model) are positively correlated, and individual differences in effort sensitivity ($k$) and $\delta$ are negatively correlated. These correlations emerge from models that share no parameters or data — only the survival function $S$ (evaluated at the choice-estimated $\lambda$) links them.

**Primary tests (independent Bayesian pipeline):**
1. Pearson $r(log(\beta_{choice}), \delta_{vigor}) > 0$, $p < .001$ one-tailed. (Exploratory: $r = +0.53$, $p < 10^{-22}$)
2. Pearson $r(log(k_{choice}), \delta_{vigor}) < 0$, $p < .01$ one-tailed. (Exploratory: $r = -0.33$, $p < 10^{-8}$)

**Secondary tests (joint model robustness):**
A joint hierarchical model with correlated random effects $[log(k_i), log(\beta_i), \alpha_i, \delta_i] \sim MVN(\mu, \Sigma)$, $\Omega \sim LKJCholesky(\eta=2)$, with $\lambda$ fixed from the choice model, must confirm:
3. $\rho(\beta, \delta)$ posterior mean > 0 with 95% CI excluding zero.
4. $\rho(k, \delta)$ posterior mean < 0 with 95% CI excluding zero.

**Support criterion:** Both primary tests (H5.1, H5.2) must be met.

**Rejection criterion:** $r(log(\beta), \delta) \leq 0$ or $r(log(k), \delta) \geq 0$.

---

### H6 — Vigor mobilization predicts metacognitive accuracy

**Statement:** Individual differences in danger-responsive vigor mobilization ($\delta$) predict how tightly participants' subjective affect tracks the survival computation $S$. Specifically, higher $\delta$ is associated with steeper within-subject slopes of anxiety on $S$ (more negative) and confidence on $S$ (more positive).

**Primary tests:**
1. Pearson $r(\delta, \text{anxiety slope on } S) < 0$, $p < .05$ one-tailed. (Exploratory: $r = -0.311$, $p < .001$)
2. Pearson $r(\delta, \text{confidence slope on } S) > 0$, $p < .05$ one-tailed. (Exploratory: $r = +0.325$, $p < .001$)

**Secondary test:**
3. $r(\delta, \text{mean anxiety}) < 0$: high-$\delta$ individuals report lower average anxiety despite stronger anxiety-$S$ coupling. (Exploratory: $r = -0.194$, $p < .001$)

**Support criterion:** Both primary tests are significant in the predicted direction.

---

## 2. Study Design

### Participants

**Confirmatory sample:** N = 350 participants recruited via Prolific. All participants completed the task using a desktop or laptop computer (mobile devices excluded). Participation was voluntary with informed electronic consent. Sessions lasted approximately 40–50 minutes. Participants were compensated at a base rate of $10 per hour plus performance-based bonuses.

**Data status:** Data collection is complete. No analyses have been performed on the confirmatory sample.

**Blinding:** The preregistered analysis plan will be finalized and submitted to AsPredicted before the confirmatory data are opened for analysis. The analysis team will be blind to any individual-level data from the confirmatory sample until after the preregistration is timestamped.

### Task Design

Identical to the exploratory sample. A virtual foraging task implemented in Unity (WebGL) and deployed via Prolific desktop browsers.

**Arena:** Circular, top-down arena. Safe zone at the center. Cookies placed at radial distances (D ∈ {1, 2, 3} corresponding to 5, 7, 9 game units from center).

**Options:** On each trial, participants choose between two cookies: a fixed low-effort option (40% of calibrated capacity, D = 1, R_L = 1 point) and a high-effort option (E_H ∈ {0.6, 0.8, 1.0} of calibrated capacity, D_H ∈ {1, 2, 3}, R_H = 5 points).

**Effort calibration:** Participants complete a pre-task calibration (3 × 10-second maximum pressing trials); the highest press count serves as calibration maximum (f_max_i). Minimum threshold: 10 presses. Below threshold → exclusion.

**Threat:** Threat probability T ∈ {0.1, 0.5, 0.9} per trial. On attack trials, the predator spawns at the perimeter and first approaches slowly (0.5 units/sec; encounter phase), then accelerates to 4× the participant's calibrated maximum speed (full attack). Strike time drawn from a Gaussian centered at twice the encounter onset time, creating temporal uncertainty.

**Effort-speed mapping:** Press rate → movement speed via discrete tiers (≥100% = full speed; ≥50% = half; ≥25% = quarter; <25% = zero). Once a cookie is clicked, participants cannot abandon—they must press through until escape or capture.

**Rewards and penalties:** Escape with high-effort cookie: +5 points. Escape with low-effort cookie: +1 point. Capture: lose current cookie reward + 5-point penalty (C = 5).

**Block structure:** 3 blocks × 27 trials = 81 total events per subject. Each block contains 15 regular behavioral trials + 12 probe trials (6 anxiety, 6 confidence). Probe trial: both options identical (forced choice); rating collected after click, before pressing begins (game pauses).

**Probe questions:**
- Anxiety: "How anxious are you about being captured on this trial?" (0–7 scale)
- Confidence: "How confident are you in your ability to reach safety on this trial?" (0–7 scale)

**Psychiatric battery:** Administered between blocks — DASS-21 (stress/anxiety/depression), PHQ-9, OASIS, STAI-Trait, AMI (behavioral/social/emotional subscales), MFIS (physical/cognitive/psychosocial), STICSA. All z-scored before analysis.

### Sample Size Justification

The confirmatory sample (N = 350) was determined to match the original planned confirmatory sample size. Post-hoc power analysis based on exploratory effect sizes indicates adequate power for all preregistered tests:

- **H1 (model comparison):** Model comparison via ELBO difference does not require a conventional power analysis; the effect sizes observed (ΔELBO = +158 additive vs. multiplicative; ΔELBO = +190 hyperbolic vs. exponential) are large relative to model complexity differences and are expected to replicate with N > 200.
- **H2 (affect LMM):** Exploratory effect sizes were very large (t = ±25.6 for S → anxiety/confidence). With N = 350 subjects × 18 probes/type = 6,300 observations per probe type, the test is heavily powered. Even a 50% reduction in effect size (t ≈ 12) would be detectable with near-certainty.
- **H3 (vigor mobilization):** The key statistic is the population posterior for μ_δ. With N = 350 subjects and approximately 15 attack trials per subject on average (N_attacks ≈ 5,250 observations), the HBM posterior will be well-constrained. The exploratory P(μ_δ > 0) = 1.0 with N = 293 leaves substantial margin for reduction.
- **H4 (dissociation):** The key correlations (H4.3 Fisher z = 5.07, H4.4 vigor β = +0.091 at N > 10,000 attack trials) are powered at N = 350. The Fisher z test comparing per-threat correlations requires only that the directional pattern holds; with N = 350, detecting a difference between r = +0.196 and r = −0.219 requires N > 80 (power > 0.99).

After applying exclusion criteria (see Section 5), the confirmatory sample is expected to yield approximately 280–330 analyzable participants, sufficient for all preregistered tests.

---

## 3. Analysis Plan

### 3.0 General Principles

1. **Same code, same parameters:** All analysis scripts used for the exploratory sample will be applied without modification to the confirmatory sample, except for file path arguments. Fixed constants (R_H = 5, R_L = 1, C = 5, effort tiers) are carried forward. Population parameters (λ, τ) are re-estimated from the confirmatory data.

2. **Sequential procedure:** Steps are executed in order: (1) preprocessing → (2) choice model fitting (→ λ, k, β) → (3) vigor HBM with λ from step 2 (→ α, δ) → (4) cross-domain correlations → (5) affect LMMs → (6) joint model robustness check. Outputs from each step feed subsequent steps; no step is modified in light of results from subsequent steps.

3. **No peeking:** The preregistration will be locked on AsPredicted before the confirmatory data are first opened for any analysis. A timestamped copy will be referenced in the paper.

4. **Statistical thresholds:** Unless specified otherwise, α = 0.05 two-tailed. All FDR corrections within a family of tests use the Benjamini–Hochberg procedure. For Bayesian tests, support is defined by the 95% credible interval excluding zero (or the directional bound specified per hypothesis).

---

### 3.1 Preprocessing (replication of exploratory pipeline)

The 5-stage preprocessing pipeline (`scripts/preprocessing/pipeline.py`) will be applied to the confirmatory raw data:

- **Stage 1:** Parse raw JSON trial data; compute trial-level variables (choice, RT, escape, effort, distance, threat, reward).
- **Stage 2:** Score all psychiatric questionnaires; compute subscale totals.
- **Stage 3:** Extract keypress timeseries; compute effort calibration (f_max_i) per subject.
- **Stage 4:** Apply exclusion criteria (see Section 5) and generate filtered subject list.
- **Stage 5:** Merge behavioral, feelings, and psychiatric files into the analysis-ready dataset.

**Outputs expected:** `behavior.csv` (one row per trial), `feelings.csv` (one row per probe rating), `psych.csv` (one row per subject), `subject_mapping.csv`.

---

### 3.2 Analysis Plan for H1 — Choice Model Comparison

**Inference method:** Stochastic variational inference (SVI) in NumPyro with the Adam optimizer (learning rate = 0.01, 30,000 steps per model). Same as exploratory.

**Models compared (same 5-model set as exploratory):**

| Model | Structure | Question |
|---|---|---|
| M1 | SV = R·exp(−kE) | Does threat matter? |
| M2 | SV = R·exp(−kE) − β·T·D | Mechanistic S or linear features? |
| M3 | SV = R·exp(−kE)·S − β·(1−S), S=exp(−λTD) | Which survival kernel? |
| M4 | SV = R·exp(−kE)·S − β·(1−S), S=(1−T)+T/(1+λD) | Which effort structure? |
| M5 | SV = R·S − k·E − β·(1−S), S=(1−T)+T/(1+λD) | **Winner** |

All models use option-specific S (S_H ≠ S_L computed from each option's distance). Per-subject k_i and β_i use non-centered parameterization with log-normal priors. λ and τ are population-level, estimated from data.

**Fixed constants:** R_H = 5, R_L = 1, C = 5, E_L = 0.4.

**Inference:** SVI with NumPyro (AutoNormal guide, 15,000 steps, Adam lr = 0.003) or MCMC (NUTS, 4 chains × 1,000 warmup + 1,000 samples). Both methods will be applied; MCMC results are primary if convergence criteria are met (all Rhat ≤ 1.05, ESS ≥ 100).

**Model fit metrics:** ELBO (from SVI) and/or WAIC (from MCMC). Prediction accuracy (proportion correct).

**Primary comparison for H1:**
- ΔELBO(M5 − M4): must be > 0 (additive > multiplicative effort)
- ΔELBO(M4 − M3): must be > 0 (hyperbolic > exponential survival)
- ΔELBO(M5 − M1): must be > 100 (survival model > effort-only baseline)

**Subject-level parameters extracted from M5:** posterior mean k_i (effort discounting) and β_i (threat bias), plus population λ, for use in all downstream analyses (H2–H6).

**No new model structures will be introduced** on the confirmatory sample; the comparison set is closed.

---

### 3.3 Analysis Plan for H3 — Vigor Hierarchical Bayesian Model

**Vigor operationalization:** Capacity-normalized smoothed pressing rate from `smoothed_vigor_ts.parquet` (20 Hz kernel-smoothed keypress timeseries). Normalization: each subject's pressing rate divided by their calibration maximum (f_max_i). Trial-level vigor is the mean capacity-normalized rate across the trial. *Excess effort* = trial vigor − effort demand of chosen option.

**Excess effort model:**

```
excess_ij = α_i + δ_i · (1 − S_ij) + ε_ij
ε_ij ~ Normal(0, σ)
```

where S_ij = (1−T) + T/(1+λ·D_chosen) uses the distance of the chosen cookie and λ from the choice model (Section 3.2).

- **α_i** (baseline excess effort): how much the participant presses above demand on average.
- **δ_i** (danger mobilization): how much excess effort increases as survival probability decreases.
- **σ**: residual standard deviation (population-level).

**Hierarchical priors:**
```
μ_α ~ Normal(0, 1),    σ_α ~ HalfNormal(1)
α_i ~ Normal(μ_α, σ_α)
μ_δ ~ Normal(0, 1),    σ_δ ~ HalfNormal(1)
δ_i ~ Normal(μ_δ, σ_δ)
σ   ~ HalfNormal(1)
```

**Inference:** NumPyro NUTS (4 chains × 1,000 warmup + 1,000 sampling, target_accept = 0.90) or SVI (AutoNormal, 10,000 steps). MCMC preferred if convergence criteria met (Rhat ≤ 1.05, ESS ≥ 100).

**Preregistered tests for H3:**
1. P(μ_δ > 0 | data) > 0.975 (95% one-sided posterior credibility). (Exploratory: P = 1.0.)
2. Proportion of subjects with posterior mean δ_i > 0 must exceed 80%. (Exploratory: 98.3%.)

**Secondary:** σ_δ > 0.05 (individual differences recoverable; exploratory: σ_δ = 0.146).

**Data filtering:** Probe trials excluded from vigor alignment using `feelings.csv` trialNumber per subject.

---

### 3.4 Analysis Plan for H2 — Affect LMMs

**Data:** `feelings.csv` from Stage 5 output. One row per probe rating. Columns: `subj`, `trialNumber`, `probe_type` (anxiety/confidence), `rating` (0–7), `p_threat`, `distance`, `distanceFromSafety`.

**S_probe computation:**
```
S_probe[i,t] = (1 − T[t]) + T[t] / (1 + λ · D[t])
```
with T = `p_threat`, D = `distance` + 1 (converting 0-indexed to 1-indexed distance level), λ = population-level estimate from the confirmatory choice model (Section 3.2).

Note: S_probe uses only population-level λ; no subject-specific parameters enter S_probe. This is the most conservative test.

**Primary LMMs (Python statsmodels, REML):**
```python
# H2.1
anxiety ~ S_probe_z + (1 + S_probe_z | subject)

# H2.2
confidence ~ S_probe_z + (1 + S_probe_z | subject)
```
where `_z` denotes z-scoring across all observations. Reported: fixed-effect β, SE, t, p (two-tailed from Satterthwaite df approximation).

**Decision rule:**
- H2.1 supported if β(S_probe_z) < 0 and |t| > 3.0
- H2.2 supported if β(S_probe_z) > 0 and |t| > 3.0

**Secondary analysis (parameter moderation, between-subjects OLS):**
```python
mean_anxiety    ~ k_z + β_z    # OLS, N = confirmatory final sample
mean_confidence ~ k_z + β_z
```
Parameters k_i and β_i are posterior means from the M5 SVI fit. Directional test: β(k_z → mean_anxiety) > 0 and β(k_z → mean_confidence) < 0. p < 0.05 one-tailed.

---

### 3.5 Analysis Plan for H4 — Choice-Vigor Dissociation

**Behavioral measures:**
- **Choice:** P(choose high-effort option) per subject — proportion of regular trials (non-probe) where the high-reward, high-effort cookie is chosen.
- **Vigor:** α_bayes — subject-level posterior mean tonic vigor from the HBM (Section 3.3).

All subject-level measures are standardized (z-scored) before regression analyses.

**H4.1 — Population-level independence**
```python
r, p = pearsonr(choice_rate, alpha_bayes)   # N = confirmatory final sample
```
Criterion: |r| < 0.10 and p > 0.05 (two-tailed; directional prediction of no correlation).

**H4.2 — k dominates choice**
```python
choice_z ~ k_z + beta_z + alpha_z    # OLS, standardized
```
Criterion: adj.R² > 0.60; k has the largest standardized β; α is non-significant (p > 0.05).

**H4.3 — Threat reverses choice–vigor coupling**
Per-threat-level Pearson correlations: r(choice_z, vigor_z | T = 0.1), r(choice_z, vigor_z | T = 0.9). Uses mean pre-encounter pressing rate per subject per threat level from `phase_vigor_metrics.parquet` (enc_pre_mean_norm column).

Fisher z-test comparing the two correlations. Criterion: r(T=0.1) > 0 AND r(T=0.9) < 0 AND Fisher z-test p < 0.01 (two-tailed).

Cross-level LMM (secondary, confirmatory with replication expected):
```python
vigor_trial ~ choice_subj_z * threat_z + dist_z + (1 | subject)
```
interaction β(choice_subj_z × threat_z) < 0, p < 0.05.

**H4.4 — Vigor predicts escape beyond choice**
```python
escaped ~ vigor_z + choice_z + threat_z + dist_z + (1 | subject)
# logistic mixed-effects, attack trials only
```
Criterion: β(vigor_z) > 0 AND β(choice_z) ≤ 0, both p < 0.001.

---

### 3.6 Quadrant Analysis (Pre-specified Descriptive)

Subjects classified into four quadrants based on median splits of P(choose high) and α_bayes (tonic vigor):
- HH: above median on both (high choice, high vigor)
- HL: above median on choice, below median on vigor
- LH: below median on choice, above median on vigor
- LL: below median on both

Pre-specified expectations (from exploratory findings):
1. k ANOVA across quadrants: F > 20, p < 0.001. k must be the primary choice-axis discriminator.
2. Escape rate ANOVA: HH and LH must escape at substantially higher rates than HL and LL (expected >25 percentage-point difference between H and L vigor groups within the same choice group).
3. Off-diagonal comparison (HL vs. LH): LH must show significantly higher β (threat bias; p < 0.001) and significantly lower k (p < 0.001) than HL.

---

## 4. Exploratory Findings from Sample 1 That Motivated These Hypotheses

All six hypotheses were generated and refined based on the exploratory sample (N=293). The key findings motivating each hypothesis are summarized below (H5 and H6 exploratory evidence is presented in Section 1 of each respective hypothesis).

### Behavioral descriptives (N=293, 13,185 choice trials)

The exploratory sample completed 81 events per participant (45 choice trials + 36 affect probes), yielding 13,185 choice observations and 10,546 probe ratings. Participants chose the high-effort/high-reward option on 43.1% of trials (SD = 20.3%). Both threat and distance strongly reduced high-effort choices: threat effect (T=0.1 vs T=0.9) = 0.484, Cohen's d = 1.63, t(292) = 27.82, p < 10^-83; distance effect (D=1 vs D=3) = 0.336, Cohen's d = 1.57, t(292) = 26.80, p < 10^-80. The 3 x 3 table of P(choose high-effort):

| | D = 1 | D = 2 | D = 3 |
|---|---|---|---|
| **T = 0.1** | 0.808 | 0.692 | 0.565 |
| **T = 0.5** | 0.633 | 0.381 | 0.188 |
| **T = 0.9** | 0.397 | 0.138 | 0.078 |

The overall capture rate was 31.7%, scaling from 11.5% (T=0.1) to 48.7% (T=0.9). Mean anxiety was 4.40 (SD=1.31) and mean confidence was 3.17 (SD=1.35) on the 0-7 scale, both strongly modulated by threat level. Capacity-normalized vigor averaged 0.686 (SD=0.164), with a modest threat-driven decrease.

### H1 — Motivated by choice model comparison results (N=293, NB03-unified)

A systematic comparison of 12 model variants using stochastic variational inference was conducted on the exploratory sample. The winning model (L4a_add) was selected based on ELBO and BIC:

| Model | ELBO | BIC |
|-------|------|-----|
| L4a_add (WINNER) | −6259.7 | 18,135.6 |
| L3_add | −6275.4 | 18,167.1 |
| L4a_hyp (multiplicative) | −6418 | — |
| L3_survival (exponential) | −6449 | — |
| L3b_surv_zi (per-subj z) | −6561 | — |
| L0_effort (effort only) | −8298 | — |

Key comparisons:
- **Additive > multiplicative effort:** ΔELBO = +158 (L4a_add vs. L4a_hyp). Additive formulation (SV = R·S − k·E) resolves a k–β identifiability problem present in the multiplicative form (SV = R·S·exp(−k·E)).
- **Hyperbolic > exponential survival:** ΔELBO = +190 (L4a_add vs. L3_survival). Hyperbolic escape probability (S = (1−T) + T/(1+λD)) indicates that participants perceive escape probability as declining gradually with distance, not as a sharp exponential fall-off.
- **Per-subject z hurts fit:** ΔELBO = −112 (L3b_surv_zi vs. L3_add). No individual differences in the distance-nonlinearity of the survival function are supported; population-level hazard scaling is sufficient.
- **α in survival helps (+15.7 ELBO):** Including tonic vigor α in the survival kernel (D/α term) improves fit marginally, suggesting motor capacity slightly modulates perceived exposure.
- **Parameter identifiability:** k–β correlation = −0.138, k–α = −0.052, β–α = +0.264. Parameters are identifiable and moderately independent.

### H2 — Motivated by affect LMM results (N=293, NB04-03, re-run 2026-03-20)

Mixed-effects models on 10,546 probe ratings (5,274 anxiety, 5,272 confidence) from N=293 subjects:

| DV | Predictor | β | SE | t | p |
|----|-----------|---|----|---|---|
| Anxiety | S_probe_z | −0.605 | 0.024 | −25.63 | <0.001 |
| Anxiety | p_threat_z | +0.575 | 0.024 | +24.45 | <0.001 |
| Anxiety | dist_safety_z | +0.226 | 0.023 | +9.71 | <0.001 |
| Confidence | S_probe_z | +0.612 | 0.024 | +25.65 | <0.001 |
| Confidence | p_threat_z | −0.586 | 0.024 | −23.99 | <0.001 |
| Confidence | dist_safety_z | −0.283 | 0.025 | −11.46 | <0.001 |

Between-subjects OLS on mean affect (N=293): k → mean anxiety (β = +0.127, p = 0.032); k → mean confidence (β = −0.154, p = 0.009). People who discount effort more (high k) show chronically higher anxiety and lower confidence.

Metacognitive calibration (per-subject Spearman r): anxiety × S_probe M(r) = −0.341, t = −16.45, p = 2.5×10⁻⁴³; confidence × S_probe M(r) = +0.340, t = +15.48, p = 8.4×10⁻⁴⁰.

### H3 — Motivated by vigor HBM results (N=293, NB16-vigor, re-run 2026-03-20)

Hierarchical Bayesian model on N=293 subjects, 23,554 trials:
- Population mean tonic vigor: μ_α = 0.315 (SB = 0.964, shrinkage = 89%, max Rhat = 1.008)
- Population mean attack mobilization: **μ_ρ = 0.067, P(ρ_i > 0) = 1.0 for all 293 subjects**
- α–ρ correlation: r = +0.016, p = 0.78 (tonic pressing and attack mobilization are independent)
- 0 divergences across all 4 chains

In the count-based (capacity-normalized, choice-binary-adjusted) confirmatory analysis, the terminal vigor threat effect reached F(2, 290) = 35.2, p = 2×10⁻¹⁵, with high-threat trials showing 104% of group mean and low-threat showing 96%. Attack trials showed a clean post-encounter increase: diff = +0.033, t = 6.95, p = 2×10⁻¹¹.

### H4 — Motivated by choice-vigor dissociation results (N=293, NB14/15, 2026-03-20)

**Core independence:** Pearson r(P(choose high), α_bayes) = +0.008, p = 0.894. Choice and tonic vigor are near-perfectly independent.

**k dominates choice:** Choice ~ k + β + α regression: adj.R² = 0.823, k β = −0.685 (dominant); α non-significant.

**Threat reverses coupling:** Per-threat correlations r(choice, vigor):
- T = 0.1: r = +0.196, p = 0.001
- T = 0.5: r = +0.013, n.s.
- T = 0.9: r = −0.219, p < 0.001
- Fisher z-test: z = 5.07, p < 0.0001

**Escape prediction (attack trials, N = 10,257):** vigor β = +0.091, p = 10⁻⁷⁷; choice β = −0.177, p ≈ 0.

**Mechanism (bootstrap, 10K iterations):**
- β → choice: −0.409 [−0.570, −0.310]
- β → vigor: +0.147 [+0.019, +0.271]
- Difference: −0.555 [−0.802, −0.380], p = 0.0000
β suppresses choice but boosts vigor — these are opposite-signed effects, formally distinguishable by bootstrap.

---

## 5. Exclusion Criteria

Exclusion criteria are identical to those applied to the exploratory sample (Stage 4 of the preprocessing pipeline). All criteria are applied before any model fitting or statistical analysis.

### Criterion 1 — Task completion
Participants must have completed all 81 trials (3 blocks × 27 trials). Participants who quit mid-task or experienced a session timeout are excluded.

### Criterion 2 — Valid effort calibration
Participants must have achieved a minimum of 10 presses during the pre-task calibration (any of the three 10-second trials). Below-threshold performance is treated as device incompatibility or failure to follow instructions.

### Criterion 3 — Valid keypress behavior
Participants with implausible keypress patterns are excluded. Specifically: (a) maximum single-trial press rate > 3 standard deviations above the sample mean (suggestive of automated input); (b) zero presses on more than 50% of regular (non-probe) trials (suggestive of task disengagement or input device failure).

### Criterion 4 — Valid predator dynamics
Trials with physically impossible predator behavior (e.g., teleportation, negative encounterTime) are flagged at the trial level. Participants with more than 10% flagged trials are excluded.

### Criterion 5 — Adequate task engagement
Participants must have an overall escape rate of at least 35% across all attack trials. Below this threshold is indicative of systematic task disengagement rather than genuine performance.

### Criterion 6 — Probe trial validity
Participants must have provided ratings on at least 80% of probe trials (≥29 of 36 total probes). Participants with excessive missing probe data are excluded from the affect analyses (H2) but retained for all other analyses (H1, H3, H4).

### Reporting
The total number of participants excluded at each criterion stage will be reported in the paper alongside a CONSORT-style flow diagram. No exclusions will be made post-hoc based on model fit quality or statistical extremity of estimates.

---

## 6. Sample Size Justification

The confirmatory sample size of N = 350 was determined prior to data collection based on three considerations:

**1. Matching the planned design.** Both samples were designed to be N = 350. The exploratory sample yielded N = 293 after exclusions (83.7% retention). The confirmatory sample is expected to yield 280–330 participants after comparable attrition, maintaining a similar final N to the exploratory sample.

**2. Power for the weakest preregistered test.** The weakest powered test among the four hypotheses is the threat-reversal of choice–vigor coupling (H4.3). In the exploratory sample, the Fisher z-test comparing r(T=0.1) = +0.196 to r(T=0.9) = −0.219 had z = 5.07. A G*Power sensitivity analysis for a two-group correlation comparison at α = 0.01, power = 0.80 requires N ≈ 64 per group; the per-threat-level N in the confirmatory sample will be approximately 280–330 subjects with full data, far exceeding this threshold.

**3. Replication of very large effects.** Hypotheses H1, H2, and H3 are based on very large exploratory effects (ΔELBO > 100 for model comparisons; t > 25 for affect LMMs; P(ρ > 0) = 1.0 for vigor mobilization). These effects are powered to replicate at any practically achievable sample size given the design. The sample size is not limiting for these tests.

**Power simulation for the affect LMM (H2):** Assuming a 50% reduction in effect size relative to exploratory (t ≈ 12.8 for S_probe on anxiety; 6,300 observations per affect type), two-tailed α = 0.025 (Bonferroni-corrected for two primary tests), simulated power > 0.999.

---

## 7. What Is Not Preregistered (Planned Exploratory Analyses)

The following analyses will be conducted on the confirmatory sample but are **not** part of the preregistered confirmatory tests. They will be clearly labeled as exploratory when reported.

**7.1 Psychiatric battery associations**

Associations between the three psychiatric factor scores (general distress, fatigue, apathy — derived from EFA of the psychiatric battery) and the choice/vigor parameters (k, β, α, ρ) will be re-examined. In the exploratory sample, the only surviving association after FDR correction was α → apathy factor (R² = 0.123, p = 3×10⁻⁹). Whether this replicates in the confirmatory sample is scientifically important but is not formally preregistered because individual psychiatric associations were largely null in the exploratory sample.

**7.2 Metacognitive calibration**

Per-subject correlations between probe ratings and model-derived S_probe, and associations between k and affective calibration accuracy to threat and distance, will be re-examined exploratorily. Exploratory findings: k × anxiety threat calibration r = −0.309; z × anxiety distance calibration r = +0.152.

**7.3 Encounter-window vigor features**

The PLS analysis relating encounter-window pressing features (dist_pre, dist_trans, tonic_pre, tonic_trans, threat_trans; partial regression slopes per subject) to choice model parameters will be re-run. Exploratory: CV R² = 0.093 (2 components), k CV R² = 0.199. This is exploratory because the PLS structure is unconstrained (component number, loadings).

**7.4 Parameter recovery simulation**

Parameter recovery for the M5 model (simulate data from known parameters → recover via SVI → compare) will be reported to validate model identifiability. Not testable as a confirmatory hypothesis but essential methodological evidence.

**7.5 Two-way (choice × vigor) ANOVA on psychiatric outcomes**

Exploratory analysis of quadrant differences in psychiatric symptoms, using the four-quadrant (HH/HL/LH/LL) grouping. In the exploratory sample, PHQ-9 showed a quadrant effect surviving FDR (p = 0.043); AMI apathy showed the expected LH > HL difference (t = −3.75, p < 0.001). These will be re-examined descriptively.

**7.6 CCA and bootstrap validation of the dissociation**

Canonical correlation analysis on {k, β, z} → {choice, vigor} and the bootstrap test of β's differential effect on choice vs. vigor will be re-run on the confirmatory sample. These tests are confirmatory in spirit but are not preregistered because they were developed iteratively during exploratory analysis.

**7.7 Apathy paradox**

The finding that high-vigor participants (high α) report higher apathy (AMI) despite better performance will be re-examined in the confirmatory sample as an exploratory replication.

---

## 8. Inference Approach

Mixed-effects models are estimated using Python `statsmodels` (version ≥ 0.14) with REML for variance components. Model comparison uses ΔELBO from NumPyro SVI with the Adam optimizer. Bayesian hierarchical models use NumPyro NUTS with convergence monitored via Rhat and effective sample size. SVI models use 30,000 steps with learning rate 0.01 and the default AutoLowRankMultivariateNormal guide.

FDR correction (Benjamini–Hochberg) is applied within each family of tests. Families are defined per hypothesis: within H1 (four model comparison criteria), within H2 (two primary LMMs, then secondaries separately), within H4 (four sub-tests). H3 uses a Bayesian inference criterion (posterior interval) rather than frequentist FDR.

Effect sizes reported alongside all frequentist tests: Cohen's d for between-subjects comparisons; partial η² for ANOVAs; marginal and conditional R² for LMMs.

---

## 9. Deviations Protocol

If any preregistered analysis cannot be executed as specified (e.g., model non-convergence, insufficient attack trials after exclusions, software version incompatibility), the deviation will be:
1. Reported transparently in the paper.
2. Accompanied by the closest feasible alternative analysis (e.g., MCMC instead of SVI if SVI fails to converge).
3. Distinguished from preregistered results in all reporting.

Any analysis that deviates from the preregistered plan will be labeled as an unplanned exploratory analysis, even if it yields the same substantive conclusion as the preregistered test.

---

## 10. Sharing and Transparency

- **Preregistration platform:** AsPredicted (DOI to be added at submission)
- **Data:** Preprocessed confirmatory data will be shared on OSF upon paper acceptance, with Prolific participant IDs removed.
- **Code:** Analysis scripts and notebooks will be shared on GitHub under an open-source license.
- **Model fits:** Posterior samples from the winning choice model and vigor HBM will be shared as compressed NumPyro NetCDF files.

---

## Appendix A — Model Equations (Complete Specification)

### Winning choice model (M5)

**Survival function (option-specific):**
```
S_o(T, D_o) = (1 − T) + T / (1 + λ · D_o)
```
- T ∈ {0.1, 0.5, 0.9}: trial-level threat probability
- D_o ∈ {1, 2, 3}: option-specific distance level (D_L = 1 always; D_H varies)
- λ: population-level hazard scaling (estimated from data; exploratory ≈ 14.0)

**Subjective value (additive effort):**
```
SV_o = R_o · S_o − k_i · E_o − β_i · (1 − S_o)
```
- R_H = 5, R_L = 1 (fixed)
- E_H ∈ {0.6, 0.8, 1.0}; E_L = 0.4 (fixed)
- k_i > 0: effort discounting (log-normal prior)
- β_i > 0: threat bias / subjective capture cost (log-normal prior)

**Choice rule:**
```
p(choose H) = σ(τ · (SV_H − SV_L))
```
- τ > 0: population-level inverse temperature (log-normal prior)

**Hierarchical priors:**
```
log k_i ~ Normal(μ_k, σ_k);   μ_k ~ Normal(0, 1);  σ_k ~ HalfNormal(1)
log β_i ~ Normal(μ_β, σ_β);   μ_β ~ Normal(0, 1);  σ_β ~ HalfNormal(1)
λ ~ LogNormal(0, 1)
τ ~ LogNormal(0, 1)
```

### Vigor HBM (excess effort model)

**Trial-level excess effort:**
```
excess_ij = α_i + δ_i · (1 − S_ij) + ε_ij
ε_ij ~ Normal(0, σ)
```
- S_ij = (1−T) + T/(1+λ·D_chosen): survival for chosen option, λ from choice model
- α_i: baseline excess effort (hierarchical Normal prior)
- δ_i: danger-responsive mobilization (hierarchical Normal prior)
- σ: residual SD (HalfNormal prior)

**Hierarchical priors:**
```
α_i ~ Normal(μ_α, σ_α);  μ_α ~ Normal(0, 1);  σ_α ~ HalfNormal(1)
δ_i ~ Normal(μ_δ, σ_δ);  μ_δ ~ Normal(0, 1);  σ_δ ~ HalfNormal(1)
σ ~ HalfNormal(1)
```

### Joint model (robustness check for H5)

```
[log(k_i), log(β_i), α_i, δ_i] ~ MVN(μ, Σ)
Σ = diag(σ) · Ω · diag(σ),  Ω ~ LKJCholesky(η = 2)
```

Choice and vigor likelihoods as above, with λ fixed from the choice-only fit.

---

## Appendix B — Variable Glossary

| Symbol | Meaning | Source | Notes |
|--------|---------|--------|-------|
| T | Threat probability | `behavior.csv`: `attackingProb` | ∈ {0.1, 0.5, 0.9} |
| D | Distance level | `behavior.csv`: `distance_H` / `distance_L` | H ∈ {1, 2, 3}; L = 1 always |
| E | Effort fraction | `behavior.csv`: `effort_H` / `effort_L` | H ∈ {0.6, 0.8, 1.0}; L = 0.4 |
| R_H, R_L | Reward | Fixed | 5, 1 points |
| C | Capture penalty | Fixed | 5 points |
| λ | Hazard scaling | Estimated from choice model | Population-level (exploratory ≈ 14.0) |
| τ | Inverse temperature | Estimated from choice model | Population-level |
| k_i | Effort discounting | Choice model posterior mean | Per-subject |
| β_i | Threat bias | Choice model posterior mean | Per-subject |
| α_i | Baseline excess effort | Vigor model posterior mean | Per-subject |
| δ_i | Danger mobilization | Vigor model posterior mean | Per-subject |
| S | Survival probability | (1−T)+T/(1+λD) | Option-specific |
| excess | Excess effort | vigor_norm − effort_chosen | Trial-level |
| vigor_norm | Capacity-normalized pressing rate | `smoothed_vigor_ts.parquet` | rate / f_max_i per subject |
| f_max_i | Calibration maximum | Preprocessing | Max presses across calibration trials |

---

*This preregistration was prepared based on exploratory analyses completed on 2026-03-22 and will be submitted to AsPredicted before the confirmatory data are opened for any analysis. All findings reported in Section 4 were derived exclusively from the exploratory sample (N=293).*
