---
name: Optimal Control Reformulation
description: Plan to reframe the paper as a stochastic optimal control problem — single cost function (c_effort, c_death) jointly determines choice and vigor
type: project
---

# Optimal Control Reformulation of Foraging Under Threat

## Context

The current paper uses descriptive computational models — a choice model (SV = R·S - k·E - β·(1-S)) and a separate vigor model (excess = α + δ·(1-S)) — then correlates their parameters post-hoc to argue that choice and vigor share a common survival computation. This approach is correlational, not mechanistic.

The reformulation treats the task as a **stochastic optimal control problem** where a single cost function (effort cost + death cost) jointly determines both which cookie to choose and how hard to press. Individual differences in cost parameters (c_effort, c_death) mechanistically generate both choice patterns and vigor patterns. The β-δ coupling becomes a structural prediction, not an observed correlation.

**Why:** The current approach fits choice and vigor with separate models, then correlates parameters post-hoc (β↔δ). The optimal control approach derives both from a single cost-minimization principle, making the coupling structural rather than correlational.

**How to apply:** All new modeling work should follow the OC framework. The descriptive models (L3_add, L4a_add, vigor HBM, joint LKJ) are superseded for the paper, though useful as comparison benchmarks.

This shifts the paper from computational psychiatry to computational neuroscience: "A single cost-minimization principle governs foraging behavior under threat."

---

## Phase 1: Recover Strike Time Distribution from Data

**Goal:** Determine sigma of the Gaussian strike time distribution N(2·encounterTime, σ²).

**Steps:**
1. Extract all `attackingTime` values from raw JSON files for attack trials
2. Group by distance level (encounterTime = 2.5s, 3.5s, 5.0s for D=5,7,9)
3. Compute `attackingTime - 2·encounterTime` residuals → fit σ
4. Verify Gaussianity (Q-Q plot, Shapiro-Wilk)

**Known facts:**
- encounterTime is deterministic per distance: D=5→2.5s, D=7→3.5s, D=9→5.0s (from `scripts/preprocessing/stage2_trial_processing.py` lines 201-207)
- predatorAttackTime = 2 × encounterTime (the mean of the strike distribution)
- σ is NOT documented anywhere in the repo — must be estimated from data
- Preliminary estimate from one participant: σ ≈ 1.0–2.0 seconds

**Files:**
- Read: `data/exploratory_350/raw/participant_*.json` (attackingTime field)
- Read: `scripts/preprocessing/stage2_trial_processing.py` (encounterTime calculation, lines 201-207)
- Write: `notebooks/06_optimal_control/01_strike_time_distribution.ipynb`
- Output: σ estimate + validation plots

---

## Phase 2: Build the Optimal Control Solver

**Goal:** For any (c_effort, c_death, T, D, E, R, σ), compute the optimal pressing policy and expected utility.

### The Control Problem

**State:** x(t) = distance from safety, x(0) = D ∈ {5, 7, 9} game units
**Control:** u(t) = press rate ∈ [0, 1] (fraction of calibrationMax)
**Speed tiers:** v(u) = {full, half, quarter, zero} depending on u relative to required rate:
- u ≥ 100% of required rate → full speed (v_max)
- u ≥ 50% → half speed (v_max/2)
- u ≥ 25% → quarter speed (v_max/4)
- u < 25% → zero speed (movement decays to stop)

**Predator dynamics:**
- Appears at t_enc (deterministic per distance)
- Strikes at t_strike ~ N(2·t_enc, σ²)
- Post-strike: moves at 4× player max speed → effectively instant capture unless player is at safety

**Survival condition:** Reach x = 0 before t_strike. So:
```
P(survive | u(.)) = (1 − T) + T · Φ((2·t_enc − t_arr) / σ)
```
where t_arr = time to reach safety under pressing policy u(.), and Φ is the standard normal CDF.

**Payoffs:**
- Survive: +R
- Captured: −(R + C) where C = 5 [lose the reward AND pay 5 penalty]

**Objective — maximize:**
```
EU(u) = P(survive|u) · R  −  (1 − P(survive|u)) · (R + C)  −  c_effort · ∫g(u(t))dt
```
where g(u) = u² (quadratic effort cost, standard in motor control — Todorov & Jordan 2002, Shadmehr & Krakauer 2008).

Rewrite with c_death scaling the capture term:
```
EU(u) = P_surv · R  −  (1 − P_surv) · c_death · (R + C)  −  c_effort · ∫u(t)² dt
```

### Why the Optimal Choice is Non-Trivial

At the crossover P(survive) = 0.5, the EV calculations show:
```
EV(H) = p × 5 − (1−p) × 10 = 15p − 10
EV(L) = p × 1 − (1−p) × 6  = 7p − 6

H dominates when p > 0.5; L dominates when p < 0.5
```
The encounterTime calibration places P(survive|attack) ≈ 0.5 at full speed, so the crossover is right in the game's operating range. Subjects pressing below max (due to effort cost) will have P(survive) < 0.5 on some trials, making L objectively better. Individual differences in c_effort shift this crossover.

### Tier-Selection Approximation (Phase 2a — primary)

Because speed is a step function with 4 levels, the optimal constant-rate policy is to press at exactly the tier threshold (excess pressing within a tier wastes effort). Evaluate EU at 4 tiers:

For tier j with threshold u_j and speed v_j:
- Arrival time: t_arr = D / v_j (or ∞ if v_j = 0)
- Survival: P_surv = (1−T) + T · Φ((2·t_enc − t_arr) / σ)
- Effort cost: c_effort · u_j² · t_arr
- EU_j = P_surv · R − (1−P_surv) · c_death · (R+C) − c_effort · u_j² · t_arr

Optimal tier: j* = argmax EU_j. Optimal choice: compare EU*(H) vs EU*(L).

This is analytically tractable, differentiable (via Φ), and trivially fast.

**Note on u_req:** The required press rate depends on cookie weight. Heavy cookie = 100% of calibrationMax, light = 40%. So the tier thresholds in absolute terms are:
- Heavy: {25%, 50%, 100%} of calibrationMax
- Light: {10%, 20%, 40%} of calibrationMax
The light cookie is always easier to reach full speed on.

### Dynamic Programming Extension (Phase 2b — if needed)

Allow tier-switching over time (coast early, sprint when predator appears). Discretize (x, t) grid (~50 position × ~100 time bins × 4 actions = 20k states), backward induction. Only pursue if constant-rate can't explain vigor timeseries.

**Files:**
- Write: `scripts/modeling/optimal_control.py` — core EU computation in JAX
  - `compute_tier_eu(D, T, R, C, u_req, t_enc, sigma, c_effort, c_death)` → (EU, u_star, P_surv) per tier
  - `optimal_choice(params_H, params_L)` → choice probability
  - All vectorized for batch trial evaluation
  - All operations differentiable via `jax.scipy.stats.norm.cdf`
- Write: `notebooks/06_optimal_control/02_oc_solver_validation.ipynb` — verify limit cases

---

## Phase 3: Hierarchical Bayesian Model

**Goal:** Fit c_effort and c_death per subject, jointly explaining choice and vigor.

### Model Specification

```
# Population priors
mu_ce ~ Normal(0, 1)           # log(c_effort) mean
mu_cd ~ Normal(0, 1)           # log(c_death) mean
sigma_ce ~ HalfNormal(0.5)
sigma_cd ~ HalfNormal(0.5)
tau ~ HalfNormal(1)            # choice temperature

# Subject parameters (non-centered parameterization)
log(c_effort_i) ~ Normal(mu_ce, sigma_ce)
log(c_death_i)  ~ Normal(mu_cd, sigma_cd)

# Per trial: compute optimal tier for H and L
EU*_H, u*_H = optimal_tier(D_H, T, R_H, C, E_H, t_enc, sigma, c_effort_i, c_death_i)
EU*_L, u*_L = optimal_tier(D_L, T, R_L, C, E_L, t_enc, sigma, c_effort_i, c_death_i)

# Choice likelihood
choice ~ Bernoulli(sigmoid(tau · (EU*_H − EU*_L)))

# Vigor likelihood (for chosen option)
observed_vigor ~ Normal(u*_chosen, sigma_vigor)
```

Key: **same c_effort and c_death generate both choice probabilities and vigor predictions**. No separate vigor parameters needed for the structural prediction.

### Parameters and Mapping

| OC Parameter | Role | Current Analog |
|---|---|---|
| c_effort | Subjective effort cost per unit pressing | κ (choice) + α (tonic vigor) |
| c_death | Subjective capture aversion | β (choice) + δ (phasic vigor) |
| σ_strike | Strike time uncertainty (population or fixed) | z / λ (hazard sensitivity) |
| τ | Choice temperature | τ |

2-3 subject-level params replace 4 (k, β, α, δ). The survival function S emerges mechanistically from Φ((2·t_enc − t_arr)/σ) rather than being an assumed functional form.

### Fitting Procedure

**Computational cost:** The tier-selection model requires evaluating 4 EU values × 2 options = 8 Φ() calls per trial. With 45 trials per subject and 293 subjects, that's ~105k Φ evaluations per MCMC step — trivially fast.

**Inference strategy:**
1. SVI first (AutoMultivariateNormal guide, 30k steps, Adam lr=0.002) for rapid iteration
2. MCMC for final fit (NUTS, 4 chains × 1000 warmup + 1000 samples, target_accept=0.95)
3. Same two-stage approach as current: choice-only first, then joint choice+vigor

**σ_strike handling options:**
- Option A: Fix at empirical estimate from Phase 1
- Option B: Fit at population level only (single σ for all subjects)
- Option C: Fit per subject (but risk identifiability issues with c_death)
- Recommendation: Start with Option A, try Option B if fit is poor

**Files:**
- Write: `scripts/modeling/oc_model.py` — NumPyro model definition
- Write: `scripts/modeling/oc_fitter.py` — fitting wrapper (parallel SVI + MCMC)
- Follow architecture of existing `base_model.py` / `fitter.py`

---

## Phase 4: Model Comparison

**Goal:** Compare OC model against current descriptive model on the same data.

1. **Fit quality:** WAIC/ELBO comparison (choice likelihood)
2. **Vigor prediction without free parameters:** Does u* from the OC model predict observed vigor? Compare against the current vigor model (which HAS free parameters α, δ)
3. **Survival function comparison:** Plot the mechanistic S = (1−T) + T·Φ(...) against the fitted S = (1−T) + T/(1+λD) — do they agree?
4. **Parameter recovery:** Simulate from the OC model, recover c_effort and c_death
5. **Escape rate prediction:** The OC model predicts P(survive) mechanistically — compare against observed escape rates by (T, D) condition

**Files:**
- Write: `notebooks/06_optimal_control/03_fit_and_compare.ipynb`
- Write: `notebooks/06_optimal_control/04_parameter_recovery.ipynb`

---

## Phase 5: Clinical and Affect Predictions

**Goal:** Show that cost parameters predict clinical outcomes and subjective affect.

1. **Clinical correlates:** Regress c_effort and c_death on psychiatric measures (DASS, PHQ-9, OASIS, STAI-T, AMI, MFIS, STICSA). Prediction: c_effort → fatigue/apathy; c_death → anxiety measures.
2. **Affect prediction:** Anxiety should track 1 − P_surv from the OC model. No additional free parameters needed — the control model predicts affect as a byproduct of the survival computation.
3. **Individual difference space:** Visualize subjects in (c_effort, c_death) space colored by clinical scores.

**Files:**
- Write: `notebooks/06_optimal_control/05_clinical_and_affect.ipynb`

---

## Phase 6: Paper Restructure

### New Results Structure

**R1: The Foraging Task as a Stochastic Optimal Control Problem.**
Define the control problem. Show that the tier-selection solution produces a mechanistic survival function. The derived S ≈ the best-fitting descriptive S (validating the functional form). Compare model fit against descriptive model.

**R2: A Single Cost Function Governs Choice and Vigor.**
c_effort and c_death jointly predict cookie selection AND pressing intensity. Key figure: predicted vs observed vigor with NO free vigor parameters. The choice-vigor coupling is structural, not correlational.

**R3: Individual Differences in the Effort-Survival Tradeoff.**
2D cost space (c_effort, c_death). High c_effort: avoid effortful cookies, press at lower tiers. High c_death: avoid risky cookies, press harder when committed. Clinical measures map onto this space (fatigue/apathy → c_effort axis; anxiety → c_death axis).

**R4: Affect as a Byproduct of the Survival Computation.**
Anxiety tracks 1 − P_surv from the control model. Higher c_death → steeper anxiety gradients (more calibrated to danger). Links motor policy to subjective experience through the common cost function.

---

## Implementation Order

1. **Strike time σ recovery** — 1 day. Quick data analysis, unblocks everything.
2. **OC solver** — 2-3 days. Core JAX functions + validation.
3. **Choice-only OC model** — 3-4 days. Hierarchical Bayesian, SVI first, then MCMC. Compare against current model.
4. **Joint choice+vigor model** — 2-3 days. Add vigor likelihood. Test structural coupling.
5. **Clinical + affect analysis** — 2 days. Rerun existing analyses with new parameters.
6. **Paper restructure** — ongoing.

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Constant-rate approximation too coarse for vigor timeseries | Add reactive perturbation term: u(t) = u* + η·h(t) where h captures encounter spike and terminal ramp |
| σ_strike trades off with c_death (identifiability) | Fix σ at empirical estimate; or fit population-level only |
| OC model fits worse than descriptive model | The descriptive model has more flexibility (4 free params vs 2-3). Worse fit with fewer params may still be scientifically preferable if predictions are correct |
| Discrete speed tiers create non-smooth EU surface | Compute EU at all 4 tiers independently (each smooth), take argmax. Use soft-max for gradient-based fitting |

---

## Verification

1. **Solver correctness:** Limit case tests (c_effort→0, c_death→0, T=0, σ→0)
2. **Model recovery:** Simulate 293 subjects, recover parameters
3. **Choice fit:** WAIC comparable to or better than current model
4. **Vigor prediction:** Correlation between u* and observed vigor WITHOUT free vigor parameters
5. **Escape rates:** Predicted P(survive) matches observed escape rates by condition
6. **Clinical:** c_effort and c_death predict psychiatric measures in expected directions

---

## Data Available for This Work

- **293 subjects**, 45 choice trials each (13,185 total behavioral trials)
- **20Hz vigor timeseries** (3.9M rows in `smoothed_vigor_ts.parquet`) with `vigor_norm` per trial
- **Per-trial timing:** encounterTime, trialEscapeTime, trialCaptureTime, isAttackTrial, trialEndTime
- **Per-subject:** calibrationMax (max press rate from calibration)
- **Raw JSON** with press-by-press timestamps (`effortRate` arrays) in `data/exploratory_350/raw/`
- **Probe ratings:** 36 per subject (anxiety + confidence) in `feelings.csv`
- **Psychiatric measures:** DASS-21, PHQ-9, OASIS, STAI-T, AMI, MFIS, STICSA in `psych.csv`

## Current Model Infrastructure (for reference/comparison)

- **Choice:** `scripts/modeling/base_model.py`, `models.py` — hierarchical Bayesian via NumPyro MCMC
- **Vigor HBM:** `scripts/run_vigor_hbm.py` — α (tonic) + ρ (phasic attack boost)
- **Joint:** `scripts/run_joint_correlated.py` — LKJ-correlated [k, β, α, δ] ~ MVN
- **Fitting:** `scripts/modeling/fitter.py` — MCMC wrapper with parallel GPU support
- **Winning descriptive model:** L4a_add: SV = R·S − k·E − β·(1−S), S = (1−T) + T/(1+λ·D/α)
