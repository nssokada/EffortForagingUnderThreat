# Threat-Effort Foraging Task: Modeling Analysis Plan v2

*Revised to use JAX/NumPyro (SVI for development, MCMC for final model), incorporate lessons from prior modeling sessions, and address identified confounds.*

---

## PART 1: CHOICE MODEL FITTING

**Step 1.** Load free choice trials only. Exclude subjects 154, 197, 208 (calibration outliers, see DATA_OVERVIEW.md). Confirm N=290 participants, 45 trials each.

**Step 2.** Define trial-level variables for each trial:
- `R_heavy` = 5, `R_light` = 1
- `C` = 5 (capture penalty, same for both cookies)
- `req_heavy` = effort demand for heavy cookie: 0.6, 0.8, or 1.0
- `req_light` = 0.4 (always)
- `p` = predator probability (0.1, 0.5, 0.9)
- `T_heavy` = exposure duration for heavy cookie (5, 7, 9 seconds for d=1, 2, 3)
- `T_light` = 5 always (light cookie always at d=1)
- `choice` = 1 if heavy selected, 0 if light

**Step 3.** Implement eight value functions. For each model compute ΔV per trial:

| Model | ΔV Formula | Free Parameters | What it tests |
|-------|-----------|-----------------|---------------|
| M1 | `4 − λ·(req_H² · D_H − req_L² · 1)` | λ, β | Pure effort discounting (no threat) |
| M2 | `4 − λ·effort(D) − γ·p` | λ, γ, β | Additive effort + linear threat (≈ our 3-param v2) |
| M3 | `5·exp(−p·T_H) − exp(−p·T_L) − λ·effort(D)` | λ, β | Objective survival × reward, effort separate |
| M4 | `[exp(−p·T_H)·5 − req_H·λ]/T_H − [exp(−p·T_L)·1 − req_L·λ]/T_L` | λ, β | Rate-of-return with survival |
| M5 | `5·exp(−κ·p^α·T_H) − exp(−κ·p^α·T_L) − λ·effort(D)` | λ, κ, α, β | Distorted survival + effort (key test) |
| M6 | `[exp(−κ·p^α·T_H)·5 − req_H·λ]/T_H − [exp(−κ·p^α·T_L)·1 − req_L·λ]/T_L` | λ, κ, α, β | Rate-of-return with distorted survival |
| M7 | `S_H·R_H − (1−S_H)·ψ·(R_H+C) − λ·effort(D) − [S_L·R_L − (1−S_L)·ψ·(R_L+C)]` where `S = exp(−κ·p^α·T)` | λ, κ, α, ψ, β | Distorted survival + per-subject penalty weighting |
| M8 | `4 − λ·effort(D) − γ·p − δ·p·D_H` | λ, γ, δ, β | Additive with threat × distance interaction |

Where:
- `effort(D) = req_H² · D_H − req_L² · 1` (LQR-inspired commitment cost differential, as in our 3-param v2)
- `T_H`, `T_L` are exposure durations in seconds
- `D_H` = distance level (1, 2, 3)

**Rationale for changes from v1:**
- **M1-M2 now use distance-scaled effort** (`req²·D` instead of fixed `0.5`). This addresses the confound: distance affects both effort and survival. Without distance in the effort term, models can't separate "avoid because far" from "avoid because dangerous."
- **M2 is our current 3-param v2 choice equation** (with γ here = β there, λ here = k there). This serves as the benchmark.
- **M7 adds a per-subject penalty-weighting parameter ψ** (= cd in the choice equation). This tests whether individual differences in capture aversion affect choice when survival depends on exposure duration. Our Phase 2 analysis showed cd's leverage in choice was collinear with T at population-fixed frac_full. M7 tests whether per-subject ψ with per-subject survival still has identifiable leverage.
- **M8 tests whether threat × distance interaction in choice requires a survival function** or can be captured by a simple linear interaction term.

**Step 4.** For each model define the choice likelihood:
```
P(choice=1) = σ(β_i · ΔV_i)
```
where σ is the logistic function and β is per-subject inverse temperature.

**Step 5.** Fit all eight models using NumPyro SVI (development) with hierarchical Bayesian estimation.

For every free parameter θ use non-centered parameterization:
- Population level: `μ_θ ~ Normal(prior_mean, prior_sd)`, `σ_θ ~ HalfNormal(0.5)`
- Individual level: `θ_i = transform(μ_θ + σ_θ · z_i)` where `z_i ~ Normal(0, 1)`
- Parameters constrained positive via exp transform (log-normal)
- Priors:
  - `μ_λ ~ Normal(0, 1)` (in log space; median ≈ 1.0)
  - `μ_β ~ Normal(0.5, 1)` (in log space; median ≈ 1.6)
  - `μ_γ ~ Normal(0, 1)` (in log space)
  - `μ_κ ~ Normal(-1, 1)` (in log space; median ≈ 0.37)
  - `μ_α ~ Normal(0, 0.5)` (in log space; median ≈ 1.0, weakly informative around no distortion)
  - `μ_ψ ~ Normal(0, 1)` (in log space)
  - `μ_δ ~ Normal(0, 1)` (in log space)

SVI settings:
- Guide: AutoNormal (mean-field)
- Optimizer: ClippedAdam (lr=0.001, clip_norm=10.0)
- Steps: 40,000 with early stopping (save best-loss parameters)
- Random seed: 42

**Step 6.** Convergence check for SVI:
- Verify loss stabilized (not diverging at best checkpoint)
- Check parameter estimates are finite and in reasonable ranges
- For the winning model: validate with NumPyro MCMC (NUTS, 4 chains × 1000 warmup + 1000 samples) as final step

> ⚠️ **STOP CONDITION:** If any model produces NaN loss or fails to converge, try reducing lr to 0.0005. If still failing, flag and exclude from comparison.

**Step 7.** Compute model comparison:
- BIC = 2 × loss + k × log(n) for each model (where loss is best ELBO, k = number of parameters, n = total trials)
- ΔBIC relative to winning model
- Report per-subject choice accuracy and r² for all models
- Return comparison table sorted by BIC

**Step 8.** Posterior predictive check for winning model:
- Generate predicted P(heavy) for each of the 9 cells using posterior mean parameters
- Return predicted versus observed choice proportions as a 3×3 table
- Flag any cell where predicted deviates from observed by more than 0.10

**Step 9.** Extract and return individual-level posterior means for all parameters of the winning model. Return as one row per participant. Save to `results/stats/`.

> **Key comparison:** Does M5/M6 (survival with probability distortion) beat M2 (linear additive, our current model)? If yes, the survival function IS needed in choice and α is the theoretically interesting parameter. If M2 wins, our current 3-param v2 architecture is validated.

---

## PART 2: SEQUENTIAL VIGOR PREDICTION

**Step 10.** Load forced choice (probe) trials only. Apply same subject exclusions as Step 1. Confirm 36 trials per participant.

**Step 11.** For each forced choice trial compute predicted ΔV:
- Use each participant's individual posterior mean parameters from Step 9
- Use the winning model's value function
- Input trial-level values: participant's own parameters and the trial's p, T_heavy, T_light
- **No new fitting** — this is pure out-of-sample prediction
- Return a predicted ΔV value per trial per participant

**Step 12.** Define within-trial epochs using raw keypress timestamps (from `alignedEffortRate`):

| Epoch | Definition | Rationale |
|-------|-----------|-----------|
| Anticipatory | Trial onset → encounterTime | Pre-predator baseline pressing |
| Reactive | encounterTime → encounterTime + 2 seconds | Immediate response to predator |
| Terminal | Last 2 seconds before trial end | Final effort push or collapse |

Use actual `encounterTime` from the data (D=1: 2.5s, D=2: 3.5s, D=3: 5.0s) rather than hardcoded values. For non-attack probe trials, encounterTime is the scheduled (but never triggered) time — use it anyway as the epoch boundary since participants don't know whether it's an attack trial during the anticipatory phase.

**Step 13.** For each trial and each epoch compute:
- Mean pressing rate in presses per second (from raw IPI data)
- Fraction at full speed (presses ≥ required rate)
- Flag trials with < 3 keypresses in any epoch as unreliable

**Step 14. Anticipatory vigor prediction test.**

Run a mixed effects model:
- Outcome: anticipatory pressing rate (z-scored within cookie type to remove the req confound)
- Fixed effects: predicted ΔV (from Step 11), cookie type (heavy=1, light=0), and their interaction
- Random effects: random intercept and random slope for ΔV by participant
- Return fixed effect estimate for ΔV with 95% CI and p-value

> **Primary test:** Is the ΔV coefficient positive and significant? This tests whether the choice model's value computation predicts how hard people press BEFORE the predator appears — a genuine out-of-sample bridge between choice and vigor.

> **Important:** ΔV is a choice-level quantity computed for the trial's conditions. On forced trials where the person didn't choose, the interpretation is: "the brain still computes the expected value of the current situation, and this drives motor preparation." If significant, this supports the EVC framework's claim that value computation governs effort allocation.

**Step 15. Delta-vigor co-regulation test.**

Compute delta-vigor per trial:
```
delta_vigor = mean pressing rate (reactive epoch) − mean pressing rate (anticipatory epoch)
```

Run mixed effects model:
- Outcome: `delta_vigor`
- Fixed effects: `predator_probability + distance + predator_probability × distance`
- Random effects: `(1 + predator_probability | participant)`
- Return full fixed effects table
- Run planned contrast: T=90% vs T=10% delta-vigor collapsed across distance
- Return contrast estimate, 95% CI, Cohen's d

> **Note from prior work:** Our encounter dynamics analysis found that encounter reactivity is threat-INDEPENDENT (F=0.04, p=.96) and trait-stable (cross-block r=0.78). The delta-vigor analysis may confirm this null for threat modulation while showing distance modulation. If delta-vigor IS threat-modulated when computed from raw epochs (vs the smoothed 20Hz timeseries used before), that would be a new finding.

**Step 16. Decoupling threshold test.**

For each participant:
1. Compute the predicted ΔV at each of the three threat levels using their individual parameters
2. Compute the empirical delta-vigor at each threat level
3. Correlate predicted ΔV with empirical delta-vigor across the three threat levels (within-subject)
4. Also compute across participants: does between-subject variation in predicted ΔV predict between-subject variation in mean delta-vigor?

Return:
- Within-subject correlation (mean across participants)
- Between-subject correlation (Pearson r)

**Step 17. Vigor collapse timing test.**

For each forced heavy cookie trial on ATTACK trials only:

1. Compute **predicted abandonment time**: time t within the trial at which EV(t) crosses zero:
```
EV(t) = S(p, T_remaining(t)) · R − λ · C_remaining(t)
```
where:
- `S(p, T_remaining) = exp(-κ · p^α · T_remaining)` (using winning model's survival function)
- `T_remaining(t)` = estimated remaining exposure based on distance and current position
- `C_remaining(t)` = approximate remaining effort cost
- Use participant's own parameters from Step 9

2. Compute **observed abandonment time**: first time pressing rate drops below 25% of participant's calibrated max for more than 1 second. If pressing never drops below threshold, code as trial_duration.

3. Correlate predicted and observed abandonment times across trials.

Return:
- Overall correlation across all trials
- Results split by predator probability level: T=10%, T=50%, T=90%
- Fraction of trials where abandonment actually occurs (from our data: ~12.3% of heavy cookie trials have trialEndState = "free")

> **Primary test:** Is the correlation significantly positive at T=90%? This is the strongest test of the model — if the value function predicts WHEN people give up, not just WHETHER they choose heavy.

---

## PART 3: INDIVIDUAL DIFFERENCES

**Step 18.** Load psychiatric battery (psych.csv). Verify N matches model fitting sample. Z-score all clinical measures within sample.

**Step 19. Primary clinical regressions.**

Using the winning model's per-subject parameters, run OLS regressions predicting each z-scored clinical measure:

If M5/M6 wins (survival with α):
```
clinical ~ α + λ + κ
```
> Primary test: Does α predict STAI-State and OASIS (anxiety measures)? Does λ predict AMI (apathy)?

If M2 wins (additive, our current architecture):
```
clinical ~ λ + γ
```
> This replicates our finding that k (≈λ) and β (≈γ) show near-null clinical associations, with only β→AMI at r=0.14.

For all models, also test:
```
clinical ~ discrepancy + calibration + [model params]
```
where discrepancy and calibration are computed from probe anxiety ratings (as in our 3-param pipeline). This tests whether the affect decomposition adds predictive power over model parameters, replicating our finding that discrepancy→clinical while params→clinical is null.

Return standardized coefficients, 95% CIs, p-values, and R² for each regression.

**Step 20. Trait vigor moderation test.**

Compute each participant's trait vigor as mean calibration pressing rate z-scored within sample.

Run mixed effects model:
- Outcome: `delta_vigor` from Step 15
- Fixed effects: `predator_probability + trait_vigor + predator_probability × trait_vigor + distance`
- Random effects: random intercept by participant
- Return full fixed effects table, focus on interaction term

> **What this tests:** Do people with higher motor capacity show LARGER threat-vigor coupling? If the encounter response scales with baseline motor output, that supports cd as "motor readiness" rather than rational survival optimization.

**Step 21. Score state dependence test.**

Compute running cumulative score at each free choice trial onset per participant.

Run mixed effects logistic regression:
- Outcome: heavy cookie choice (0/1)
- Fixed effects: `predicted_ΔV + current_score + predicted_ΔV × current_score`
- Random effects: random intercept by participant
- Return full fixed effects table, focus on interaction term

> **Note:** This is descriptive — do not refit structural models. Tests whether people adjust strategy based on accumulated outcomes (loss aversion, risk homeostasis).

**Step 22. Anxiety choice surface visualization.**

Split participants into terciles by STAI-State score. For each tercile:
1. Compute mean P(heavy) for each of the 9 cells
2. Compute winning model predicted P(heavy) for each cell using that tercile's mean parameter estimates

Return three 3×3 observed matrices and three 3×3 predicted matrices — one pair per tercile — for figure.

> **What this shows:** Do anxious people show a shifted choice surface (uniformly more cautious), a rotated surface (selectively more cautious at high threat), or no difference? If the winning model has α, does tercile-mean α predict the observed surface shift?

---

## PART 4: CONNECTING CHOICE AND VIGOR MODELS

**Step 23. Bridge to frac_full.**

Using the winning choice model's per-subject parameters:
- Correlate each parameter with per-subject mean frac_full (fraction at full speed)
- If M5/M6 wins: does α predict frac_full? (High α = more threat-distorted → more vigorous pressing?)
- Does λ predict frac_full inversely? (High effort cost → lower frac_full?)

Compare to our known correlations:
- cd → frac_full: r=0.710
- β → frac_full: r=0.249
- k → frac_full: r=0.044

> **This bridges Parts 1 and 2:** If the choice model's parameters also predict vigor (without being fit to vigor), it supports a unified computation across choice and action.

**Step 24. Parameter correspondence test.**

Correlate the winning model's parameters with the 3-param v2 model's parameters:
- λ ↔ k (effort cost)
- γ or α ↔ β (threat sensitivity)
- Model-derived survival at each condition ↔ model-derived ΔV

> **This tests whether the new models recover the same individual differences we already found**, just under a different parameterization.

---

## EXECUTION PLAN

### Phase A: SVI Development (on this machine)
1. Implement all 8 models in NumPyro
2. Fit via SVI (ClippedAdam, 40K steps, early stopping)
3. Model comparison by BIC
4. Run Parts 2-4 with winning model
5. Parameter recovery for winning model (3 × 50 subjects)

### Phase B: MCMC Finalization (GPU or extended run)
6. Refit winning model with NUTS (4 chains × 2000 warmup + 2000 samples)
7. Verify R-hat < 1.01, ESS > 400
8. Confirm SVI-MCMC parameter correlation > 0.99
9. Compute WAIC (requires MCMC samples)
10. Update all downstream analyses with MCMC parameters

### Phase C: Confirmatory (after exploratory is complete)
11. Preregister the winning model and all tests
12. Run on confirmatory sample (N=350)
13. Report discovery and replication side by side

---

## OUTPUT REQUIREMENTS

**Step 25.** Return the following outputs in order:

| # | Output |
|---|--------|
| 1 | Convergence summary for all eight models |
| 2 | Model comparison table: BIC, ΔBIC, choice accuracy, choice r² for all models |
| 3 | Winning model identity clearly labeled |
| 4 | Posterior predictive check: predicted vs observed 3×3 choice proportions |
| 5 | Individual parameter estimates table (N=290 rows) |
| 6 | Fixed effects table from Step 14 anticipatory vigor |
| 7 | Fixed effects table and contrast from Step 15 delta-vigor |
| 8 | Decoupling threshold results from Step 16 |
| 9 | Abandonment timing results from Step 17 |
| 10 | Clinical regression tables from Step 19 |
| 11 | Trait vigor moderation table from Step 20 |
| 12 | Score state dependence table from Step 21 |
| 13 | Three 3×3 anxiety tercile matrices from Step 22 |
| 14 | Parameter-frac_full correlations from Step 23 |
| 15 | Parameter correspondence from Step 24 |

**Step 26.** Flag any step where sample size drops below N=200.

**Step 27.** Save all results to `results/stats/model_comparison_v2/`.

---

## PARAMETER REFERENCE

| Parameter | Model(s) | Interpretation |
|-----------|---------|----------------|
| λ | All | Effort cost — how strongly effort/distance deters choice. ≈ k in 3-param v2 |
| β | All | Inverse temperature — choice stochasticity. ≈ τ in 3-param v2 |
| γ | M2, M8 | Threat sensitivity — linear scaling of threat on value. ≈ β in 3-param v2 |
| κ | M5-M7 | Survival scaling constant — baseline hazard rate |
| α | M5-M7 | Probability distortion — α > 1 means overweighting high threat. ≈ γ^(-1) in 2+2 model |
| ψ | M7 | Capture penalty weighting — individual sensitivity to loss from capture. ≈ cd entering choice |
| δ | M8 | Threat × distance interaction — does threat deter more at far distances? |

## MODEL COMPARISON GUIDE

| If this wins... | Implication | Relation to prior work |
|----------------|-------------|----------------------|
| M1 | Effort explains everything; threat adds nothing | Falsified by data (threat is massive effect) |
| M2 | Additive effort + linear threat | **Validates our 3-param v2** — no survival function needed in choice |
| M3 | Objective survival function needed | Survival computation drives choice, not just linear threat |
| M4 | Rate-of-return logic | People optimize reward per unit time under threat |
| M5 | **Key test:** Distorted survival needed | Probability distortion (like our old γ) is necessary, but WITHIN a survival function |
| M6 | Rate-of-return + distortion | Full integration: reward-rate × distorted survival |
| M7 | Individual penalty weighting in choice | cd CAN enter choice through survival (contradicts our Phase 2 null) |
| M8 | Linear T×D interaction sufficient | The threat × distance interaction doesn't need a survival function — a simple product term works |
