# Threat-Effort Foraging Task: Modeling Analysis Plan

---

## PART 1: CHOICE MODEL FITTING

**Step 1.** Load free choice trials only. Exclude subjects 154, 197, 208. Confirm N=290 participants, 45 trials each.

**Step 2.** Define trial-level variables for each trial:
- `R_heavy` = 5, `R_light` = 1
- `C_heavy` = 0.9, `C_light` = 0.4
- `p` = predator probability (0.1, 0.5, 0.9)
- `T_heavy` = trial duration for heavy cookie (5, 7, 9 seconds for d=1, 2, 3)
- `T_light` = 5 always
- `choice` = 1 if heavy selected, 0 if light

**Step 3.** Implement six value functions. For each model compute ΔV per trial:

| Model | ΔV Formula | Free Parameters |
|-------|-----------|-----------------|
| M1 | `4 − 0.5λ` | λ, β |
| M2 | `4(1 − γp) − 0.5λ` | λ, γ, β |
| M3 | `5·exp(−p·T_heavy) − exp(−p·5) − 0.5λ` | λ, β |
| M4 | `[exp(−p·T_heavy)·5 − 0.9λ]/T_heavy − [exp(−p·5)·1 − 0.4λ]/5` | λ, β |
| M5 | `5·exp(−κ·p^α·T_heavy) − exp(−κ·p^α·5) − 0.5λ` | λ, κ, α, β |
| M6 | `[exp(−κ·p^α·T_heavy)·5 − 0.9λ]/T_heavy − [exp(−κ·p^α·5)·1 − 0.4λ]/5` | λ, κ, α, β |

**Step 4.** For each model define the choice likelihood:
```
P(choice=1) = σ(β · ΔV)
```
where σ is the logistic function.

**Step 5.** Fit all six models in Stan using hierarchical Bayesian estimation. For every free parameter θ use:
- Population level: `μ_θ ~ prior`, `σ_θ ~ HalfNormal(0,1)`
- Individual level: `θ_i ~ Normal(μ_θ, σ_θ)`
- Priors:
  - `μ_λ ~ Normal(1,2)` truncated positive
  - `μ_β ~ Normal(2,2)` truncated positive
  - `μ_γ ~ Normal(0,2)` truncated positive
  - `μ_κ ~ Normal(1,1)` truncated positive
  - `μ_α ~ Normal(1,1)` truncated positive
- Likelihood: `choice_i ~ Bernoulli(σ(β_i · ΔV_i))`

**Step 6.** Run four chains, 2000 warmup, 2000 sampling iterations. Check R-hat < 1.01 and ESS > 400 for all parameters before proceeding. Flag any model that fails convergence.

> ⚠️ **STOP CONDITION:** Do not proceed past Step 6 if any model fails convergence. Report the failure and stop.

**Step 7.** Compute model comparison:
- WAIC at trial level within participants, summed to individual level
- Return mean WAIC relative to winning model per participant as a table
- Compute protected exceedance probability across all six models using Bayesian model selection
- Return winning model identity clearly labeled

**Step 8.** Posterior predictive check for winning model:
- Generate predicted P(heavy) for each of the 9 cells using posterior mean parameters
- Return predicted versus observed choice proportions as a 3×3 table
- Flag any cell where predicted deviates from observed by more than 0.10

**Step 9.** Extract and return individual-level posterior means and 89% HDIs for all parameters of the winning model. Return as one row per participant.

---

## PART 2: SEQUENTIAL VIGOR PREDICTION

**Step 10.** Load forced choice trials only. Apply same subject exclusions as Step 1. Confirm 36 trials per participant, 10,548 trials total.

**Step 11.** For each forced choice trial compute predicted ΔV:
- Use each participant's individual posterior mean parameters from Step 9
- Use the winning model's value function
- Input trial-level values: participant's own λ, α, κ and the trial's p, T_heavy, T_light
- **No new fitting** — this is pure prediction
- Return a predicted ΔV value per trial per participant

**Step 12.** Define within-trial epochs using timestamped IKI data:

| Epoch | Definition |
|-------|-----------|
| Anticipatory | Trial onset → predator spawn (spawn at T_heavy/2: 2.5s, 3.5s, 4.5s for d=1,2,3) |
| Reactive | Spawn time → spawn + 2 seconds |
| Terminal | Final 3 seconds before strike time |

**Step 13.** For each trial and each epoch compute mean pressing rate in presses per second. Flag trials where fewer than 3 keypresses occur in any epoch as unreliable — exclude from that epoch's analysis but retain for others.

**Step 14. Anticipatory vigor prediction test.**

Run a mixed effects model:
- Outcome: anticipatory pressing rate
- Fixed effects: predicted ΔV (from Step 11), forced option (heavy=1, light=0), and their interaction
- Random effects: random intercept and random slope for ΔV by participant
- Run separately for heavy forced trials and light forced trials
- Return fixed effect estimate for ΔV with 95% CI and p-value for each

> **Primary test:** Is the ΔV coefficient positive and significant on heavy forced trials?

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
- Return plot data: mean delta-vigor ± SE by predator probability level for figure

**Step 16. Decoupling threshold test.**

For each participant:
1. Compute the predicted ΔV threshold — the ΔV value below which continued pressing is not expected-value positive (where ΔV = 0 in the winning model given the participant's parameters)
2. Compute the empirical predator probability at which delta-vigor transitions from positive to negative by fitting a linear interpolation through their three delta-vigor values at T=10%, 50%, 90%

Correlate predicted threshold ΔV with empirical transition probability across participants. Return Pearson r and p-value.

**Step 17. Vigor collapse timing test.**

For each forced heavy cookie trial:

1. Compute **predicted abandonment time**: time t within the trial at which EV(t) crosses zero:
```
EV(t) = S(p, T_remaining(t)) · 5 − λ · C_remaining(t)
```
where:
- `T_remaining(t)` = strike_time − t
- `C_remaining(t)` = 0.9 · (presses_remaining / total_required_presses)
- Use participant's own parameters from Step 9

2. Compute **observed abandonment time**: first time pressing rate drops below 25% of participant's calibrated max for more than 1 second. If pressing never drops below threshold, code as trial_duration.

3. Correlate predicted and observed abandonment times across trials within each participant using Pearson r.

Return:
- Mean within-participant correlation and its distribution across participants
- Results split by predator probability level: T=10%, T=50%, T=90%

> **Primary test:** Is the mean correlation significantly greater than zero at T=90%?

---

## PART 3: INDIVIDUAL DIFFERENCES

**Step 18.** Load trait anxiety scores. Verify N matches model fitting sample. Z-score anxiety within sample.

**Step 19. Primary anxiety regression.**

Run three OLS regressions predicting z-scored anxiety:

| Model | Predictors |
|-------|-----------|
| A | `anxiety ~ α` |
| B | `anxiety ~ λ` |
| C | `anxiety ~ α + λ` |

Return standardized coefficients, 95% CIs, p-values, and R² for all three.

> **Primary test:** Is α significant in Model C while λ is not?

**Step 20. Trait vigor moderation test.**

Compute each participant's trait vigor as mean calibration pressing rate z-scored within sample.

Run mixed effects model:
- Outcome: `delta_vigor` from Step 15
- Fixed effects: `predator_probability + trait_vigor + predator_probability × trait_vigor + distance`
- Random effects: random intercept by participant
- Return full fixed effects table, focus on interaction term

**Step 21. Score state dependence test.**

Compute running cumulative score at each free choice trial onset per participant.

Run mixed effects logistic regression:
- Outcome: heavy cookie choice (0/1)
- Fixed effects: `predicted_ΔV + current_score + predicted_ΔV × current_score`
- Random effects: random intercept by participant
- Return full fixed effects table, focus on interaction term

> **Note:** This is descriptive — do not refit structural models.

**Step 22. Anxiety choice surface visualization.**

Split participants into terciles by trait anxiety score. For each tercile:
1. Compute mean P(heavy) for each of the 9 cells
2. Compute winning model predicted P(heavy) for each cell using that tercile's mean parameter estimates

Return three 3×3 observed matrices and three 3×3 predicted matrices — one pair per tercile — for figure.

---

## OUTPUT REQUIREMENTS

**Step 23.** Return the following outputs in order:

| # | Output |
|---|--------|
| 1 | Convergence diagnostics table for all six models |
| 2 | Model comparison table: WAIC and protected exceedance probability for all six models |
| 3 | Winning model identity clearly labeled |
| 4 | Posterior predictive check: predicted vs observed 3×3 choice proportions for winning model |
| 5 | Individual parameter estimates table from winning model (N=290 rows) |
| 6 | Fixed effects table from Step 14 anticipatory vigor analysis |
| 7 | Fixed effects table and contrast result from Step 15 delta-vigor analysis |
| 8 | Plot data from Step 15 for figure |
| 9 | Correlation result from Step 16 decoupling threshold test |
| 10 | Abandonment timing correlation results from Step 17 by threat level |
| 11 | Anxiety regression tables from Step 19 |
| 12 | Trait vigor moderation table from Step 20 |
| 13 | Score state dependence table from Step 21 |
| 14 | Three 3×3 anxiety tercile choice matrices from Step 22 |

**Step 24.** Flag any step where sample size drops below N=200 due to missing data or exclusions. Report exact N used for each analysis.

**Step 25.** Do not proceed past Step 6 if any model fails convergence. Report the failure and stop.

---

## PARAMETER REFERENCE

| Parameter | Model(s) | Interpretation |
|-----------|---------|----------------|
| λ | All | Effort discount rate — how strongly effort cost reduces subjective value |
| β | All | Inverse temperature — choice stochasticity |
| γ | M2 | Threat sensitivity — how strongly predator probability scales reward |
| κ | M5, M6 | Survival scaling constant |
| α | M5, M6 | Probability distortion exponent — α > 1 means overweighting of high threat probabilities |

## MODEL COMPARISON GUIDE

| If this model wins... | The theoretical implication is... |
|----------------------|----------------------------------|
| M1 | Effort-reward tradeoff explains choices; threat adds nothing |
| M2 | Threat and effort combine additively and independently |
| M3 | Objective survival probability explains threat-effort integration |
| M4 | Global capture rate logic governs choices under threat |
| M5 | Probability distortion is necessary — subjective threat exceeds objective survival calculations |
| M6 | Both capture rate structure and probability distortion are required |
