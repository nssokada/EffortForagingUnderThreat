# Full Analysis Plan: Threat-Effort Foraging Task
## Integrating Effort and Threat Across the Predatory Imminence Continuum

*Target: Nature Communications*
*N=290 after exclusions (subjects 154, 197, 208 removed)*

---

## OVERVIEW

Four parts building sequentially. Each part has clear outputs and stopping criteria. Do not proceed past a stopping criterion without reporting the failure.

**Part 1:** Computational model of patch selection — establish separable k and β parameters and their integration in choice.

**Part 2:** Vigor decomposition — establish clean epoch-level vigor signals across the threat imminence continuum.

**Part 3:** The threat imminence gradient — the central finding. Test where effort-threat integration occurs and where it dissolves.

**Part 4:** Optimal policy, suboptimality decomposition, and clinical mechanism — connect the computational model to behavioral errors and clinical outcomes.

---

## PART 1: COMPUTATIONAL MODEL OF PATCH SELECTION

---

### Step 1.1 — Fit the 3-param v2 model

Load free choice trials only. Confirm 45 trials per participant, N=290.

**Model specification:**

Choice equation:
```
ΔV = 4 - k_i · effort(D) - β_i · p
P(heavy) = sigmoid(ΔV / τ)
```

Where:
- effort(D) = req_heavy · T_heavy − req_light · T_light
- req_heavy = {0.6, 0.8, 1.0} for d=1,2,3; req_light = 0.4 always
- T_heavy = {5, 7, 9} seconds for d=1,2,3; T_light = 5 always
- p = predator probability {0.1, 0.5, 0.9}
- τ = population-level temperature only (not per-subject)
- k_i and β_i are per-subject parameters

**Fitting procedure:**
- NumPyro SVI, AutoNormal guide
- ClippedAdam lr=0.001, clip_norm=10.0
- 40,000 steps with early stopping at best loss
- Non-centered parameterization for k_i and β_i
- Log-normal priors: μ_k ~ Normal(0,1), μ_β ~ Normal(0,1) in log space
- σ_k ~ HalfNormal(0.5), σ_β ~ HalfNormal(0.5)

**MCMC validation (run AFTER full pipeline completes — do not block on this):**
- NUTS, 4 chains × 1000 warmup + 1000 samples on winning model
- Confirm SVI-MCMC correlation > 0.99 for k and β
- R-hat < 1.01, ESS > 400 for all parameters
- All downstream analyses use SVI parameters; MCMC validates that SVI is accurate

**Return:**
- Population-level parameter estimates with 89% HDI
- Individual-level k_i and β_i posterior means for all N=290 participants as one row per participant
- Choice accuracy, r², BIC
- Save to results/stats/

> ⚠️ **STOPPING CRITERION:** If choice r² < 0.90 OR MCMC R-hat > 1.05 for any parameter, stop and report. Do not proceed to Step 1.2.

---

### Step 1.2 — Parameter orthogonality check

Compute Pearson r between k_i and β_i across all N=290 participants.

Return:
- r value and p-value
- 95% CI on r

**Expected:** r ≈ −0.006. Confirms effort sensitivity and threat sensitivity are separable individual differences.

---

### Step 1.3 — Parameter recovery simulation

Simulate 100 synthetic datasets. For each simulation:
1. Draw k, β, cd values from their prior distributions
2. Generate synthetic choices from the model
3. Fit the model to the synthetic data
4. Compute correlation between generating and recovered parameters

Return:
- r(generating k, recovered k) — expected > 0.80
- r(generating β, recovered β) — expected > 0.80
- Cross-recovery r(generating k, recovered β) — expected < 0.20
- Cross-recovery r(generating β, recovered k) — expected < 0.20

> ⚠️ **STOPPING CRITERION:** If recovery for k or β < 0.70, stop and report which parameters are confounded. Do not proceed.

---

### Step 1.4 — Triple dissociation

Run three separate OLS regressions predicting behavioral outcomes from z-scored individual parameters (k_z, β_z, cd_z):

**Regression A — Overcaution:**
```
overcautious_rate ~ k_z + β_z + cd_z
```
Where overcautious_rate = proportion of free choice trials choosing light when heavy was objectively optimal (see Part 4 for optimal policy derivation — run Part 4 Step A first if not yet computed, or use precomputed values if available).

**Regression B — Threat slope:**
```
threat_slope ~ k_z + β_z + cd_z
```
Where threat_slope = per-participant slope of P(heavy) regressed on predator_probability across all free choice trials.

**Regression C — Vigor execution:**
```
frac_full ~ k_z + β_z + cd_z
```
Where frac_full = per-participant fraction of trials pressing at or above required rate.

Return standardized coefficients, 95% CIs, p-values, and R² for all three regressions.

**Expected pattern:** k predicts overcaution, β predicts threat slope, cd predicts frac_full. Each parameter maps onto a distinct behavioral output.

---

### Step 1.5 — Choice integration test

This is the integration finding in choice.

Run mixed effects logistic regression on free choice trials:

```
P(heavy) ~ k_z + β_z + k_z × β_z
         + predator_probability + distance
         + trial_number + current_score
         + (1 | participant)
```

Where current_score = running cumulative score at trial onset.

Return full fixed effects table: all terms with coefficients, z-values, p-values.

**Primary test:** Is k_z × β_z significant? Positive interaction means high effort-sensitive AND high threat-sensitive individuals show partial compensation — their combined avoidance is less than the additive prediction. This is genuine integration in the deliberative system.

---

### Step 1.6 — Posterior predictive check

Generate predicted P(heavy) for each of the 9 cells using posterior mean parameters.

Return predicted vs observed as a 3×3 table:
```
         D=1         D=2         D=3
T=10%  obs/pred   obs/pred   obs/pred
T=50%  obs/pred   obs/pred   obs/pred
T=90%  obs/pred   obs/pred   obs/pred
```

Flag any cell where |predicted − observed| > 0.10.

---

### Step 1.7 — M3 robustness check

Fit M3 (objective survival function, population-level only):
```
ΔV = 5·exp(−p·T_heavy) − 1·exp(−p·5) − λ·effort(D)
```

Return BIC for M3 versus 3-param model. Report ΔBIC.

Note: The central findings (epoch-specific vigor interactions) do not depend on which choice model wins. If M3 wins on BIC, report both models — M3 for population-level mechanism, 3-param for individual differences.

---

## PART 2: VIGOR DECOMPOSITION

---

### Step 2.1 — Compute epoch-level pressing rates

Load ALL trials — free choice and forced. N=290. All 81 trials per participant.

**Epoch definitions using actual encounterTime from data:**

| Epoch | Start | End | Imminence Stage |
|-------|-------|-----|-----------------|
| Anticipatory | Trial onset | encounterTime | Pre-encounter |
| Reactive | encounterTime | encounterTime + 2s | Post-encounter |
| Terminal | Strike_time − 2s | Strike_time | Circa-strike |

Notes:
- For non-attack trials: use scheduled encounterTime as epoch boundary. Participants did not know trial outcome during pressing.
- Terminal epoch: attack trials only. Use actual strike_time.
- Flag and exclude trials with fewer than 3 keypresses in any epoch from that epoch's analysis. Retain for other epochs.

For each trial and epoch compute:
- Mean pressing rate in presses per second
- Fraction at full speed (presses ≥ required rate)

Report N of trials contributing to each epoch analysis.

---

### Step 2.2 — Residualize each epoch

For each epoch separately run:

```
pressing_rate ~ cookie_type + trial_number
              + (1 + cookie_type | participant)
```

Where cookie_type = 1 for heavy, 0 for light. Trial_number continuous 1–81.

**Do NOT include distance or predator_probability** — these contain the signal to be predicted.

Save residuals:
- vigor_resid_anticipatory
- vigor_resid_reactive
- vigor_resid_terminal

---

### Step 2.3 — Verify residuals contain signal

For each epoch return:
- Pearson r between vigor_resid and predator_probability with p-value
- Pearson r between vigor_resid and distance with p-value
- Between-subject SD of mean residualized vigor (mean across all trials per participant)
- Mean and SD of residuals overall (should be near zero mean)

> ⚠️ **STOPPING CRITERION:** If predator_probability does not significantly predict vigor_resid in BOTH anticipatory and reactive epochs (p < .05), stop Part 2 and report. The vigor signal is not present.

---

### Step 2.4 — Between-subject variance across epochs

Compute mean residualized vigor per participant per epoch collapsed across all trials.

Return:
- SD of participant means: anticipatory, reactive, terminal
- Ratio: anticipatory SD / reactive SD
- Levene's test comparing variance across three epochs: F-statistic and p-value

**Expected:** Reactive SD approximately 6-7× lower than anticipatory. Reflects conserved post-encounter response.

---

### Step 2.5 — Cross-epoch correlation matrix

Correlate participant mean vigor residuals across epochs.

Return full 3×3 matrix with Pearson r and p-values for each pair:
- anticipatory vs reactive
- anticipatory vs terminal
- reactive vs terminal

---

## PART 3: THE THREAT IMMINENCE GRADIENT

---

### Step 3.1 — Core epoch-by-epoch interaction test

**This is the central analysis.** Run the following mixed effects model three times — once per epoch — with residualized vigor as outcome:

```
vigor_resid ~ predator_probability + distance
            + β_z + k_z + cd_z
            + predator_probability × β_z
            + predator_probability × k_z
            + predator_probability × cd_z
            + distance × β_z
            + distance × k_z
            + distance × cd_z
            + (1 | participant)
```

Where β_z, k_z, cd_z are z-scored individual parameters from Step 1.1.

Return full fixed effects table for each epoch with coefficients, z-values, p-values.

**The predicted pattern — this is the finding:**

| Interaction | Anticipatory | Reactive | Terminal |
|------------|--------------|----------|----------|
| threat × β | Significant | Null | Null |
| threat × k | Null | Null | Null |
| distance × cd | Null | Significant | Significant |
| cd main effect | Null | Null | Significant |

Any deviation from this pattern should be reported and interpreted. Partial confirmation is still meaningful.

> ⚠️ **STOPPING CRITERION:** If threat × β is null in anticipatory AND distance × cd is null in reactive, the central claim fails. Stop and report.

---

### Step 3.2 — Threat independence test

Run on reactive epoch only, using all trials (attack and non-attack):

```
vigor_resid_reactive ~ attack_trial + predator_probability
                     + attack_trial × predator_probability
                     + (1 | participant)
```

Where attack_trial = 1 if predator actually appeared on that trial, 0 otherwise.

Return full fixed effects table.

**Expected:** attack_trial large and significant (predator presence drives reactive vigor). predator_probability null or small. Interaction null.

This distinguishes the post-encounter reactive system (triggered by predator presence) from the pre-encounter deliberative system (modulated by stated probability).

---

### Step 3.3 — Sliding window analysis

**Purpose:** Show in real time when the system transition occurs within trials.

Using raw (non-residualized) pressing rate from all trials, for each 500ms window from trial onset to strike_time:

1. Compute partial correlation between predator_probability and pressing_rate controlling for cookie_type and distance across all trials
2. Compute partial correlation between cd_z and pressing_rate controlling for cookie_type, distance, predator_probability across all trials

Return two time series with one value per 500ms window:
- r(predator_probability, vigor | controls) across time
- r(cd_z, vigor | controls) across time

Also return 95% confidence bands for each time series.

**This is Figure 3 of the paper.** The crossing of these two time series at approximately encounterTime is the behavioral signature of the system handoff from deliberative to reactive control.

**Crossing-point test:** Identify the time window where r(cd_z, vigor) first exceeds r(predator_probability, vigor). Test whether this crossing point falls within ±1 second of the mean encounterTime across distances. Bootstrap the crossing point 1000 times (resampling participants) to generate a 95% CI. If the CI includes the mean encounterTime, the handoff is temporally locked to predator appearance.

---

### Step 3.4 — Pre-encounter integration figure data

Split participants into tertiles by β_z: low (bottom third), medium (middle third), high (top third).

For each tertile compute mean anticipatory vigor residual ± SE at each threat level {0.1, 0.5, 0.9}.

Also compute the same for the reactive epoch.

Return two tables:

**Anticipatory epoch:**
```
         T=0.1      T=0.5      T=0.9
Low β    mean±SE    mean±SE    mean±SE
Mid β    mean±SE    mean±SE    mean±SE
High β   mean±SE    mean±SE    mean±SE
```

**Reactive epoch (same structure).**

**This is Figure 4A.** Three lines diverging with threat in anticipatory epoch (high-β people ramp up more), converging to the same slope in reactive epoch (β no longer matters after encounter).

---

### Step 3.5 — Post-encounter reactive figure data

Split participants into tertiles by cd_z: low, medium, high.

For each tertile compute mean reactive vigor residual ± SE at each distance level {1, 2, 3}.

Also compute the same for the anticipatory epoch.

Return two tables:

**Reactive epoch:**
```
         D=1        D=2        D=3
Low cd   mean±SE    mean±SE    mean±SE
Mid cd   mean±SE    mean±SE    mean±SE
High cd  mean±SE    mean±SE    mean±SE
```

**Anticipatory epoch (same structure).**

**This is Figure 4B.** Lines flat across distance in anticipatory, diverging with distance in reactive (high-cd people show steeper distance gradient after encounter).

---

### Step 3.6 — Parameter handoff summary table

Compile all key interaction results into a single summary table. This is Table 2 of the paper.

Format:
```
Parameter    | Anticipatory         | Reactive             | Terminal             | Interpretation
-------------|----------------------|----------------------|----------------------|---------------
β (threat)   | threat×β: coef, z, p | threat×β: coef, z, p | threat×β: coef, z, p | Pre-encounter only
k (effort)   | threat×k: coef, z, p | threat×k: coef, z, p | threat×k: coef, z, p | Choice only
cd (capture) | dist×cd: coef, z, p  | dist×cd: coef, z, p  | dist×cd: coef, z, p  | Post-encounter
cd (main)    | main: coef, z, p     | main: coef, z, p     | main: coef, z, p     | Circa-strike
```

---

## PART 4: OPTIMAL POLICY, SUBOPTIMALITY, AND CLINICAL MECHANISM

---

### Step 4A — Derive optimal policy

For each of the 9 cells in the design compute expected value of heavy and light choice using objective task parameters.

**For heavy cookie in cell (p, d):**
```
EV_heavy = S(p,d) × 5 − (1 − S(p,d)) × 5 − effort_cost_heavy(d)
```

**For light cookie in cell (p, d):**
```
EV_light = S(p,1) × 1 − (1 − S(p,1)) × 5 − effort_cost_light
```

Where:
- S(p, d) = exp(−p × T(d)) using M3 survival function
- T(d) = {5, 7, 9} for d=1,2,3
- effort_cost_heavy(d) = req_heavy × T_heavy = {0.6×5, 0.8×7, 1.0×9} = {3.0, 5.6, 9.0}
- effort_cost_light = 0.4 × 5 = 2.0

Optimal choice in each cell: heavy if EV_heavy > EV_light, light otherwise.

Return:
- 3×3 optimal policy matrix (heavy/light for each cell)
- 3×3 EV margin matrix (|EV_heavy − EV_light| for each cell — how strongly optimal one way or the other)
- Under this policy, what proportion of trials favor heavy? What proportion favor light?

---

### Step 4B — Compute individual deviation profiles

For each participant and each free choice trial classify as:
- **Optimal:** Chose according to the optimal policy for that cell
- **Overcautious error:** Chose light when heavy was optimal
- **Overrisky error:** Chose heavy when light was optimal

Per participant compute:
- optimality_rate = proportion of trials matching optimal policy
- overcautious_rate = proportion of trials making overcautious errors
- overrisky_rate = proportion of trials making overrisky errors

Note: optimality_rate + overcautious_rate + overrisky_rate = 1.0

Return:
- Distribution of all three rates across N=290 participants
- Mean, SD, median, IQR for each
- Histogram data for each

Also compute cell-level deviation per participant: for each of the 9 cells, signed difference between observed P(heavy) and optimal P(heavy) (0 or 1). Return 9-value deviation profile per participant.

---

### Step 4C — Compute affective measures from probe trials

Load anxiety and confidence ratings from forced choice (probe) trials. 36 ratings per participant.

**Per participant compute:**

**Mean anxiety:** Mean anxiety rating across all 36 probe trials.

**Mean confidence:** Mean confidence rating across all 36 probe trials.

**Calibration (affect-tracking):** Within-participant Pearson r between anxiety rating and objective danger (1 − S(p,d)) across all anxiety probe trials. Higher calibration = anxiety tracks the computational danger signal more accurately trial-to-trial.

**Calibration (confidence-accuracy):** Within-participant correlation between confidence rating and whether the assigned option matched the optimal policy for that cell. Higher calibration = participant is more confident when forced into the optimal choice.

Compute BOTH calibration measures. Report their intercorrelation.

**Discrepancy:** Mean anxiety rating minus predicted anxiety from the behavioral survival computation.
Predicted anxiety for a trial = 1 − S(p, d) where S is the objective survival probability for that trial's p and d.
Discrepancy = mean(anxiety_rated) − mean(1 − S(p,d)) across all probe trials.
Positive discrepancy = feels more threatened than the objective danger level implies.
Negative discrepancy = feels less threatened than the objective danger level implies.

Return:
- Distribution of each measure across N=290 participants
- Pearson r between calibration and discrepancy — confirm orthogonality

> **Note:** If calibration and discrepancy correlate at |r| > 0.40, flag this before running clinical regressions. High correlation would indicate they are measuring the same construct.

---

### Step 4D — Model comparison stack for overcautious errors

Run seven nested regression models predicting overcautious_rate. Use OLS. Report AIC and BIC for all seven.

```
M1 (null):        overcautious_rate ~ 1
M2 (effort):      overcautious_rate ~ k_z
M3 (threat):      overcautious_rate ~ β_z
M4 (additive):    overcautious_rate ~ k_z + β_z
M5 (interaction): overcautious_rate ~ k_z + β_z + k_z × β_z
M6 (affective):   overcautious_rate ~ discrepancy
M7 (full):        overcautious_rate ~ k_z + β_z + k_z × β_z + discrepancy + calibration
```

Return for each model: all coefficients with 95% CIs, p-values, R², AIC, BIC, ΔBIC relative to best model.

**Primary questions:**
- Does M5 beat M4? Tests whether effort and threat sensitivity interact in producing suboptimality.
- Does M7 beat M5? Tests whether affective channels explain variance beyond computational parameters.
- What is the unique ΔR² for discrepancy in M7 above M5?

---

### Step 4E — Same stack for overrisky errors and overall optimality

Run the identical seven-model stack predicting overrisky_rate.

Run the identical seven-model stack predicting optimality_rate.

Return the same outputs as Step 4D for each.

---

### Step 4F — Residual suboptimality analysis

From M5 in Step 4D, extract residuals: the portion of overcautious_rate not explained by k_z, β_z, and their interaction. Call these overcautious_resid.

Run:
```
overcautious_resid ~ discrepancy + calibration
```

Return: coefficients, 95% CIs, p-values, R².

**This is the key affective mechanism test.** If discrepancy predicts overcautious_resid, affective miscalibration drives avoidance errors that the computational parameters cannot account for.

---

### Step 4G — Clinical mechanism test

**Primary regressions:**

```
STAI-State ~ overcautious_resid + overcautious_rate + k_z + β_z + discrepancy
OASIS ~ overcautious_resid + overcautious_rate + k_z + β_z + discrepancy
AMI ~ overrisky_resid + overrisky_rate + k_z + β_z + discrepancy
```

Return full tables with standardized coefficients, 95% CIs, p-values, R².

**Also run the discrepancy ΔR² test:**

For STAI-State and OASIS:
1. Fit: clinical ~ k_z + β_z + overcautious_rate. Record R².
2. Add discrepancy. Record new R².
3. ΔR² = improvement from adding discrepancy.

Return ΔR² with F-test for significance.

**Primary question:** Does affective discrepancy predict anxiety symptoms above and beyond computational parameters and raw behavioral errors? If yes, this establishes the clinical mechanism chain: affective miscalibration → avoidance errors → anxiety symptoms.

---

### Step 4H — Threat imminence and suboptimality

Test whether the pre-encounter vigor signal carries information about suboptimality above and beyond choice parameters.

For each participant compute their personal anticipatory vigor slope: regress their anticipatory vigor residuals on predator_probability within-participant. This gives a per-person estimate of how strongly β modulates their pre-encounter motor preparation.

Run:
```
overcautious_rate ~ anticipatory_vigor_slope + k_z + β_z
```

Return full regression table.

**Question:** Does the anticipatory motor signal predict overcaution above and beyond the choice parameters? If yes, the pre-encounter system carries behavioral information that the choice model alone does not capture — connecting the imminence gradient finding to foraging outcomes.

---

### Step 4I — Behavioral profile analysis

Classify each participant into one of four profiles based on median split of k_z and β_z:

| Profile | k | β | Prediction |
|---------|---|---|------------|
| Vigilant | Low | Low | Low avoidance, responsive to threat, best earners |
| Effort-averse | High | Low | Avoids distant patches, less threat-reactive |
| Threat-sensitive | Low | High | Avoids high-threat conditions specifically |
| Overcautious | High | High | Avoids most conditions, worst earners |

For each profile compute:
- Mean optimality_rate
- Mean overcautious_rate
- Mean overrisky_rate
- Mean total earnings
- Mean discrepancy
- Mean STAI-State

Return profile summary table and N per profile.

**Note:** The k × β interaction in choice (Step 1.5) predicts the Overcautious profile should be less severe than naive additivity predicts — report whether this is visible in the earnings data.

---

## OUTPUT REQUIREMENTS

Return all outputs in this order. Label each output clearly with its Step number.

| # | Step | Output |
|---|------|--------|
| 1 | 1.1 | 3-param model parameter estimates table, accuracy, r², BIC |
| 2 | 1.1 | MCMC validation: R-hat, ESS, SVI-MCMC correlations |
| 3 | 1.1 | Individual k_i and β_i table (N=290 rows) |
| 4 | 1.2 | k × β orthogonality correlation |
| 5 | 1.3 | Parameter recovery simulation results |
| 6 | 1.4 | Triple dissociation regression tables (A, B, C) |
| 7 | 1.5 | Choice integration test full fixed effects table |
| 8 | 1.6 | Posterior predictive check 3×3 table |
| 9 | 1.7 | M3 robustness check BIC comparison |
| 10 | 2.3 | Vigor residual verification: correlations and SDs per epoch |
| 11 | 2.4 | Between-subject variance table across epochs |
| 12 | 2.5 | Cross-epoch correlation matrix |
| 13 | 3.1 | Anticipatory epoch full fixed effects table |
| 14 | 3.1 | Reactive epoch full fixed effects table |
| 15 | 3.1 | Terminal epoch full fixed effects table |
| 16 | 3.2 | Threat independence test table |
| 17 | 3.3 | Sliding window time series data (two series) |
| 18 | 3.4 | Pre-encounter integration figure data (β tertiles × threat, both epochs) |
| 19 | 3.5 | Post-encounter reactive figure data (cd tertiles × distance, both epochs) |
| 20 | 3.6 | Parameter handoff summary table |
| 21 | 4A | Optimal policy 3×3 matrix and EV margin matrix |
| 22 | 4B | Individual deviation profile distributions |
| 23 | 4C | Affective measure distributions and calibration × discrepancy correlation |
| 24 | 4D | Seven-model comparison table for overcautious rate |
| 25 | 4E | Seven-model comparison table for overrisky rate |
| 26 | 4E | Seven-model comparison table for optimality rate |
| 27 | 4F | Residual suboptimality regression table |
| 28 | 4G | Clinical mechanism regression tables (STAI, OASIS, AMI) |
| 29 | 4G | Discrepancy ΔR² tables for STAI and OASIS |
| 30 | 4H | Threat imminence and suboptimality regression table |
| 31 | 4I | Behavioral profile summary table |

---

## STOPPING CRITERIA

| After Step | Stop if |
|-----------|---------|
| 1.1 | Choice r² < 0.90 OR MCMC R-hat > 1.05 |
| 1.3 | Parameter recovery for k or β < 0.70 |
| 2.3 | predator_probability does not predict vigor_resid in anticipatory AND reactive epochs |
| 3.1 | threat × β null in anticipatory AND distance × cd null in reactive |

At each stopping criterion: report the failure clearly, report what was found, and stop. Do not continue to subsequent steps.

---

## ADDITIONAL REQUIREMENTS

**Sample size:** Flag any analysis where N drops below 200. Report exact N used for every analysis.

**Multiple comparisons:** Apply Benjamini-Hochberg FDR correction within each analysis family (within Part 1, within Part 3, within Part 4). Report both uncorrected and FDR-corrected p-values for all tests.

**Effect sizes:** Report Cohen's d or partial r for all significant effects. Flag effects with |r| < 0.10 or d < 0.20 as small even if significant.

**Convergence:** Flag any mixed effects model with convergence warnings. Report singular fit warnings. Do not interpret results from models that failed to converge.

**Save all outputs to:** results/stats/full_analysis/

---

## WHAT THE RESULTS DETERMINE

| Pattern | Target Journal |
|---------|---------------|
| Steps 1.5 AND 3.1 both hold as predicted | **Nature Communications** — effort-threat integration in pre-encounter phase, demonstrated in choice and anticipatory vigor, with dissolution at encounter |
| Step 1.5 null, Step 3.1 holds | **Nature Communications** — sequential system handoff story, additive in choice but qualitatively distinct in execution |
| Step 3.1 partially holds (β anticipatory + cd reactive significant) | **PNAS** — computational decomposition of pre-encounter and post-encounter systems |
| Step 3.1 fails | **Science Advances** — separable parameters in choice, behavioral observations, honest account of model limitations |

---

## KEY FIGURES (for reference when writing)

**Figure 1:** Task schematic — 3×3 design, timeline within trial, three epoch definitions

**Figure 2:** Choice model results — 3×3 observed choice surface, parameter distributions, triple dissociation scatter plots, k×β orthogonality

**Figure 3:** Sliding window analysis — two time series crossing at encounterTime showing system handoff in real time

**Figure 4A:** β tertiles × threat level in anticipatory vs reactive epochs — diverging lines in anticipatory, converging in reactive

**Figure 4B:** cd tertiles × distance in anticipatory vs reactive epochs — flat lines in anticipatory, diverging in reactive

**Figure 5:** Suboptimality decomposition — model comparison stack, residual mechanism, clinical chain

**Figure 6:** Behavioral profiles — 2×2 k×β space with earnings, optimality, and discrepancy
