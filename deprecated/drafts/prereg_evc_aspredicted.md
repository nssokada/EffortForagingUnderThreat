**Hypothesis**
The overarching question is: where on the threat imminence continuum does effort-threat integration occur in human foraging? We propose that integration happens specifically in the pre-encounter deliberative phase and dissolves upon predator encounter, when a separate reactive defense system — orthogonal to effort sensitivity — takes over.

We test this using a task where subjects forage for food items in a virtual environment under threat of capture by a virtual predator, choosing between options that vary in reward, effort cost, and escape distance, then physically executing their foraging bout by pressing keys. On a subset of trials (probe trials), subjects report their subjective anxiety and confidence. Within-trial vigor is decomposed into anticipatory (pre-encounter), reactive (post-encounter), and terminal (circa-strike) epochs.

The computational framework estimates three per-subject parameters from a hierarchical Bayesian model:
- **k** — effort cost: how strongly effort/distance deters choice (identified from the distance gradient in choice)
- **β** — threat aversion: how strongly threat probability deters choice (identified from the threat gradient in choice)
- **c_d** — capture aversion: how strongly survival incentive drives motor vigor (identified from vigor data)

The choice equation is:
```
ΔEU = 4 - k × effort(D) - β × T
P(heavy) = sigmoid(ΔEU / τ)
```
where effort(D) = req_H × T_H − req_L × T_L (effort demand × exposure duration differential), T is the threat probability, and τ is a population-level inverse temperature.

The vigor equation optimizes:
```
EU(u) = S(u) × R - (1 - S(u)) × c_d × (R + C) - c_e_vigor × (u - req)² × D
S(u) = (1 - T) + T × p_esc × sigmoid((u - req) / σ)
```
where u is pressing rate, req is the effort demand, R is the reward, C is the capture penalty, p_esc is the population-level escape probability, and σ controls the sigmoid steepness.

Vigor residuals are computed by residualizing epoch-level pressing rate (median normalized rate = median(1/IPI) / calibrationMax) on cookie type and trial number with random slopes:
```
pressing_rate ~ cookie_type + trial_number + (1 + cookie_type | participant)
```

Specific hypotheses are as follows:
1. Threat will reduce high-effort choice, increase excess motor effort, and shift subjective anxiety upward and confidence downward
    a. High-effort choice will decrease with threat probability and with escape distance
    b. Excess effort (pressing rate minus chosen option's demand), conditioned on chosen cookie type, will increase with threat probability
    c. Trial-level anxiety will increase and confidence will decrease with threat probability T
2. The 3-parameter model will capture choice with two recoverable per-subject parameters (k, β) and vigor with a third (c_d)
    a. Per-subject choice predictions will correlate r > 0.85 with observed choice proportions (exploratory: r = 0.985)
    b. Parameter recovery will yield correlations r > 0.70 for k and β in simulation (exploratory: k = 0.860, β = 0.883)
    c. k and β will be approximately independent: |r| < 0.15 (exploratory: r = 0.105)
    d. The k × β interaction in choice will be significant (exploratory: z = 7.23, p < .0001), indicating genuine integration of effort and threat costs in the deliberative system
3. The three parameters will show a triple dissociation in behavioral signatures
    a. log(k) will predict overcautious rate: r > 0, p < 0.01 (exploratory: t = 18.6)
    b. log(β) will predict threat sensitivity in choice (slope of P(heavy) on threat): p < 0.01 (exploratory: t = −15.5)
    c. log(c_d) will predict vigor consistency (mean fraction of trial at full speed): p < 0.01 (exploratory: t = 23.3)
    d. k and β will explain unique variance in overcaution: joint regression R² > 0.70 (exploratory: R² = 0.818)
4. The population-level choice surface is better described by an objective survival function than by an additive threat penalty
    a. M3 (ΔV = 5 × exp(−p × T_H) − exp(−p × T_L) − λ × effort) will have lower BIC than M2 (ΔV = 4 − λ × effort − γ × p) (exploratory: ΔBIC = 1,772 in favor of M3)
    b. Adding probability distortion (M5: p^α) will NOT improve over M3 (exploratory: M5 ΔBIC = +4,797 worse)
    c. The probability distortion parameter α will show near-zero individual variance (exploratory: α = 2.76, SD = 0.007), indicating that probability weighting is a fixed population property, not an individual difference
5. Effort-threat integration occurs specifically in the pre-encounter deliberative phase and dissolves at predator encounter (the threat imminence gradient)
    a. The threat × β interaction on residualized vigor will be significant in the anticipatory epoch (pre-encounter) and null in the reactive epoch (post-encounter): anticipatory p < 0.05, reactive p > 0.10 (exploratory: anticipatory z = 2.46, p = .014; reactive z = 0.56, p = .576)
    b. The distance × c_d interaction on residualized vigor will be null in the anticipatory epoch and significant in the reactive epoch: anticipatory p > 0.10, reactive p < 0.01 (exploratory: anticipatory z = −0.33, p = .739; reactive z = −6.14, p < .0001)
    c. The c_d main effect will be null in the anticipatory epoch and significant in the terminal epoch (circa-strike): anticipatory p > 0.10, terminal p < 0.001 (exploratory: anticipatory z = 0.08, p = .936; terminal z = 6.17, p < .0001)
    d. Threat probability will predict vigor in the anticipatory epoch (p < 0.01) but not in the terminal epoch (p > 0.10) (exploratory: anticipatory r = .053, p < 10^−12; terminal r = .007, p = .525)
    e. Reactive vigor will be driven by actual predator presence (attack_trial), not stated probability, and the interaction attack_trial × predator_probability will be null (exploratory: attack_trial z = 8.96, p < .0001; interaction z = −1.95, p = .051)
    f. Between-subject variance in residualized vigor will be compressed in the reactive epoch relative to the anticipatory epoch by a factor > 3 (exploratory: 6.6×)
6. Affective calibration and discrepancy will show a differential prediction pattern for performance versus anxiety symptoms
    a. Calibration (within-subject r between anxiety and threat probability T) and discrepancy (mean anxiety residual after removing the T–anxiety relationship) will be approximately orthogonal: |r| < 0.15 (exploratory: r = 0.024)
    b. Discrepancy will predict STAI-State controlling for k, β, and overcautious rate: p < 0.01, ΔR² > 0.05 (exploratory: β = 0.334, p < .0001, ΔR² = 0.108)
    c. Calibration will predict residual overcaution (excess conservative choice beyond what k, β, and their interaction explain): p < 0.01 (exploratory: β = −0.007, p = .0007)
    d. Discrepancy will predict at least two additional symptom measures from {OASIS, STICSA, PHQ-9, DASS-Anxiety, DASS-Stress, DASS-Depression} at p < 0.05 uncorrected (exploratory: OASIS β = 0.211, p = .0004)
    e. Model parameters (k, β, c_d) will NOT independently predict clinical measures: all |r| < 0.10 or p > 0.05 (exploratory: all params null)

**Dependent variable**
Listed according to hypotheses:
1. (a) Binary choice of high- vs. low-effort cookie on each of 45 behavioral trials. (b) Excess effort: the difference between actual pressing rate and effort demand, centered by cookie type (trial-level, continuous). (c) Self-reported anxiety and confidence ratings on probe trials (0–7 scale), 18 per affect type per subject.
2. (a) Per-subject predicted versus observed choice proportion. (b) Simulated-and-recovered parameter correlations in log space for k and β. (c) Pearson r between log(k) and log(β). (d) Fixed effect of k_z × β_z interaction from mixed effects logistic regression on choice.
3. (a–c) Per-subject behavioral outcomes regressed on z-scored log-transformed parameters. (d) R² from joint regression of overcautious rate on log(k) and log(β).
4. (a) BIC from M3 versus M2 choice models. (b) BIC from M5 versus M3. (c) SD of individual α estimates from M5.
5. (a) Fixed effects from mixed effects models: vigor_resid ~ predator_probability × β_z + distance × k_z + ... + (1 | participant), run separately for anticipatory, reactive, and terminal epochs. (b–c) Same models, specific interaction terms. (d) Correlation of vigor residuals with threat per epoch. (e) Fixed effects from reactive vigor ~ attack_trial × predator_probability + (1 | participant). (f) Between-subject SD of mean residualized vigor per epoch.
6. (a) Pearson r between calibration and discrepancy. (b) OLS regression of z-scored clinical measure on discrepancy controlling for k, β, and overcautious rate. (c) OLS regression of overcaution residual on calibration. (d) Pearson r between discrepancy and clinical measures. (e) Pearson r between each model parameter and each clinical measure.

**Conditions**
One experiment with a within-subjects design. Each subject completes 81 trials (3 blocks × 27 trials) in a virtual foraging task. Conditions are fully crossed within subject:
- **Threat probability:** T ∈ {0.1, 0.5, 0.9} — the probability of a predator attack on each trial
- **Distance:** D ∈ {1, 2, 3} corresponding to 5, 7, 9 game units from the safe zone — the distance of the high-effort cookie
- **Effort demand:** E_H ∈ {0.6, 0.8, 1.0} of calibrated pressing capacity — the effort required for the high-effort cookie (confounded with distance: D = 1 → E = 0.6, D = 2 → E = 0.8, D = 3 → E = 1.0)

The low-effort option is fixed (E_L = 0.4, D_L = 1, R_L = 1 point). The high-effort option yields R_H = 5 points. Capture penalty is C = 5 points.

Each block contains 15 regular behavioral trials and 12 probe trials (6 anxiety, 6 confidence). Probe trials use forced-choice (identical options) and collect ratings before pressing begins. A psychiatric battery (DASS-21, PHQ-9, OASIS, STAI-State, AMI, MFIS, STICSA) is administered between blocks.

Within-trial epochs defined using actual encounterTime and strike_time from the game data:
- **Anticipatory:** Trial onset → encounterTime (pre-encounter, deliberative)
- **Reactive:** encounterTime → encounterTime + 2 seconds (post-encounter, defensive)
- **Terminal:** strike_time − 2 seconds → strike_time (circa-strike, attack trials only)

**Analyses**
H1.
1. Logistic mixed-effects model on trial-level binary choice: `choice ~ threat_z + dist_z + threat_z:dist_z + (1 | subject)`. Tests: β(threat) < 0 and β(dist) < 0, both p < 0.01.
2. Linear mixed-effects model on trial-level cookie-centered excess effort, conditioned on chosen cookie type: `excess_cc ~ threat_z + (1 | subject)`. Test: β(threat) > 0, p < 0.05.
3. Linear mixed-effects models on probe ratings: `anxiety ~ T_z + (1 + T_z | subject)` and `confidence ~ T_z + (1 + T_z | subject)`. Tests: β(T) > 0 for anxiety and β(T) < 0 for confidence, both |t| > 3.0.

H2.
1. Fit the 3-parameter model via NumPyro SVI (AutoNormal guide, 40,000 steps, ClippedAdam lr = 0.001, early stopping at minimum loss). Choice: all 81 trials (probes contribute P(H) = 0.5). Vigor: all 81 trials. Choice equation: ΔEU = 4 − k × effort(D) − β × T. Effort = req_H × T_H − req_L × T_L. Vigor equation: EU(u) = S(u) × R − (1 − S(u)) × c_d × (R + C) − c_e_vigor × (u − req)² × D. Validate with MCMC (NUTS, 4 chains).
2. Report per-subject choice r (test > 0.85).
3. Parameter recovery: 3 × 50 subjects simulated and recovered. Report r for k, β (test > 0.70 each).
4. Report r(log(k), log(β)) (test |r| < 0.15).
5. Mixed effects logistic regression: choice ~ k_z × β_z + threat + distance + trial_number + current_score + (1 | subj). Report k_z × β_z interaction z and p.

H3.
1. OLS regressions: overcautious_rate ~ k_z + β_z + cd_z; threat_slope ~ k_z + β_z + cd_z; frac_full ~ k_z + β_z + cd_z. Report t-values for each parameter in each regression.
2. Report R² from joint regression of overcaution on k + β.

H4.
1. Fit M3 (ΔV = 5 × exp(−p × T_H) − exp(−p × T_L) − λ × effort) and M2 (ΔV = 4 − λ × effort − γ × p) via SVI. Compare BIC.
2. Fit M5 (M3 + probability distortion p^α). Compare BIC to M3.
3. Report SD of individual α from M5.

H5.
1. Compute residualized vigor per epoch (residualize pressing rate on cookie_type + trial_number with random slopes by participant).
2. Run the core model three times, once per epoch: vigor_resid ~ predator_probability + distance + β_z + k_z + cd_z + predator_probability:β_z + predator_probability:k_z + predator_probability:cd_z + distance:β_z + distance:k_z + distance:cd_z + (1 | participant). Report all interaction terms.
3. Report threat × β in anticipatory (test p < 0.05) and reactive (test p > 0.10).
4. Report distance × c_d in anticipatory (test p > 0.10) and reactive (test p < 0.01).
5. Report c_d main effect in anticipatory (test p > 0.10) and terminal (test p < 0.001).
6. Report r(vigor_resid, threat) per epoch. Test that threat predicts anticipatory (p < 0.01) but not terminal (p > 0.10).
7. Reactive epoch: vigor_resid ~ attack_trial + predator_probability + attack_trial:predator_probability + (1 | participant). Test attack_trial significant, interaction null.
8. Report between-subject SD per epoch. Test reactive SD < anticipatory SD (ratio > 3).

H6.
1. Compute calibration (within-subject r between anxiety and T) and discrepancy (mean anxiety residual from population T–anxiety regression). Report r(calibration, discrepancy) (test |r| < 0.15).
2. OLS: STAI_State_z ~ k_z + β_z + overcautious_rate + discrepancy_z. Report discrepancy coefficient (test p < 0.01, ΔR² > 0.05).
3. OLS: overcaution_residual ~ discrepancy_z + calibration_z. Report calibration (test p < 0.01).
4. Report r(discrepancy, clinical_measure) for all clinical measures. Test ≥ 2 significant at p < 0.05.
5. Report r(k, clinical), r(β, clinical), r(c_d, clinical) for all clinical measures. Test all |r| < 0.10 or p > 0.05.

**Exploratory analyses (not preregistered as directional hypotheses):**
- **E1:** Sliding window analysis of parameter-vigor correlations across within-trial time, testing whether the β → vigor association crosses the cd → vigor association near encounterTime.
- **E2:** In the combined exploratory + confirmatory sample (N ≈ 570–620), test whether log(k), log(β), and log(c_d) predict anxiety symptoms at p < .05 uncorrected. The exploratory sample showed null results for all parameters. The combined sample provides 83% power to detect r = 0.12.
- **E3:** Behavioral profiles (k × β median split) — descriptive analysis of optimality, overcaution, discrepancy, and STAI by profile.
- **E4:** Simpson's paradox demonstration in unconditional vigor.
- **E5:** Anticipatory vigor slope (individual threat–vigor slope in anticipatory epoch) as predictor of overcaution above k and β (exploratory: p = .035).

**Outliers and Exclusions**
Exclusion criteria applied before any model fitting:
1. Incomplete task (did not finish all 81 trials)
2. Invalid calibration (fewer than 10 presses across calibration trials)
3. Implausible keypresses: max single-trial press rate > 3 SD above sample mean (automated input), or zero presses on > 50% of regular trials (disengagement)
4. Invalid predator dynamics: > 10% of trials with physically impossible predator behavior
5. Low engagement: overall escape rate < 35% across attack trials
6. Insufficient probes: < 80% probe completion (< 29/36). These subjects excluded from affect analyses (H1c, H4, H6) only.
7. Calibration outliers: subjects with mean IKI > 2.5 SD from sample mean (exploratory: 3 subjects excluded: 154, 197, 208)

No post-hoc exclusions based on model fit quality or statistical extremity.

**Sample Size**
N = 350 recruited via Prolific. Expected N ≈ 280–330 after exclusions (exploratory retention: 83.7%). At N = 280, power > 0.95 for all preregistered tests (H1–H6) based on exploratory effect sizes. The weakest preregistered test is H5a (threat × β in anticipatory; exploratory z = 2.46); at N = 280 with the same effect size, power > 0.90.

For the exploratory analyses (E1–E5), the combined exploratory + confirmatory sample (N ≈ 570–620) provides greater sensitivity for small effects.

**Other**
The exploratory sample (N = 293, after exclusion of 3 calibration outliers: N = 290) was used to develop all hypotheses and specify all analysis plans. The confirmatory sample has been collected but not analyzed. The preregistration will be timestamped on AsPredicted before the confirmatory data are opened. All analysis code is available at [repository URL]; data will be shared on OSF upon acceptance.

**Name**
Effort-threat integration on the threat imminence continuum in human foraging

**Finally**
Experiment
