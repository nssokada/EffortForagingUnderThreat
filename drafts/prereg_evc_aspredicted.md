**Hypothesis**
The overarching questions being asked are A) how do humans integrate energetic cost and predation risk when foraging, and B) does the affective relationship between computed danger and experienced anxiety predict anxiety symptoms?

We test this using a task where subjects forage for food items in a virtual environment under threat of capture by a virtual predator, choosing between options that vary in reward, effort cost, and escape distance, then physically executing their foraging bout by pressing keys. On a subset of trials (probe trials), subjects report their subjective anxiety and confidence.

The computational framework estimates three per-subject parameters from a joint hierarchical Bayesian model:
- **k** — effort cost: how strongly effort/distance deters choice (identified from choice data)
- **β** — threat aversion: residual threat aversion beyond survival-weighted expected value (identified from choice data)
- **c_d** — capture aversion: how strongly the survival incentive drives motor vigor (identified from vigor data)

The choice equation is:
```
ΔEU = 4 - k × effort(D) - β × T
P(heavy) = sigmoid(ΔEU / τ)
```
where effort(D) encodes the effort–distance cost of the high-effort option relative to the low-effort option, T is the threat probability, and τ is a population-level inverse temperature. Note that S (survival probability), γ (probability weighting), and ε (effort efficacy) do not appear in the choice equation; the choice model uses raw threat probability T directly.

The vigor equation optimizes:
```
EU(u) = S(u) × R - (1 - S(u)) × c_d × (R + C) - c_e_vigor × (u - req)² × D
S(u) = (1 - T) + T × p_esc × sigmoid((u - req) / σ)
```
where u is pressing rate, req is the effort demand, R is the reward, C is the capture penalty, p_esc is the population-level escape probability, and σ controls the sigmoid steepness.

Specific hypotheses are as follows:
1. Threat will reduce high-effort choice, increase excess motor effort, and shift subjective anxiety upward and confidence downward
    a. High-effort choice will decrease with threat probability and with escape distance
    b. Excess effort (pressing rate minus chosen option's demand), conditioned on chosen cookie type, will increase with threat probability
    c. Trial-level anxiety will increase and confidence will decrease with threat probability T
2. The 3-parameter model will jointly capture choice and vigor with three recoverable per-subject parameters
    a. Per-subject choice predictions will correlate r > 0.85 with observed choice proportions (exploratory: r = 0.990)
    b. Trial-level vigor predictions will explain r² > 0.30 of excess effort variance (exploratory: r² = 0.424)
    c. The model will capture distance gradients in choice: predicted P(choose heavy) will decrease with distance within each threat level
    d. Parameter recovery will yield correlations r > 0.70 for all three parameters (k, β, c_d) in simulation (exploratory: k = 0.850, β = 0.841, c_d = 0.927)
    e. k and β will be approximately independent: |r| < 0.10 (exploratory: r = -0.006)
    f. Cross-recovery k → β: |r| < 0.15 (exploratory: r = 0.030)
3. The three parameters will show a triple dissociation in behavioral signatures
    a. log(k) will predict overcautious rate (proportion of trials where the subject chose the low-effort option despite it being EV-suboptimal): r > 0, p < 0.01 (exploratory: r = 0.885)
    b. log(β) will predict threat sensitivity in choice (ΔP(heavy) between low and high threat): r > 0, p < 0.01 (exploratory: r = 0.574)
    c. log(c_d) will predict the vigor gap (difference in excess effort between high- and low-threat conditions): r > 0, p < 0.01 (exploratory: r = 0.580)
    d. k and β will explain unique variance in overcaution: partial R² each > 0.05 (exploratory: k partial R² = 0.768, β partial R² = 0.102)
    e. k dominates low-threat choice: r(log(k), P(heavy) at T = 0.1) < -0.50 (exploratory: r = -0.937)
    f. β dominates high-threat choice: r(log(β), P(heavy) at T = 0.9) < -0.50 (exploratory: r = -0.840)
4. Between-subject confidence will correlate with foraging ambition but not with task performance
    a. Between-subject mean confidence will correlate positively with P(choose heavy), r > 0, p < 0.05
    b. Between-subject mean confidence will NOT correlate with choice quality (proportion of EV-optimal choices), |r| < 0.10
    c. Between-subject mean confidence will NOT correlate with survival rate, |r| < 0.10
5. Affective calibration and discrepancy will show a differential prediction pattern for performance versus anxiety symptoms
    a. Calibration (within-subject r between anxiety and danger, operationalized as threat probability T) and discrepancy (mean anxiety residual after removing the T–anxiety relationship) will be approximately orthogonal: |r| < 0.15 (exploratory: r = 0.032)
    b. Calibration will predict task performance: calibration predicts policy alignment (correspondence between subject's choices and model-optimal choices) controlling for k, β, and c_d, p < 0.05
    c. Discrepancy will predict residual overcaution (excess conservative choice beyond what k, β, and c_d explain) controlling for k, β, and c_d, p < 0.01
    d. Discrepancy will predict anxiety symptoms: r(discrepancy, STAI-State) > 0, p < 0.01
    e. Discrepancy will predict at least two additional symptom measures from {OASIS, STICSA, PHQ-9, DASS-Anxiety, DASS-Stress, DASS-Depression} at p < 0.05 uncorrected
6. Within-trial encounter dynamics will dissociate from choice parameters
    a. Encounter reactivity (vigor increase upon predator appearance) will predict c_d: r > 0, p < 0.05 (exploratory: r = 0.390)
    b. Encounter reactivity will be independent of β: |r| < 0.15 (exploratory: r = 0.078)
    c. Encounter reactivity will be independent of k: |r| < 0.15 (exploratory: r = -0.111)

**Dependent variable**
Listed according to hypotheses:
1. (a) Binary choice of high- vs. low-effort cookie on each of 45 behavioral trials (trial-level). (b) Excess effort: the difference between the actual pressing rate and the effort demand of the chosen option, both in proportion-of-calibrated-capacity units, centered by cookie type (trial-level, continuous). (c) Self-reported anxiety and confidence ratings on probe trials (0–7 scale, trial-level), 18 per affect type per subject.
2. (a) Per-subject predicted versus observed choice proportion, from the jointly fitted 3-parameter model. (b) Trial-level predicted versus observed excess effort. All 81 trials enter the vigor likelihood (probe trials contribute P(H) = 0.5 to choice). (c) Condition-level (threat × distance) predicted versus observed P(choose heavy). (d) Simulated-and-recovered parameter correlations in log space for k, β, and c_d. (e–f) Pearson r between log(k) and log(β), and cross-recovery correlations.
3. (a) Per-subject overcautious rate and log(k). (b) Per-subject threat sensitivity (ΔP(heavy) low–high T) and log(β). (c) Per-subject vigor gap and log(c_d). (d) Partial R² from joint regression of overcaution on log(k) and log(β). (e–f) Per-threat-level choice proportions and parameter correlations.
4. (a) Per-subject Pearson r between mean confidence and P(choose heavy). (b) Per-subject Pearson r between mean confidence and choice quality (proportion of EV-optimal choices). (c) Per-subject Pearson r between mean confidence and survival rate (1 − capture rate).
5. (a) Pearson r between calibration and discrepancy. (b) Partial regression of calibration on policy alignment controlling for k, β, c_d. (c) Partial regression of discrepancy on residual overcaution controlling for k, β, c_d. (d–e) Pearson r between discrepancy and z-scored psychiatric subscale scores.
6. (a) Pearson r between encounter reactivity and log(c_d). (b–c) Pearson r between encounter reactivity and log(β) or log(k).

**Conditions**
One experiment with a within-subjects design. Each subject completes 81 trials (3 blocks × 27 trials) in a virtual foraging task. Conditions are fully crossed within subject:
- **Threat probability:** T ∈ {0.1, 0.5, 0.9} — the probability of a predator attack on each trial
- **Distance:** D ∈ {1, 2, 3} corresponding to 5, 7, 9 game units from the safe zone — the distance of the high-effort cookie
- **Effort demand:** E_H ∈ {0.6, 0.8, 1.0} of calibrated pressing capacity — the effort required for the high-effort cookie

The low-effort option is fixed (E_L = 0.4, D_L = 1, R_L = 1 point). The high-effort option yields R_H = 5 points. Capture penalty is C = 5 points.

Each block contains 15 regular behavioral trials and 12 probe trials (6 anxiety, 6 confidence). Probe trials use forced-choice (identical options) and collect ratings before pressing begins. A psychiatric battery (DASS-21, PHQ-9, OASIS, STAI-State, AMI, MFIS, STICSA) is administered between blocks.

**Analyses**
H1.
1. Logistic mixed-effects model on trial-level binary choice: `choice ~ threat_z + dist_z + threat_z:dist_z + (1 | subject)`. Tests: β(threat) < 0 and β(dist) < 0, both p < 0.01. Monotonicity of P(choose heavy) across adjacent threat levels within each distance, confirmed by paired t-tests (p < 0.01, one-tailed).
2. Linear mixed-effects model on trial-level cookie-centered excess effort, conditioned on chosen cookie type: `excess_cc ~ threat_z + (1 | subject)`. Test: β(threat) > 0, p < 0.05.
3. Linear mixed-effects models on probe ratings: `anxiety ~ T_z + (1 + T_z | subject)` and `confidence ~ T_z + (1 + T_z | subject)`, where T is threat probability. Tests: β(T) > 0 for anxiety and β(T) < 0 for confidence, both |t| > 3.0.

H2.
1. Fit the 3-parameter model (3 per-subject parameters: k, β, c_d; population: c_e_vigor, τ, p_esc, σ_motor, σ_v) via NumPyro SVI (AutoNormal guide, 40,000 steps, Adam lr = 0.002). Choice data: all 81 trials (probe trials contribute a constant P(H) = 0.5 because both options are identical). Vigor data: all 81 trials (types 1, 5, 6), with probe distances from startDistance. Cookie-type centering of excess effort using choice-trial means. The choice equation is ΔEU = 4 − k × effort(D) − β × T; c_d is excluded from the choice equation because the capture penalty C = 5 is the same for both options, making c_d's contribution to the option differential zero. The vigor equation optimizes EU(u) = S(u) × R − (1 − S(u)) × c_d × (R + C) − c_e_vigor × (u − req)² × D via softmax-weighted grid search. As a planned robustness check, we will validate the SVI approximation by refitting the model using NumPyro MCMC (NUTS sampler, 4 chains) and comparing per-subject parameter estimates.
2. Report per-subject choice r (test > 0.85) and trial-level vigor r² (test > 0.30).
3. Condition-level PPC: verify that predicted P(heavy) declines with D within each T.
4. Parameter recovery: simulate 3 datasets × 50 subjects × 81 trials from the fitted population distribution, refit, report Pearson r in log space for k, β, and c_d (test > 0.70 for each).
5. Report Pearson r between log(k) and log(β) (test |r| < 0.10) and cross-recovery k → β (test |r| < 0.15).

H3.
1. For each subject, compute overcautious rate, threat sensitivity (ΔP(heavy) between T = 0.1 and T = 0.9), and vigor gap (excess effort difference between high and low threat). Report Pearson r(log(k), overcautious rate), r(log(β), threat sensitivity), and r(log(c_d), vigor gap). All three tests: r > 0, p < 0.01.
2. Joint regression of overcaution on log(k) and log(β): report partial R² for each (test > 0.05).
3. Within-threat correlations: r(log(k), P(heavy) at T = 0.1) < -0.50, and r(log(β), P(heavy) at T = 0.9) < -0.50.

H4.
1. For each subject, compute mean confidence, P(choose heavy), choice quality (proportion of EV-optimal choices), and survival rate. Report Pearson r(confidence, P(choose heavy)), r(confidence, choice quality), and r(confidence, survival rate).
2. Test whether confidence tracks ambition (P(heavy)) but not accuracy (choice quality) or survival.

H5.
1. Compute per-subject calibration (within-subject r between anxiety rating and threat probability T) and discrepancy (mean residual from population-level anxiety ~ T regression).
2. Test orthogonality: |r(calibration, discrepancy)| < 0.15.
3. Test calibration → policy alignment: regression of calibration on policy alignment (correspondence between observed choices and model-optimal choices) controlling for k, β, and c_d, p < 0.05.
4. Test discrepancy → residual overcaution: regression of discrepancy on overcaution residual (excess conservative choice beyond model predictions) controlling for k, β, and c_d, p < 0.01.
5. Test discrepancy → anxiety symptoms: r(discrepancy, STAI-State) > 0, p < 0.01.
6. Count additional clinical measures with r(discrepancy, measure) > 0, p < 0.05 (test ≥ 2 from {OASIS, STICSA, PHQ-9, DASS-Anxiety, DASS-Stress, DASS-Depression}).
7. Report any cross-associations (calibration → clinical, discrepancy → performance) to characterize leakage in the differential prediction pattern.

H6.
1. Compute per-subject encounter reactivity (vigor change upon predator appearance within trial).
2. Report Pearson r(reactivity, log(c_d)) > 0, p < 0.05.
3. Report |r(reactivity, log(β))| < 0.15 and |r(reactivity, log(k))| < 0.15.

**Exploratory analyses (not preregistered as directional hypotheses):**
- **Exploratory E1: β and apathy.** In the combined exploratory + confirmatory sample (N ≈ 570–620), we will test whether log(β) predicts apathy (AMI) at p < .05.
- **Exploratory E2: Computational parameters and clinical symptoms.** In the combined exploratory + confirmatory sample (N ≈ 570–620), we will test whether log(k), log(β), and log(c_d) predict anxiety symptoms (STAI, OASIS, STICSA, DASS-Anxiety) and apathy (AMI) at p < .05 uncorrected. The exploratory sample showed small, inconsistent associations (|r| = 0.08–0.12) that did not survive correction for multiple comparisons, but the exploratory sample was underpowered for effects of this magnitude (50% power at r = 0.12). The combined sample provides 83% power to detect r = 0.12, resolving whether these reflect true nulls or underpowered signals.
- Bayesian ROPE analysis of model parameters vs. clinical measures (combined sample)
- Within-trial encounter dynamics: encounter reactivity as predictor of apathy (AMI)
- Cross-validated machine learning prediction of clinical outcomes
- Factor analysis of psychiatric battery and parameter-factor associations
- Behavioral profile (k × β × c_d) analysis (descriptive, supplementary)
- Simpson's paradox demonstration in unconditional vigor

**Outliers and Exclusions**
Exclusion criteria applied before any model fitting:
1. Incomplete task (did not finish all 81 trials)
2. Invalid calibration (fewer than 10 presses across calibration trials)
3. Implausible keypresses: max single-trial press rate > 3 SD above sample mean (automated input), or zero presses on > 50% of regular trials (disengagement)
4. Invalid predator dynamics: > 10% of trials with physically impossible predator behavior
5. Low engagement: overall escape rate < 35% across attack trials
6. Insufficient probes: < 80% probe completion (< 29/36). These subjects excluded from affect analyses (H1c, H4, H5) only.

No post-hoc exclusions based on model fit quality or statistical extremity.

**Sample Size**
N = 350 recruited via Prolific. Expected N ≈ 280–330 after exclusions (exploratory retention: 83.7%). The weakest powered preregistered test is H4a (r = 0.25; at N = 280, power > 0.99). At N = 280, power > 0.95 for all preregistered tests (H1–H6) based on exploratory effect sizes.

For the exploratory clinical analyses (E1–E2), the combined exploratory + confirmatory sample (N ≈ 570–620) provides substantially greater sensitivity. At N = 580: power = 83% for r = 0.12, 96% for r = 0.15. This combined approach follows recommendations for maximizing sensitivity in computational psychiatry where individual parameter–symptom effect sizes are typically small (r = 0.10–0.20; Gillan et al., 2016; Wise & Dolan, 2020). We will report clinical associations from both the confirmatory sample alone (preregistered H1–H6) and the combined sample (exploratory E1–E2).

**Other**
The exploratory sample (N = 293) was used to develop all hypotheses and specify all analysis plans. The confirmatory sample has been collected but not analyzed. The preregistration will be timestamped on AsPredicted before the confirmatory data are opened. All analysis code is available at [repository URL]; data will be shared on OSF upon acceptance.

**Name**
Three separable cost signals govern foraging under threat: effort, threat aversion, and defensive vigor

**Finally**
Experiment
