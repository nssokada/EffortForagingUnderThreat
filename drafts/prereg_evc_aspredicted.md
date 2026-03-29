**Hypothesis**
The overarching questions being asked are A) how do humans integrate energetic cost and predation risk when foraging, and B) does the affective relationship between computed danger and experienced anxiety predict anxiety symptoms?

We test this using a task where subjects forage for food items in a virtual environment under threat of capture by a virtual predator, choosing between options that vary in reward, effort cost, and escape distance, then physically executing their foraging bout by pressing keys. On a subset of trials (probe trials), subjects report their subjective anxiety and confidence.

The computational framework estimates two per-subject parameters from a joint hierarchical Bayesian EVC model with LQR-inspired cost structure:
- **c_e** — effort cost: how strongly effort/distance deters choice (identified from choice data)
- **c_d** — capture aversion: how strongly the survival incentive drives motor vigor (identified from vigor data)
- **S** — survival probability: S = (1 − T^γ) + ε × T^γ × p_esc, where γ (probability weighting) and ε (effort efficacy) are population-level parameters shared across subjects

The model uses a cost structure inspired by linear-quadratic optimal control (LQR), where effort enters choice as a commitment cost (req² × D) and vigor as a deviation cost ((u − req)² × D). This is an analogy to LQR theory, not a formal implementation: the model has no state dynamics, no feedback law, and no Riccati equation.

Specific hypotheses are as follows:
1. Threat will reduce high-effort choice, increase excess motor effort, and shift subjective anxiety upward and confidence downward
    a. High-effort choice will decrease with threat probability and with escape distance
    b. Excess effort (pressing rate minus chosen option's demand), conditioned on chosen cookie type, will increase with threat probability
    c. Trial-level anxiety will increase and confidence will decrease with model-derived survival probability S
2. The EVC model with LQR-inspired cost structure will jointly capture choice and vigor with two recoverable per-subject parameters
    a. Per-subject choice predictions will correlate r > 0.85 with observed choice proportions
    b. Trial-level vigor predictions will explain r² > 0.30 of excess effort variance
    c. The model will capture distance gradients in choice: predicted P(choose heavy) will decrease with distance within each threat level
    d. Parameter recovery will yield correlations r > 0.70 for both log(c_e) and log(c_d) in simulation
    e. log(c_e) and log(c_d) will be approximately independent: |r| < 0.25
3. Between-subject confidence will correlate with foraging ambition but not with task performance
    a. Between-subject mean confidence will correlate positively with P(choose heavy), r > 0, p < 0.05
    b. Between-subject mean confidence will NOT correlate with choice quality (proportion of EV-optimal choices), |r| < 0.10
    c. Between-subject mean confidence will NOT correlate with survival rate, |r| < 0.10
4. Affective calibration and discrepancy will show a differential prediction pattern for performance versus anxiety symptoms
    a. Calibration (within-subject r between anxiety and model-derived danger 1−S) and discrepancy (mean anxiety residual after removing the S-anxiety relationship) will be approximately orthogonal: |r| < 0.15
    b. Calibration will predict task performance: r(calibration, choice quality) > 0 OR r(calibration, survival rate) > 0, p < 0.05
    c. Discrepancy will predict anxiety symptoms: r(discrepancy, STAI-State) > 0, p < 0.01
    d. Discrepancy will predict at least two additional symptom measures from {OASIS, STICSA, PHQ-9, DASS-Anxiety, DASS-Stress, DASS-Depression} at p < 0.05 uncorrected

**Dependent variable**
Listed according to hypotheses:
1. (a) Binary choice of high- vs. low-effort cookie on each of 45 behavioral trials (trial-level). (b) Excess effort: the difference between the actual pressing rate and the effort demand of the chosen option, both in proportion-of-calibrated-capacity units, centered by cookie type (trial-level, continuous). (c) Self-reported anxiety and confidence ratings on probe trials (0–7 scale, trial-level), 18 per affect type per subject.
2. (a–b) Per-subject predicted versus observed choice proportion and trial-level excess effort, from the jointly fitted EVC model with LQR-inspired cost structure. All 81 trials enter both likelihoods (probe trials contribute P(H)=0.5 to choice). (c) Condition-level (threat × distance) predicted versus observed P(choose heavy). (d) Simulated-and-recovered parameter correlations in log space. (e) Pearson r between log(c_e) and log(c_d).
3. (a) Per-subject Pearson r between mean confidence and P(choose heavy). (b) Per-subject Pearson r between mean confidence and choice quality (proportion of EV-optimal choices). (c) Per-subject Pearson r between mean confidence and survival rate (1 − capture rate).
4. (a) Pearson r between calibration and discrepancy. (b) Pearson r between calibration and choice quality or survival rate. (c–d) Pearson r between discrepancy and z-scored psychiatric subscale scores.

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
3. Linear mixed-effects models on probe ratings: `anxiety ~ S_z + (1 + S_z | subject)` and `confidence ~ S_z + (1 + S_z | subject)`, where S is the EVC model's survival probability. Tests: β(S) < 0 for anxiety and β(S) > 0 for confidence, both |t| > 3.0.

H2.
1. Fit the EVC model with LQR-inspired cost structure (2 per-subject parameters: c_e, c_d; population: γ, ε, c_e_vigor, τ, p_esc, σ_motor, σ_v) via NumPyro SVI (AutoNormal guide, 40,000 steps, Adam lr=0.002). Choice data: all 81 trials (probe trials contribute a constant P(H)=0.5 because both options are identical). Vigor data: all 81 trials (types 1, 5, 6), with probe distances from startDistance. Cookie-type centering of excess effort using choice-trial means. The choice equation is ΔEU = S × 4 − c_e × (0.81 × D_H − 0.16); c_d is excluded from the choice equation because its contribution to the option differential is collinear with the reward term (both scale with R_H − R_L), making c_d empirically unidentifiable from choice data. The vigor equation optimizes EU(u) = S(u) × R − (1 − S(u)) × c_d × (R + C) − c_e_vigor × (u − req)² × D via softmax-weighted grid search. As a planned robustness check, we will validate the SVI approximation by refitting the model using NumPyro MCMC (NUTS sampler, 4 chains) and comparing per-subject parameter estimates.
2. Report per-subject choice r² (test > 0.85) and trial-level vigor r² (test > 0.30).
3. Condition-level PPC: verify that predicted P(heavy) declines with D within each T.
4. Parameter recovery: simulate 3 datasets × 50 subjects × 81 trials from the fitted population distribution, refit, report Pearson r in log space for c_e and c_d (test > 0.70).
5. Report Pearson r between log(c_e) and log(c_d) (test |r| < 0.25).

H3.
1. For each subject, compute mean confidence, P(choose heavy), choice quality (proportion of EV-optimal choices), and survival rate. Report Pearson r(confidence, P(choose heavy)), r(confidence, choice quality), and r(confidence, survival rate).
2. Test whether confidence tracks ambition (P(heavy)) but not accuracy (choice quality) or survival.

H4.
1. Compute per-subject calibration (within-subject r between anxiety rating and 1−S) and discrepancy (mean residual from population-level anxiety~S regression).
2. Test orthogonality: |r(calibration, discrepancy)| < 0.15.
3. Test calibration → performance (differential prediction: calibration predicts performance): r(calibration, choice quality) or r(calibration, survival), p < 0.05.
4. Test discrepancy → anxiety symptoms (differential prediction: discrepancy predicts symptoms): r(discrepancy, STAI-State) > 0, p < 0.01.
5. Count additional clinical measures with r(discrepancy, measure) > 0, p < 0.05 (test ≥ 2 from {OASIS, STICSA, PHQ-9, DASS-Anxiety, DASS-Stress, DASS-Depression}).
6. Report any cross-associations (calibration → clinical, discrepancy → performance) to characterize leakage in the differential prediction pattern.

**Exploratory analyses (not preregistered):**
- Bayesian ROPE analysis of model parameters vs. clinical measures
- Cross-validated machine learning prediction of clinical outcomes
- Factor analysis of psychiatric battery and parameter-factor associations
- Behavioral profile (c_e × c_d quadrant) analysis (descriptive, supplementary)
- Simpson's paradox demonstration in unconditional vigor
- Optimality and Pareto analysis
- Anxiety sensitization across blocks

**Outliers and Exclusions**
Exclusion criteria applied before any model fitting:
1. Incomplete task (did not finish all 81 trials)
2. Invalid calibration (fewer than 10 presses across calibration trials)
3. Implausible keypresses: max single-trial press rate > 3 SD above sample mean (automated input), or zero presses on > 50% of regular trials (disengagement)
4. Invalid predator dynamics: > 10% of trials with physically impossible predator behavior
5. Low engagement: overall escape rate < 35% across attack trials
6. Insufficient probes: < 80% probe completion (< 29/36). These subjects excluded from affect analyses (H1c, H3, H4) only.

No post-hoc exclusions based on model fit quality or statistical extremity.

**Sample Size**
N = 350 recruited via Prolific. Expected N ≈ 280–330 after exclusions (exploratory retention: 83.7%). The weakest powered test is H3c (Steiger's test for dependent correlations; exploratory z = 3.14). At N = 280, power > 0.95 for all preregistered tests based on exploratory effect sizes.

**Other**
The exploratory sample (N = 293) was used to develop all hypotheses and specify all analysis plans. The confirmatory sample has been collected but not analyzed. The preregistration will be timestamped on AsPredicted before the confirmatory data are opened. All analysis code is available at [repository URL]; data will be shared on OSF upon acceptance.

**Name**
Metacognitive bias, not threat computation, bridges foraging decisions to anxiety symptoms

**Finally**
Experiment
