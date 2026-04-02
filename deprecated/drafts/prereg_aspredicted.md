**Hypothesis**
The overarching questions being asked are A) how do humans integrate energetic cost and predation risk when foraging, and B) does this integration manifest coherently across choice behavior, motor execution, and subjective experience?

We will test this using a task where subjects forage for food items in a virtual environment under threat of capture by a virtual predator, choosing between options that vary in reward, effort cost, and escape distance, then physically executing their foraging bout by pressing keys.

The computational framework estimates four per-subject parameters from independently fit hierarchical Bayesian models:
- **k** — effort sensitivity: how strongly effort cost deters choice (from choice model)
- **β** — threat bias: how strongly danger deters choice beyond expected value (from choice model)
- **α** — baseline vigor: tonic pressing rate above task demand (from vigor model)
- **δ** — danger mobilization: how much excess effort increases with model-derived danger, 1 − S (from vigor model)
- **S** — survival probability: S = (1 − T) + T/(1 + λD), integrating threat probability T and distance D (shared across models)

Specific hypotheses are as follows:
1. Threat will reduce high-effort choice, increase excess motor effort, and shift subjective anxiety upward and confidence downward
    a. High-effort choice will decrease with threat probability and with escape distance
    b. Excess effort (pressing rate minus chosen option's demand) will increase with threat, including within constant-demand trials
    c. Trial-level anxiety will increase and confidence will decrease with threat probability
2. Choice shift and vigor shift under threat will be coherently coupled across individuals
    a. Participants who shift choices most toward safety will show the largest increase in excess effort
    b. This coupling will remain significant when computed from independent trial halves (odd vs. even)
3. The reallocation strategy will approximate the expected-value-maximizing policy
    a. Participants who reallocate more will achieve higher foraging earnings
    b. The dominant deviation from optimal will be excessive caution rather than excessive risk-taking
4. Choices will be best explained by a model in which effort enters as an additive physical cost and survival probability follows a hyperbolic function of distance
    a. Additive effort will outperform multiplicative effort discounting
    b. A hyperbolic survival kernel will outperform an exponential kernel
    c. The model-derived survival probability S will predict trial-level anxiety (negatively) and confidence (positively) within subjects
5. Model-derived danger (1 − S) will drive excess motor effort at the population level, with meaningful individual variation in the strength of this response
    a. The population-mean danger mobilization parameter δ will be positive with a credible interval excluding zero
6. Computational parameters governing the effort-danger trade-off will covary across independently estimated models
    a. Threat bias in choice (β) will positively correlate with vigor mobilization (δ); effort sensitivity (k) will negatively correlate with δ
    b. Both couplings will have 95% posterior bootstrap credible intervals excluding zero
    c. β and δ will jointly predict closer approximation to the optimal policy
7. Participants whose motor effort is more danger-responsive (higher δ) will show more accurate subjective threat appraisal
    a. δ will predict steeper within-subject anxiety slopes on S (more negative) and confidence slopes on S (more positive)
    b. Threat bias β will show the same pattern, but effort sensitivity k will not — dissociating threat-responsive parameters from effort cost sensitivity

**Dependent variable**
Listed according to hypotheses:
1. (a) Binary choice of high- vs. low-effort cookie on each of 45 behavioral trials (trial-level). (b) Excess effort: the difference between the actual pressing rate and the effort demand of the chosen option, both in proportion-of-calibrated-capacity units (trial-level, continuous). (c) Self-reported anxiety and confidence ratings on probe trials (0–7 scale, trial-level), 18 per affect type per subject.
2. Per-subject threat-driven shift scores: Δchoice = P(choose high | T=0.9) − P(choose high | T=0.1) and Δvigor = mean excess effort at T=0.9 − mean excess effort at T=0.1.
3. (a) Total foraging earnings (sum of points across all trials) and a reallocation index = |Δchoice| + |Δvigor| (standardized). (b) Trial-level classification of suboptimal choices as "too cautious" (chose safe when risky had higher EV) or "too risky" (the reverse).
4. (a, b) Model fit measured by ELBO from stochastic variational inference (SVI) across five models (see Analysis section). (c) Fixed-effect coefficient from mixed-effects linear regression of probe ratings on model-derived survival probability S_probe (z-scored).
5. Subject-level posterior mean δ_i from a hierarchical Bayesian model of excess effort, and the population-level posterior for μ_δ.
6. Pearson correlations between posterior means of choice model parameters (log(β), log(k)) and vigor model parameter (δ), estimated from independently fit models.
7. Per-subject within-subject regression slopes of anxiety (and confidence) on S_probe, correlated with δ, β, and k.

**Conditions**
One experiment with a within-subjects design. Each subject completes 81 trials (3 blocks × 27 trials) in a virtual foraging task. Conditions are fully crossed within subject:
- **Threat probability:** T ∈ {0.1, 0.5, 0.9} — the probability of a predator attack on each trial
- **Distance:** D ∈ {1, 2, 3} corresponding to 5, 7, 9 game units from the safe zone — the distance of the high-effort cookie
- **Effort demand:** E_H ∈ {0.6, 0.8, 1.0} of calibrated pressing capacity — the effort required for the high-effort cookie

The low-effort option is fixed (E_L = 0.4, D_L = 1, R_L = 1 point). The high-effort option yields R_H = 5 points. Capture penalty is C = 5 points + loss of current cookie reward.

Each block contains 15 regular behavioral trials and 12 probe trials (6 anxiety, 6 confidence). Probe trials use forced-choice (identical options) and collect ratings before pressing begins. A psychiatric battery (DASS-21, PHQ-9, OASIS, STAI-T, AMI, MFIS, STICSA) is administered between blocks.

**Analyses**
H1.
1. Logistic mixed-effects model on trial-level binary choice: `choice ~ threat_z + dist_z + threat_z:dist_z + (1 | subject)`. Tests: β(threat) < 0 (threat reduces high-effort choice), β(dist) < 0 (distance reduces high-effort choice), β(threat × dist) < 0 (threat amplifies the distance effect). All three coefficients must be significant at p < 0.01. Additionally, monotonicity of subject-mean P(choose high) across all adjacent threat levels within each distance: P(T=0.1) > P(T=0.5) > P(T=0.9) for each D, confirmed by paired t-tests on subject means (all p < 0.01, one-tailed).
2. Linear mixed-effects model on trial-level excess effort: `excess_effort ~ threat_z * dist_z + effort_chosen_z + (1 | subject)`. Excess effort is the difference between the actual pressing rate and the effort demand of the chosen option (both in proportion-of-capacity units). Including effort_chosen_z as a covariate controls for the composition effect (subjects choosing easier options under high threat). Tests: β(threat) > 0, p < 0.05 (threat increases pressing beyond demand); β(threat × dist) < 0, p < 0.05 (the threat-driven excess effort boost diminishes at farther distances, where sustained high-rate execution approaches a physical ceiling).
3. Linear mixed-effects models on trial-level probe ratings: `anxiety ~ threat_z + dist_z + (1 + threat_z | subject)` and `confidence ~ threat_z + dist_z + (1 + threat_z | subject)`. Tests: β(threat) > 0 for anxiety, β(threat) < 0 for confidence, both at p < 0.001. The logistic GLMM (H1a) is estimated via variational Bayes (Laplace approximation) in statsmodels; linear LMMs (H1b, H1c) are estimated with REML.

H2.
1. Pearson r(Δchoice, Δvigor) < 0, one-tailed, p < 0.01.
2. Split-half robustness: Pearson r(Δchoice_odd, Δvigor_even) < 0, p < 0.05 one-tailed. Same for reversed split.

H3.
1. Pearson r(reallocation_index, total_earnings) > 0, one-tailed, p < 0.01.
2. Among suboptimal trials, the proportion classified as "too cautious" exceeds 50%. One-sample t-test, p < 0.05.

H4.
1. We will fit 5 computational models to subjects' choices using stochastic variational inference (SVI) in NumPyro: (M1) effort-only: SV = R·exp(−kE), (M2) linear threat features: SV = R·exp(−kE) − β·T·D, (M3) exponential survival + multiplicative effort: SV = R·exp(−kE)·S − β·(1−S) with S = exp(−λTD), (M4) hyperbolic survival + multiplicative effort: SV = R·exp(−kE)·S − β·(1−S) with S = (1−T)+T/(1+λD), (M5) hyperbolic survival + additive effort: SV = R·S − k·E − β·(1−S) with S = (1−T)+T/(1+λD). All models use option-specific S and per-subject k_i, β_i with log-normal hierarchical priors. Model fit assessed by ELBO. Our primary hypothesis is that M5 outperforms M4 (ΔELBO > 0, testing additive vs. multiplicative effort) and M4 outperforms M3 (ΔELBO > 0, testing hyperbolic vs. exponential survival). MCMC (NumPyro NUTS, 4 chains × 1000 warmup + 1000 samples) will also be run; MCMC results are primary if convergence criteria are met (Rhat ≤ 1.05, ESS ≥ 100).
2. Mixed-effects linear regression: anxiety ~ S_probe_z + (1 + S_probe_z | subject). Test: β(S_probe_z) < 0, |t| > 3.0. Same for confidence (β > 0). S_probe uses only population-level λ from the choice model.

H5.
1. A hierarchical Bayesian model: excess_ij = α_i + δ_i · (1 − S_ij) + ε_ij, with hierarchical Normal priors on α_i and δ_i, fit via NumPyro NUTS or SVI. S_ij uses λ from the choice model. Test: P(μ_δ > 0 | data) > 0.975, and proportion of subjects with δ_i > 0 exceeds 80%.

H6.
1. Pearson r(log(β), δ) > 0, p < 0.001 one-tailed. Pearson r(log(k), δ) < 0, p < 0.01 one-tailed. Parameters from independently fit choice and vigor models.
2. A joint hierarchical model with correlated random effects [log(k_i), log(β_i), α_i, δ_i] ~ MVN(μ, Σ), Ω ~ LKJCholesky(η=2), λ fixed from choice model. Test: ρ(β,δ) posterior 95% CI > 0 and ρ(k,δ) posterior 95% CI < 0.
3. OLS regression: optimality_index ~ β_z + δ_z + k_z. Both β and δ must be significant predictors (p < 0.05).

H7.
1. For each subject, compute the within-subject regression slope of anxiety (and confidence) on S_probe_z. Test: Pearson r(δ, anxiety_slope) < 0 and r(δ, confidence_slope) > 0, p < 0.05 one-tailed.
2. Pearson r(β, anxiety_slope) < 0, p < 0.05 one-tailed. Pearson r(k, anxiety_slope) must be non-significant (p > 0.05, two-tailed).

**Outliers and Exclusions**
Exclusion criteria applied before any model fitting:
1. Incomplete task (did not finish all 81 trials)
2. Invalid calibration (fewer than 10 presses across calibration trials)
3. Implausible keypresses: max single-trial press rate > 3 SD above sample mean (automated input), or zero presses on > 50% of regular trials (disengagement)
4. Invalid predator dynamics: > 10% of trials with physically impossible predator behavior
5. Low engagement: overall escape rate < 35% across attack trials
6. Insufficient probes: < 80% probe completion (< 29/36). These subjects excluded from affect analyses (H4c, H7) only.

No post-hoc exclusions based on model fit quality or statistical extremity.

**Sample Size**
N = 350 recruited via Prolific. Expected N ≈ 280–330 after exclusions (exploratory retention: 83.7%). The weakest powered test is the choice-vigor coupling (H2a, moderate r ≈ 0.2–0.3); detecting r = 0.20 at α = 0.01 one-tailed requires N > 200 (power > 0.95). All other tests are based on very large exploratory effects (choice: d > 1.5; affect LMM: t > 25; vigor: P(δ > 0) = 1.0; parameter coupling: r > 0.33).

**Other**
The exploratory sample (N = 293) was used to develop all hypotheses and specify all analysis plans. The confirmatory sample has been collected but not analyzed. The preregistration will be timestamped on AsPredicted before the confirmatory data are opened. All analysis code will be shared on GitHub and data on OSF upon acceptance.

**Name**
Effort reallocation under threat in continuous foraging

**Finally**
Experiment
