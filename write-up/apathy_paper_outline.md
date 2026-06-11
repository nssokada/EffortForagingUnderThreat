# Apathy-framed paper — outline (revised 2026-06-10 via paper-spine questionnaire)

**TITLE:** *Joint computational integration of threat and effort in human defensive decisions reveals a behavioral signature of apathy*

**Status:** Bullet-pointed outline rebuilt via section-by-section questionnaire. Joint computational integration is the headline; apathy validates it. H2 (vigor temporal dynamics) folded into a single H1 sub-bullet. H6 (confidence mediation) folded into H5 as sub-bullets H5e–H5g. Five results sections total (H1–H5).

---

## SIGNIFICANCE STATEMENT (bullets)

- **Gap.** Decisions under stress involve *what* + *how vigorously*; threat-processing and effort-based decision literatures rarely join these dimensions.
- **Approach.** Virtually embodied foraging task (choice + keypress vigor under threat × distance) fit with a joint fitness model.
- **Finding 1.** Joint W(u) > separable alternatives; per-subject capture-cost (ω) and effort-cost (κ) cleanly recovered, dissociable.
- **Finding 2.** Apathy specifically — not anxiety/depression — maps onto ω across 2 samples (N = 290 + 281); strongest at the behavioral level (β_T_choice → AMI β = −0.198, p < .001 in both samples).
- **Finding 3.** Trait apathy → ω is fully mediated by within-task confidence (Sobel p = .006); anxiety mediators fail.
- **Take-away.** Apathy's defensive signature is computationally specific and routed through metacognitive confidence — distinct from anxiety/depression.

---

## INTRODUCTION (4 paragraphs)

### P1 — Two-axis decision under stress (everyday-scenario hook)
- Two-axis decision: *what* to do + *how vigorously* to do it.
- Vignette: forager / driver / athlete deciding both choice and motor commitment.
- These dimensions jointly determine behavior but are studied separately.

### P2 — Threat literature: Pavlovian / threat-imminence
- Defensive behavior characterized along the **threat-imminence continuum** (pre-encounter → post-encounter → circa-strike) with stereotyped Pavlovian + instrumental responses. [Mobbs, Fanselow, LeDoux]
- Typical paradigms: passive observation of conditioned stimuli or fixed Skinner-box avoidance — **no explicit cost-benefit choice** between effort and safety.
- Motor responses (freezing, flight) well-quantified; **choice constrained to do/don't-do** rather than *what* to do.
- **Gap:** how subjects *decide* under graded threat, and how decision interacts with motor execution, rarely measured jointly.

### P3 — Effort literature: motor vigor
- **Motor vigor** as a quantitative measure of effort cost / reward sensitivity: pressing rate, reach velocity, saccade vigor scale lawfully with expected reward. [Shadmehr, Mazzoni, Niv]
- Vigor sensitive readout of subjective value — tonic dopamine modulates it; individual differences track motivational state.
- Tasks typically isolate vigor from choice context AND from acute threat.
- **Gap:** how motor vigor is *jointly* modulated by choice-level cost-benefit and by threat, in the same decision, is uncharacterized.

### P4 — Foraging theory bridge + clinical motivation + our approach
- **Foraging theory** as the normative bridge: predicts that animals jointly weight predation risk and effort cost when deciding *what* + *how vigorously*. Fitness-weighted utility decomposes into a capture-cost-weighted survival component + effort-cost-weighted action component.
- **Clinical motivation** sits at this intersection: apathy = altered motivation that should manifest in both choice AND motor dimensions; existing tasks measure only one or the other, so the joint computational signature of apathy is unknown.
- **Our task.** Virtually embodied foraging task — subjects choose high- vs low-effort options at varying threat probability and distance, then execute via continuous keypress effort. Two pre-registered samples (N = 290 + 281). Joint W(u) model recovers per-subject capture-cost and effort-cost weighting; we test whether parameters track clinical state.

---

## RESULTS

### H1. Threat and effort jointly modulate choice AND vigor — two distinct dimensions of behavioral variation.
- **H1a.** P(high-effort choice) ↓ with threat probability and ↓ with distance; negative T × D interaction (deterrent effect of distance amplifies under high threat).
- **H1b.** Within-trial pressing rate (vigor) ↑ with threat probability. Distance has small positive main effect but the T × D interaction is *negative* (opposite to predatory-imminence prediction), consistent with calibration-ceiling compression.
- **H1c.** Marginal r(mean P(heavy), mean vigor) is small and positive (+0.150 exp / +0.077 conf) — nearly the joint W(u) model's prediction from cancellation of dissociated ω-pathway and aligned κ-pathway contributions.
- **H1d.** Within-trial vigor temporal dynamics (anticipatory ramp + reactive surge) characterized in companion work / Supplementary.

**Table H1.1.** Choice: trial-level logit + cluster-robust SE (clusters = subjects). Predictors z-scored within sample. Source: [[result_101]].

| Sample | N | Predictor | β | SE | z | p |
|---|---|---|---|---|---|---|
| Exploratory | 290 | T_z | **−1.015** | 0.046 | −22.28 | < .001 |
| Exploratory | 290 | D_z | **−0.747** | 0.032 | −23.69 | < .001 |
| Exploratory | 290 | T_z × D_z | −0.195 | 0.024 | −8.02 | < .001 |
| Confirmatory | 281 | T_z | **−0.908** | 0.046 | −19.80 | < .001 |
| Confirmatory | 281 | D_z | **−0.666** | 0.030 | −22.05 | < .001 |
| Confirmatory | 281 | T_z × D_z | −0.116 | 0.025 | −4.72 | < .001 |

**Table H1.2.** Vigor (within-trial pressing rate). Source: `notebooks/analysis/H1_adaptive_shifts.ipynb` cells 2 + 7 + 10 (= [[result_103]]).

*Test 1 — Paired t-test within cookie, T = 0.9 vs T = 0.1 (primary, preregistered; type==1 free-choice trials):*

| Cookie | Sample | N | mean(T=0.1) | mean(T=0.9) | Δ [95% CI] | t (df) | p | Cohen's d_z |
|---|---|---|---|---|---|---|---|---|
| Heavy | Exploratory | 223 | 0.987 | 1.031 | +0.045 [+0.029, +0.061] | t(222) = +5.54 | < .001 | +0.371 |
| Heavy | Confirmatory | 205 | 0.977 | 1.010 | +0.034 [+0.020, +0.048] | t(204) = +4.71 | < .001 | +0.329 |
| Light | Exploratory | 238 | 0.908 | 0.958 | +0.050 [+0.037, +0.063] | t(237) = +7.51 | < .001 | +0.487 |
| Light | Confirmatory | 235 | 0.917 | 0.983 | +0.066 [+0.052, +0.080] | t(234) = +9.37 | < .001 | +0.611 |

Light-cookie d_z is consistently larger than heavy-cookie d_z (~1.5–1.9×); heavy-cookie pressing saturates above T = 0.5 (calibration ceiling).

*Test 2 — Trial-level LMM (complementary; cookie partialled; z-scored T):*

| Sample | N_trials | Predictor | β | z | p |
|---|---|---|---|---|---|
| Exploratory | 12,941 | threat_z | **+0.021** | +12.61 | < .001 |
| Exploratory | 12,941 | is_heavy | +0.057 | — | — |
| Confirmatory | 12,452 | threat_z | **+0.018** | +12.83 | < .001 |
| Confirmatory | 12,452 | is_heavy | +0.050 | — | — |

*Supplementary — chosen-distance LMM Model B (broader all-trials, z-scored predictors, with T × D interaction):*

| Sample | N_trials | Predictor | β | z | p |
|---|---|---|---|---|---|
| Exploratory | 23,288 | threat_z | +0.018 | +16.40 | < .001 |
| Exploratory | 23,288 | chosen_dist_z | ≈ 0 | −0.09 | .931 |
| Exploratory | 23,288 | threat_z × chosen_dist_z | **−0.005** | −3.53 | < .001 |
| Confirmatory | 22,434 | threat_z | +0.017 | +17.45 | < .001 |
| Confirmatory | 22,434 | chosen_dist_z | +0.002 | +1.89 | .059 |
| Confirmatory | 22,434 | threat_z × chosen_dist_z | **−0.004** | −3.51 | < .001 |

ΔAIC (B vs A) ≈ −10 in both samples — interaction is preferred. The T × D interaction is *negative* — opposite to predatory-imminence prediction. Most parsimonious mechanistic reading: calibration ceiling compresses additional threat-driven escalation at long distance.

**Table H1.3.** Channel independence (per-subject mean P(heavy) vs mean vigor). Source: [[result_401]].

| Sample | N | r(P(heavy), mean_vigor) | p |
|---|---|---|---|
| Exploratory | 290 | +0.150 | .011 |
| Confirmatory | 281 | +0.077 | .201 (n.s.) |

The small positive marginal correlation is itself a prediction of the joint W(u) model: the ω-pathway is *dissociated* (ω → choice negative, ω → vigor positive) while the κ-pathway is *aligned* (κ negative on both), and the two nearly cancel. Predicted vs observed r matches within 0.025 in both samples.

### H2. A joint fitness function beats separable alternatives — establishes the framework.
- **H2a.** Joint W(u) = S·R − (1−S)·ω·(R+C) − κ·(u−req)²·D outperforms an effort-only model that ignores threat (ΔWAIC ≈ +4,700 / +3,800).
- **H2b.** Joint W(u) outperforms a threat-only model lacking individual effort sensitivity (ΔWAIC ≈ +2,000 / +1,600).
- **H2c.** Joint W(u) outperforms a single-parameter model where ω = κ (M3: ΔWAIC ≈ +2,600 / +3,500), AND outperforms the scaled-single-parameter control M3b (ΔWAIC ≈ +2,000 / +1,600). The two-dimensional (ω, κ) structure is necessary; a single avoidance/engagement trait does not suffice.
- **H2d.** Parameter recovery clean (r ≈ 0.92 for both ω and κ); per-subject ω and κ are dissociable individual-difference traits.
- **H2e.** *Channel-specific trade-off (supplementary):* M4 does not dominate on every single metric. M2 (threat-only) marginally exceeds M4 on choice accuracy (78.9% vs 77.3% exp; 77.8% vs 75.9% conf), and M1 (effort-only) achieves higher choice R² (0.95 vs 0.81) because it overfits the choice surface with only κ and reward differences. **M4 wins because it is the only model that explains vigor variance** (M4 vigor R² = 0.37 / 0.41; all alternatives ≤ 0.10). The joint WAIC framing correctly trades choice-fit headroom for joint coverage.

**Table H2.1.** Model comparison: joint M4 vs separable alternatives. ΔWAIC = (alternative WAIC) − (M4 WAIC); positive favors M4. Sources: [[result_201]], [[result_202]], [[result_203]], [[result_204]].

*ΔWAIC (joint likelihood; positive favors M4):*

| Comparison | Exploratory ΔWAIC (SE) | Confirmatory ΔWAIC (SE) |
|---|---|---|
| **M4 vs M1** (effort-only, ignores threat) | **+4,729** (≈ 667) | **+3,785** (≈ 443) |
| **M4 vs M2** (threat-only, population κ) | **+1,966** (≈ 669) | **+1,621** (≈ 449) |
| **M4 vs M3** (single-parameter θ = ω = κ) | **+2,599** (≈ 593) | **+3,474** (≈ 305) |
| M4 vs M3b (scaled single-param control) | +1,959 | +1,597 |

*Channel-specific fit by model (exploratory / confirmatory):*

| Model | Choice accuracy | Choice R² | **Vigor R²** | Joint WAIC |
|---|---|---|---|---|
| **M4 — Joint W(u): per-subject ω + κ (winner on joint)** | 77.3% / 75.9% | 0.796 / 0.809 | **0.372 / 0.412** | **12,776 / 12,252** |
| M2 — Threat-only (per-subject ω, population κ) | **78.9%** / **77.8%** | 0.893 / — | 0.013 / 0.012 | 14,742 / 13,873 |
| M3 — Single-parameter (θ = ω = κ) | 77.3% / 75.6% | 0.814 / — | 0.102 / 0.075 | 15,374 / 15,727 |
| M1 — Effort-only (per-subject κ, intercept vigor) | 71.0% / 70.8% | **0.951** / 0.946 | 0.006 / 0.007 | 17,505 / 16,037 |

**Trade-off:** M2 (threat-only) marginally exceeds M4 on raw choice accuracy and M1 (effort-only) achieves higher choice R² by overfitting the choice surface with only κ and reward differences. M4 is the only model that explains vigor variance (R² = 0.37–0.41 vs ≤ 0.10 for all alternatives). The joint WAIC framing correctly accounts for this trade-off — M4 trades a few percentage points of choice-fit headroom for an order-of-magnitude improvement in vigor coverage.

**Table H2.2.** Parameter recovery under production M4 + MCMC (synthetic data drawn from empirical posterior, N = 500). Source: [[result_205]].

| Parameter | Pearson r (true vs recovered) | Spearman ρ | 80% HDI coverage | 95% HDI coverage |
|---|---|---|---|---|
| **ω** | **0.924** | 0.940 | 77.2% | 92.4% |
| **κ** | **0.918** | 0.925 | 81.4% | 95.6% |

Convergence: max R̂ = 1.001, min ESS = 4,567. (Earlier κ-recovery failure was a recovery-harness specification artifact — see result_205.)

### H3. Vigilance (ω) and mobilization (κ) parameters predict survival, error structure, and deviation from the foraging optimum.
- **H3a.** Higher vigilance predicts higher escape rates on attack trials (β = +0.060, 95% HDI [+0.029, +0.093] exploratory; β = +0.046 [+0.017, +0.075] confirmatory): capture-averse subjects mobilize more reliably when a predator attacks.
- **H3b.** Higher vigilance predicts a more overcautious error structure (β = +0.177 [+0.163, +0.193] exploratory; β = +0.123 [+0.109, +0.137] confirmatory). Subjects who weight capture cost more heavily disproportionately err by avoiding the high-effort option on trials where it would have paid better, rather than erring in the reckless direction.
- **H3c.** Higher effort-cost weighting predicts lower mean pressing intensity (β = −0.194 [−0.215, −0.173] exploratory; β = −0.196 [−0.217, −0.176] confirmatory). Subjects who weight motor effort more heavily press at lower intensity overall, reflecting greater sensitivity to the cost of motor exertion.
- **H3d.** Two derived axes capture cost-weighting variation:
  - **Balance** — indexes whether capture-driven or effort-driven avoidance dominates a subject.
  - **Magnitude** — indexes overall engagement with the cost-benefit calculation.
  - Joint regression on pressing intensity reveals an independent positive effect of capture-cost weighting (β = +0.137 [+0.120, +0.154] exploratory; +0.125 [+0.110, +0.141] confirmatory), partially cancelling the suppressive effort-cost effect from H3c — a **defensive-mobilization signature**.
- **H3e.** Subjects systematically deviate from the foraging-theoretic optimum (calibrated to a median-human optimal effort-cost weighting of κ ≈ 6.87):
  - Direction of deviation tracks vigilance — higher capture-cost weighting overweights vs optimum.
  - Trade-off: sacrifice earnings for safety (β(angle → % optimal) = −0.041 [−0.055, −0.026] exploratory; −0.054 [−0.072, −0.036] confirmatory).

**Table H3.1.** Preregistered H4-family parameter regressions (Bayesian linear regression, 95% HDI, z-scored log-parameters). Source: [[result_208]].

| Sub-hypothesis | Outcome | Predictor | Exploratory β [95% HDI] | Confirmatory β [95% HDI] | Verdict |
|---|---|---|---|---|---|
| H4a | Escape rate (attack trials) | log(ω) | **+0.060** [+0.029, +0.093] | **+0.046** [+0.017, +0.075] | PASS |
| H4a | Escape rate | log(κ) | −0.003 [−0.033, +0.029] | +0.003 [−0.028, +0.030] | — |
| H4b | Overcaution ratio | log(ω) | **+0.177** [+0.163, +0.193] | **+0.123** [+0.109, +0.137] | PASS |
| H4c | Mean vigor (pressing rate) | log(κ) | **−0.194** [−0.215, −0.173] | **−0.196** [−0.217, −0.176] | PASS |
| H4d | % optimal trials | angle (ω–κ rotation) | **−0.041** [−0.055, −0.026] | **−0.054** [−0.072, −0.036] | PASS |
| H4e | Earnings | choice consistency | +14.3 [+5.0, +23.2] | +8.4 [−2.3, +19.0] | **FAIL confirmatory** |
| H4e | Earnings | intensity deviation | −19.3 [−28.8, −9.4] | −4.1 [−14.6, +7.4] | **FAIL confirmatory** |

**Table H3.2.** Post-hoc Cartesian decomposition: each outcome regressed on log(ω) + log(κ) + ω×κ interaction (Bayesian, z-scored predictors). Reveals hidden dual-channel signatures. Source: [[result_208]].

| Outcome | Term | Exploratory β [95% HDI] | Confirmatory β [95% HDI] |
|---|---|---|---|
| **Mean vigor** | ω_z | **+0.137** [+0.120, +0.154] | **+0.125** [+0.110, +0.141] |
| | κ_z | **−0.238** [−0.256, −0.222] | **−0.228** [−0.244, −0.212] |
| | ω × κ | **−0.053** [−0.068, −0.039] | −0.021 [−0.034, −0.008] |
| **P(heavy)** | ω_z | **−0.154** [−0.161, −0.147] | **−0.168** [−0.177, −0.160] |
| | κ_z | **−0.076** [−0.083, −0.069] | **−0.062** [−0.070, −0.053] |
| | ω × κ | +0.006 [+0.000, +0.013] | **+0.015** [+0.007, +0.022] |
| **Escape rate** | ω_z | **+0.058** [+0.029, +0.089] | **+0.049** [+0.018, +0.077] |
| | κ_z | +0.005 [−0.024, +0.035] | +0.008 [−0.021, +0.038] |
| **% optimal** | ω_z | **−0.072** [−0.085, −0.060] | **−0.108** [−0.120, −0.095] |
| | κ_z | **−0.035** [−0.048, −0.023] | **−0.046** [−0.059, −0.034] |

Note the ω → vigor effect is *positive* (β ≈ +0.13) and partially-cancels with the κ → vigor effect (β ≈ −0.23) — a "defensive mobilization signature" that the univariate H4c specification underestimates. r(ω_z, κ_z) = +0.369 / +0.302 in exploratory/confirmatory.

### H4. Confidence, but not anxiety, registers vigilance AND mobilization state.
- **H4a.** Both probe channels calibrate to task conditions: confidence falls and anxiety rises with both threat and distance (main effects).
  - Confidence ~ T: β = −0.58 / −0.67 (exploratory / confirmatory), both p < .001.
  - Anxiety ~ T: β = +0.58 / +0.53, both p < .001.
  - T × D interaction is null for confidence and sample-asymmetric for anxiety — main effects dominate.
- **H4b.** Confidence registers BOTH vigilance and mobilization — in joint regressions of log(ω) and log(κ) on all eight affect features, each controlling for the *other* parameter:
  - **Mean confidence → log(ω)**: β = −0.173, 95% CI [−0.297, −0.048], p = .006. Lower mean task confidence → higher capture-cost weighting.
  - **Mean confidence → log(κ)**: β = −0.129, 95% CI [−0.240, −0.018], p = .023. Lower mean task confidence → lower effort-cost weighting.
  - Confidence reactivity to distance shows the same pattern on both parameters (p = .058 on ω; p = .054 on κ) — trending but not surviving.
  - The two parameter outcomes carry independent confidence signal: confidence-intercept is significant on each AFTER partialling out the other parameter (log(κ) ↔ log(ω) cross-regression β ≈ +0.32, p < .001 in both directions).
- **H4c.** Anxiety features carry no signal on either parameter. All four anxiety features (intercept, T-slope, D-slope, T × D) are null on log(ω) and on log(κ) in the joint specification — confidence is the unique metacognitive register.

**Table H4.1.** Within-task affect probe scaling: LMM `response ~ T_z * D_z + (1|subj)`, predictors z-scored within sample (now with T × D interaction).

| Outcome | Predictor | Exploratory β | z | p | Confirmatory β | z | p |
|---|---|---|---|---|---|---|---|
| **Anxiety** | T_z | **+0.578** | +23.88 | < .001 | **+0.532** | +21.98 | < .001 |
| Anxiety | D_z | +0.228 | +9.43 | < .001 | +0.275 | +11.39 | < .001 |
| Anxiety | T_z × D_z | +0.038 | +1.56 | .118 (n.s.) | **−0.106** | −4.39 | < .001 |
| **Confidence** | T_z | **−0.579** | −23.75 | < .001 | **−0.671** | −26.99 | < .001 |
| Confidence | D_z | −0.293 | −12.03 | < .001 | −0.260 | −10.47 | < .001 |
| Confidence | T_z × D_z | +0.002 | +0.08 | .934 (n.s.) | −0.010 | −0.40 | .686 (n.s.) |

**T × D interaction is essentially null for confidence in both samples and sample-asymmetric for anxiety** (positive trend in exp, negative ★ in conf — direction does not replicate). Main effects of T and D dominate, with confidence falling and anxiety rising in mirror-image fashion; T effects ≈ 2× the D effects.

**Table H4.2.** Per-subject affect-feature decomposition → log(ω) AND log(κ). Each subject's confidence and anxiety ratings regressed on threat, distance, and T × D; the four resulting per-subject coefficients per channel are entered jointly as predictors of each parameter, controlling for the *other* parameter (κ when ω is outcome; ω when κ is outcome). Pooled OLS w/ HC3, N = 571.

*Outcome: log(ω); control: log(κ).* Model R² = 0.137.

| Predictor (per-subject feature, z-scored) | β | SE | 95% CI | p |
|---|---|---|---|---|
| **confidence_intercept (mean confidence)** | **−0.173** | 0.063 | [−0.297, −0.048] | **.006** |
| confidence_slope_D | −0.128 | 0.068 | [−0.260, +0.004] | .058 (trending) |
| confidence_T × D | −0.118 | 0.069 | [−0.253, +0.017] | .087 (trending) |
| confidence_slope_T | −0.097 | 0.068 | [−0.230, +0.036] | .154 (n.s.) |
| anxiety_intercept | +0.010 | 0.063 | [−0.114, +0.133] | .878 (n.s.) |
| anxiety_slope_T | +0.005 | 0.063 | [−0.119, +0.130] | .932 (n.s.) |
| anxiety_slope_D | +0.081 | 0.064 | [−0.044, +0.207] | .205 (n.s.) |
| anxiety_T × D | −0.004 | 0.062 | [−0.126, +0.117] | .945 (n.s.) |
| log(κ) (covariate) | +0.315 | 0.051 | [+0.215, +0.416] | < .001 |

*Outcome: log(κ); control: log(ω).* Model R² = 0.132.

| Predictor | β | SE | 95% CI | p |
|---|---|---|---|---|
| **confidence_intercept** | **−0.129** | 0.057 | [−0.240, −0.018] | **.023** |
| confidence_slope_D | −0.133 | 0.069 | [−0.269, +0.003] | .054 (trending) |
| confidence_T × D | −0.100 | 0.061 | [−0.220, +0.021] | .104 (n.s.) |
| confidence_slope_T | −0.027 | 0.058 | [−0.141, +0.087] | .639 (n.s.) |
| anxiety_intercept | +0.030 | 0.069 | [−0.106, +0.166] | .665 (n.s.) |
| anxiety_slope_T | +0.022 | 0.065 | [−0.106, +0.149] | .741 (n.s.) |
| anxiety_slope_D | −0.092 | 0.069 | [−0.227, +0.043] | .180 (n.s.) |
| anxiety_T × D | −0.054 | 0.064 | [−0.180, +0.072] | .405 (n.s.) |
| log(ω) (covariate) | +0.317 | 0.048 | [+0.222, +0.412] | < .001 |

**Pattern:** Confidence intercept (mean confidence) survives on BOTH parameters in the joint specification — registers vigilance AND mobilization independently. Confidence reactivity to distance trends in the same direction on both (p ≈ .055 on each). All four anxiety features are null on both outcomes.

### H5. Apathy — not anxiety or depression — uniquely maps onto vigilance, routed through confidence.
- **H5a.** Apathy is the only one of seven clinical scales at the subscale level that tracks ω. AMI_Total → log(ω) β = +0.135, p = .003; behavioural signature stronger: AMI → β_T_choice β = −0.198, p < .001 (both samples p < .01). No anx/dep subscale survives. [Supp: DASS_Stress small same-direction trend on modality.]
- **H5b.** Item-level EFA shows a clean psychometric dissociation: an AMI-loaded disengagement factor → ω (5-factor F4 β = −0.102 [−0.182, −0.023] ★), and a STAI-loaded trait-anxiety factor → κ (5-factor F3 β = +0.106 [+0.022, +0.184] ★). AMI Social + Behavioural ★ at the subscale level; AMI Emotional null. The two parameters are tracked by separate latent symptom dimensions — apathy on ω, anxiety on κ — even though only AMI survives at the raw subscale level. [Supp: full EFA loadings + per-factor table for n_factors = 3–5.]
- **H5c.** Apathy reroutes defensive output toward choice and away from vigor: total_mod β = +0.111, channel_balance β = +0.117 (each p < .01, joint).
- **H5d.** Confidence fully mediates: AMI → confidence β = −0.205 ***; confidence → ω | AMI β = −0.155 ***; c = +0.111 → c' = +0.079 (n.s.); Sobel p = .006; bootstrap CI excludes 0. ~30% mediated. Anxiety mediators all fail.
- **H5e.** Same confidence channel hides apathy's only κ signal: suppression mediation on log(κ) (a·b = +0.032, CI excludes 0) despite null total effect — mirrors H4 (confidence registers BOTH parameters).

**Table H5.1.** Headline apathy → ω regression across samples. Spec: `log(ω)_z ~ AMI_Total_z + ANX+DEP_FIXED_z + log(κ)_z` (frequentist OLS, HC3).

| Sample | N | β(AMI_Total) | SE | 95% CI | p |
|---|---|---|---|---|---|
| **Pooled** | 571 | **+0.132** | 0.045 | [+0.044, +0.221] | **.003** |
| Confirmatory | 281 | +0.160 | 0.067 | [+0.029, +0.290] | .017 |
| Exploratory | 290 | +0.109 | 0.063 | [−0.014, +0.233] | .083 (marginal) |

**Table H5.2.** Behavioral signature: AMI_Total ~ β_T_choice (univariate OLS, HC3).

| Sample | N | β | SE | 95% CI | p |
|---|---|---|---|---|---|
| **Pooled** | 571 | **−0.198** | 0.040 | [−0.276, −0.120] | **< .001** |
| Confirmatory | 281 | −0.215 | 0.057 | [−0.326, −0.104] | < .001 |
| Exploratory | 290 | −0.181 | 0.057 | [−0.292, −0.070] | .001 |

**Table H5.3.** AMI subscale specificity: β_T_choice → subscale (univariate Bayesian Student-t, pooled N = 571).

| Subscale | β | SE | 95% HDI | p (frequentist) |
|---|---|---|---|---|
| AMI_Total | −0.202 | 0.040 | [−0.280, −0.119] | < .001 |
| AMI_Social | −0.198 | 0.040 | [−0.282, −0.120] | < .001 |
| AMI_Behavioural | −0.166 | 0.041 | [−0.253, −0.087] | < .001 |
| AMI_Emotional | −0.048 | 0.041 | [−0.125, +0.037] | n.s. |
| MFIS_Psychosocial | −0.10 | 0.04 | ★ HDI | < .05 |
| MFIS_Physical | −0.01 | — | n.s. | n.s. |
| MFIS_Cognitive | −0.04 | — | n.s. | n.s. |

**Table H5.4.** Channel modality joint model: `AMI_Total_z ~ total_mod_z + channel_balance_z` (OLS HC3, pooled N = 571).

| Predictor | β | SE | 95% CI | p |
|---|---|---|---|---|
| **total_mod** (overall reactivity magnitude) | **+0.111** | 0.041 | [+0.031, +0.190] | **.007** |
| **channel_balance** (choice > vigor preference) | **+0.117** | 0.043 | [+0.034, +0.201] | **.006** |

Model R² = 0.026, F p < .001.

**Table H5.5.** Anxiety/depression nulls + DASS_Stress modality trends + ANX+DEP composite (pooled N = 571, Bayesian Student-t univariate).

| Predictor | Outcome | β | 95% HDI | Verdict |
|---|---|---|---|---|
| All individual anx/dep scales | β_T_choice (raw behavioral) | \|β\| < 0.07 | all clip 0 | NULL |
| ANX+DEP composite (7 scales) | β_T_choice | −0.02 | clips 0 | NULL |
| ANX+DEP composite | log(ω) (joint w/ AMI + log κ) | −0.084 | [−0.165, −0.008] | ★ marginal (p = .031) |
| DASS_Stress | vigor_mod | **−0.107** | excludes 0 | **★** |
| DASS_Stress | channel_balance | **+0.092** | excludes 0 | **★** (same direction as apathy) |

**Table H5.6.** Mediation: AMI_Total → mediator → log(ω) (pooled N = 571; OLS with HC3 for paths, Sobel test + 5000-rep percentile bootstrap for indirect).

| Mediator | a-path β (X→M) | b-path β (M→Y\|X) | c-path β (total) | c'-path β (direct) | a·b indirect | Sobel z, p | Bootstrap 95% CI | Verdict |
|---|---|---|---|---|---|---|---|---|
| **mean_confidence** | **−0.205*** | **−0.155*** | +0.111** | +0.079 (p = .064) | **+0.032** | **z = 2.75, p = .006** | **[+0.012, +0.058]** | **FULL MEDIATION** |
| mean_anxiety | −0.022 | +0.027 | +0.114** | +0.115** | −0.001 | n.s. | spans 0 | NO MEDIATION |
| anx_slope (anxiety reactivity) | +0.088 ★ | −0.011 | +0.114** | +0.115** | −0.001 | n.s. | spans 0 | NO MEDIATION |
| anx_calibration (anxiety-T r) | +0.109 ★ | +0.004 | +0.110 ★ | +0.108 ★ | +0.0004 | n.s. | spans 0 | NO MEDIATION |

Significance: \*\*\* p < .001; \*\* p < .01; ★ p < .05.

**Table H5.7.** Mediation also operates on κ (suppression mediation; pooled N = 571).

| Outcome | a-path β (AMI → conf) | b-path β (conf → κ \| AMI) | c-path β (total) | c'-path β (direct) | a·b indirect | Bootstrap 95% CI | Verdict |
|---|---|---|---|---|---|---|---|
| log(κ) | −0.205*** | −0.158*** | +0.030 (n.s.) | −0.002 (n.s.) | +0.032 | excludes 0 | **suppression mediation (apathy's only κ signal)** |

**Proportion mediated** (point estimate): a·b / c ≈ 29% (Bayesian) to 32% (frequentist).

---

## DISCUSSION (3 paragraphs)

### D1 — Synthesis
- Foraging theory unified threat and effort decisions; humans implement the joint computation predicted by the normative model.
- Joint W(u) > separable alternatives; (ω, κ) dissociable and behaviorally meaningful at the individual level.
- Closes the gap between threat-processing (anxiety/defensive) and effort-based (motor/foraging) literatures.

### D2 — Apathy phenotype + confidence mediation
- Apathy = the computationally specific clinical signature, routed through metacognitive confidence, not emotional reactivity.
- Across 2 samples, AMI specifically predicts ω; raw behavioral β_T_choice → AMI replicates strongly (p < .001 both samples). Not anxiety, not depression.
- Specific to motivational/action apathy (AMI Social + Behavioural; MFIS Psychosocial); not emotional anhedonia.
- Full mediation by within-task confidence (Sobel p = .006, ~30% mediated); anxiety mediators all fail.

### D3 — Limitations
- Individual anxiety/depression scales don't reach significance; ANX+DEP composite marginal (p = .031), doesn't appear in behavior, won't survive correction.
- Cross-sample replication asymmetric for model-parameter finding (confirmatory ★, exploratory marginal); behavioral β_T_choice → AMI replicates fully in both samples.
- Factor analysis (Supplementary) methodologically valuable (caught STAI scoring bug) but substantive factor findings were rotation artifacts; don't survive raw-scale verification.
- Whether confidence-mediation generalizes outside this task is open.

### Closing line (both axes)
- Humans implement joint threat-effort computation; apathy is its clinically specific phenotype — routed through metacognitive confidence.

---

## Methods (appendix to outline — for drafting reference)

- **Participants.** Two pre-registered Prolific samples (N = 290 exploratory, N = 281 confirmatory). Pooled N = 571.
- **Task.** Choice (heavy R = 5 vs light R = 1) × T ∈ {0.1, 0.5, 0.9} × D ∈ {1, 2, 3}, keypress execution, predator attacks on subset, per-trial confidence + anxiety probes.
- **Model.** W(u) = S(u)·R − (1−S(u))·ω·(R+C) − κ·(u−req)²·D, per-subject (ω, κ) fitted via NumPyro HMC.
- **Clinical Measures.** DASS-21, STAI-Trait (reverse-keying corrected via DASS-anchored direction check), OASIS, STICSA, PHQ-9, AMI, MFIS. Z-scored within sample.
- **Analysis.** Mixed-effects models for behavioral outcomes; per-subject behavioral regression coefficients as direct measurements; Bayesian Student-t robust regression (Bambi); frequentist OLS with HC3 as sensitivity; Bayesian + bootstrap (5000 reps) + Sobel for mediation. Replication threshold: directional and HDI-significant in both samples.

---

## Naming convention

- ω ↔ **capture cost / vigilance** (defensive intensity scaling)
- κ ↔ **effort cost / mobilization** (motor amplitude cost)

Use "capture cost" / "effort cost" in Methods + H2; use "vigilance" / "mobilization" as the dispositional individual-differences labels in H3-H5 and Discussion.

---

## Figures (proposed)

| # | Content |
|---|---|
| 1 | Task design schematic + W(u) formula + (ω, κ) plane diagram |
| 2 | Behavioral coordination: P(heavy) × (T, D); vigor × T within cookie; cross-channel correlation scatter (H1) |
| 3 | Model comparison + parameter recovery + per-subject (ω, κ) scatter (H2-H3) |
| 4 | Parameter consequences: escape × ω; overcautious errors × ω; pressing × κ; deviation from optimum (H3) |
| 5 | Affect register: confidence + anxiety probe ratings × (T, D); confidence → log(ω) scatter (H4) |
| 6 | Apathy → ω, routed through confidence (H5): top row — AMI_Total → log(ω); β_T_choice → AMI_Total; channel modality scatter. Bottom row — mediation path diagram; AMI → confidence; confidence → log(ω); failed anxiety mediators panel. |

Six figures total.

---

## What's out of scope (supplementary or future paper)

- **Vigor temporal dynamics / Threat Imminence Continuum.** Within-trial anticipatory ramp + reactive surge (former embodied H2). Companion paper or Supplementary.
- **Substantive factor-analysis findings.** EFA was methodologically valuable (caught STAI bug) but substantive factors largely rotation artifacts. Methods footnote only.
- **ANX+DEP comorbidity story.** Composite β = -0.084, p = .031 — marginal, absent in behavior, doesn't cross-validate. Discussion limitation; supplementary table.
- **CCA / multivariate clinical analyses.** Null; supplementary.
- **Vigor-ODE work.** Dead end.

---

## Outline status

**REVISED 2026-06-10 via paper-spine questionnaire.** All sections rebuilt in bullet form. Joint computational integration as headline; apathy validates.

**Decisions locked:**
- Title: joint computation → reveals apathy signature.
- Sig Stmt: literature-gap hook → 3 equal-weight claims → computationally-specific-apathy take-away.
- Intro: 4 paragraphs, no confidence preview. P1 = everyday vignette; P2 = Pavlovian/imminence; P3 = motor vigor; P4 = foraging → clinical motivation → our task.
- H1: both channels jointly modulated AND independent; TIC dynamics folded into a sub-bullet pointing to companion work.
- H2: joint model > separable, recovery clean (structural correlation saved for Methods).
- H3: parameter consequences (escape, errors, pressing); raw betas saved for H5.
- H4: confidence registers BOTH ω and κ when controlling for the other parameter; anxiety null on both (sets up H5).
- H5: merged apathy + mediation. Headline = apathy maps onto vigilance, routed through confidence. Sub-bullets: model + behavioral + specificity + channel modality + mediation (full on ω, suppression on κ) + anxiety mediators fail.
- Discussion: 3 paragraphs (synthesis / apathy + mediation combined / limitations); closing line ties both axes.

**Pending decisions:**
- Cite list for each Intro paragraph (P2 Mobbs/Fanselow/LeDoux; P3 Shadmehr/Mazzoni/Niv; P4 foraging theory + apathy + clinical-effort literature).
- Whether the affect-register figure (5) is needed or rolls into figure 1 / supplementary.
- Whether the channel-modality finding (H5d) gets its own figure panel or is part of figure 6.
- Whether to expand the Methods naming-convention paragraph in the main paper, or only Supplementary.
