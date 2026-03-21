# Discoveries

Summary of all empirical findings from the exploratory sample (N=293). Organized by analysis domain.

---

## 1. Choice Modeling

### Winning model: Unified Additive-Effort Hyperbolic-Survival (L4a_add)

**Value function:** SV = R·S − k·E − β·(1−S)
**Survival:** S = (1−T) + T / (1 + λ·D/α)
**Choice:** softmax(τ · ΔSV)

- k = additive effort cost (per-subject). β = subjective capture cost (per-subject). α = tonic vigor (observed, from vigor HBM). τ, λ = population-level.
- Best by ELBO (−6260) and BIC across 12 models tested via SVI (NB03-choice, 2026-03-20)
- Beats multiplicative effort (L4a_hyp) by +158 ELBO — additive effort is the correct form
- Hyperbolic escape kernel beats exponential by +207 ELBO — people perceive escape probability as declining gradually
- Mechanistic S: separates attack probability (T) from escape probability (f(D/α)), not conflated as in original FET
- Parameters cleanly identified: k-β r=−0.11 (n.s.), k-α r=−0.08 (n.s.), β-α r=+0.14

**Model comparison hierarchy (SVI ELBO):**
1. L4a_add (unified additive): −6260 ← WINNER
2. L3_add (additive, no α): −6275 (α helps +16)
3. L4a_hyp (multiplicative): −6418
4. L3_survival (mult, no α): −6449
5. L3b_surv_zi (per-subj z): −6561 (per-subj z HURTS)
6. L2_TxD (feature): −6780
7. L0_effort (effort only): −8298

**Key findings from comparison:**
- Additive >> multiplicative effort (+158 ELBO, solves k-β identifiability)
- Hyperbolic >> exponential escape kernel (+207 ELBO)
- α in survival helps (+16 ELBO) but α in effort hurts (L4b_add: −6554)
- Per-subject z hurts (−112 ELBO vs L3) — no individual differences in distance nonlinearity needed
- S = (1−T) + T·f(D/α) >> S = exp(−λ·T·D/α) — must separate attack probability from escape probability

**Previous model (FETExponentialBias):**
- WAIC=12,063, R²=0.454, Accuracy=82.5% — original fit with multiplicative effort + old S formulation
- Superseded by L4a_add. Old parameter estimates in results/stats/FET_Exp_Bias_*.csv for reference.

### Model-free behavior
- Effort (distance) reduces high-reward choice probability
- Threat reduces high-reward choice probability
- Effort × threat interaction: effort's deterrent effect amplified under high threat

### Population parameters
| Param | μ | Subject mean | Subject SD | Range |
|-------|---|-------------|-----------|-------|
| τ (temperature) | 0.437 | — | — | population-level only |
| k (effort disc.) | μ_k=1.54 | 1.95 | 1.26 | [0.33, 4.76] |
| z (hazard sens.) | μ_z=0.30 | 0.38 | 0.22 | [0.14, 1.46] |
| β (threat bias) | μ_β_log=−0.19 | 1.44 | 1.89 | [0.20, 13.58] — very right-skewed |

### Parameter independence
- Minimal posterior correlations between z, k, β — independently identifiable

---

## 2. Vigor Analysis

### 2a. Model-free vigor effects (NB09)
| Finding | Effect | p |
|---------|--------|---|
| Threat scales anticipatory vigor (onset slope) | β=+0.029 | 2×10⁻¹¹ |
| Attack triggers phasic spike (vigor_norm) | β=+0.015 | 0.009 |
| Attack spike disappears after demand removal (vigor_resid) | β=+0.003 | 0.644 n.s. |
| Terminal vigor predicts escape | β=+0.097 | 10⁻⁹⁸ |
| Tonic–phasic tradeoff on attack trials | r=−0.36 to −0.48 | — |

### 2b. Choice–vigor mapping (NB06)
Subject-level regression of vigor dimensions on model parameters (z, k, β):

| Vigor dimension | k | z | β | adj. R² |
|-----------------|---|---|---|---------|
| Tonic vigor | −0.19** | 0.09 trend | 0.05 | 0.049 |
| Anticipatory mobilization | −0.11* | 0.09* | 0.11** | 0.039 |
| Reactive spike | n.s. | n.s. | n.s. | −0.007 |
| Terminal persistence | n.s. | n.s. | n.s. | −0.009 |

**Key:** k is a global suppressor; z and β selectively predict anticipatory mobilization; reactive and terminal phases are dissociated from choice parameters.

### 2c. Parameter dissociation (NB08) — 8/39 tests survive FDR
| DV | Predictor | β | p_fdr |
|----|-----------|---|-------|
| onset_slope | k | −0.045 | 0.002 |
| onset_slope | β | +0.032 | 0.037 |
| onset_mean | z | +0.055 | 0.002 |
| onset_peak | z | +0.055 | 0.002 |
| post_encounter_vigor | k | −0.049 | 0.004 |
| terminal_mean | k | −0.044 | 0.002 |
| encounter_spike × threat | k | −0.026 | 0.049 |
| post_encounter_vigor × threat | k | −0.031 | 0.004 |

**Summary:** z → onset/anticipatory vigor; k → global suppression across phases; β → anticipatory slope boost.

### 2d. Trial-level survival → vigor (NB10, survival_vigor_lmm.csv)
- Terminal mean: S_trial β=−0.011, p_fdr=0.0002 — **lower survival → higher terminal vigor**
- S_trial × z_i interaction: marginal (p=0.12), not significant after FDR
- All other phase DVs (anticipatory slope/mean, encounter spike, post-encounter mean): n.s.

### 2e. PLS analysis (NB10)
- PLSCanonical(n=3): Component 1 significant by permutation (p<0.05)
- Structure: z and β load positively, k loads negatively; correlates with anticipatory/onset vigor
- Effect sizes modest (r≈0.13–0.25)

### 2f. Functional regression: params → vigor(t) (NB13)
Time-resolved LMMs at 0.1s bins across onset, encounter, and terminal windows:
- **z**: ramps from β≈0 at t=0 to β≈+0.065 by t=0.75s in onset; positive throughout encounter; turns **negative** in terminal (high-z subjects front-load vigor)
- **k**: persistently negative β≈−0.04 to −0.08 across ALL phases (global suppressor)
- **β**: modest positive in onset and encounter phases

### 2g. Phase-based imminence diagnostics (NB12) — KEY NEW RESULTS

#### ICC dissociation across phases (strongest vigor finding)
| Phase DV | ICC | Interpretation |
|----------|-----|----------------|
| onset_mean | **0.737** | 74% trait — who you are |
| onset_slope | **0.480** | 48% trait |
| enc_pre_mean | 0.418 | moderate |
| enc_post_mean | 0.418 | moderate |
| enc_spike | 0.183 | mostly trial-driven |
| term_mean | 0.329 | moderate |
| term_slope | **0.029** | 97% trial-driven — what's happening to you |

Anticipatory phases are person-level traits; reactive phases are state-driven.

#### Threat does NOT modulate any phase DV at group level
Between-subject ANOVAs on subject×threat means: ALL non-significant (p=0.20–0.89). Threat effects exist at the trial level (NB09 LMMs) but are swamped by individual differences when averaged per subject.

#### Model params predict ALL phase DVs (small R²)
| DV | z | k | β | adj.R² | p |
|----|---|---|---|--------|---|
| onset_mean | +0.215 | −0.128 | +0.078 | 0.048 | <0.001 |
| onset_slope | +0.128 | −0.208 | +0.175 | 0.062 | <0.001 |
| enc_pre_mean | +0.175 | −0.165 | +0.073 | 0.042 | 0.002 |
| enc_post_mean | +0.077 | −0.185 | +0.135 | 0.037 | 0.003 |
| enc_spike | −0.119 | −0.073 | +0.120 | 0.024 | 0.019 |
| term_mean | −0.045 | −0.221 | +0.067 | 0.044 | 0.001 |
| term_slope | −0.193 | +0.015 | −0.095 | 0.035 | 0.004 |

z flips sign: positive for onset/pre-encounter, negative for spike/term_slope. k is negative everywhere (global suppressor, strongest at terminal). β boosts onset slope and post-encounter.

#### Split-half reliability: all DVs reliable except term_slope
All SB r > 0.83 except term_slope (0.40, marginal). Individual differences are real — the low R² in param regressions is not a measurement problem.

#### Attack contrast — mostly confounded with threat (NB12, Check 6)
**Uncorrected:** Attack trials show higher onset mean (p<0.001), onset slope (p=0.001), enc_post (p=0.007), enc_spike (p=0.033), term_slope (p<0.001).

**After controlling for threat level:** Almost everything disappears. The attack effect on onset was entirely driven by high-threat trials having more attacks.

**One surviving effect: terminal slope** (combined p=0.00001). On attack trials, people show steeper terminal ramp at threat=0.5 (t=5.21, p=3.6×10⁻⁷) and threat=0.9 (t=2.95, p=0.003). This is genuine — when being chased, you sprint harder toward safety.

**Attack-trial-only analysis:** Pre→post transition on attack trials is non-significant (t=1.11, p=0.268). No reliable encounter spike even when the predator actually appears (in residualized vigor).

**Model params do NOT predict the attack effect:** After threat control, no model predicts the per-subject attack effect for any phase DV (all p > 0.07). The terminal sprint is generic.

### 2h. Count-based vigor (press counts per phase) — KEY METHODOLOGICAL FINDING

#### Residualization matters enormously — and was done wrong in the 20Hz pipeline

**The problem with 20Hz smoothed vigor:**
- 20Hz samples at 50ms intervals, but people press ~3-5/s (IPI 200-500ms). Most datapoints are kernel interpolation, not observed behavior.
- The vigor pipeline residualized per-subject against effort × distance. This removed both demand AND real between-subject variance, killing the ICC.

**The correct approach: capacity-normalized press counts, choice-binary demand adjustment.**
1. Count raw keypresses per phase (onset 0-2s, pre-encounter, post-encounter 0-2s, terminal last 2s)
2. Convert to rate (presses/sec) and divide by subject's maximum capacity (95th percentile of 1s-bin press rates)
3. Demand-adjust by dividing by the group mean rate for that choice level (choice=0 or choice=1) — NOT per-subject regression, NOT effort×distance

**Why choice-binary and not effort×distance:**
- Cookie weight (effort) is what determines pressing demand — heavier cookies require more presses
- Distance determines how LONG you press, not how FAST — if people press faster for farther cookies, that's a real signal
- Effort_H has 3 levels but what matters is the chosen cookie: choice=0 (easy) or choice=1 (hard). Binary is the right granularity.

**Why NOT per-subject residualization:**
- Per-subject regression removes each subject's mean and slope → ICC goes to zero
- Group-level or ratio-based normalization preserves between-subject variance

#### Results: choice-ratio normalized counts

**Threat ANOVA (between-subject):**
| Phase | F | p | Direction |
|---|---|---|---|
| Onset | 0.08 | 0.92 | null |
| Pre-encounter | 0.01 | 0.99 | null |
| Post-encounter | 0.20 | 0.82 | null |
| **Terminal** | **35.2** | **2×10⁻¹⁵** | **High threat → 104% of group mean, low → 96%** |

Terminal threat effect was INVISIBLE in 20Hz residualized data (F=0.33, p=0.72). The 20Hz pipeline was removing the signal.

**Param → subject-level press rates (choice-ratio normalized):**
| Phase | z | k | β | adj.R² | p |
|---|---|---|---|---|---|
| Onset | +0.230 | −0.076 | +0.042 | 0.045 | 0.001 |
| Pre-encounter | +0.154 | −0.196 | +0.112 | 0.051 | <0.001 |
| **Post-encounter** | **+0.173** | **−0.211** | **+0.230** | **0.093** | **<0.001** |
| Terminal | −0.012 | −0.162 | +0.124 | 0.026 | 0.013 |

Post-encounter R²=0.093 is nearly double the best 20Hz result. Same pattern: z boosts onset, k suppresses globally, β boosts post-encounter.

**ICC (choice-ratio normalized):**
| Phase | ICC |
|---|---|
| Onset | 0.714 |
| Pre-encounter | 0.316 |
| Post-encounter | 0.394 |
| Terminal | 0.161 |

**Attack effect (within threat level):**
Only terminal survives: t=18.0, p=10⁻⁴⁹, +9.5% of choice-group average.

**Distance effect:** Null everywhere after choice normalization (all p>0.3). Distance affects duration, not rate.

#### What this changes
Both stories coexist with the right demand correction:
1. **Individual differences** (trait): z, k, β predict pressing style (R²=0.03–0.09), stable across trials (ICC 0.16–0.71)
2. **State-driven terminal mobilization**: people deploy more capacity when threat is high (F=35) and when attacked (t=18), independent of parameters

The 20Hz residualized approach obscured the terminal threat effect and inflated the apparent uniqueness of the trait signal. Count-based capacity-normalized measures are the correct vigor operationalization for this task.

### 2i. CRITICAL BUG: encounterTime was in wrong reference frame

The vigor pipeline (`vigor_data_prep.py` line 110) computed `encounterTime = encounterTime - firstEffortTime`, shifting from trial-start-relative to effort-onset-relative. But keypress times in `keypress_events.parquet` are trial-start-relative. This frame mismatch caused the pre-encounter window to be placed incorrectly, making it appear that most trials had no pre-encounter pressing data.

**Original encounterTime (stage2, trial-start-relative):** M=3.22s, range=[2.50, 5.00]. Fixed values: 2.5s (D≤5), 3.5s (D=7), 5.0s (D=9).
**Shifted encounterTime (vigor pipeline):** M=1.64s (after subtracting firstEffortTime M=1.71s), clipped to 0 for 2,876 trials.

**Fix:** Use original encounterTime from `processed_trials.pkl` or add firstEffortTime back. Keypress times are already in trial-start frame.

### 2j. Encounter-centered analysis (CORRECTED FRAME) — KEY RESULTS

**Design:** 4s window centered on encounter (enc−2s to enc+2s). Count-based, capacity-normalized, choice-ratio adjusted. N=20,592 trials (89% coverage), 293 subjects. Per-subject features computed via partial regression slopes (rate ~ threat + distance + attack).

#### Group-level effects

**Threat modulation (between-subject ANOVA):**
| Phase | Low | Med | High | F | p |
|---|---|---|---|---|---|
| Pre | 0.990 | 0.946 | 0.953 | 2.54 | 0.079 |
| **Post** | **0.977** | **0.995** | **1.017** | **6.88** | **0.001** |
| **Transition** | **−0.013** | **+0.049** | **+0.064** | **7.84** | **0.0004** |

Post-encounter pressing and the pre→post transition are threat-modulated. People press harder after the encounter point on high-threat trials.

**Attack effects (within threat level):**
| Phase | diff | t | p |
|---|---|---|---|
| Pre | −0.009 | −1.01 | 0.314 |
| **Post** | **+0.033** | **6.95** | **2×10⁻¹¹** |
| **Transition** | **+0.042** | **5.29** | **2×10⁻⁷** |

Clean imminence signal: when predator actually appears, post-encounter pressing increases by 3.3% of capacity. Pre-encounter is unaffected (correctly — participants don't know yet).

**Distance effects:**
| Phase | D=1 | D=2 | D=3 | F | p |
|---|---|---|---|---|---|
| **Pre** | **0.984** | **0.900** | **0.954** | **6.59** | **0.001** |
| Post | 0.995 | 1.002 | 0.998 | 0.20 | 0.82 |
| **Transition** | **0.011** | **0.102** | **0.044** | **7.60** | **0.0005** |

#### Individual differences: feature × param correlations

| Feature | z | k | β |
|---|---|---|---|
| **dist_pre** | **−0.270** | **−0.435** | **−0.212** |
| **dist_trans** | **+0.198** | **+0.407** | **+0.227** |
| tonic_pre | +0.174 | −0.163 | +0.103 |
| tonic_trans | −0.200 | +0.104 | −0.053 |
| threat_trans | −0.117 | −0.131 | −0.128 |

**dist_pre × k = −0.435 is the strongest single vigor→param correlation.** High-k people show less distance-dependent pressing before encounter. dist_trans × k = +0.407 flips sign — high-k people show bigger distance-dependent encounter transitions.

**threat_trans is significant across all three params** (r ≈ −0.12, all p<0.05). People with higher z/k/β show smaller threat-dependent transitions — more uniform across threat levels.

#### PLS: encounter-window features → model params

- 2 comp: train R²=0.144, **CV R²=0.093**, permutation p=0.000
- 3 comp: train R²=0.162, **CV R²=0.117**
- Per-param (2 comp): **k CV R²=0.199**, z CV R²=0.072, β CV R²=−0.039
- Component 1: dist_pre (+0.668) and dist_trans (−0.658) — distance modulation
- Component 2: tonic_trans (+0.640) and tonic_pre (−0.569) — tonic levels

**Interpretation:** Encounter-window vigor features predict k well from just 12 features in a 4s window. Distance modulation of pressing around the encounter is the primary bridge between vigor and choice model parameters. k is the best-predicted parameter — effort discounting manifests clearly in how pressing rate varies with distance around the encounter point.

### 2k. Dead ends
- **ODE vigor analysis (NB11):** kinetics degenerate, no new findings
- **Continuous within-trial temporal alignment:** confounded by trial duration × distance × effort; phase approach is the right level of analysis
- **Encounter spike (20Hz residualized):** demand artifact
- **Per-subject effort×distance residualization:** removes between-subject variance, zeroes ICC — wrong approach
- **Onset/terminal phases as primary vigor measures:** onset has variable start times (34% zero-press in first 2s), terminal is messy. Encounter-centered window is cleaner.
- **Spearman correlations for features:** confounded marginal associations. Use partial regression slopes controlling for threat + distance + attack.
- **threat_mod_onset × k (r=+0.38 from Spearman):** was a confound — disappeared with partial slopes. Do not claim.

---

## 3. Affect Analysis

### 3a. Core LMM results (NB12, N=293, 10,546 ratings)
| DV | Predictor | β | p |
|----|-----------|---|---|
| Anxiety | p_threat | +0.575 | <0.001 |
| Anxiety | S_probe | −0.602 | <0.001 |
| Anxiety | dist_safety | +0.226 | <0.001 |
| Confidence | p_threat | −0.586 | <0.001 |
| Confidence | S_probe | +0.632 | <0.001 |
| Confidence | dist_safety | −0.283 | <0.001 |

Model-derived survival predicts anxiety (−) and confidence (+) at trial level.

### 3b. Parameter moderation of affect
- z → chronic confidence deficit (main effect): β=−0.199, p_fdr=0.013 ✅
- **dist_safety × z → confidence**: β=−0.056, p=0.023, p_fdr=0.068 (marginal, does not survive FDR)
- **dist_safety × z → anxiety**: β=+0.046, p=0.051, p_fdr=0.068 (directional, n.s.)
- **p_threat × β → anxiety**: β=+0.050, p=0.038, p_fdr=0.068 (marginal, does not survive FDR)
- **p_threat × β → confidence**: β=−0.037, p=0.134, n.s.
- p_threat × z interaction: NULL (p>0.09 for both affect types)
- **Interpretation**: Directional tendencies consistent with draft claims but none survive FDR. The robust finding is z → chronic confidence main effect. Draft updated to reflect this.

### 3c. State-trait decomposition (NB12, Section 7)
Between-subjects OLS (trait ~ z + k + β):

| DV | z | k | β | adj. R² |
|----|---|---|---|---------|
| Trait anxiety (mean task) | −0.581, p=0.097 n.s. | +0.146, p=0.019* | n.s. | 0.020 |
| Trait confidence (mean task) | −0.719, p=0.044* | −0.163, p=0.010* | n.s. | 0.036 |

Within-subjects (state ~ p_threat + dist_safety): phasic state responses robust and parameter-independent.

**Key:** k predicts BOTH trait anxiety (+) and confidence (−); z predicts only confidence. Phasic state responses are driven by task variables, not model parameters.

### 3d. Cross-domain: vigor × affect — NULL
- 15 vigor × affect pairs tested; **none survive FDR** (highest: tonic_vigor ~ anx_threat_slope r=+0.124, p_fdr=0.196)
- Motor vigor and affective systems are functionally parallel but NOT cross-correlated at individual-difference level

### 3e. Metacognitive calibration (NB05-psych)

#### Ratings track model-derived survival
- Per-subject r(anxiety, S_probe): M=−0.341, t=−16.45, p=2.5×10⁻⁴³
- Per-subject r(confidence, S_probe): M=+0.340, t=+15.48, p=8.4×10⁻⁴⁰
- Subjective reports are internally consistent with the model's survival computation

#### Calibration to objective conditions
- Anxiety × threat: M(r)=+0.316, p=10⁻³⁹; anxiety × distance: M(r)=+0.133, p=10⁻¹⁷
- Confidence × threat: M(r)=−0.308, p=10⁻³⁶; confidence × distance: M(r)=−0.140, p=10⁻¹⁹
- Large individual variability: only ~38% of subjects reach individual significance for threat calibration

#### k predicts calibration accuracy (strongest metacognitive finding)
- k × anxiety threat calibration: r=−0.309, p<0.001
- k × confidence threat calibration: r=−0.210, p<0.001
- Joint model (anxiety threat calibration ~ z+k+β): R²=0.121, p<0.001, driven by k (β=−0.342)
- **Interpretation:** Higher k (more effort discounting) → worse affective calibration to threat. Low-k people are more engaged AND more metacognitively accurate.

#### z predicts distance-specific calibration
- z × anxiety distance calibration: r=+0.152, p=0.010
- z governs D^z in the model; high-z people also differentiate distance more in their anxiety ratings. Clean, specific correspondence.

#### β modestly boosts anxiety calibration
- β × anxiety threat calibration: r=+0.135, p=0.022
- β × anxiety survival calibration: r=+0.140, p=0.017

#### Ratings → outcomes: weak / condition-driven
- Confidence → escape (attack trials): β=+0.022, p=0.017 raw, but **n.s. after controlling for threat/distance** (p=0.78)
- Anxiety → success (all trials): β=+0.013, p=0.038 after controls — small residual effect, possibly anxiety → more effort → better outcome
- **Bottom line:** No strong metacognitive prediction of outcomes beyond what conditions explain

### 3f. Psychiatric battery × model parameters

#### Factor analysis of psychiatric battery (NB06-psych, 2026-03-20)

**14 subscales → 3 factors** (parallel analysis suggested 2; 3-factor solution used on theoretical grounds). KMO=0.931, Bartlett's p≈0. Varimax rotation. 69.2% total variance explained.

| Factor | Var% | Key loaders | Interpretation |
|---|---|---|---|
| F1 | 37.1% | STICSA (0.88), DASS_Anx (0.82), DASS_Stress (0.78), OASIS (0.72), PHQ9 (0.72), STAI_State (0.72) | General distress |
| F2 | 20.4% | MFIS_Phys (−0.79), MFIS_Cog (−0.74), MFIS_Psychosoc (−0.66) | Fatigue |
| F3 | 11.7% | AMI_Social (−0.62), AMI_Behav (−0.56), DASS_Dep (−0.48), STAI_Trait (+0.39) | Apathy/amotivation |

Factors near-orthogonal (max inter-factor r=0.107).

**5 params → 3 factor scores:**

| Factor | R² | p | Significant predictors |
|---|---|---|---|
| F1 (Distress) | 0.015 | 0.51 | None |
| F2 (Fatigue) | 0.030 | 0.12 | z marginal (β=+0.138, p=0.013) |
| **F3 (Apathy)** | **0.155** | **3×10⁻⁹** | **α (β=−0.343, p<0.001)** |

**Key finding:** α alone explains 15.5% of the apathy factor — stronger than the raw AMI result (R²=0.097). The dissociation is sharp: choice params {k, z, β} predict no psychiatric factor. Tonic vigor α predicts only apathy. Phasic vigor ρ predicts nothing. The task captures motivational drive (apathy) but not mood (distress) or energy (fatigue).

#### Individual scale results (pre-factor analysis)
- 0/39 bivariate correlations survive FDR on individual scales
- z shows consistent negative pattern across anxiety/fatigue (r=−0.10 to −0.18, uncorrected)
- STAI-Trait still has compressed range (SD=5.8 vs expected ~10-12) despite scoring fix

---

## 4. Anxiety–Vigor Coupling (NB13)

### All levels — NULL
- Subject-level: r=+0.03, n.s.
- Trial-level concurrent: β=+0.002, p=0.674
- Predictive: β=+0.007, p=0.053 (marginal)
- Phase-specific: 16 tests, all null after FDR
- Functional regression: complete null across all time bins
- PLS: overfits (CV R²=−0.071)

**Interpretation:** Trial-by-trial anxiety carries NO information about vigor beyond threat level. The common structure is in the shared INPUT (survival computation), not in serial affect→motor causation.

---

## 5. Choice-Vigor Dissociation — MAJOR FINDING

### Core result: choice and vigor are uncorrelated (r=−0.018, p=0.76)
P(choose high-effort cookie) and pre-encounter pressing rate (capacity-normalized, choice-adjusted) are independent. N=293. Four quadrants roughly equally populated (22-28% each).

### Quadrant profiles — k and β separate the groups

| Quadrant | N | z | k | β | Escape | Earnings |
|---|---|---|---|---|---|---|
| HH (choose hard, press hard) | 63 | 0.31 | **1.00** | 0.89 | 53% | **+95.5** |
| HL (choose hard, press soft) | 83 | 0.26 | 1.33 | **0.59** | **19%** | −3.8 |
| LH (choose easy, press hard) | 78 | 0.43 | 2.35 | **2.20** | **60%** | +71.4 |
| LL (choose easy, press soft) | 69 | 0.41 | **3.04** | 1.66 | 25% | **−25.8** |

- k ANOVA: F=51.6, **p=10⁻²⁷**
- β ANOVA: F=18.3, **p=10⁻¹¹**
- z ANOVA: F=13.3, **p=10⁻⁸**
- Escape ANOVA: F=115.6, **p<10⁻⁴¹**
- Earnings ANOVA: F=91.1, **p<10⁻⁴¹**

### Vigor dominates outcomes, not choice
- Escape ~ choice + vigor + interaction: R²=0.66. **Vigor β=+0.795, choice β=−0.160.** Vigor explains ~5× more variance. Choosing hard actually hurts escape (farther from safety).
- Earnings ~ choice + vigor + interaction: R²=0.60. Vigor β=+0.758, choice β=+0.208.

### β is the dissociation parameter
β acts on choice (r=−0.488 with p_high) but NOT on vigor (r=+0.109, n.s.). High-β people choose safe options (threat bias suppresses risky choice) but their motor system is unaffected — they press just as hard when they do engage. This creates the LH pattern: choose easy, press hard, escape well.

### Off-diagonal comparison (HL vs LH, N=161)

| Variable | HL (choose hard, press soft) | LH (choose easy, press hard) | p |
|---|---|---|---|
| β | 0.59 | **2.20** | <0.001 |
| k | 1.33 | **2.35** | <0.001 |
| Escape rate | **18.7%** | **60.4%** | <0.001 |
| Confidence | **3.40** | 2.76 | 0.003 |
| Anxiety calibration | 0.22 | **0.36** | 0.007 |
| AMI (apathy) | 25.4 | **32.2** | <0.001 |

HL people are overconfident — high confidence but worst escape rate. LH people report higher apathy (AMI) but are better calibrated in their anxiety responses and survive far more often. The "apathetic" group is actually the more adaptive one.

### Threat reverses the choice-vigor relationship

| Threat | Choice × Vigor r | p |
|---|---|---|
| 0.1 (low) | **+0.196** | 0.001 |
| 0.5 (med) | +0.013 | 0.82 |
| 0.9 (high) | **−0.219** | <0.001 |

At low threat, choice and vigor align (r=+0.20). At high threat, they decouple (r=−0.22). Threat drives the dissociation — under danger, conservative-but-capable (LH) strategies emerge.

### Formal validation (Phase 0 — all gates passed)
- **Vigor reliability:** split-half r=0.835, SB=0.910. Block reliability SB=0.891. (Choice SB=0.369 — vigor is MORE reliable than choice)
- **Independence robust to operationalization:** 4/5 vigor measures show |r(choice)| < 0.10. Only total presses (r=+0.30) is correlated — a demand confound, not true coupling.
- **Parameter identifiability:** k-β posterior median correlation = +0.143 (low). Parameters are independently identifiable.

### Formal statistical tests (Phase 1-2)

**Multiple regression asymmetry:**
- Choice ~ z + k + β: **adj.R²=0.823**, F=453, p≈0. k dominant (β=−0.685)
- Vigor ~ z + k + β: **adj.R²=0.075**, F=8.9, p=1.2×10⁻⁵. 11× weaker.

**CCA: two significant canonical dimensions**
- Dim 1: r=0.909, p=10⁻¹¹³. Loads on k/β → Choice (nearly perfectly)
- Dim 2: r=0.289, p=5×10⁻⁷. Loads on z/β → Vigor (weak but significant)
- The param space maps onto behavior through TWO independent pathways.

**Bootstrap test (10K iterations): β selectively predicts choice**
- β → choice: −0.409 [−0.570, −0.310] (CI excludes zero)
- β → vigor: +0.147 [+0.019, +0.271] (CI barely excludes zero on POSITIVE side — opposite direction)
- β_choice − β_vigor = −0.555 [−0.802, −0.380], **p=0.0000**. β's effect on choice is significantly stronger and opposite-signed from vigor.

**z goes in opposite directions:**
- z → choice: −0.276 (choose safe), z → vigor: +0.199 (press harder). Compensatory.

**Diagonal/off-diagonal decomposition:**
- Diagonal (HH↔LL, general effort): k dominant (β=−0.896), adj.R²=0.481
- Off-diagonal (LH↔HL, dissociation): all three contribute equally (z=+0.47, k=+0.47, β=+0.54), adj.R²=0.417

### Threat modulation (Phase 3)

**Cross-level LMM interaction (N=20,658 trials):**
- `vigor ~ choice_subj_z * threat_z + dist_z + (1|subj)`
- choice × threat: **β=−0.022, z=−3.54, p=0.0004**. Survives with random slopes (p=0.002).
- Formally: the choice-vigor coupling reverses with threat.

**Per-threat correlations with bootstrap CIs:**
| Threat | r | 95% CI | p |
|---|---|---|---|
| 0.1 | +0.196 | [+0.073, +0.314] | 0.001 |
| 0.5 | +0.013 | [−0.096, +0.121] | 0.821 |
| 0.9 | −0.219 | [−0.317, −0.120] | <0.001 |
Fisher z-test: z=5.07, **p<0.0001**

**β mediates the reversal:** β→choice strengthens from r=−0.21 (low threat) to r=−0.52 (high threat), while β→vigor stays flat (~+0.10). β's choice suppression amplifies with threat because β enters through (1−S) which grows as threat increases.

**Full interaction model:** z×threat (+0.012, p=0.050) and k×threat (+0.014, p=0.023) significant. β×threat not significant (p=0.118). Threat modulates vigor through z and k, not β.

### Outcome prediction (Phase 4)

**Trial-level escape LMM (N=10,257 attack trials):**
- vigor: **β=+0.091, p=10⁻⁷⁷**
- choice: **β=−0.177, p≈0** (choosing hard HURTS escape)
- Adding vigor to choice-only model: ΔAIC=341

**Pairwise (same choice, different vigor):**
- HH vs HL: 53% vs 19% escape, t=12.7, p=10⁻²⁵
- LH vs LL: 60% vs 25% escape, t=12.6, p=10⁻²⁵
- **Vigor triples escape rate within the same choice group.**

**Pairwise (same vigor, different choice):**
- HH vs LH: 53% vs 60%, t=−2.29, p=0.024. Choosing easy slightly helps.
- HL vs LL: 19% vs 25%, t=−2.34, p=0.020. Same pattern.

**Subject-level earnings:** R²=0.60. Vigor β=+0.758, choice β=+0.208.

### Confidence miscalibration (Phase 5) — strongest affect finding

**Confidence bias = conf_z − escape_z (F=50.2, p=10⁻²⁶):**
| Quadrant | Conf bias | Interpretation |
|---|---|---|
| HL | **+0.981** | Massively overconfident |
| LL | +0.377 | Mildly overconfident |
| HH | −0.247 | Slightly underconfident |
| LH | **−1.177** | Most underconfident / well-calibrated |

**R²=0.415:** choice drives overconfidence (β=+0.423), vigor drives underconfidence (β=−0.783). The choice-vigor dissociation directly predicts metacognitive accuracy.

### Psychiatric (Phase 6)
- **Apathy factor (F3)** from 3-factor EFA: α β=−0.343, R²=0.155, p=3×10⁻⁹. Stronger than raw AMI result.
- AMI_Total bivariate: α r=+0.340, p<0.001 (Bayesian estimates).
- No other param predicts any factor. General distress and fatigue are orthogonal to task behavior.
- PHQ-9 shows quadrant effect (FDR p=0.043): HH=9.0, HL=7.5, LH=7.9, LL=5.4. High-α people report more depression.

#### Mental health → behavioral profiles (predictive direction, 2026-03-20)
- MH features predict **vigor** (high/low α): 62% accuracy, AUC=0.675. Driven by AMI.
- MH features predict **HL vs LH**: 61% accuracy, AUC=0.645. AMI Social→LH (+0.64), STAI Trait→HL (−0.44).
- MH features do NOT predict choice (49%, chance) or coupled/decoupled (51%, chance).
- Clinical instruments see the motor channel (α) but are blind to the decision channel (k, β).

#### PLS: 5 params → mental health + affect (NB07-psych, 2026-03-20)

**X:** {k, z, β, α, ρ}. **Y:** 3 psychiatric factors + mean anxiety/confidence + threat sensitivity of anxiety/confidence. N=285.

**Overall:** R²=0.073, permutation p=0.0000 (5000 perms), CV R²=0.039. Significant but modest.

**3 PLS components:**
| Comp | r | p | X loadings | Y loadings |
|---|---|---|---|---|
| 1 | 0.538 | 10⁻²³ | α (+0.85), k (−0.66) | Anx threat sens (+0.36), Apathy (−0.30), Conf threat sens (−0.27) |
| 2 | 0.300 | 10⁻⁷ | z (+0.66), k (+0.64), β (+0.43) | Apathy (−0.19), Mean confidence (−0.20) |
| 3 | 0.228 | 10⁻⁴ | ρ (+0.86), β (+0.42) | Weak — fatigue (−0.10) |

**Per-Y R² (in-sample):** Anx threat sens=0.145, Apathy=0.130, Conf threat sens=0.088, Mean anxiety=0.065, Mean confidence=0.052, Fatigue=0.020, Distress=0.013.

**Interpretation:** Comp 1 is "engaged effort" — high α + low k maps onto better anxiety calibration, more apathy, lower mean anxiety. This is the adaptive profile (LH/HH). Comp 2 loads on all choice params → lower confidence. Comp 3 (ρ) barely predicts anything in Y. The params move together primarily through the α-k axis to predict affect calibration and apathy, but not distress or fatigue.

---

## 6. Why Vigor Is a Single Trait (Not a Computation)

### The variance budget tells the story

After removing demand (choice-ratio normalization):

| Source | Choice | Vigor |
|---|---|---|
| Person (who you are) | 5% | **26%** |
| Conditions (threat, distance, attack) | 13% | **4%** |
| Noise/unexplained | 82%* | 70% |

*Choice "noise" is mostly irreducible Bernoulli variance from binary data, not actual noise.

**Choice is condition-driven:** The FET model captures how threat, distance, and effort conditions shape each trial's decision (model R²=0.45). Individual differences (ICC=0.05) are small — people flexibly adjust their choices to conditions.

**Vigor is person-driven:** After demand removal, conditions (threat, distance, attack) explain only 4% of within-person variance. Individual tonic pressing level dominates (ICC=0.26). People bring their own pressing rate to every trial regardless of how dangerous it is.

### Condition effects on vigor are real but tiny

| Condition | Effect (% of mean) | p |
|---|---|---|
| Threat (high − low) | −3.9% | 6×10⁻⁶ |
| Distance (far − near) | −3.1% | 0.002 |
| Attack (atk − noatk) | −2.8% | 10⁻⁴ |

A person pressing at 60% of capacity on a safe trial presses at ~58% on a dangerous trial. Statistically significant with 20K trials, but the signal-to-noise ratio is ~1:5 (effect ~0.04, within-person SD ~0.23). This is why per-subject modulation slopes (δ_T, δ_D) have negative split-half reliability — the within-person condition effects are too small relative to trial-to-trial noise to estimate stably.

### Motor capacity does NOT predict vigor

r(capacity, vigor) = −0.04, n.s. How fast you *can* press doesn't predict how fast you *do* press. α is a strategic effort allocation decision, not a physical constraint.

### The vigor model: two parameters (α tonic, ρ phasic) — Bayesian HBM

**Bayesian hierarchical model (NB16):** Two-window model with separate likelihoods, fit with NumPyro NUTS (4 chains × 2000, 0 divergences, 58s).
```
pre_enc_rate_ij  ~ Normal(α_i, σ_pre)          # [enc-2, enc]
terminal_rate_ij ~ Normal(γ_i + ρ_i·attack, σ_term)  # [trialEnd-2, trialEnd]
```

**α (tonic vigor):** Pre-encounter pressing rate, capacity-normalized = fraction of motor capacity deployed.
- μ_α=0.519 (52% of capacity). σ_α=0.196. SB=0.925. Bayes-OLS r=1.000.
- Window: [enc−2s, enc]. Capacity = 95th percentile of 1s-bin press rates.
- **NOT motor ability:** capacity→α r=+0.03, CalMax→α r=+0.10, onset_rate→escape r=−0.04 (all null). Motor ability predicts nothing; the fraction of capacity deployed predicts everything.
- **NOT task engagement:** Controlling for questionnaire RT, choice entropy, affect variability does not change α→escape (R²=0.71→0.72) or α→AMI (R²=0.12→0.12).
- **NOT strategic:** Onset pressing rate is flat across threat, distance, and choice (after removing mechanical demand confound of heavier cookies). No dynamic reallocation. Speed tier structure (within a tier, pressing faster doesn't change movement speed) removes incentive for fine-grained adjustment.
- **IS a stable default motor setting:** What fraction of what you CAN do, you actually DO. Like walking speed — a habitual set point, not a deliberate choice.
- Predicts: escape (r=+0.84), AMI apathy (r=+0.34), anxiety threat calibration (r=+0.26), mean anxiety (r=−0.16). All survive engagement controls.
- Shrinkage: 2.1% (already very reliable).

**ρ (phasic vigor):** Terminal attack boost, capacity-normalized.
- μ_ρ=0.526 (53% of capacity boost). P(μ_ρ>0)=1.0000. σ_ρ=0.129. SB=0.762. Bayes-OLS r=0.991.
- Window: [trialEnd−2s, trialEnd]. Captures defensive sprint under active predator pursuit.
- γ_i is a nuisance per-subject terminal baseline (μ_γ=0.092 — low, people idle in terminal non-attack).
- Shrinkage: 16.8% (ρ benefits substantially from hierarchical regularization).
- Does NOT predict outcomes, choice params, mental health, or quadrant identity — universal defensive response.

**α-ρ correlation: r=−0.237 (p<0.001).** Moderately anticorrelated. Fast tonic pressers have smaller sprint boosts — likely a ceiling effect (less room to accelerate when already pressing at ~70% capacity). This is real, not a structural artifact.

**Why two separate windows with separate likelihoods:**
- Pre-encounter and terminal are different behavioral states at different trial phases.
- Sharing a single α across both creates artifacts: terminal non-attack rate (0.09) ≠ pre-enc rate (0.52), forcing α to compromise and inducing spurious α-ρ correlations.
- Encounter-aligned post-enc [enc, enc+2] doesn't work — attack effect is too small in first 2s (ρ SB=0.28).
- γ_i absorbs the terminal baseline, letting ρ purely capture the attack contrast.

**Outputs:** `results/stats/vigor_hbm_posteriors.csv` (per-subject α, ρ with posterior SDs), `vigor_hbm_population.csv`, `results/model_fits/exploratory/vigor_hbm_idata.nc` (full MCMC trace).

**What was tested and rejected:**
- **δ_T (threat sensitivity):** SB=−0.03. Not a reliable individual difference. Conditions only explain 4% of within-person variance.
- **δ_D (distance sensitivity):** SB=−0.30. Not reliable.
- **Survival-guided model (vigor = α + δ·(1−S_trial)):** Per-subject R²=0.021. The survival computation doesn't govern vigor.
- **Trial-level rate (total presses / duration):** Conflates phases, produces paradoxical results (reversed escape predictions). Window approach is necessary.
- **Onset (first 2s of pressing):** SB=0.975 (very reliable) but correlates r=−0.50 with pre-enc α — different construct (motor initiation vs sustained effort) and only r=0.39 with pre-enc α. Potentially confounded with encounter time overlap.
- **Variable-length windows (first_press→enc, enc→end):** ρ SB=0.43 but uneven windows are reviewer-vulnerable.
- **RT (first press latency):** SB=0.988 but r=−0.81 with α — redundant, not a new dimension. Report in supplementary.

### This asymmetry IS the dissociation

Choice is computed fresh each trial from threat × distance → survival → softmax (13% condition-driven). Vigor is set by α and barely adjusts (4% condition-driven). The FET model captures the condition-responsive channel (choice) nearly perfectly. The trait-like channel (vigor) is outside the model's scope. And it's vigor — the channel the model doesn't capture — that determines survival.

---

## 7. The Unified 3-Parameter Framework: {k, β, α}

### The two models (separate, not unified)

**Choice model (L3_add):**
```
SV = R·S - k·E - β·(1-S)
S = (1-T) + T / (1 + λ·D)
choice ~ softmax(τ · ΔSV)
```
- k (per-subject): additive effort cost. "How much do I care about the cost of the hard option?"
- β (per-subject): subjective capture cost. "How much do I fear being caught?"
- λ, τ (population): escape kernel scale, choice temperature
- S mechanistically separates attack probability (1-T) from escape probability T·f(D)
- Additive effort and hyperbolic kernel both strongly favored by model comparison
- **α does NOT enter** — exhaustively tested, data rejects every placement

**Vigor model (Bayesian HBM):**
```
pre_enc_rate ~ Normal(α_i, σ_pre)
terminal_rate ~ Normal(γ_i + ρ_i·attack, σ_term)
```
- α (per-subject): fraction of motor capacity deployed. NOT motor ability.
- ρ (per-subject): universal phasic sprint during attack
- α is invisible to the choice computation but determines survival

### The three parameters

| Param | Source | Meaning | Key prediction | SB |
|---|---|---|---|---|
| k | Choice model (unified) | Effort discounting | Choice (R²=0.88), earnings | Posterior |
| β | Choice model (unified) | Threat bias (pure, after α handles distance) | Conf miscalibration, threat sensitivity | Posterior |
| α | Vigor HBM + choice survival function | Motor engagement / effective exposure | Escape (R²=0.73), apathy (R²=0.155) | 0.925 |

**z** (distance-threat scaling) → population-level structural parameter, supplementary.
**ρ** (phasic sprint) → universal defensive response (P(μ_ρ>0)=1.0), uniform across profiles, supplementary.

### Inter-correlations (L3_add model, α separate)
- **k-β: r = −0.14 (p=0.018)** — slight negative, effort-averse people are slightly less threat-sensitive
- **k-α: r = −0.05 (n.s.)** — effort aversion and motor setting are independent
- **β-α: r = +0.26 (p<0.001)** — people who deploy more capacity also have higher threat sensitivity in their affect (see anxiety calibration)

**k and α are independent.** The choice system (what you avoid) and the motor system (how hard you work) don't communicate. Tested within a unified model where α enters the survival function — the model rejects α (λ→∞, S≈(1-T)). People compute choice value from conditions (threat, difficulty) without incorporating their motor capability.

### α does NOT enter the choice model (exhaustively tested)
- α in survival function f(D/α): λ→∞, f→0, α effectively drops out (+16 ELBO marginal)
- α in effort (E/α): HURTS (−25 ELBO). Creates k-α confound (r=+0.25).
- α in effort×distance (E·D/α): HURTS (−294 ELBO)
- α in both effort and survival: cancels out (−3 ELBO)
- Normalizing D/3 doesn't fix: λ scales up to compensate, same degenerate solution
- Speed tier analysis: tier_diff predicts choice (49% vs 29%) but this is just the difficulty gradient
- **Conclusion:** People compute SV from visible conditions (difficulty, threat) + personal dispositions (k, β). Motor capability (α) is invisible to the choice computation.

### Quadrant profiles (unified 3-param model)

| Quad | N | k | β | α | Escape | Conf bias | AMI | PHQ-9 |
|---|---|---|---|---|---|---|---|---|
| HH | 74 | 1.4 | 0.9 | **0.68** | 52% | −0.34 | 30.8 | 9.0 |
| HL | 83 | 1.4 | **1.4** | 0.35 | **17%** | **+1.05** | 24.9 | 7.5 |
| LH | 72 | **4.6** | 0.4 | **0.67** | **62%** | **−1.19** | **31.8** | 7.9 |
| LL | 62 | **5.2** | 0.4 | 0.37 | 24% | +0.38 | 24.2 | 5.4 |

ANOVAs: k (F=161, η²=0.63), α (F=211, η²=0.69), β (F=17, η²=0.15), Escape (F=145, η²=0.60), Conf bias (F=54, η²=0.36), Apathy (F=14, η²=0.13), AMI (F=12, η²=0.11), PHQ-9 (F=3.8, FDR p=0.043).

**β distinguishes HH from HL** (same k ~1.4, same choice rate ~58%, but β=0.9 vs 1.4). High threat bias without high vigor = overconfidence + worst survival.

**ρ is uniform across all profiles** (~0.50-0.55). Universal defensive sprint, supplementary.

### LDA: two dimensions explain 99.8% (ρ contributes ~0%)

- **LD1 (72%):** α dominates (weight=−1.80). Separates high-vigor from low-vigor.
- **LD2 (28%):** k dominates (+0.99), with z (+0.52) and β (+0.56). Separates high-choice from low-choice.
- **LD3 (0.2%):** ρ loads here but explains nothing.
- **CV accuracy: 83% with 4 params, 83% with 5** — ρ doesn't help quadrant prediction.

### Supplementary: RT (first-press latency)

- SB=0.988. Extremely reliable but r=−0.81 with α — same underlying trait.
- RT by quadrant: HH=1.05s, HL=2.56s, LH=0.97s, LL=2.15s (mirrors α perfectly).
- Adds 1% to quadrant prediction (82.6→83.6%). Not a new dimension.
- Report as supplementary validation that α captures a general motor engagement trait that also manifests in initiation speed.

### Each quadrant's unique signature (Cohen's d vs rest)

- **HH:** low k (d=−1.15), high α (d=+1.43) — willing to work and does
- **HL:** low everything: z(−0.73), k(−0.74), β(−0.78), α(−1.49) — undifferentiated low effort
- **LH:** high everything: z(+0.48), k(+0.42), β(+0.64), α(+1.49) — selective, capable, cautious
- **LL:** high k (d=+1.12), low α (d=−1.28) — effort-averse everywhere

### Mental health

**Only AMI (apathy) survives FDR (3/16 measures).** Driven entirely by α (β=+0.325). Choice params contribute nothing. High α → more self-reported apathy — the "adaptive apathy" paradox.

**No interactions survive FDR.** k×α, z×α, β×α interaction terms add nothing (0/16 significant). The mental health link is α → AMI, full stop.

**Dissociation score (α residualized from k) → AMI:** r=+0.316, FDR-significant. People who press more than their k predicts report more apathy.

---

## 8. Key Effect Sizes (Final)

| Finding | Effect | Significance |
|---|---|---|
| Choice model fit | R²=0.45 | WAIC comparison |
| Unified 3-param → choice | R²=0.88 | k, β, α |
| Unified 3-param → vigor | R²=0.08 | 11× asymmetry |
| k-α independence (unified) | r=0.006 | Dissociation within unified model |
| Choice-vigor independence | r=−0.02 | CI [−0.13, +0.10] |
| β selectivity | diff=−0.56 | bootstrap p<0.0001 |
| CCA dimensions | r=0.91, r=0.29 | two pathways |
| Threat reversal | Δr=0.42 | Fisher z=5.07, LMM p=0.0004 |
| Vigor → escape | β=+0.09 | p=10⁻⁷⁷ |
| Choice → escape | β=−0.18 | hurts survival |
| Vigor triples escape | 53% vs 19% | p=10⁻²⁵ |
| Confidence miscalibration | R²=0.42 | F=50, p=10⁻²⁶ |
| 5-param → quadrant | 83% CV | chance=25% |
| α → Apathy factor | R²=0.155 | p=3×10⁻⁹ (factor analysis) |
| PLS 5 params → MH+affect | CV R²=0.039 | perm p=0.0000 |
| PLS Comp1 (α,k) → anx calib | R²=0.145 | r=0.538 |
| Affect ~ S_probe | β ≈ ±0.6 | p<10⁻¹³⁰ |
| Vigor variance: conditions | 4% | After demand removal |
| Vigor variance: person | 26% | ICC=0.26 |
| α reliability | SB=0.925 | Pre-enc 2s window (Bayesian HBM) |
| ρ reliability | SB=0.762 | Terminal 2s window (Bayesian HBM) |
| μ_α | 0.519 (52% capacity) | 95% HDI [0.497, 0.541] |
| μ_ρ | 0.526 (53% capacity) | P(>0)=1.0000, HDI [0.509, 0.542] |
| α-ρ correlation | r=−0.237 | Ceiling effect, not artifact |
| ρ across quadrants | F=0.6, n.s. | Universal response |
| RT-α redundancy | r=−0.81 | Same trait, supplementary |

---

## Technical Notes

### Vigor operationalization (FINAL)

**Two vigor parameters from two fixed 2s windows, estimated via Bayesian HBM (NB16):**
- **α (tonic):** Pre-encounter window [enc − 2s, enc]. Count keypresses / 2s / capacity (95th pctile 1s-bin rates). Hierarchical: α_i ~ Normal(μ_α, σ_α). Posterior mean per subject.
- **ρ (phasic):** Terminal window [trialEnd − 2s, trialEnd]. Same normalization. Hierarchical: terminal_rate ~ Normal(γ_i + ρ_i·attack, σ_term). γ_i is nuisance baseline. Posterior mean per subject.
- **OLS fallback:** α = subject mean of pre-enc rate. ρ = subject mean of (terminal attack rate − terminal non-attack rate). Correlates r>0.99 with Bayesian estimates.

**Why these windows:**
- Pre-encounter captures sustained anticipatory effort during active foraging (SB=0.91)
- Terminal captures defensive mobilization under maximal active threat (SB=0.46)
- Both fixed 2s — equal duration, no asymmetry concerns
- Both anchored to clear task events (encounter = predator spawn; trial end = escape/capture)
- Different behavioral states by design (anticipation vs active defense)

**Capacity normalization:** 95th percentile of each subject's 1-second bin press rates across all trials. This is NOT motor ability — capacity does not predict α (r=−0.04).

**Choice-ratio normalization:** Divide rate by group mean for choice=0 or choice=1. This removes demand (hard cookies require more pressing) while preserving between-subject variance. Do NOT use per-subject normalization (kills ICC) or effort×distance residualization.

### Task design constraints
- **Effort × distance confound:** E=0.6/D=1, E=0.8/D=2, E=1.0/D=3. Only 3 difficulty levels, not crossed. Cannot separate effort aversion from distance aversion within the choice model. k captures both.
- **Only 9 unique trial types:** 3 difficulty × 3 threat. Model comparison is fitting parametric functions to 9 cells.
- **Speed tiers kill fine-grained vigor adjustment:** Within a tier, pressing faster doesn't change movement speed. Most people are in the same tier for E=0.4 and E=0.6 (100% full speed). Differentiation only at E=1.0 (38% full, 61% half).
- **Pre-encounter window confound:** [enc-2, enc] overlaps with onset [firstPress, firstPress+2] for D=1 (67% overlap) and D=2 (59% overlap). Only D=3 has clean separation. The apparent "effort maintenance" dropoff is largely this overlap artifact.
- **Low cookie is always D=1, E=0.4:** No variation in the safe option. Choice is always "risky-varied vs safe-fixed."

### Critical bugs/issues
- **encounterTime frame mismatch (see 2i).** Vigor pipeline shifted encounterTime to effort-onset-relative, but keypress times are trial-start-relative. Use original encounterTime from `processed_trials.pkl` (stage2). Values: 2.5s (D≤5), 3.5s (D=7), 5.0s (D=9).
- **Trial-level rate is misleading.** Total presses / trial duration conflates phases and produces reversed escape predictions. Always use windowed measures.

### Environment
- Conda env: `effort_foraging_threat` (python 3.11, pyarrow 23.0.1)
- Base anaconda pyarrow 19.0.0 cannot read parquet files from this env
- `encounter_time` exists in ALL trials (scheduled predator time), not just attack trials

### Dead ends
ODE vigor model, continuous temporal alignment, per-subject effort×distance residualization, Spearman marginal features, 20Hz smoothed vigor pipeline, 20Hz encounter spike, vigor condition-modulation params as individual differences (δ_T SB=−0.03, δ_D SB=−0.30), survival-guided vigor model (R²=0.02 per subject), trial-level vigor rate, onset window as α (captures initiation not sustained effort), variable-length windows

### Supplementary analyses to include
- RT (first-press latency): SB=0.988, r=−0.81 with α, mirrors quadrant structure
- Vigor variance budget: 26% person, 4% conditions, 70% noise
- Window comparison table: fixed enc±2s, terminal, variable, onset+terminal, trial-level
- Encounter-centered threat/attack effects on pre/post pressing
- PLS: encounter-window features → choice params (CV R²=0.12)
