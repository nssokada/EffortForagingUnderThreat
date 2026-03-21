# Hypotheses

Tracking all hypotheses — tested, supported, refuted, and open. Each entry includes the hypothesis, test method, result, and implications.

Status key: ✅ Supported | ❌ Refuted | ⚠️ Partial | 🔲 Untested | 🚫 Dead end

---

## H1 — Effort and threat are integrated into a unified value computation
**Statement:** Humans combine energetic cost and exposure-dependent threat into a single subjective value signal when making foraging decisions, rather than treating these dimensions independently.
**Test:** Compare effort-only, threat-only, and combined models via WAIC.
**Result:** ✅ Supported. Combined effort–threat model (FETExponentialBias) substantially outperforms all single-dimension models. Exponential effort discounting + survival function + threat bias is the winning form.
**Notebooks:** NB02-01, NB02-02

## H2 — Effort discounting follows an exponential form
**Statement:** Among candidate discount functions (exponential, hyperbolic, quadratic, linear), exponential best describes how effort reduces subjective reward value.
**Test:** WAIC comparison across 4 discount forms within the combined effort–threat model.
**Result:** ✅ Supported. Exponential discounting yields lowest WAIC.
**Notebooks:** NB02-01

## H3 — There is residual threat sensitivity beyond expected value
**Statement:** Threat influences choice beyond its effect on survival probability — captured by a subject-specific threat bias parameter (β).
**Test:** Compare FETExponential (no bias) vs FETExponentialBias.
**Result:** ✅ Supported. Bias-extended model wins on WAIC. β is right-skewed (mean=1.44) indicating most subjects show some additional threat aversion.
**Notebooks:** NB02-01, NB02-02

## H4 — Model-derived survival predicts subjective affect
**Statement:** Trial-level survival probability computed from fitted parameters predicts self-reported anxiety (−) and confidence (+).
**Test:** Mixed-effects models: affect ~ S_probe + (1|subj).
**Result:** ✅ Supported. Anxiety: β=−0.602, p<0.001. Confidence: β=+0.632, p<0.001.
**Notebooks:** NB04-03

## H5 — Individual differences in model parameters modulate emotional reactivity
**Statement:** z, k, and β differentially shape how affect scales with task variables.
**Test:** Moderation models: affect ~ task_var × param + (1|subj); between-subjects OLS for trait affect.
**Result:** ⚠️ Partial.
- z → chronic confidence deficit (β=−0.199, p_fdr=0.013) ✅
- z × p_threat interaction: NULL (p>0.09) ❌
- k → trait anxiety (+0.146, p=0.019) and trait confidence (−0.163, p=0.010) ✅
- β: no significant moderation of affect ❌
**Notebooks:** NB04-03

## H6 — Model-derived survival predicts defensive action vigor
**Statement:** Lower survival probability → higher vigor, even after controlling for effort level.
**Test:** Trial-level LMM: vigor ~ S_trial + effort + (1|subj).
**Result:** ⚠️ Partial. Terminal mean only: S_trial β=−0.011, p_fdr=0.0002. Other phases n.s. This is a trial-level within-subject effect; at the between-subject level, threat does not significantly modulate any phase DV (all ANOVAs p>0.20).
**Notebooks:** NB03-10, NB03-12

## H7 — Model parameters dissociate across vigor phases
**Statement:** z, k, and β make distinct contributions to different temporal phases of vigor.
**Test:** Subject-level regressions (NB06, NB08, NB12 Check 4); functional regression at 0.1s bins (NB13).
**Result:** ✅ Supported. Clean dissociation pattern:
- z: positive for onset/anticipatory (+0.13 to +0.22), flips negative for enc_spike (−0.12) and term_slope (−0.19)
- k: global suppressor across all phases (−0.07 to −0.22), strongest at terminal
- β: boosts onset slope (+0.18) and post-encounter (+0.14)
- Effect sizes small: R²=0.024–0.062 across phases
**Notebooks:** NB03-06, NB03-08, NB03-12, NB04-04

## H8 — z modulates survival–vigor coupling
**Statement:** Participants with higher hazard sensitivity (z) show stronger trial-by-trial coupling between survival and vigor.
**Test:** LMM: vigor ~ S_trial × z_i + (1|subj).
**Result:** ❌ Refuted. S_trial × z_i interaction: p=0.12 for terminal mean, n.s. after FDR.
**Notebooks:** NB03-10

## H9 — Anxiety causally drives vigor (serial affect→motor)
**Statement:** Trial-level anxiety ratings predict subsequent vigor beyond what threat level explains.
**Test:** Concurrent LMMs, predictive LMMs, residual anxiety models, phase-specific LMMs, functional regression across time bins.
**Result:** ❌ Refuted at every level tested. Complete null.
**Interpretation:** The common structure is in the shared INPUT (survival computation), not in sequential affect→motor causation.
**Notebooks:** NB04-04

## H10 — Vigor and affect are cross-correlated at individual-difference level
**Statement:** Subjects who show stronger vigor responses also show stronger affective responses.
**Test:** 15 vigor × affect correlation pairs with FDR correction.
**Result:** ❌ Refuted. None survive FDR (highest r=+0.124, p_fdr=0.196).
**Notebooks:** NB04-03

## H11 — Attack triggers a selective phasic vigor spike
**Statement:** Predator encounter triggers a vigor spike beyond what effort demands explain.
**Test:** Compare vigor_norm vs vigor_resid at encounter; attack vs non-attack within threat levels.
**Result:** ❌ Refuted. Spike disappears in vigor_resid (p=0.644). After controlling for threat, the attack effect on enc_spike is non-significant (p=0.126). The uncorrected attack effect was driven by the threat level confound (high threat → more attacks AND more vigor).
**Notebooks:** NB03-09, NB03-12

## H12 — ODE kinetics capture individual differences in vigor mobilization speed
**Statement:** Exponential rise time (α) to encounter is individually parameterizable and relates to model parameters.
**Result:** 🚫 Dead end. α is degenerate.
**Notebooks:** NB03-11

## H13 — Psychiatric symptom dimensions relate to model parameters
**Statement:** Transdiagnostic variation (DASS-21, PHQ-9, OASIS, STAI-T, AMI, MFIS) correlates with z, k, β.
**Result:** 🔲 Untested (blocked). NB07 requires factor analysis of psych battery.
**Notebooks:** NB03-07, NB04-01, NB04-02

## H14 — Two-system architecture (deliberative vs. reactive)
**Statement:** Behavior is organized into (a) a deliberative system governed by fitted parameters (z, k, β → trait affect, tonic vigor) and (b) a reactive system driven by immediate state (threat/distance → phasic affect, encounter → vigor surge), with the two systems operating in parallel but NOT cross-correlated.
**Test:** NB12 phase-based diagnostics; ICC dissociation; param × phase regressions; attack contrast analysis.
**Result:** ⚠️ Reframed. The "two systems" label overstates the evidence. What we actually have:
- **Individual differences in pressing style** are stable (ICC up to 0.74), reliably measured (SB r > 0.83), and linked to choice model parameters (R²=2–6%)
- **Trial-level threat effects** exist in within-subject LMMs but wash out at the between-subject level
- **Reactive encounter response** is not a clean signal: disappears in demand-residualized vigor and after threat control
- **One genuine reactive effect:** terminal slope increases on attack trials (p=0.00001 after threat control), but this is generic (not parameter-linked)
- **Better framing:** Trait-level correspondence between choice valuation and execution style, plus a generic terminal sprint response. Not two "systems."
**Notebooks:** NB03-06, NB03-08, NB03-09, NB03-12, NB04-03, NB04-04

## H15 — Vigor story is about individual differences, not real-time tracking
**Statement:** The vigor data's contribution to the paper is showing that stable individual differences in how people value effort (k) and threat (z) manifest in their motor execution style, rather than showing that a survival computation runs in real time during action execution.
**Test:** NB12 phase-based diagnostics.
**Result:** ✅ Supported.
- ICC dissociation: onset_mean=0.74 (trait) vs enc_spike=0.18 vs term_slope=0.03 (state)
- Threat ANOVAs: all n.s. at group level (p=0.20–0.89)
- Param regressions: all significant (p<0.02) but small R² (2–6%)
- Attack contrast after threat control: only terminal slope survives, parameter-independent
**Implication:** Reframe the paper's vigor section from "the same computation governs choice and vigor" to "individual differences in threat/effort valuation predict individual differences in motor execution strategy."
**Notebooks:** NB03-12

---

## Dead Ends (closed)
- **H11 — Encounter spike:** demand-driven artifact + threat confound
- **H12 — ODE kinetics:** degenerate parameters
- **Continuous temporal alignment of vigor:** confounded by trial structure (distance × duration)
- **Attack contrast as imminence signal:** confounded with threat; only terminal slope survives and is generic
- **Within-trial survival tracking:** not testable with this task design

---

## Open / Future Hypotheses

### H13 — Psychiatric symptom dimensions relate to model parameters
**Statement:** Transdiagnostic variation (DASS-21, PHQ-9, OASIS, STAI-T, AMI, MFIS) correlates with z, k, β.
**Test:** Bivariate correlations + multiple regression, N=293.
**Result:** ❌ Essentially null. 0/39 bivariate correlations survive FDR. z shows consistent weak negative associations with anxiety/fatigue measures (r=−0.10 to −0.18) but no individual test survives correction. k and β show no relationships.
**Note:** STAI-Trait scoring bug found and fixed (State reverse items were applied to Trait). Even after fix, SD=5.8 is low and STAI-T correlates negatively with all distress measures — may still be problematic.
**Interpretation:** Model parameters capture task-specific individual differences, largely orthogonal to broad psychiatric symptom dimensions.
**Notebooks:** NB psych analysis

## H19 — Subjective affect ratings track model-derived survival (metacognitive consistency)
**Statement:** Per-subject correlation between affect ratings and model-derived S_probe is substantial, indicating that subjective reports are internally consistent with the survival computation.
**Test:** Per-subject Spearman r(rating, S_probe) using each subject's own fitted z.
**Result:** ✅ Supported.
- Anxiety × S_probe: M(r)=−0.341, t=−16.45, p=2.5×10⁻⁴³
- Confidence × S_probe: M(r)=+0.340, t=+15.48, p=8.4×10⁻⁴⁰
**Notebooks:** NB04-05

## H20 — k predicts metacognitive calibration accuracy
**Statement:** People who discount effort more (high k) show worse calibration of their affect ratings to objective threat conditions.
**Test:** Per-subject threat calibration (Spearman r between rating and threat) regressed on z, k, β.
**Result:** ✅ Supported.
- k × anxiety threat calibration: r=−0.309, p<0.001
- k × confidence threat calibration: r=−0.210, p<0.001
- Joint model for anxiety: R²=0.121, driven by k (β=−0.342)
- **Interpretation:** k captures engagement — high-k people discount effort, press less, AND report less differentiated affect. Low-k people are more engaged across choice, motor, and affective domains.
**Notebooks:** NB04-05

## H21 — z predicts distance-specific affect calibration
**Statement:** People with higher z (who differentiate distance more in choices) also show better anxiety calibration to distance in their probe ratings.
**Test:** Per-subject distance calibration × z.
**Result:** ✅ Supported. r=+0.152, p=0.010. Clean, specific: z governs D^z in the choice model, and high-z people's anxiety ratings also differentiate distance more.
**Notebooks:** NB04-05

## H22 — Confidence predicts escape outcomes (metacognitive accuracy)
**Statement:** Higher confidence ratings predict actual escape on subsequent attack trials.
**Test:** LMM: escaped ~ rating_z + (1|subj), with and without controls.
**Result:** ❌ Refuted. Raw effect (p=0.017) disappears after controlling for threat and distance (p=0.78). Confidence tracks conditions, not outcomes beyond conditions.
**Notebooks:** NB04-05

## H23 — Multivariate encounter-window vigor features predict model params (PLS)
**Statement:** A multivariate set of vigor features from the encounter window (enc−2s to enc+2s) predicts choice model parameters with cross-validated generalization.
**Test:** PLS regression on 12 count-based encounter-window features → z, k, β. 5-fold CV + permutation.
**Result:** ✅ Supported (corrected frame).
- 2 comp: train R²=0.144, CV R²=0.093, permutation p=0.000
- 3 comp: train R²=0.162, CV R²=0.117
- Per-param (2 comp): **k CV R²=0.199**, z CV R²=0.072, β CV R²=−0.039
- Compare: 20Hz PLS had CV R²=−0.071 (overfit)
- **Note:** Earlier run with wrong encounterTime frame gave different feature correlations. All results below use the corrected trial-start-relative encounter time.
**Notebooks:** To be formalized

## H24 — Distance modulation of pre-encounter pressing is primary bridge to k
**Statement:** Individual differences in how much people modulate pressing rate by distance in the pre-encounter window is the strongest single predictor of choice model parameters — particularly k (effort discounting).
**Test:** Partial regression slopes (rate ~ threat + distance + attack per subject), correlated with params.
**Result:** ✅ Supported.
- **dist_pre × k: r=−0.435** (strongest single vigor→param correlation)
- dist_pre × z: r=−0.270
- dist_pre × β: r=−0.212
- dist_trans × k: r=+0.407 (flipped: high-k people show bigger distance-dependent encounter transitions)
- PLS Component 1 dominated by dist_pre (+0.668) and dist_trans (−0.658)
- **Interpretation:** k governs effort discounting in choices AND manifests in distance-dependent pressing. High-k people press at more uniform rates across distances before encounter. The distance modulation is the motor expression of effort sensitivity.
**Notebooks:** To be formalized

## H25 — ❌ RETRACTED: threat_mod_onset × k was a confound
**Original claim:** threat_mod_onset × k: r=+0.382 (Spearman).
**Correction:** With partial regression slopes (controlling for distance and attack), this effect disappears. The marginal Spearman was picking up distance effects misattributed to threat because threat covaries with chosen distance.
**Lesson:** Always use partial slopes, not marginal correlations, when predictors are correlated.

## H26 — Encounter transition is threat-modulated (corrected frame)
**Statement:** The pre→post encounter vigor transition scales with threat probability.
**Test:** Between-subject ANOVA on subject×threat means of encounter transition (corrected encounter time).
**Result:** ✅ Supported.
- Transition: F=7.84, **p=0.0004**
- Low: −0.013, Med: +0.049, High: +0.064
- Post-encounter pressing also threat-modulated: F=6.88, p=0.001
- **Note:** This was non-significant (F=0.91, p=0.40) in the wrong-frame analysis. The time frame bug was hiding a real effect.

## H27 — Attack-driven encounter transition (clean imminence signal)
**Statement:** When the predator actually appears, post-encounter pressing increases relative to pre-encounter, beyond what threat level predicts.
**Test:** Attack effect on encounter transition, within threat level.
**Result:** ✅ Supported.
- Post-encounter: diff=+0.033, t=6.95, **p=2×10⁻¹¹**
- Transition: diff=+0.042, t=5.29, **p=2×10⁻⁷**
- Pre-encounter: diff=−0.009, t=−1.01, p=0.31 (correctly null — no foreknowledge)
- **Note:** Previous wrong-frame result showed pre-encounter attack suppression (t=−4.60) which was likely artifactual from the misaligned window. With corrected frame, pre-encounter is properly null.

## H28 — Threat modulation of encounter transition is param-linked
**Statement:** Per-subject threat slope of the encounter transition correlates with all three model parameters.
**Test:** Partial regression slope of transition ~ threat (controlling for distance + attack), correlated with z, k, β.
**Result:** ✅ Supported (modest effects).
- threat_trans × z: r=−0.117, p<0.05
- threat_trans × k: r=−0.131, p<0.05
- threat_trans × β: r=−0.128, p<0.05
- **Interpretation:** Higher z/k/β → smaller threat-dependent encounter transitions. People with more differentiated choice parameters show more uniform encounter transitions across threat levels.

## H29 — Choice and vigor are independent behavioral dimensions
**Statement:** How often a person chooses the high-effort cookie is uncorrelated with how hard they press during execution.
**Test:** Pearson correlation between p(choose high) and mean pre-encounter pressing rate (capacity-normalized, choice-ratio adjusted).
**Result:** ✅ Supported. r=−0.018, p=0.76. Four quadrants roughly equally populated (22-28%).

## H30 — β creates the choice-vigor dissociation
**Statement:** β (threat bias) suppresses risky choice but does not affect motor execution, producing a dissociation between valuation and action.
**Test:** β → p_high vs β → vigor; off-diagonal quadrant comparison.
**Result:** ✅ Supported.
- β → choice: r=−0.488, p<0.001 (strong suppression)
- β → vigor: r=+0.109, p=0.062 (null)
- HL group β=0.59 vs LH group β=2.20 (t=−6.18, p<0.001)
- **Interpretation:** β acts on the survival computation that governs choice but does not constrain the motor system. High-β people are threat-biased in decisions but physically capable — they choose safe but press hard.

## H31 — Vigor dominates escape outcomes over choice
**Statement:** How hard you press predicts escape from predators far better than what you chose.
**Test:** escape ~ choice_z + vigor_z + choice×vigor: R²=0.66.
**Result:** ✅ Supported.
- Vigor β=+0.795, choice β=−0.160
- Choosing hard actually hurts escape (farther from safety)
- HH escape=53%, LH escape=60%, HL escape=19%, LL escape=25%

## H32 — Threat reverses the choice-vigor relationship
**Statement:** At low threat, choice and vigor align (people who choose hard press hard). At high threat, they decouple (people who choose easy press hard).
**Test:** Per-threat-level correlation between p_high and vigor.
**Result:** ✅ Supported.
- Threat=0.1: r=+0.196, p=0.001
- Threat=0.5: r=+0.013, n.s.
- Threat=0.9: r=−0.219, p<0.001
- **Interpretation:** Under high threat, the conservative-but-capable (LH) strategy emerges. β-driven threat bias suppresses choice but not execution, and this effect intensifies with threat.

## H33 — Off-diagonal groups differ in confidence, calibration, and apathy
**Statement:** HL (choose hard, press soft) and LH (choose easy, press hard) differ in metacognitive accuracy and self-reported apathy.
**Test:** t-tests on affect/psych measures between HL and LH groups.
**Result:** ✅ Supported.
- Confidence: HL=3.40 > LH=2.76, p=0.003 (HL is overconfident relative to ability)
- Anxiety calibration: LH=0.36 > HL=0.22, p=0.007 (LH is better calibrated)
- AMI apathy: LH=32.2 > HL=25.4, p<0.001 (LH reports more apathy but performs better)

## H34 — Formal dissociation: β → choice ≠ β → vigor (bootstrap test)
**Statement:** β's regression coefficient for choice is significantly different (and opposite-signed) from its coefficient for vigor.
**Test:** 10K bootstrap, difference in standardized betas.
**Result:** ✅ Supported.
- β → choice: −0.409 [−0.570, −0.310]
- β → vigor: +0.147 [+0.019, +0.271]
- Difference: −0.555 [−0.802, −0.380], p=0.0000
- β suppresses choice but slightly BOOSTS vigor — opposite directions.

## H35 — CCA recovers two independent param→behavior pathways
**Statement:** CCA on {z,k,β} → {choice,vigor} yields two significant canonical dimensions — one for choice, one for vigor.
**Test:** sklearn CCA + MANOVA.
**Result:** ✅ Supported.
- Dim 1: r=0.909, p=10⁻¹¹³ → maps almost exclusively to Choice
- Dim 2: r=0.289, p=5×10⁻⁷ → maps almost exclusively to Vigor
- MANOVA Wilks' λ significant for all three params (all p≈0)

## H36 — Cross-level LMM confirms threat reverses choice-vigor coupling
**Statement:** The interaction between subject-level choice tendency and trial-level threat on vigor is significant.
**Test:** LMM: vigor_trial ~ choice_subj_z × threat_z + dist_z + (1|subj)
**Result:** ✅ Supported.
- choice × threat: β=−0.022, z=−3.54, p=0.0004
- Survives with random slopes: p=0.002
- Fisher z-test on per-threat correlations: z=5.07, p<0.0001

## H37 — Trial-level vigor predicts escape beyond choice
**Statement:** Pre-encounter pressing rate predicts escape on attack trials, controlling for choice, threat, and distance.
**Test:** LMM: escaped ~ vigor_z + choice_z + threat_z + dist_z + (1|subj). N=10,257 attack trials.
**Result:** ✅ Supported.
- Vigor: β=+0.091, p=10⁻⁷⁷
- Choice: β=−0.177, p≈0 (choosing hard HURTS escape)
- ΔAIC for adding vigor to choice-only model: 341

## H38 — Confidence miscalibration tracks the choice-vigor dissociation
**Statement:** The choice-vigor quadrants predict confidence miscalibration (confidence relative to actual escape rate).
**Test:** ANOVA on confidence bias (conf_z − escape_z) across quadrants; regression on continuous choice/vigor.
**Result:** ✅ Supported. Strongest affect finding.
- ANOVA: F=50.2, p=10⁻²⁶
- HL overconfident (+0.98), LH underconfident (−1.18)
- R²=0.415: choice β=+0.423 (drives overconfidence), vigor β=−0.783 (drives accurate calibration)

## H39 — AMI apathy tracks vigor, not choice
**Statement:** Self-reported apathy (AMI) correlates with vigor (positive — high pressers report more apathy) but not choice.
**Test:** AMI ~ choice + vigor + interaction, with FDR correction.
**Result:** ✅ Supported. Survives FDR.
- Vigor β=+0.311, R²=0.093
- Choice β=−0.056 (n.s.)
- "Adaptive apathy" — high-vigor people report not wanting to do things but execute well.

---

### Open / Future

### H15 — Confirmatory replication (N=350)
**Status:** 🔲 Untested. Data not yet added to repo. Highest priority — the full Phase 0-6 pipeline needs replication.

### H18 — Terminal sprint as generic defensive response
**Status:** ✅ Supported. Supplementary finding.
