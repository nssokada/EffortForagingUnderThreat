# Pipeline State

Current execution status of each notebook and script in the analysis pipeline.
Last updated: 2026-06-08 (log(ω/κ) robust pooled Bayesian → AMI_Social replicates).

---

## 2026-06-09 (followup-2) — ★★★ Behavioral threat-sensitivity → AMI Social/Behavioural REPLICATES CLEAN both samples; clinical typology visible in behavior

**Script:** `scripts/analysis/behavioral_betas_deep_dive.py` ✅. Tests behavioral betas against all 13 clinical scales, AMI subscales, per-sample, clinical-typology quadrants.

**Output:** `results/stats/affect_analysis/behavioral_betas_deep_dive.csv` ✅.

**HEADLINE finding — cleanest cross-sample replication anywhere in this session:**

β_T_choice → AMI_Total:
- Exploratory: β = -0.185 ★ (HDI [-0.295, -0.069])
- Confirmatory: β = -0.222 ★ (HDI [-0.338, -0.110])

threat_sens_composite (β_T_choice + β_T_vigor mean) → AMI_Total:
- Exploratory: β = -0.157 ★
- Confirmatory: β = -0.263 ★

**Both behavioral signatures survive 95% HDI in BOTH samples.** No (ω, κ)-based finding ever did this.

**Specificity findings:**
- Effect specific to AMI Social (β = -0.198 ★) and AMI Behavioural (β = -0.166 ★)
- AMI Emotional null (β = -0.048, n.s.)
- ALL anxiety/depression scales null (including ANX+DEP composite from §4.79)

**Clinical typology (Pure Apathy vs Pure Distress on behavioral threat-sensitivity):**
- threat_sens_composite contrast = -0.442 ★ (HDI [-0.716, -0.177]) — 0.44 SD separation

**Implication for paper:** Lead with behavioral findings (β_T_choice → AMI). §4.79 ANX+DEP comorbidity finding is fragile — doesn't appear in raw behavior.

See discoveries §4.81.

---

## 2026-06-09 (followup) — ★★★ Raw behavioral betas outperform (ω, κ) at predicting AMI; ANX+DEP doesn't cross-validate

**Script:** `scripts/analysis/behavioral_betas_predict_clinical.py` ✅. Per-subject behavioral betas extracted from `data/model_input_{exp,conf}/choice_trials.csv` and `vigor_cell_means.csv`.

**Key results (pooled N=571):**

| Behavioral beta | β → AMI_Total | β → ANX+DEP |
|---|---|---|
| **beta_T_choice** | **-0.202 ★** | -0.019 (null) |
| beta_TxD_choice | +0.116 ★ | +0.010 (null) |
| **beta_T_vigor** | **-0.110 ★** | -0.040 (null) |

**Predictive R² comparison (pooled, AMI_Total):**
- (ω, κ) only: in-sample 0.012, CV -0.028
- **5 behavioral betas only: in-sample 0.053, CV +0.001**
- Both: in-sample 0.061, CV -0.003

**Behavioral betas predict AMI ~4× better in-sample than (ω, κ), only behavioral betas reach positive CV R².**

ANX+DEP doesn't CV from either approach (all R² negative). §4.75-§4.79 comorbidity finding is real in Bayesian but doesn't generalize.

See discoveries §4.80.

---

## 2026-06-09 — ★★★ STAI direction FIX + final pooled headline (use STAI_Trait_FIXED, not STAI_Trait_corrected)

**Script:** `scripts/analysis/fix_stai_and_rerun_full.py` ✅. Fixes STAI direction bug (§4.78) by anchoring to DASS_Anxiety, then reruns headline.

**The bug:** §4.67's PC1-sign reverse-coding flipped STAI items to align with PC1, but PC1's sign was arbitrary and pointed in the "calmness" direction. Result: STAI_Trait_corrected was measuring opposite of intended construct.

**Outputs:**
- `results/stats/affect_analysis/stai_fixed_exp.csv` ✅
- `results/stats/affect_analysis/stai_fixed_con.csv` ✅

**FINAL HEADLINE (pooled N=571, Student-t):**

`log(ω)_z ~ AMI_Total + ANX+DEP_FIXED + log(κ)`
- AMI_Total β=+0.135 ★ [+0.055, +0.214]
- ANX+DEP_FIXED β=-0.084 ★ [-0.165, -0.008]
- log_κ β=+0.370 ★

**Clinical typology (median-split quadrants on AMI × ANX+DEP_FIXED):**
- Pure Apathy (N=104): ω_z = +0.189 (highest)
- Comorbid (N=172): ω_z = +0.015
- Healthy (N=182): ω_z = -0.063
- Pure Distress (N=113): ω_z = -0.094 (lowest)

PureApathy − PureDistress contrast: β=+0.266 ★ (HDI [+0.025, +0.505]).

**Use FIXED STAI in all future analyses, NOT STAI_Trait_corrected.** Saved to `clinical_scores_corrected_*.csv` is now superseded by `stai_fixed_*.csv`.

See discoveries §4.78 (STAI bug) and §4.79 (final corrected analysis).

---

## 2026-06-08 (followup-12) — ⚠ Cross-sample replication of ANX+DEP composite: confirmatory ★, exploratory directional only

**Script:** `scripts/analysis/anxdep_composite_replication.py` ✅. Re-fit the §4.75 model in each sample.

**Results:**
- Pooled (N=571): AMI_Total ★ β=+0.134, ANX+DEP ★ β=-0.092
- Confirmatory (N=281): AMI_Total ★ β=+0.163, ANX+DEP ★ β=-0.114
- Exploratory (N=290): AMI_Total β=+0.108 (clips 0, P>0=0.97), ANX+DEP β=-0.069 (clips 0, P<0=0.89)

Direction consistent in both samples; only confirmatory meets full 95% HDI in both. Effects probably real but at threshold of detectability in exploratory.

See discoveries §4.76.

---

## 2026-06-08 (followup-11) — ★★★ ANX+DEP composite (7-scale) reveals OPPOSITE-DIRECTION clinical effect to AMI

**Script:** `scripts/analysis/anxiety_depression_one_more_look.py` ✅. Honest re-look at anxiety/depression effects: directional probability + composites + WAIC + subgroup.

**Headline:** In a Bayesian regression `log(ω) ~ AMI_Total + ANXDEP_composite_7 + log(κ)` on pooled N=571:
- AMI_Total β=+0.134 ★ (positive — apathy increases ω)
- **ANX+DEP composite β=-0.092 ★ (negative — anxiety+depression decreases ω)**
- Both effects survive 95% HDI in the SAME model

Composite = z-mean of 7 scales (DASS_Anx, DASS_Dep, DASS_Stress, STAI_corrected, OASIS, STICSA, PHQ9).

This is the **comorbidity finding the paper has been searching for**: two distinct clinical dimensions, opposite directions, same parameter.

WAIC says individual scales hurt the model (kitchen-sink worst); composite is the right level of aggregation.

See discoveries §4.75.

---

## 2026-06-08 (followup-10) — ⚠ (ω, κ) profile typology: Type A vs Type B AMI_Social contrast β=+0.29 ★ but fully captured by linear ω

**Script:** `scripts/analysis/omega_kappa_profile_clinical.py` ✅. Tests median-split quadrants + polar decomposition + typology-beyond-linear test.

**Outputs:**
- `results/stats/affect_analysis/omega_kappa_profile_clinical.csv` ✅
- `results/figs/affect_analysis/omega_kappa_AMI_scatter.png` ✅

**Key result:** Type A (high ω, low κ) vs Type B (low ω, high κ) differ by 0.29 SD on AMI_Social ★. But after controlling for continuous log(ω) + log(κ), typology indicators are null — the contrast IS the linear ω effect coarsened.

**Paper recommendation:** Use typology as a narrative device in Discussion/figures; report continuous AMI_Total → log(ω) in Results. The κ axis adds nothing beyond the structural ω-κ correlation.

See discoveries §4.74.

---

## 2026-06-08 (followup-9) — ★★★ Kitchen-sink TOTALS (recommended PRIMARY specification): AMI_Total → log(ω) β=+0.148 ★

**Script:** `scripts/analysis/kitchen_sink_totals.py` ✅. Replaces 11-subscale kitchen-sink with 7-total kitchen-sink (one total per questionnaire). All inputs use corrected STAI.

**Output:** `results/stats/affect_analysis/kitchen_sink_totals.csv` ✅.

**Primary spec (recommended for paper):**
- 7 clinical totals + log(κ) covariate, Student-t, N=570
- AMI_Total → log(ω) β=+0.148 ★ (HDI [+0.062, +0.225])
- All other clinical totals NULL; κ axis NULL

**Sensitivity:** Swap AMI_Total → AMI_Social. β=+0.168 ★ (HDI [+0.087, +0.246]). Effect slightly stronger because Social subscale carries the unique variance; Behavioural/Emotional dilute slightly when bundled in Total.

See discoveries §4.73.

---

## 2026-06-08 (followup-8) — ★★★ FINAL clean rerun with corrected STAI: AMI_Social → log(ω) is the ONLY robust clinical signal

**Script:** `scripts/analysis/recompute_corrected_clinical_and_rerun.py` ✅. Properly applies STAI reverse-coding fix BEFORE running headline analyses (prior runs were on broken STAI).

**Outputs:**
- `results/stats/affect_analysis/clinical_scores_corrected_exp.csv` ✅
- `results/stats/affect_analysis/clinical_scores_corrected_con.csv` ✅
- `results/stats/affect_analysis/headline_corrected_results.csv` ✅

**Final headline (corrected, kitchen-sink ω with 11 clinical scales + log_κ control, Student-t, N=570):**
- AMI_Social → log(ω) β=+0.159 ★ — **the only surviving clinical predictor**
- All other clinical scales (DASS subscales, OASIS, STICSA, MFIS, PHQ9, AMI other subscales, corrected STAI): NULL
- κ axis: no surviving clinical predictor

**Retracted as STAI-broken-scale artifacts:**
- §4.63 multivariate STAI → ω β=+0.114 ★
- §4.71 OASIS suppression → ω β=-0.120 ★ (doesn't survive kitchen-sink with corrected STAI)
- Previous κ kitchen-sink STAI → κ β=-0.133 ★ (corrected STAI: β=-0.074 n.s.)

**Correlation between broken and corrected STAI: r = +0.057.** They're essentially independent scales — the broken one was measuring noise.

See discoveries §4.72.

---

## 2026-06-08 (followup-7) — ⚠️ PARTIAL RECOVERY: F3 direction replicates under suppression + focused-subset specifications

**Script:** `scripts/analysis/anxiety_replication_attempts.py` ✅. Five replication attempts (kitchen-sink, AMI_Social-control, F3-focused STAI subset, PC1, sample-split).

**Findings:**
- **OASIS_Total → log(ω) β=-0.120 ★** when controlling for AMI_Social (suppression effect)
- **F3-focused STAI subset → log(κ) β=-0.084 ★** (8 items, items 0/1/4/7/10/14/15/19)
- Confirmatory-only STAI → log(κ) β=-0.134 ★ (fails cross-sample)
- All anxiety scales trend in F3 direction in kitchen-sink but none survive univariately

**Revised verdict:** §4.70 was too strong. F3 effect is directionally real but weak; effect size is at the threshold for individual subscales. Latent factor aggregates the diffuse signal.

**Recommended paper framing:** Two raw-scale findings on log(ω):
1. AMI_Social → +ω (apathy increases vigilance)
2. OASIS_Total → -ω after AMI_Social control (anxiety decreases vigilance)

Opposite directions on same parameter. Substantively interpretable.

See discoveries §4.71.

---

## 2026-06-08 (followup-6) — ✗ RETRACTION: trait-anxiety → both-parameters finding does NOT replicate to raw scales

**Script:** `scripts/analysis/raw_anxiety_scales_vs_params.py` ✅. Tests raw STAI_Trait (with PC1-sign reverse-coding), DASS_Anx, DASS_Stress, OASIS, STICSA, and 4-scale anxiety composite against log_omega and log_kappa.

**Output:** `results/stats/affect_analysis/raw_anxiety_vs_params.csv` ✅.

**Result: ALL anxiety scales null on both ω and κ.** The 3-factor F3 → both-parameters finding from §4.69 does NOT replicate to any raw scale. Conclusion: F3 was a varimax-rotation artifact, not a real construct effect.

**The "two-axis clinical story" framing is retracted.** Only the apathy axis (AMI_Social → log ω) is real.

**Confirmed remaining finding:** AMI_Social → log ω β = +0.122 ★. The only robust raw-scale clinical signal across all analyses.

See discoveries §4.70.

---

## 2026-06-08 (followup-5) — ★★★ EFA factors against ALL THREE outcomes: anxiety is a PARALLEL (both-parameter) effect invisible to balance

**Script:** `scripts/analysis/item_efa_vs_balance.py` ✅. Tests each factor (3/4/5-factor solutions) on log(ω), log(κ), AND log(ω/κ) side-by-side.

**Output:** `results/stats/affect_analysis/item_efa_vs_balance.csv` ✅.

**Headline:** Balance metric is a *filter*, not an amplifier. It splits clinical signals into:
- **Differential** (apathy on ω): balance picks them up
- **Parallel** (anxiety on BOTH ω and κ): balance CANCELS them

Trait anxiety (F3, 3-factor): β(log ω)=+0.084 ★, β(log κ)=+0.097 ★, **β(log ω/κ)=+0.010 (null!)**. We missed this in every prior analysis (§4.63-§4.68) because we only tested balance.

Non-social apathy (5-factor F4): β(log ω)=-0.102 ★, β(log κ)=+0.017, β(log ω/κ)=-0.079 ★. Differential — survives all three.

For paper: report the full table (ω, κ, ω/κ) per factor. The pattern of which columns survive is the substantive story.

See discoveries §4.69.

---

## 2026-06-08 (followup-4) — ★★★ 5-factor EFA reveals SOCIAL vs NON-SOCIAL APATHY dissociation on ω + trait anxiety → κ

**Script:** `scripts/analysis/item_efa_reduced_factors.py` ✅. Re-fit at n_factors=3, 4, 5.

**Outputs:**
- `results/stats/affect_analysis/item_efa_{3,4,5}factor_loadings.csv` ✅
- `results/stats/affect_analysis/item_efa_reduced_summary.csv` ✅
- `results/figs/affect_analysis/item_efa_scree.png` ✅

**Recommended solution: 5 factors** (varimax). Cleanest interpretation with strongest signals.

| Factor | Content | β(log ω) | β(log κ) |
|---|---|---|---|
| F1 | Fatigue (MFIS) | -0.04 | -0.03 |
| F2 | Somatic anxiety (DASS+STICSA) | -0.06 | -0.01 |
| F3 | Trait NA inverse (STAI -) | +0.06 | **+0.106 ★** |
| **F4** | **Non-social apathy (AMI subset)** | **-0.102 ★** | +0.02 |
| F5 | Residual | +0.00 | -0.03 |

**Key dissociation:** AMI_Social subscale → ω β=+0.124 (positive). Item-EFA F4 (non-social AMI items) → ω β=-0.102 (negative). Opposite directions within AMI.

**New κ finding:** Trait anxiety predicts lower κ (mobilization). First robust κ result.

See discoveries §4.68.

---

## 2026-06-08 (followup-3) — ★★★ Item-level EFA on transdiagnostic items: F6 (apathy) replicates AMI_Social finding; STAI reverse-coding fixed

**Script:** `scripts/analysis/item_level_efa_on_params.py` ✅. N=571 pooled, 106 items (PHQ-9 excluded), Student-t regressions.

**Outputs:**
- `results/stats/affect_analysis/item_efa_loadings.csv` ✅ (factor loadings)
- `results/stats/affect_analysis/item_efa_subject_scores.csv` ✅ (per-subject factor scores)
- `results/stats/affect_analysis/item_efa_param_regressions.csv` ✅ (factor → ω/κ tests)
- `results/stats/affect_analysis/item_efa_parallel_analysis.csv` ✅

**Key results (see discoveries §4.67):**
- **STAI reverse-coding bug confirmed and fixed**: 11/20 STAI items needed reversal. All prior STAI_Trait analyses should be re-checked.
- Parallel analysis recommended 8 factors. F1=Fatigue, F2=Somatic Anx, F3=Trait NA, F4=Depression, F5/F6/F8=Apathy clusters, F7=Mixed Anxiety.
- **F6 (apathy variant) → log(ω) β=+0.114 ★** (univariate); β=+0.109 ★ (multivariate). Replicates AMI_Social baseline (β=+0.124) at the latent level.
- F8 → log(ω) β=-0.099 ★ univariate; barely surviving in multivariate. Fragile — opposite-direction apathy cluster.
- No κ signal robust across 8 factors.

**Caveats:** Used varimax only (sklearn FactorAnalysis lacks oblimin; factor_analyzer 0.5.1 incompatible with sklearn 1.8). Oblique rotation sensitivity check pending.

---

## 2026-06-08 (followup-2) — ★★ AMI_Social effect is METRIC-INVARIANT across 4 balance parameterizations

**Script:** `scripts/analysis/balance_metric_comparison.py` ✅. N=571 (no outlier filter), Student-t likelihood.

**Output:** `results/stats/affect_analysis/balance_metric_comparison.csv` ✅. Also diagnostic figure `results/figs/affect_analysis/log_ratio_distribution_diagnostic.png`.

**Setup:** 4 metrics tested as outcomes:
- M1: log(ω/κ) — current
- M2: z(log ω) − z(log κ) — standardized difference (recommended primary)
- M3: ω/(ω+κ) — bounded proportion
- M4: arctan(log κ / log ω) — angle in degrees

AMI_Social univariate β [HDI]:
- M1: +0.084 [+0.019, +0.144] ★
- M2: +0.111 [+0.045, +0.178] ★ (strongest; Spearman p=0.002)
- M3: +0.062 [+0.016, +0.111] ★
- M4: +0.082 [+0.027, +0.141] ★

DASS_Stress dead on all 4 (univariate + multivariate Student-t).

See discoveries §4.65. Outlier filter from §4.63/§4.64 is no longer needed — AMI_Social robust without it.

---

## 2026-06-08 (followup) — ✗ DASS / comorbidity null; AMI_Social is the unique signal

**Script:** `scripts/analysis/log_ratio_dass_comorbidity.py` ✅. Three phases (DASS diagnostic, factor analysis + composites, comorbidity tests), N=561 pooled.

**Outputs (all ✅):**
- `results/stats/affect_analysis/log_ratio_dass_diagnostic.csv` — Spearman/Huber/trimmed/Student-t/Normal on DASS subscales
- `results/stats/affect_analysis/log_ratio_composites.csv` — ANX/DEP/APATHY/STRESS composites
- `results/stats/affect_analysis/log_ratio_comorbidity.csv` — polar + quadrant + joint univariate
- `results/stats/affect_analysis/factor_analysis_parallel.csv` — Horn's parallel analysis (2 factors confirmed)
- `results/figs/affect_analysis/dass_vs_log_ratio.png`

**Key finding:** No DASS effect, no comorbidity effect, no composite effect. AMI_Social from §4.63 is the only surviving signal. STAI_Trait reverse-coding bug flagged (loads -0.67 on F1; should be positive). See discoveries §4.64.

---

## 2026-06-08 — ★★ log(ω/κ) "vigilance–mobilization balance" pooled Bayesian on clinical scales

**Script:** `scripts/analysis/log_ratio_clinical_robust.py` ✅. Pooled N=561 (286 exp + 275 conf after dropping 10 |log_ratio_z|>3 subjects).

**Output:** `results/stats/affect_analysis/log_ratio_clinical_robust.csv` ✅. Univariate + 4 multivariate model families (A_subscales, B_factors, C_dass_only, D_totals).

**Setup:** Student-t (ν=3) likelihood for outlier robustness. Predictors z-scored WITHIN each sample, then pooled. Log_sum engagement covariate dropped (orthogonal to all clinical scales, |r|<0.06).

**Surviving effects:**
- AMI_Social univariate β=+0.103 (HDI [+0.025, +0.173]); multivariate β=+0.141 (HDI [+0.047, +0.237]) — STRONGER in kitchen-sink ★
- AMI_Total univariate β=+0.069 (driven by Social subscale)
- STAI_Trait multivariate β=+0.114 (univariate β=+0.050 just misses) — partial suppression

**Null:** DASS-21 (all subscales + total), F1/F2 factors, PHQ9, MFIS, OASIS, STICSA.

**Prior passes (superseded):**
- `log_ratio_bayes_multimodel.csv` (Normal likelihood, pooled-z, with log_sum covariate) — showed AMI_Social + DASS_Stress borderline
- `log_ratio_bayes_no_engagement.csv` (Normal likelihood, pooled-z, no log_sum) — showed AMI_Social + STAI_Trait + AMI_Behavioural

See discoveries §4.63.

---

## 2026-06-07 — ★★★ Clinical → vigor dynamics: AMI (apathy) reaches anticipatory phase

**Script:** `scripts/analysis/clinical_predict_dynamics.py` ✅. Both samples N=569 (anticipatory); exploratory N=290 (reactive).

**Output:** `results/stats/clinical/clinical_predict_dynamics.csv` ✅ (180 rows: 15 scales × 5 anticip outcomes × 2 samples + 15 × 2 reactive × 1 sample, after dropping cells lacking data).

**Setup:** Each clinical scale (DASS21_{Anx,Dep,Stress}, PHQ9, OASIS, STAI_{Trait,State}, STICSA, AMI_{Total,Beh,Soc,Emo}, MFIS_Total) plus EFA factors (F1=distress, F2=engagement/anti-apathy) z-scored within sample. Regressed on each dynamics outcome, partialing (ω_z, κ_z). For peak_post, baseline (pre_mean) as additional covariate.

**Replicating findings (p<0.05 BOTH samples, same sign):**
- AMI_Total / AMI_Behavioural / AMI_Social → pre_at_lowT/midT/highT: β ≈ +0.20 to +0.32, p < 1e-5 both samples (apathy → higher anticipatory baseline at all threat levels)
- F2 (engagement factor) → pre_at_lowT/midT/highT: β ≈ −0.17 to −0.23, p < 0.01 both samples (consistent with AMI; F2 has negative AMI loadings)
- PHQ9_Total → pre_at_midT: marginal replication (p < 0.05 both)

**Follow-up diagnostic on AMI → abs_peak_strike (initially looked replicated, β = −0.16/−0.26):**
- Re-ran with pre_mean as covariate (baseline disentangle): exp collapses to null (β=−0.09, p=0.09), conf survives (β=−0.24, p<1e-5). Single-sample only — fails cross-sample bar.
- This mirrors the §4.59 anxiety→peak diagnostic. The naive AMI→peak claim is dropped.

**Nulls:**
- DASS21_Anxiety / Depression / Stress: nothing replicates
- STAI_Trait, OASIS, STICSA: confirmatory-only effects, don't replicate
- All clean reactive measures (peak_post baseline-controlled, accel_post in exp): NO clinical hits (best accel_post p = 0.15, best peak_post p = 0.07)
- AMI_Emotional: doesn't reach the dynamics

**Status:** ★★★ SUPPORTED on anticipatory baseline only. Apathy → higher uniform anticipatory pressing, INDEPENDENT of (ω, κ). Clinical signal the parameter-level analysis (§4.6) missed lives in the *anticipatory baseline*, not the reactive component. Anxiety-spectrum scales remain null across all dynamics measures.

---

## 2026-06-04 — result_604 Stage 3: HiTOP-style factor analysis confirms clinical null

**Script:** `scripts/analysis/embodied_clinical_factor_analysis.py` ✅

**Method:** Parallel analysis (Horn's, 500 perms) + EFA with varimax rotation on N = 568 pooled. Tested whether (ω, κ) predicts factor scores rather than individual subscales (addresses the HiTOP / p-factor concern).

**Parallel analysis:** 2 factors retained. F1 eigenvalue = 8.52 (dominant — consistent with p-factor literature). F2 eigenvalue = 1.42 (modest). Subsequent eigenvalues drop below random 95% threshold.

**Factor structure (varimax):**
- **F1 (general internalising distress)**: positive loadings on DASS21_Anxiety (+0.86), STICSA (+0.86), DASS21_Stress (+0.82), OASIS (+0.77), PHQ9 (+0.76), DASS21_Depression (+0.72), STAI_State (+0.71). STAI_Trait anomalously loads opposite (−0.59).
- **F2 (apathy/fatigue/anhedonia)**: negative loadings on MFIS_Psychosocial (−0.75), MFIS_Cognitive (−0.72), MFIS_Physical (−0.69), AMI_Behavioural (−0.57), DASS21_Depression (−0.47), PHQ9 (−0.46).

**Structure matches HiTOP's internalising distress + somatic-form distinction. Theoretically sensible.**

**(ω, κ) → factor scores: ALL NULL.**
- F1: β(ω) = −0.073 [−0.160, +0.016] marginal wrong-direction; β(κ) = +0.027; β(ω×κ) = +0.029
- F2: β(ω) = −0.003; β(κ) = −0.016; β(ω×κ) = +0.033

**Verdict across three analyses:** Stage 1 (subscales, pooled-z): 1 hit in interpretable direction. Stage 2 (comorbidity groups): all null. Stage 3 (factor analysis): all null. **The clinical decomposition is genuinely absent in this data, not just underpowered or hidden by comorbidity confound.**

**Outputs:** `results/stats/clinical/parallel_analysis.csv`, `factor_loadings.csv`, `factor_scores.csv`, `factor_param_regressions.csv`.

---

## 2026-06-04 — result_604 verification: analysis correct, but cross-sample heterogeneous + result_602 doesn't replicate at cell-mean vigor

**Script:** `scripts/analysis/verify_clinical_decomp.py` ✅

**Triggered by:** Skepticism about the result_604 Stage 1 finding (ω → AMI Social/Emotional/Total positive, β ≈ +0.10–0.14, 95% HDI excludes zero). Question: was the analysis right, or was there a scoring/sign error?

**Verdict: the analysis was correct.** The bambi regression coefficients (β ≈ +0.11 on ω → AMI_Total) are properly within-sample partial standardised slopes; the raw pooled Pearson (r = +0.062, p = 0.14) appears smaller because pooled-raw is contaminated by between-sample heterogeneity in both AMI and ω means. Within-sample z-scoring (which result_604 uses) is the correct way to handle this.

**But verification surfaced two real concerns:**

1. **Cross-sample heterogeneity in the AMI → ω signal.** Within-sample raw Pearson:
   - Exploratory: r(AMI_Total, log_ω) = +0.076 (weak, would not pass HDI test alone)
   - Confirmatory: r(AMI_Total, log_ω) = +0.146 (moderate)
   - The pooled β ≈ +0.114 is the *average* of these. The "ω → social/emotional apathy" headline is driven primarily by confirmatory. Exploratory shows same-sign but much smaller effects. Not a strong "both samples" replication.

2. **The legacy result_602 (AMI → vigor) does NOT replicate at the master-table mean_vigor metric.** r(AMI_Total, mean_vigor):
   - Exploratory: +0.012 (p = 0.84, NULL)
   - Confirmatory: −0.029 (p = 0.63, NULL)
   - Result_602's claim "AMI apathy tracks vigor (negative)" relied on a different vigor operationalisation (pre-encounter capacity-normalised). At the cell-mean aggregate used by result_208 / result_401 / result_604, AMI and vigor are uncorrelated. The κ → apathy chain (κ → vigor → AMI) breaks at the second link.

**Parameter distributions verified reasonable** in both samples (ω log-normal-ish, κ log-normal-ish, r(log_ω, log_κ) = +0.369 / +0.302 matching result_208).

**No item-level AMI data** available in psych.csv (only subscales + total). Cannot rule out scoring-direction issues by inspecting items. r(AMI, mean_vigor) ≈ 0 in both samples is *consistent with* either standard scoring (AMI ↑ = apathy ↑, behavioural correlate just absent) or reverse-scoring; without items, the direction is determined by convention (result_602 framing assumes standard).

**Implications for the paper:**
- The result_604 headline "ω → social/emotional apathy" survives mathematically but with the honest caveat that confirmatory drives most of the signal
- The result_602 finding (κ → apathy via vigor) does not hold at the metric consistent with the rest of the paper
- The clinical story stays narrow: a small ω → social/emotional apathy effect, replicated in direction but heterogeneous in magnitude. Not strong enough to anchor Frame A.

---

## 2026-06-03 — H4 choice decomposition + cross-channel r(choice, vigor) prediction (result_208 update)

**Script:** `scripts/analysis/h4_choice_decomp.py` ✅ run on both samples.

**Outputs:**
- `results/stats/individual_diffs/h4_choice_decomp.csv` ✅ (32 coefficient rows: P(heavy) + mean_vigor × polar + Cartesian + main-effects × both samples)
- `results/stats/individual_diffs/h4_predicted_r_cv.csv` ✅ (2 rows: predicted vs observed r(choice, vigor) per sample)

**Key findings:**

Choice partial coefficients (Cartesian with interaction):
- Exploratory: β(ω → P(heavy)) = −0.154 [−0.161, −0.147], β(κ → P(heavy)) = −0.076 [−0.083, −0.069]
- Confirmatory: β(ω → P(heavy)) = −0.168 [−0.177, −0.160], β(κ → P(heavy)) = −0.062 [−0.070, −0.053]
- Both samples: ω ≈ 2× stronger than κ on choice; small positive ω × κ interaction (+0.006 / +0.015)

Predicted vs observed r(choice, vigor) — embodied W(u) framework prediction:
- Exploratory: predicted = +0.143, observed = +0.150 (p=0.011) — match within 0.007
- Confirmatory: predicted = +0.052, observed = +0.077 (p=0.201) — match within 0.025

Pathway decomposition (both samples):
- ω-pathway Cov (β_ωc·β_ωv): consistently ≈ −0.020 (negative; ω lowers choice but raises vigor)
- κ-pathway Cov (β_κc·β_κv): +0.018 / +0.014 (positive; κ lowers both)
- Cross term (driven by r(ω,κ) ≈ +0.30–0.37): +0.010 / +0.010
- Near-cancellation between channel pathways; small positive residual from cross term explains observed r

**Implication:** Marginal r(choice, vigor) is *quantitatively* predicted by the embodied two-parameter framework, not just qualitatively consistent. Anchors the embodiment argument for [[result_401]] rewrite (Phase 2, deferred).

**Sampling diagnostics:** R̂ = 1.000 across all 12 fits; ESS_bulk ≥ 7,001. Each fit ≈ 1 s wall time (bambi/NumPyro).

**Note on operationalization:** `mean_vigor` here is the M4 cell-mean aggregate, NOT the legacy "pre-encounter capacity-normalized + choice-ratio-adjusted" metric used in legacy H29 (which reported r ≈ −0.018). Different operationalization → different sign on marginal r. The current operationalization is the one that matches 208's vigor partial coefficients, so it is the consistent metric for the cross-channel prediction.

---

## 2026-05-28/29 — M1 effort-kernel discount-form selection (result_206)

**Script:** `scripts/modeling/joint_optimal/m1_effort_kernels.py` ✅ run on both samples.

**Outputs (Axis B — discount function, headline):**
- `results/stats/joint_optimal/m1_effort_kernels_exploratory.csv` ✅
- `results/stats/joint_optimal/m1_effort_kernels_confirmatory.csv` ✅

**Outputs (Axis A — effort exponent robustness):**
- `results/stats/joint_optimal/m1_effort_exponent_exploratory.csv` ✅
- `results/stats/joint_optimal/m1_effort_exponent_confirmatory.csv` ✅

**Figures:** `results/figs/paper/fig_s_m1_effort_kernels_{exploratory,confirmatory}.{pdf,png}` ✅

**Result:** Linear discount on `E = req²·D` wins decisively in BOTH samples. ΔBIC = 0 / 266 / 333 / 861 (linear / exp / quad / hyp) in exploratory; ΔBIC = 0 / 273 / 386 / 952 (linear / quad / exp / hyp) in confirmatory. Subject-level choice R² ≈ 0.95 for linear in both samples. Free-power optimum p̂ ≈ 5.2–5.4 in both (uninterpretable high; M1's p=2 is the principled near-optimum). **result_206 status upgraded `supported_exploratory → supported`.** Runtime: ~2 min per sample on CPU.

### Bug discovered + fixed: `data/model_input/` was mislabeled

While running the confirmatory M1 sweep, discovered that `data/model_input/` — which the prior result_206 frontmatter labeled "exploratory, N=281" — in fact contained the **confirmatory sample's data** (281 subjects, 12,645 choice trials, p_heavy=0.414, matching `stage5_filtered_data_20260403_142413/behavior_rich.csv`). True exploratory has 293 subjects, 13,185 trials, p_heavy=0.431.

**Fix:** Generated explicit per-sample model_input snapshots:
- `data/model_input_exploratory/` — true exploratory (N=293, 13,185 trials, 3,935 vigor cells). Built 2026-05-29 from `data/exploratory_350/processed/stage5_filtered_data_20260403_133425/` via `prepare_model_input.py`.
- `data/model_input_confirmatory/` — confirmatory (N=281, 12,645 trials, 3,822 vigor cells). Built 2026-05-29 from `stage5_filtered_data_20260403_142413/`.

`data/model_input/` left in place as the confirmatory snapshot for backward compatibility. **All new code should reference the explicit dirs, not `data/model_input/`.**

The cached MCMC fits in `results/stats/joint_optimal/exploratory/mcmc_m4_params.csv` and `results/stats/joint_optimal/confirmatory/mcmc_m4_params.csv` use a separate data-loading path (`scripts/run_mcmc_pipeline.py`) and are NOT affected — their subject counts (290 expl, 281 conf) match the actual sample they purport to be.

---

## 2026-06-02 — H2 vigor dynamics (result_104) blocked by two bugs

**Status:** Deferred. Notebook execution attempted; cannot ship a valid both-sample result_104 yet.

**Investigation:**

1. Patched a missing `outputs: []` / `execution_count: null` on the "H2 Summary" cell of `notebooks/analysis/H2_vigor_dynamics.ipynb` (via a sub-agent). This fixed `nbformat.validator.NotebookValidationError`. No source-code change.
2. Re-executing the notebook end-to-end fails on the H2c GAM cell with `LinAlgError: Singular matrix` — the cell tries to fit a MixedLM with `K = min(K_SPLINE, 4)` cubic-spline basis on a `t_epoch` column with only 4 unique values, which is rank-deficient. The cell's own comment notes that the original analysis used the raw `alignedEffortRate` timecourse, not 4 discrete epochs. This is a substantive analysis bug, not an environment issue.
3. Extracted cells 1 + 3 + 5 (imports, H2a paired-t, H2b encounter spike) as a standalone Python script and ran them. Output for the two samples is **byte-for-byte identical**: Heavy Δ=+0.0349 / t=7.72 / p=1.89e-13 / d=+0.454 and Light Δ=+0.0541 / t=12.95 / p=1.45e-30 / d=+0.762 in both samples; H2b mean spike +0.0358 / t=11.01 / p=8.73e-24 / d=+0.647 in both. The H2b numbers match `confirmatory_hypothesis_results.csv` exactly, indicating `vigor_metrics` is loaded from a single confirmatory source regardless of which sample is requested.

**Diagnosis of the data-loading bug.** `len(d['vigor_metrics'])` is `93960` for both "exploratory" and "confirmatory" in `load_both()` output, while `len(d['trials'])` correctly differs (23,490 expl vs 22,761 conf). Same shape of bug as the `data/model_input/` mislabel resolved 2026-05-29 — a data-loading path returning one sample's data under both labels.

**Fix needed (next session):**
- Trace `vigor_metrics` source inside `notebooks/analysis/load_data.py` and fix sample dispatch.
- Decide the H2c GAM specification (restore raw `alignedEffortRate` timecourse with K=10 matching prereg, or accept degenerate epoch-level with K ≤ 3 and document deviation).
- Re-execute H2 notebook, T3-validate confirmatory against the cached CSV (d=0.647, GAM enc χ²=1024.8, GAM threat χ²=114.8), write up result_104 as a full lab report.

**At-risk results.** Any result that reads `d['vigor_metrics']` is at the same risk. Trial-level paths (`d['vigor']`, `d['vigor_valid']`) used by H1 (results 101–103) and H8 (result 402) are NOT affected.

---

## 2026-06-07 (final) — Disentangle: anxiety→peak fully mediated by baseline; confidence_intercept→peak EMERGES

**Script:** `scripts/analysis/anxiety_peak_disentangle.py` ✅ (exploratory only)

**Output:** `results/stats/joint_optimal/anxiety_peak_disentangle.csv` ✅

**Setup.** For each affect predictor × each reactive measure (peak_post, accel_post, time_to_peak), fit two models: with and without pre_mean as covariate. Compare β.

**Headline:**
- anxiety_intercept → peak_post: β = −0.143 (p=0.005) → β = −0.045 (p=0.43, NULL) when baseline controlled — FULLY MEDIATED
- All other anxiety→peak effects: same pattern, all mediated
- **confidence_intercept → peak_post: EMERGES with baseline control** β = −0.093 (p=0.079) → β = −0.125 (p=0.001) ★★★
- Real effect: higher baseline confidence → lower absolute peak, controlling anticipatory baseline

**Status:** Clean affect-reactive finding identified: confidence baseline → lower peak. Anxiety effects on reactive measures are all baseline-mediated. Needs confirmatory replication via vigor_ts pipeline.

---

## 2026-06-07 (latest) — ⚠️ Anxiety does NOT modulate reactive acceleration; interactions null

**Script:** `scripts/analysis/anxiety_modulates_reactive_dynamics.py` ✅ (exploratory only)

**Output:** `results/stats/joint_optimal/anxiety_modulates_reactive_dynamics.csv` ✅

**Setup.** Tested 6 anxiety features (intercept, slope_T, slope_D, mean, sd, range) on 4 reactive measures (accel_post, peak_post, time_to_peak, latency), as direct effects (controlling ω, κ) and as ω×anx / κ×anx interactions.

**Headline:**
- accel_post: only anxiety_slope_D marginal (β = −0.119, p = 0.041); all others null
- peak_post (baseline-confounded): anxiety_intercept → lower peak (β = −0.14, p = 0.005); anxiety_slope_T → higher peak (β = +0.13, p = 0.015)
- All interactions on accel_post: NULL
- ω, κ effects on acceleration stable across anxiety profiles

**Status:** Anxiety doesn't robustly modulate the clean reactive measure. Paper §3.7 simplifies to one paragraph noting affect calibrates to conditions but does not modulate parameter-dynamics coupling.

---

## 2026-06-07 (later) — ★ Reactive acceleration from timecourse: ω, κ reach reactive phase with opposite effects

**Script:** `scripts/analysis/reactive_dynamics_from_timecourse.py` ✅ (exploratory only — confirmatory vigor_ts not processed)
**Env:** /opt/anaconda3/envs/limaAnalysis/bin/python3.11 (pyarrow required for parquet)

**Output:** `results/stats/joint_optimal/reactive_dynamics_timecourse.csv` ✅

**Setup.** Loaded smoothed_vigor_ts.parquet (3.9M rows, 20Hz). Per-subject mean across attack trials of: pre_mean, peak_post, time_to_peak, accel_post (slope over first 500ms after encounter), latency. Tested ω + κ + affect predictors.

**Baseline-confound check:**
- accel_post × pre_mean: r = −0.19 (clean, baseline-independent)
- Other measures: r = −0.35 to −0.75 (heavily confounded)

**Headline:**
- ω → faster reactive acceleration (β = +0.178, p = 0.005)
- κ → slower reactive acceleration (β = −0.174, p = 0.006)
- R² = 0.04 (small but clean)
- Affect features null on acceleration

**Status:** SUPPORTED in exploratory. ω is NOT confined to anticipatory phase — both parameters reach both anticipatory and reactive phases with opposite signs. Need confirmatory replication via vigor_ts pipeline processing.

---

## 2026-06-07 — ⚠️ Spike measurement diagnostic: anxiety→spike LARGELY ARTIFACT; ω/κ → absolute peak emerges

**Script:** `scripts/analysis/spike_measurement_diagnostic.py` ✅

**Output:** `results/stats/joint_optimal/spike_measurement_diagnostic.csv` ✅

**Setup.** Tested 5 spike measures (absolute peak, subtractive delta, ratio, normalized to calibrationMax, peak-to-peak) plus key test of subtractive spike with baseline as covariate. Within-sample replication.

**Headline:**
- Baseline (pre_mean) strongly negatively correlates with subtractive spike measures (r = −0.58 to −0.81) — ceiling problem confirmed
- ABSOLUTE peak strike is uncorrelated with baseline (r = +0.06)
- anxiety_slope_T effect on spike DISAPPEARS when controlling for baseline in exp (β = +0.003, p = 0.94); ~40% reduction in conf
- ω → absolute peak: REPLICATES POSITIVE (β = +0.13/+0.22)
- κ → absolute peak: REPLICATES STRONGLY NEGATIVE (β = −0.49/−0.56, R² up to 0.31)

**Status:** §4.55's "anxiety front-loading" finding is LARGELY ARTIFACT. §3.7 paper section needs rework. κ → reactive damping emerges as the cleaner replicating finding.

---

## 2026-06-06 — ★★ Affect modulates vigor dynamics beyond (ω, κ): anxiety_slope_T → smaller reactive spike, replicates

**Script:** `scripts/analysis/affect_modulates_dynamics.py` ✅

**Output:** `results/stats/affect_analysis/affect_modulates_dynamics.csv` ✅

**Setup.** Per-subject dynamics features (anticipatory baseline at low/mid/high T; spike magnitudes) regressed on ω + κ alone vs ω + κ + 6 affect features. Within-sample replication.

**Replicating affect predictors (both samples, p<0.05, same sign, after controlling ω, κ):**
- anxiety_slope_T → anticipatory baseline at all T levels: POSITIVE (β ≈ +0.15)
- anxiety_slope_T → reactive spike: NEGATIVE (β ≈ −0.15) — counterintuitive but replicates
- anxiety_intercept → reactive spike: POSITIVE (β ≈ +0.12)
- confidence_slope_D → anticipatory baseline: NEGATIVE (β ≈ −0.17)
- confidence_slope_D → reactive spike: POSITIVE (β ≈ +0.20)

**R² lifts substantial:** spike_mag_mean R² from 0.02 to 0.24 with affect added.

**Status:** SUPPORTED. Affect modulates dynamics beyond parameters with a non-obvious counterintuitive pattern (anxiety reactivity REDUCES rather than INCREASES reactive spike). This is the substantive empirical finding the embodied paper needs.

---

## 2026-06-06 — ★★ Parameters predict embodied vigor dynamics (strategic/reactive dissociation)

**Script:** `scripts/analysis/parameters_predict_vigor_dynamics.py` ✅

**Output:** `results/stats/joint_optimal/parameters_predict_vigor_dynamics.csv` ✅

**Setup.** Per-subject features from beh phase-segmented effort columns (mean_preEncounter_effort, mean/peak_strike_effort, mean_postEncounter_effort). Computed anticipatory slope on T, baseline at low T, reactive spike (3 metrics). Within each sample.

**Headline (replicates both samples):**
- ω → anticipatory steepness on T: exp β = +0.215 p = 6e-4; conf β = +0.188 p = 3e-3 ★
- κ → baseline anticipatory at low T: exp β = −0.458 p < 10⁻¹³; conf β = −0.512 p < 10⁻¹⁶ ★★★
- ω → baseline anticipatory at low T (positive): β ≈ +0.26 both p < 10⁻⁵
- ω → reactive spike NULL in both samples on all 3 spike metrics (predicted dissociation ✓)
- κ → reactive: marginal/null on peak; replicates negative on post-pre ramping

**Status:** SUPPORTED. The substantive embodied finding the paper has been missing. Parameters control within-trial dynamics, not just averages. Strategic/reactive dissociation maps onto predatory imminence continuum.

---

## 2026-06-05 — Multivariate (ω, κ) MMR + CCA: no new signal

**Script:** `scripts/analysis/multivariate_omega_kappa.py` ✅ (after manual Pillai/Wilks computation; statsmodels MANOVA formula crashed initially)

**Output:** `results/stats/affect_analysis/multivariate_omega_kappa.csv` ✅

**Headline:**
- MMR clinical → (ω, κ) joint: Pillai p = 0.62 exp, 0.12 conf (NULL/marginal)
- CCA (ω, κ) ↔ affect top r = 0.247 exp, drops to 0.157 conf-projected
- CCA (ω, κ) ↔ clinical top r = 0.207 exp, drops to 0.108 conf-projected
- Confidence_intercept loads dominantly on the shared (ω, κ) conservative dimension but the dimension itself is modest

**Status:** Confirms clinical decoupling at multivariate level (5th independent confirmation). Confirms confidence baseline as joint-substrate but with weak replication. No new finding.

---

## 2026-06-05 — CCA behavior × affect/clinical: affect replicates (r=0.41), clinical collapses (r=0.06)

**Script:** `scripts/analysis/behavior_clinical_cca.py` ✅

**Output:** `results/stats/affect_analysis/behavior_clinical_cca.csv` ✅

**Setup.** Per-subject behavioral features (12: choice GLM coefs, vigor GLM coefs, autocorrelations, vigor SD) × clinical (10 scales) and affect (6 features). CCA within each sample + cross-sample projection.

**Headline:**
- Behavior × Clinical: exp top r = 0.32, conf-projected r = 0.06 (COLLAPSES — no replicable dimension)
- Behavior × Affect: exp top r = 0.49, conf-projected r = 0.41 ★ (REPLICATES)
- The replicating affect dimension = joint behavioral and metacognitive calibration to T and D

**Status:** SUPPORTED. Confirms multivariate clinical decoupling. Confirms behavioral-affective calibration as the cleanest replicating between-subject signature.

---

## 2026-06-05 — ⚠️ Controlled regressions: substrate not parameter-specific; clinical does NOT predict ω/κ

**Scripts:**
- `scripts/analysis/affect_clinical_controlled.py` ✅ — Tests A, B (affect substrate after controlling other parameter)
- `scripts/analysis/clinical_predict_params.py` ✅ — Test C corrected: ω/κ ~ clinical scales jointly

**Outputs:**
- `results/stats/affect_analysis/affect_clinical_controlled.csv` ✅
- `results/stats/affect_analysis/clinical_predict_params.csv` ✅

**Headline:**
- Affect substrate (§4.49) DOES NOT survive controlling for the other parameter — collapses to shared (ω,κ) variance
- Only cross-parameter effects replicate (ω→κ and κ→ω at β≈0.2, p<0.001)
- Clinical scales joint F-test for ω: exp p=0.60, conf p=0.075 — NOT replicating
- Clinical scales joint F-test for κ: exp p=0.70, conf p=0.44 — NULL
- No single clinical predictor replicates across both samples for either parameter

**Status:** SUPPORTED clinical decoupling finding. Substrate story needs revision: not ω-specific but joint conservative-style.

---

## 2026-06-05 — ★★ Heavy-minus-light confidence_intercept → ω: sharper substrate, replicates

**Script:** `scripts/analysis/affect_heavy_minus_light_predict_params.py` ✅

**Output:** `results/stats/affect_analysis/affect_heavy_minus_light_predict_params.csv` ✅

**Setup.** Per-subject (heavy − light) contrasts for intercept, slope_T, slope_D in anxiety and confidence. Two models: contrasts-only (6 predictors), contrasts+light baselines (12 predictors).

**Headline:** confidence_intercept_HvL → ω replicates in BOTH model variants:
- Contrasts-only: exp β = −0.23, p = 0.028; conf β = −0.40, p = 4e-4
- Contrasts+light: exp β = −0.34, p = 0.005; conf β = −0.48, p = 2e-4

Effect strengthens with light baseline as covariate → variance is in the contrast, not the absolute level.

**Status:** SUPPORTED. The substrate of ω is sharpened from "global low confidence" (§4.46) to "confidence selectively suppressed on heavy/demanding cookies" (§4.49). κ side remains weaker.

---

## 2026-06-05 — ★ Cookie-stratified affect → params: confidence_heavy_intercept → ω replicates

**Script:** `scripts/analysis/affect_TD_by_cookie_predict_params.py` ✅

**Output:** `results/stats/affect_analysis/affect_TD_by_cookie_predict_params.csv` ✅

**Setup.** Per-subject `response ~ T + D` fit separately for heavy and light probe trials, anxiety and confidence. 12 affect features. N ≈ 9 trials per cookie × question. Second-level regressions within each sample.

**Headline:**
- confidence_heavy_intercept → ω: exp β = −0.342 (p = 0.005), conf β = −0.543 (p = 0.0002) ★ REPLICATES
- Near-replications on ω: confidence_heavy_slope_T, confidence_heavy_slope_D (strong in conf, weaker in exp, same direction)
- κ side weaker: confidence_heavy_intercept hits exp (p = 0.014), marginal conf (p = 0.099)
- Cookie stratification roughly doubles R² in confirmatory (ω 0.07; κ 0.14)

**Status:** SUPPORTED. The substrate finding from §4.46 localizes to heavy-cookie trials. The honest paper claim is: lower baseline confidence on HEAVY trials specifically (not globally) predicts higher ω.

---

## 2026-06-05 — Separate anxiety vs confidence regressions: confirms anxiety effects are not hidden

**Script:** `scripts/analysis/affect_TD_predict_params_separate.py` ✅

**Output:** `results/stats/affect_analysis/affect_TD_predict_params_separate.csv` ✅

**Setup.** Three models per outcome × sample: anxiety_only (3 predictors), confidence_only (3 predictors), joint (6 predictors).

**Headline:** Anxiety effects null in both separate AND joint models. Confidence_intercept replicates in both separate and joint models. The asymmetry is genuine, not a methodological artifact.

**Status:** §4.46's finding stands. No new replicating predictors.

---

## 2026-06-05 — ★ Simplified affect→params (T, D only): only confidence_intercept replicates

**Script:** `scripts/analysis/affect_TD_predict_params.py` ✅. Predictors: 6 features (anxiety/confidence × intercept, slope_T, slope_D) from `phenotype_metacog_slopes_subjects.csv`.

**Output:** `results/stats/affect_analysis/affect_TD_predict_params.csv` ✅

**Setup.** Per-subject affect slopes from `response ~ T + D` (no cookie reward). Second-level regression: ω_z and κ_z (log-z within sample) on the 6 affect features, within each sample.

**Headline:**
- confidence_intercept → ω: REPLICATES (exp β=−0.20 p=0.001; conf β=−0.14 p=0.025)
- confidence_intercept → κ: REPLICATES (exp β=−0.15 p=0.011; conf β=−0.15 p=0.016)
- All slope predictors (anxiety/confidence × T, D) null in both samples
- All anxiety predictors null in both samples

R² = 0.03–0.05. Modest but replicated.

**Status:** SUPPORTED but conservative. The previous reward-reactivity finding (§4.45) was driven by a heavy-vs-light intercept difference that collapses without cookie weight in the regression. The honest substrate finding is baseline confidence, not reactivity.

---

## 2026-06-05 — ★ Affect features predict ω and κ (metacognitive substrate)

**Script:** `scripts/analysis/affect_features_predict_params.py` ✅. Within-sample regressions, N = 290 + 281.

**Output:** `results/stats/affect_analysis/affect_features_predict_params.csv` ✅

**Setup.** ω_z and κ_z (log-z-scored within sample) regressed on 8 affect predictors: anxiety/confidence × T-slope, D-slope, reward-slope, intercept. All predictors z-scored within sample.

**Headline (replicates in both samples):**
- confidence_slope_reward → ω: β = −0.22 (exp), −0.30 (conf); p = 0.002, 5e-5
- confidence_slope_reward → κ: β = −0.20 (exp), −0.16 (conf); p = 0.008, 0.025
- confidence_intercept → κ: β = −0.22 (exp), −0.20 (conf); p = 0.025, 0.039

Anxiety features and threat-reactivity slopes mostly null. R² = 0.04–0.08.

**Status:** SUPPORTED with replication. Both ω and κ share a confidence-based metacognitive substrate — specifically, how steeply confidence drops with cookie reward. Affect is NOT a parallel readout; it's the substrate of conservative computational style.

---

## 2026-06-05 — ★★★ Fitness landscape over (ω, κ) parameter space

**Script:** `scripts/analysis/fitness_landscape.py` ✅. 30×30 grid (900 points).

**Outputs:**
- `results/stats/joint_optimal/fitness_landscape.csv` ✅
- `results/figs/joint_optimal/fitness_landscape.png` ✅ (3-panel heatmap)

**Setup.** For each (ω, κ) on log-spaced grid: solve W = S·R − (1−S)·ω·(R+C) − κ(u−req)²·D for optimal vigor on each branch, softmax choice with τ=2.01, compute objective E[earnings] = S·R − (1−S)·C and E[survival] = S, averaged across 9 (T, D_heavy) conditions.

**Headline.** Three distinct optima:
- Earnings max: ω = 0.12, κ = 0.05
- Survival max: ω = 10 (boundary), κ = 0.49
- Combined fitness: ω = 0.26, κ = 0.05

Observed N = 571 subject median (ω = 1.42, κ = 0.21) sits in an intermediate Pareto position — over-cautious vs EV-max but not at survival-max extreme.

**Status:** SUPPORTED — the conceptual centerpiece for the paper's normative section. Frames human computational individual differences as departures from a multi-objective fitness landscape, not from a single "right" optimum.

---

## 2026-06-05 — ★★ Foraging-optimum grid analysis (calibrated κ_opt)

**Script:** `scripts/analysis/foraging_optimum_grid.py` ✅. N=571, within-sample analysis.

**Output:** `results/stats/joint_optimal/foraging_optimum_grid.csv` ✅

**Setup.** Foraging objective W = S·R − (1−S)(R+C) − κ_opt(u−req)²D, ω=1 fixed. κ_opt calibrated via SSE-minimization against group-median observed vigor: κ_opt* = 6.87. Sensitivity bounds: 3.43 and 13.73. Per-subject signed deviations in Δ_choice, Δ_vigor_heavy, Δ_vigor_light. Tested with params alone and + affect slopes/intercepts.

**Headline (replicates in both samples, all κ_opt values):**
- ω → over-avoidance + over-pressing (signature of cautious arousal)
- κ → over-avoidance + under-pressing (signature of effort conservation)
- R² params alone: 0.88–0.92 (choice), 0.61–0.78 (vigor)
- Affect signal small (ΔR² 0.005–0.025), inconsistent across affect predictors, weakest at calibrated κ_opt*

**Status:** SUPPORTED — strong normative validation of the framework. Calibrated foraging optimum is the cleanest definition of "optimal" for the paper. Affect-reactivity → residual deviation is modest under this framing (much weaker than under pct_opt).

---

## 2026-06-05 — ★★ Optimal switching + affect reactivity → optimality

**Script:** `scripts/analysis/optimal_switching_affect.py` ✅. N = 571 pooled.

**Output:** `results/stats/joint_optimal/optimal_switching_affect.csv` ✅

**Setup.** Group-level adaptive switching shown (P(heavy) drops with T, vigor rises). pct_opt regressed on (ω, κ) base + each per-subject affect slope on (T, D, cookie reward).

**Headline.** Base R² = 0.57 from ω + κ. Adding all affect slopes + intercepts: R² = 0.65 (ΔR² = +0.08). Strongest: confidence_slope_threat β = −0.23, p = 3×10⁻¹⁷ (subjects whose confidence drops more with rising threat are more optimal). Anxiety_slope_threat β = +0.18, p = 5×10⁻¹¹. Confidence_slope_distance β = −0.15, p = 2×10⁻⁸.

**Status:** SUPPORTED — confirms user's proposed Nuzzi-style framing: humans approximately optimal at group level; deviations from optimality driven by affect reactivity to task conditions (esp. confidence to threat). Within-sample replication needed before headline status.

---

## 2026-06-05 — ★ ω → survival: normative validation, replicates both samples

**Script:** `scripts/analysis/omega_survival.py` ✅. N = 571 pooled, both samples separately.

**Output:** `results/stats/joint_optimal/omega_survival.csv` ✅

**Headline:** ω → escape_rate β = +0.222, p = 1×10⁻⁶ (pooled). Replicates exploratory (β=+0.24, p=2e-4) and confirmatory (β=+0.19, p=2e-3). κ null. Holds at all three threat levels (strongest at T=0.5: β=+0.27, p=2e-9). Also: ω → fewer total captures per trial (β=−0.22, p=1e-6).

**Status:** SUPPORTED with strong within-sample replication. Major normative-validation finding. Belongs in paper §2.4 or new §2.5.

---

## 2026-06-05 — Param-vs-behavior comparison + Fung bug fix

**Scripts:** 
- `scripts/analysis/param_vs_behavior_clinical.py` ✅ (N=571, merge bug avoided from the start)
- `scripts/analysis/fung_style_condition_clinical.py` ✅ (re-run after fixing merge bug)

**Outputs:**
- `results/stats/clinical/param_vs_behavior_clinical.csv` ✅
- `results/stats/clinical/fung_style_condition_clinical.csv` ✅ (corrected)

**Bug discovered:** Both scripts originally used `groupby(["subj", "T_round"])` aliasing exp/conf subj=1. Fixed to include sample.

**Headline (corrected):**
- 36/264 Bonferroni hits (up from 18). Top: confidence_T0.5 → AMI_Social β=−0.24 p=1e-8.
- (ω, κ) predict vigor_shift (R²=0.072) and p_heavy_shift (R²=0.023) — model captures behavior.
- (ω, κ) DO NOT carry clinical signal that survives when shift or affect intercept is in the model.
- Bridge: parameters → behavior → clinical is the mechanism; parameters → clinical directly is null.

**Status:** SUPPORTED with sample-replication caveat. Headline now: model describes behavior; behavior + subjective state predict clinical apathy; ω/κ do not directly predict clinical scales but do explain behavioral deployment.

---

## 2026-06-05 — ★ Fung-style condition × clinical: 18/264 survive Bonferroni — clinical signal lives in affect readouts and behavioral shifts

**Script:** `scripts/analysis/fung_style_condition_clinical.py` ✅. Pooled N = 571, sample-controlled.

**Output:** `results/stats/clinical/fung_style_condition_clinical.csv` ✅

**Setup.** 264 univariate regressions: clinical_z ~ predictor_z + sample. Three predictor groups: (a) per-condition behavior/affect (P_heavy_T0.1/0.5/0.9, vigor_T*, anxiety_T*, confidence_T*), (b) condition shifts (high-T minus low-T), (c) per-subject affect reactivity (intercept, slope_T, slope_D, cal_T from result_510).

**Headline.** 18 survive Bonferroni at α = 0.000189. Two findings matter most:
- **p_heavy_shift_THighLow → AMI_Behavioural** (β = −0.154, p = 2e-4): Fung-style behavior × clinical hit. Apathy = failure to modulate choice across threat.
- **confidence_intercept → apathy scales** (AMI_Total β = −0.20, AMI_Behavioural β = −0.18, AMI_Social β = −0.22): low task-baseline confidence ↔ broad apathy/engagement deficit.

Anxiety_intercept ↔ anxiety scales hits at p < 1e-7 but partly method-variance.

**Status:** SUPPORTED in pooled data; within-sample replication required before headline status.

---

## 2026-06-05 — Mediation v3: trial-level (state) mediation

**Script:** `scripts/analysis/trial_level_mediation.py` ✅. Probe-trial mixed-effects. Monte Carlo CI (20k iter). N=293, ~10k trials per question.

**Output:** `results/stats/affect_analysis/trial_level_mediation.csv` ✅

**Setup.** Probe-trial pooled. Three mixed-effects fits per question: c-total (vigor ~ T+D+ω+κ), a-path (affect ~ same), b-path (vigor ~ same + affect). All with random intercept by subject. Affect decomposed into between (subject mean) + within (trial deviation). MC CI on indirect = a × b assuming independence.

**Headline.** Anxiety: null (a-paths ≈ 0). Confidence: state-level indirect statistically robust (p < 0.001 for both ω and κ) but quantitatively small (β ≈ +0.002–0.004, prop_mediated ~0.5%). κ pathway is a suppressor. Between-subject mediation null by design (random intercept absorbs trait variance).

**Result status:** PARTIAL/SMALL. Concludes the mediation thread. Verb is "*adaptively tunes*" / "*participates in*" — NOT "organizes." Confidence is part of a real but small within-trial regulatory loop; it is not the pipeline through which (ω, κ) shape behavior.

---

## 2026-06-05 — Mediation v2: confidence STRUCTURE (intercept + slope_T + slope_D + cal_T)

**Script:** `scripts/analysis/confidence_structure_mediation.py` ✅. Pooled N = 559. 5000-iter bootstrap.

**Outputs:**
- `results/stats/affect_analysis/confidence_structure_mediation.csv` ✅ (single-mediator)
- `results/stats/affect_analysis/confidence_structure_mediation_multi.csv` ✅ (multi-mediator)

**Setup.** Used per-subject confidence/anxiety structure decomposition from [[result_510]] (intercept after partialing T, D; slope on T; slope on D; r(T, response)). Multivariate mediation with both ω_z and κ_z as exposures simultaneously, sample-controlled. Tested each structure measure singly and the three (intercept + slope_T + slope_D) jointly.

**Headline.** Slopes uniformly null. Intercept (baseline) carries small bootstrap-significant indirect effects for p_heavy, escape_rate, earnings (β ≈ −0.005 to −0.019 on z-scaled outcomes). Anxiety structure completely null.

**Result status:** PARTIAL. Picks the verb between §4.35's "reflects only" and full "organizes" — confidence *level* (not reactivity) carries a small piece of the (ω, κ) → behavior link, but total indirects are null and intercept interpretation has a circularity risk (intercept partly downstream of choice patterns).

---

## 2026-06-05 — Mediation test: confidence/anxiety as (ω, κ) → behavior mediators (NULL)

**Script:** `scripts/analysis/confidence_mediation.py` ✅. Pooled N = 571. 5000-iter bootstrap subject resamples.

**Output:** `results/stats/affect_analysis/confidence_mediation.csv` ✅

**Setup.** Multivariate mediation: ω_z and κ_z entered together as exposures, mediator (mean_confidence or mean_anxiety), outcome ∈ {earnings, pct_opt, p_heavy, mean_vigor, escape_rate}. Sample-controlled. Bootstrap CI on indirect (a × b).

**Headline numbers.** All indirect effects |β| ≤ 0.007 (z-scaled). c′ ≈ c throughout — no path through affect. The one bootstrap CI that excludes zero (ω → p_heavy via mean_confidence, p_boot = 0.039) gives prop_mediated < 1%, substantively null.

**Result status:** NULL. Picks the verb: "confidence *reflects* / *tracks* / *tunes*" are supported by other analyses; "confidence *organizes* / *mediates*" is NOT. The integrative-readout framing remains valid only as correlation + moment-to-moment regulation, not as a pipeline.

---

## 2026-06-05 — Affect reshapes within-trial vigor (probe trials, pooled)

**Script:** `scripts/analysis/affect_reshapes_behavior.py` ✅ run pooled (exp + conf).

**Output:** `results/stats/affect_analysis/affect_reshapes_behavior.csv` ✅

**Approach.** Long-form probe-trial table (each probe asks one question — anxiety OR confidence; not both). Per-trial vigor (`norm_rate`) joined to per-trial rating, plus per-subject (ω, κ) from M4. Within-question z-score for affect. Fit `statsmodels.MixedLM` with random intercept by subject (REML=False so logL comparable). Three nested models: M_base (T, D, ω, κ) → M_affect (+aff_z) → M_int (+ T:aff + D:aff). Fit separately per question (n ~10k trials each, N = 293).

**Headline numbers:**

| Question | aff β | aff p | ΔAIC (M_affect vs M_base) | Notable interaction |
|---|---|---|---|---|
| Anxiety | +0.021 | 0.002 ★★ | −7.5 | T:aff p = 0.06 (marginal) |
| Confidence | −0.041 | 2.6e-9 ★★★ | −33.4 | D:aff β = +0.020, p = 8e-4 ★★★ |

Trait param effects (ω β ≈ +0.38, κ β ≈ −0.75) dwarf the affect signal but the affect-of-the-moment carries small, well-resolved incremental within-subject structure. Anxiety nudges vigor up; confidence nudges it down on average but supports vigor at high D.

**Result status:** SUPPORTED. Used to back the "metacognitive signals are structured alongside the W(u) computation" framing. Honest magnitude caveats apply: affect adds incremental, not dominant, predictive value.

---

## 2026-05-29 — Trial-level affect ~ S_probe LMM (result_501)

**Script:** `scripts/analysis/affect_survival_lmm.py` ✅ run on both samples.

**Outputs:**
- `results/stats/affect_analysis/s_probe_affect_lmm_exploratory.csv` ✅
- `results/stats/affect_analysis/s_probe_affect_lmm_confirmatory.csv` ✅

**Approach.** For each probe trial: compute u* = argmax_u W(u; T, D, ω, κ) using the subject's fitted M4 (ω, κ) and the M4 posterior-mean population params (γ, h, σ_sp). S_probe = S(u*, T, D). Z-score within sample. Fit `response ~ S_probe_z + (1|subj)` separately for anxiety and confidence using `statsmodels.mixedlm` (ML).

**Result — both signs as predicted (higher survival → less anxiety, more confidence), replicates across both samples:**

| Sample | Channel | β(S_probe_z) | SE | z | p | N obs | N subj |
|---|---|---|---|---|---|---|---|
| Exploratory | Anxiety | −0.584 | 0.025 | −23.74 | 1.5e-124 | 5,220 | 290 |
| Exploratory | Confidence | +0.625 | 0.025 | +25.30 | 3.1e-141 | 5,218 | 290 |
| Confirmatory | Anxiety | −0.545 | 0.025 | −22.25 | 1.0e-109 | 5,068 | 281 |
| Confirmatory | Confidence | +0.680 | 0.025 | +27.09 | 1.3e-161 | 5,068 | 281 |

**Validates the legacy NB04-03 numbers from `instructions/memory/hypotheses.md` § H4** (anxiety β = −0.602, confidence β = +0.632 on the older N=293 exploratory). Current exploratory (β = −0.584 / +0.625) matches to within rounding — confirms the M4-derived S_probe behaves like the deprecated framework's S_probe.

**Population params used (from M4 mcmc_convergence_diagnostics.csv posterior means):**

- Exploratory: γ = 0.846, h = 0.550, σ_sp = 0.247
- Confirmatory: γ = 0.826, h = 0.381, σ_sp = 0.243

**Implication:** result_501 upgraded from `untested` (deferred stub) → `supported`. The trial-level affect-survival coupling is a robust population-level effect that operates through the model-derived survival quantity, not just through raw threat/distance. Mechanistically distinct from the threat-only LMMs in [[result_102]] because S_probe is a model-derived nonlinear function of (T, D, ω, κ) rather than the raw conditions.

---

## Historical / older state below — last refreshed 2026-03-20

---

## Preprocessing (`notebooks/01_preprocessing/`)

| Notebook | Status | Output |
|----------|--------|--------|
| `01_run_pipeline.ipynb` | ✅ Complete | `data/exploratory_350/processed/stage{1-5}_*/` |
| `02_data_prep.ipynb` | ✅ Complete | Various |
| `03_data_prep_stage1_analysis_table.ipynb` | ✅ Complete | `analysis_table.parquet` (deprecated for vigor) |
| `04_behavior_overview.ipynb` | ✅ Complete | `results/figs/behavior/fig{1-5}_*.{pdf,png}` |

**Active stage5 output:** `data/exploratory_350/processed/stage5_filtered_data_20260320_191950/`
- `behavior.csv` — N=293 trials
- `psych.csv` — psychiatric battery (all subscales scored), N=293 subjects
- `feelings.csv` — 10,546 rows, 293 subjects (5,274 anxiety + 5,272 confidence)
- `subject_mapping.csv` — participantID → subj integer

---

## Choice Modeling (`notebooks/02_choice_modeling/`)

| Notebook | Status | Notes |
|----------|--------|-------|
| `01_fit_compare_ppc.ipynb` | ✅ Complete | FETExponentialBias fit (superseded by L3_add) |
| `02_parameter_recovery.ipynb` | ⚠️ Not run | Needs to run against L3_add fit |
| `03_unified_model_comparison.ipynb` | ✅ Complete | **11-model SVI comparison. Winner: L4a_add (α in survival, additive effort, hyperbolic kernel).** Saved: `unified_model_comparison.csv`, `unified_3param_clean.csv` |
| `scripts/run_unified_model_comparison.py` | ✅ Complete | Standalone re-run on new data path (stage5_20260320_191950). Results consistent with NB03. |

**Current winning model: L4a_add** (by ELBO and BIC)
```
SV = R·S - k·E - β·(1-S)
S = (1-T) + T/(1+λ·D/α)
```
Note: L3_add (no α) is still primary for subject-level parameter extraction (unified_3param_clean.csv) since α comes from vigor independently. L4a_add wins by 15.7 ELBO over L3_add.

- k, β per-subject; λ, τ population-level
- α (from vigor HBM) enters survival kernel — marginal gain (+15.7 ELBO vs L3_add)
- Additive >> multiplicative (+158 ELBO)
- Hyperbolic >> exponential (+190 ELBO vs L3_survival)

**Key model comparison findings (2026-03-20 re-run, N=293, 13185 trials):**
- L4a_add: ELBO=−6259.7, BIC=18135.6 (best)
- L3_add:  ELBO=−6275.4, BIC=18167.1 (primary parameter source)
- Per-subject z hurts (−112 ELBO) — not needed
- α in effort only (L4c): hurts (−24 ELBO vs L3_add)
- α in effort+survival (L4d): hurts (−2.6 ELBO vs L3_add)
- k-β r=−0.138 (p=0.018), k-α r=−0.052 (p=0.37), β-α r=+0.264 (p<0.001)

---

## Vigor Data Prep

| Script | Status | Output |
|--------|--------|--------|
| `scripts/vigor_data_prep.py` | ✅ Complete | `data/exploratory_350/processed/vigor_prep/` |

**vigor_prep contents:**
- `keypress_events.parquet` — 899,936 rows (one per keypress)
- `trial_events.parquet` — 23,733 rows (one per trial)
- `effort_ts.parquet` — 293 rows (calibrationMax)
- `subject_mapping.csv` — 293 rows

---

## Vigor Analysis (`notebooks/03_vigor_analysis/`)

| Notebook | Status | Key Output | Notes |
|----------|--------|------------|-------|
| `01_single_trial_visualization.ipynb` | ✅ Fixed | — | Column harmonization done |
| `02_kernel_smoothing.ipynb` | ✅ Complete | `smoothed_vigor_ts.parquet` (48.2 MB) | EVAL_HZ=20 |
| `03_tonic_phasic_decomposition.ipynb` | ✅ Fixed | — | Column harmonization done |
| `04_phase_extraction.ipynb` | ✅ Complete | `phase_vigor_metrics.parquet`, `phase_trial_metrics.parquet` | |
| `05_subject_features.ipynb` | ✅ Complete | `subject_vigor_table.csv` | |
| `06_choice_vigor_mapping.ipynb` | ✅ Complete | `results/choice_vigor_mapping_results.csv` | |
| `07_clinical_prediction.ipynb` | ✅ Unblocked | — | Factor scores now available from NB06-psych |
| `08_parameter_dissociation.ipynb` | ✅ Complete | `results/tables/table_s2_parameter_dissociation.csv/.tex` | |
| `09_final_stats.ipynb` | ✅ Complete | `results/step1_modelfree_results.csv` | |
| `10_pls_vigor_params.ipynb` | ✅ Complete | `results/stats/pls_vigor_params_results.csv` | PLS + trial-level LMM |
| `11_vigor_ode.ipynb` | ✅ Dead end | — | ODE kinetics degenerate, no new findings |
| `12_imminence_diagnostics.ipynb` | ✅ Complete | — | Phase-based encounter diagnostics |
| `13_encounter_vigor_counts.ipynb` | ✅ Complete | — | Encounter-centered count-based vigor |
| `14_choice_vigor_dissociation.ipynb` | ✅ Complete | `results/figs/fig_*.png` | 6-figure dissociation visualization |
| `15_dissociation_formal_tests.ipynb` | ✅ Complete | — | Phase 0-6 statistical pipeline |
| `16_bayesian_vigor_model.ipynb` | ✅ Complete | `vigor_hbm_posteriors.csv`, `vigor_hbm_population.csv`, `vigor_hbm_idata.nc` | **Two-window HBM: α (pre-enc) + ρ (terminal)** |

**Vigor model (final) — re-run 2026-03-20 via scripts/run_vigor_hbm.py:**
```
pre_enc_rate  ~ Normal(α_i, σ_pre)                     # [enc-2, enc], vigor_norm
terminal_rate ~ Normal(γ_i + ρ_i·attack, σ_term)       # [trialEnd-2, trialEnd], vigor_norm
```
Data source: `smoothed_vigor_ts.parquet` (mean vigor_norm per window), N=293, 23,554 trials.
- μ_α=0.315, SB=0.964, shrinkage=89%, max Rhat=1.008
- μ_ρ=0.067, P(>0)=1.0, SB=0.635, shrinkage=37%, max Rhat=1.006
- α-ρ: r=+0.016, p=0.78 (independent)
- 0 divergences. idata.nc saved (549 MB).

---

## Psychological Analysis (`notebooks/04_psych_analysis/`)

| Notebook | Status | Notes |
|----------|--------|-------|
| `01_bayesian_mental_health_regressions.ipynb` | ⚠️ Unknown | Not checked recently |
| `02_psychological_analysis.ipynb` | ⚠️ Unknown | Not checked recently |
| `03_affect_survival.ipynb` | ✅ Complete (re-run 2026-03-20) | S_probe (L3_add, λ=2.0) → anxiety/confidence LMM; state-trait decomposition |
| `04_anxiety_vigor_coupling.ipynb` | ✅ Complete | Anxiety → vigor coupling NULL at all levels |
| `05_metacognitive_calibration.ipynb` | ✅ Complete | Probe-trial linkage, S_probe→ratings, k→calibration |
| `06_factor_analysis.ipynb` | ✅ Complete (re-run 2026-03-20) | 3-factor EFA (distress/fatigue/apathy), α→F3(apathy) R²=0.123, t=−6.11 |
| `07_pls_params_mental_health.ipynb` | ✅ Complete | PLS 5 params→MH+affect, CV R²=0.039, perm p<0.001 |
| `08_mixture_model_subtypes.ipynb` | ✅ Complete | GMM k=3; coupled/decoupled hypothesis NULL |

---

## Publication Figures (`notebooks/05_figures/`)

| Notebook | Status | Notes |
|----------|--------|-------|
| `01_publication_figures.ipynb` | ⚠️ Needs update | Will need rerun after draft rewrite |

---

## Results Files

**`results/stats/` (key files):**
- `unified_model_comparison.csv` ✅ (12-model SVI comparison)
- `unified_3param_clean.csv` ✅ (L3_add subject parameters: k, β)
- `vigor_hbm_posteriors.csv` ✅ (per-subject α, ρ, γ with posterior SDs; re-run 2026-03-20 via smoothed_vigor_ts)
- `vigor_hbm_population.csv` ✅ (population hyperparameters + split-half reliability)
- `affect_lmm_results.csv` ✅ (re-run 2026-03-20, L3_add S_probe)
- `affect_trait_scores.csv` ✅ (re-run 2026-03-20, per-subject mean affect + k/β)
- `affect_vigor_cross_domain.csv` ✅ (all n.s.)
- `psych_factor_scores.csv` ✅ (re-run 2026-03-20, 3-factor EFA, N=291)
- `psych_factor_loadings.csv` ✅ (re-run 2026-03-20)
- `psych_params_to_factors.csv` ✅ (re-run 2026-03-20, 3-param + 4-param OLS)
- `choice_vigor_dissociation_results.csv` ✅ (2026-03-20, 20-row stats table: correlations, ANOVAs, t-tests)
- `choice_vigor_dissociation_subjects.csv` ✅ (2026-03-20, N=293 subject-level data with quadrant labels)
- `pls_mh_*.csv` ✅ (PLS params→MH)
- `joint_correlated_correlations.csv` ✅ (2026-03-21, LKJ ρ posteriors for all 6 param pairs)
- `joint_correlated_subjects.csv` ✅ (2026-03-21, per-subject k, β, α, δ from joint model)
- `joint_correlated_population.csv` ✅ (2026-03-21, population hyperparameters + ELBO)
- `joint_correlated_omega_samples.csv` ✅ (2026-03-21, 4000 posterior samples of correlation matrix)

**EVC+gamma parameter recovery (2026-03-26):**
- `evc_parameter_recovery.csv` ✅ (5 synthetic datasets × 50 subjects; c_death r=0.946, epsilon r=0.926, c_effort r=0.04 NOT recoverable, gamma=0.262 vs true 0.283)

**EVC Option 2 parameter recovery (2026-03-27):**
- `evc_option2_recovery.csv` ✅ (5 datasets × 50 subj; ce r=0.941 PASS, cd r=0.917 PASS, eps r=-0.025 FAIL — no individual variance, gamma=0.274 vs true 0.210 slight positive bias)
- `fig_s_option2_recovery.png` ✅ (3-panel scatter: ce, cd, eps true vs recovered)
- Script: `scripts/analysis/evc_option2_recovery.py`

**EVC-LQR full pipeline (2026-03-27):**
- `evc_lqr_recovery.csv` ✅ (5 datasets × 50 subj; cd r=0.888, eps r=0.933, gamma 0.314 vs true 0.318)
- `evc_lqr_ppc.csv` ✅ (Choice acc=75.4%, AUC=0.819, subj choice r=0.901, vigor r=0.510, subj vigor r=0.717)
- `evc_lqr_clinical.csv` ✅ (No FDR survivors; best uncorrected: cd→AMI_Emotional r=0.121 p=0.039)
- `evc_lqr_clinical_interactions.csv` ✅ (No significant cd×eps interactions)
- `evc_lqr_clinical_factors.csv` ✅ (F1/F2/F3 all null)
- `evc_lqr_affect.csv` ✅ (Anxiety beta=-0.786 t=-13.09; Confidence beta=0.848 t=13.40)
- `evc_lqr_metacognition.csv` ✅ (Conf-CQ r=0.012 null; Conf-SR r=-0.048 null; Steiger z=0.82 ns)
- `evc_lqr_dissociation.csv` ✅ (Partial dissociation: cal→CQ r=0.239, disc→STAI-State r=0.308)
- `evc_lqr_profiles.csv` ✅ (4 quadrants; P(heavy) R²=0.877; Helpless archetype lowest earnings)

**Figures (2026-03-27):**
- `fig_s_lqr_recovery.png` ✅ (2-panel scatter: cd and eps recovery)
- `fig_ppc_lqr.png` ✅ (6-panel PPC)
- `fig_s_lqr_clinical.png` ✅ (Forest plot)
- `fig_lqr_metacognition.png` ✅ (4-panel metacognition)
- `fig_lqr_quadrants.png` ✅ (4-panel profiles)

**Draft:**
- `drafts/draft003/evc_lqr_paper.md` ✅ (Full paper + critical review)

**DEFINITIVE EVC 2+2 model (2026-03-27) — population epsilon:**
- Model: `scripts/modeling/evc_final_2plus2.py` ✅
- `oc_evc_final_params.csv` ✅ (N=293, per-subject ce and cd)
- `oc_evc_final_population.csv` ✅ (epsilon=0.098, gamma=0.210, ce_vigor=0.003, tau=0.476)
- **Fit:** BIC=17,768, Choice acc=79.3%, Subj choice r²=0.951, Vigor r²=0.511, Subj vigor r²=0.687
- `evc_final_recovery.csv` ✅ (3 datasets×50 subj; ce r=0.916 PASS, cd r=0.943 PASS, gamma PASS)
- `evc_final_ppc.csv` ✅ (Choice acc=79.3%, AUC=0.876, subj choice r=0.976, vigor r=0.722, subj vigor r=0.836)
- `evc_final_affect.csv` ✅ (Anxiety beta=-0.557 t=-14.04; Confidence beta=0.575 t=13.48)
- `evc_final_metacognition.csv` ✅ (Conf-CQ r=-0.081 null; Conf-SR r=-0.048 null; Steiger z=-0.50 ns)
- `evc_final_dissociation.csv` ✅ (cal→CQ r=0.230 p<.001, disc→STAI-State r=0.327 p<.0001)
- `evc_final_clinical.csv` ✅ (No FDR survivors; no significant interactions)
- `evc_final_clinical_factors.csv` ✅ (F1/F2/F3 all null)
- `evc_final_profiles.csv` ✅ (4 quadrants: Cautious/Lazy/Vigilant/Bold; P(heavy) R²=0.953)
- Figures: fig_s_final_recovery.png, fig_ppc_final.png, fig_s_final_clinical.png, fig_final_metacognition.png, fig_final_quadrants.png

**Superseded (keep for reference):**
- `FET_Exp_Bias_*.csv` — old model, replaced by L3_add
- `joint_model_*.csv` — old joint model (independent priors, σ_δ collapsed), replaced by joint_correlated_*

**`results/model_fits/exploratory/`:**
- `vigor_hbm_idata.nc` ✅ (full MCMC trace, 549 MB, re-run 2026-03-20 via smoothed_vigor_ts)
- `FET_Exp_Bias_fit.pkl` — superseded

**`results/tables/`:**
- `table_s2_parameter_dissociation.csv/.tex` ✅
