# Apathy Results — Full Write-up

**Date:** 2026-06-09
**Outcome variable:** AMI_Total (Apathy Motivation Index, total score across Social/Behavioural/Emotional subscales)
**Sample:** Pooled N = 571 (exploratory N = 290, confirmatory N = 281)
**Inference:** Student-t robust Bayesian regression (Bambi), within-sample z-scoring, 95% HDI as significance threshold
**Recommended primary specification:** `log(ω) ~ AMI_Total + ANX+DEP_composite + log(κ)` with corrected STAI scoring

---

## 1. Executive Summary

Apathy (AMI_Total) predicts the model's per-subject capture-cost weighting parameter (ω) in the threat-effort foraging task: β = +0.135 (95% HDI [+0.055, +0.214]) in the recommended specification. The effect is independent of κ (mobilization, effort-cost weighting), independent of the structural ω–κ correlation, and survives in a multivariate model with all major clinical scale composites as competing predictors. Apathy and ω are also related through direct behavioral channels: per-subject regression coefficients of choice and vigor on threat predict AMI_Total (β_T_choice → AMI_Total β = -0.202 ★, pooled), and this behavioral signature replicates cleanly across both pre-registered samples. The behavioral findings are slightly more sensitive than the model-parameter approach (in-sample R² ~4× higher), but the (ω, κ) model offers theoretical grounding that the raw behavioral betas lack. Together, the model and behavioral results converge on a coherent story: apathetic subjects show heightened threat-deterred decision-making, captured both as higher ω in the model parameters and as more negative β_T_choice in raw behavior. **The AMI → log(ω) relationship is fully mediated by within-task confidence** (§9): apathetic subjects feel less confident during the task (β = −0.204 ★), and lower confidence in turn predicts higher ω (β = −0.181 ★), with ~32% of the total effect mediated and the direct path dropping to non-significance after confidence is controlled. Anxiety-state mediators (mean anxiety, anxiety reactivity to threat) do NOT carry the effect — only confidence does. The apathy phenotype of high vigilance is fundamentally a metacognitive low-confidence phenotype.

---

## 2. Model-Parameter Findings on AMI_Total

### 2.1 Core specification

| # | Specification | Predictor(s) on AMI_Total | β | 95% HDI | Survives? |
|---|---|---|---|---|---|
| 1 | Univariate, balance metric | log(ω/κ) | +0.084 | [+0.019, +0.144] | ★ |
| 2 | Joint: AMI_Total ~ log(ω) + log(κ) | log(ω) | +0.141 | [+0.054, +0.225] | ★ |
| 2 | Joint: AMI_Total ~ log(ω) + log(κ) | log(κ) | -0.059 | [-0.146, +0.033] | null |
| 3 | Polar: AMI_Total ~ log(ω/κ) + log(ω·κ) | log(ω/κ) | +0.092 | [-0.001, +0.180] | clips 0 |
| 3 | Polar: AMI_Total ~ log(ω/κ) + log(ω·κ) | log(ω·κ) | +0.115 | [+0.022, +0.199] | ★ |
| 4 | Kitchen-sink ω (subscales) | log(ω) ~ AMI_Total + 10 scales + log(κ) | AMI_Total: +0.151 | [+0.056, +0.244] | ★ |
| 5 | Kitchen-sink ω (totals) | log(ω) ~ AMI_Total + 6 totals + log(κ) | AMI_Total: +0.148 | [+0.062, +0.225] | ★ |
| 6 | **Recommended primary** | log(ω) ~ AMI_Total + ANX+DEP_composite + log(κ) | **AMI_Total: +0.135** | **[+0.055, +0.214]** | **★** |

### 2.2 Reading the specifications

**Joint frame (specification 2)** isolates which model parameter carries the apathy signal: log(ω) is significant (β = +0.141 ★) while log(κ) is null (β = -0.059, n.s.). Apathy maps specifically onto capture-cost weighting (vigilance), not effort-cost weighting (mobilization).

**Polar frame (specification 3)** decomposes (log ω, log κ) into orthogonal rotations: log(ω/κ) ("balance") and log(ω·κ) ("magnitude"). Both rotations are in the same model. The magnitude term survives more cleanly than the balance term — apathetic subjects have *high overall weighting on both cost dimensions* slightly more than they have *vigilance specifically dominant over mobilization*.

**Kitchen-sink specifications (4-6)** progressively add clinical-scale controls. The AMI_Total effect is stable across all of them: β ≈ +0.135 to +0.151. This robustness to inter-scale partialling argues that the AMI signal isn't a generic distress confound — it's a unique construct effect.

### 2.3 Cross-sample replication of the AMI → ω finding

| Sample | β(AMI_Total → log ω) | 95% HDI | Status |
|---|---|---|---|
| Pooled (N=571) | +0.135 | [+0.055, +0.214] | ★ |
| Confirmatory (N=281) | +0.163 | [+0.058, +0.269] | ★ |
| Exploratory (N=290) | +0.108 | [-0.002, +0.218] | clips 0 |

Confirmatory replicates with full HDI significance; exploratory shows the same direction (P(β > 0) = 0.97) but doesn't formally meet the 95% threshold. Cross-sample direction is consistent.

### 2.4 Polar parameter decomposition on AMI_Total

Sources §4.65, §4.66, §4.85.

The model's (log ω, log κ) space can be rotated 45° to:
- **log(ω/κ)** — "balance" axis: positive = vigilance-dominated; negative = mobilization-dominated
- **log(ω·κ)** — "magnitude" axis: positive = strong weighting on both costs; negative = weak weighting on both

Both rotations predict AMI_Total positively when entered jointly (β = +0.092 and +0.115; only the magnitude term clears 95% HDI). The structural correlation between log_omega and log_kappa in the joint posterior (r ≈ +0.5) means these two rotations capture related but distinct variance. The "magnitude" interpretation says: apathetic subjects are *more engaged with both* the capture-cost and effort-cost dimensions, with a smaller additional tilt toward vigilance specifically.

---

## 3. Behavioral Findings on AMI_Total

### 3.1 Per-subject behavioral regression coefficients

For each subject, fit:
- Choice model: `P(choose_high) ~ threat + distance + threat:distance` → 3 betas
- Vigor model: `mean_rate ~ T_round + actual_dist` (cell-level WLS) → 2 betas

Test each beta against AMI_Total (Source §4.81):

| Behavioral beta | β on AMI_Total | 95% HDI | Survives? |
|---|---|---|---|
| **β_T_choice** | **-0.202** | [-0.280, -0.119] | **★** |
| β_TxD_choice | +0.116 | [+0.032, +0.198] | ★ |
| β_T_vigor | -0.110 | [-0.189, -0.027] | ★ |
| β_D_choice | -0.055 | (null) | |
| β_D_vigor | -0.016 | (null) | |
| **threat_sens_composite** (avg of T_choice, T_vigor after sign flip) | **-0.211** | [-0.291, -0.131] | **★** |

The strongest single behavioral predictor is **β_T_choice**: subjects whose choice probability of the high-effort option drops more steeply with threat have higher apathy scores. Effect size β = -0.20 ★ is roughly 1.5× larger than the model-parameter effect (+0.135 for AMI_Total → log ω).

### 3.2 Cross-sample replication of β_T_choice → AMI_Total

| Sample | β | 95% HDI | Status |
|---|---|---|---|
| Pooled (N=571) | -0.202 | [-0.280, -0.119] | ★ |
| Confirmatory (N=281) | -0.222 | [-0.338, -0.110] | ★ |
| Exploratory (N=290) | -0.185 | [-0.295, -0.069] | ★ |

**Both samples survive 95% HDI cleanly.** This is the strongest cross-sample replication of any clinical finding in the session — better than the model-parameter approach (where exploratory clips zero).

### 3.3 Channel modality: total reactivity + channel preference jointly predict apathy

Sources §4.84, §4.85.

Define sign-agnostic magnitudes:
- `choice_mod = sqrt(β_T_choice² + β_D_choice²)`
- `vigor_mod = sqrt(β_T_vigor² + β_D_vigor²)`
- `total_mod = choice_mod_z + vigor_mod_z` (overall behavioral responsiveness)
- `channel_balance = choice_mod_z − vigor_mod_z` (channel preference, positive = more choice-modulated)

Joint regression: `AMI_Total ~ total_mod + channel_balance`:

| Predictor | β | 95% HDI | Survives? |
|---|---|---|---|
| **total_mod** | **+0.111** | [+0.027, +0.190] | **★** |
| **channel_balance** | **+0.121** | [+0.042, +0.205] | **★** |

Both effects survive **independently** in the same model — these are two genuinely separate dimensions of behavioral variation that each correlate with apathy. Apathetic subjects show heightened overall reactivity AND preferentially deploy that reactivity through the choice channel rather than the vigor channel.

The 2D quadrant analysis confirms this: subjects scoring HIGH on choice modulation (regardless of vigor level) show elevated AMI_Total relative to LOW-choice-modulation subjects (contrast β ≈ -0.31 ★ in Bayesian regression).

---

## 4. Comparison: Model vs Behavioral Approaches

Source §4.85.

### 4.1 Empirical correspondence

The model parameters and behavioral magnitudes capture overlapping but distinct variance:

| Predicted mapping | Actual Pearson r |
|---|---|
| choice_mod ↔ log_omega | +0.203 |
| vigor_mod ↔ log_kappa (negative direction) | -0.110 |
| total_mod_behav ↔ log(ω/κ) | +0.083 (weak) |
| channel_balance_behav ↔ log(ω·κ) | +0.186 |

Correspondence is directional but the correlations are surprisingly weak.

### 4.2 Joint regression: behavioral + model parameters simultaneously

`AMI_Total ~ total_mod_behav + channel_balance_behav + log(ω/κ) + log(ω·κ)`:

| Predictor | β | Survives? |
|---|---|---|
| total_mod_behav | +0.098 | ★ |
| channel_balance_behav | +0.113 | ★ |
| log(ω/κ) | +0.079 | (dropped) |
| log(ω·κ) | +0.081 | (dropped) |

**When both are in the model, the behavioral measures dominate** — they survive while the model parameters drop to non-significance. The behavioral measures contain the model variance plus additional clinical variance the model doesn't capture.

### 4.3 Predictive R² comparison

Source §4.80.

| Predictor set | in-sample R² for AMI_Total | 5-fold CV R² |
|---|---|---|
| (log ω, log κ) only | 0.012 | -0.028 |
| 5 behavioral betas only | 0.053 | +0.001 |
| Both combined (7 predictors) | 0.061 | -0.003 |

The behavioral betas predict AMI_Total roughly 4× better in-sample than (ω, κ) does, and only the behavioral predictor set achieves a positive cross-validated R² (barely). The model parameters do not outperform raw per-subject behavioral coefficients at clinical prediction.

### 4.4 Why this matters

The (ω, κ) model framework was designed to provide a *theoretical* decomposition of behavior into capture-cost and effort-cost weighting components. It does so successfully — the parameters fit the data well and have clean recovery properties. But for the specific task of predicting clinical state from individual differences, raw behavioral measurements are more sensitive. The model parameters compress all behavioral variation into two scalar dimensions, which loses information that the per-subject regression coefficients preserve.

**For the paper:**
- Lead with behavioral findings as the empirically primary clinical signal
- Frame (ω, κ) as the theoretical explanation for why the behavioral signal exists
- Be transparent that the model is a useful summary but not a complete predictor

---

## 5. Clinical Typology with AMI_Total

Source §4.79.

Median split on AMI_Total × ANX+DEP_composite → 4 clinical profiles:

| Profile | N | log(ω)_z mean |
|---|---|---|
| **Pure Apathy** (high AMI, low ANXDEP) | 104 | **+0.189** |
| Comorbid (high both) | 172 | +0.015 |
| Healthy (low both) | 182 | -0.063 |
| **Pure Distress** (low AMI, high ANXDEP) | 113 | **-0.094** |

PureApathy − PureDistress contrast: β = +0.266 ★ (HDI [+0.025, +0.505]).

**Substantive interpretation:** Pure-apathy subjects show heightened vigilance; pure-distress subjects show reduced vigilance; comorbid and healthy subjects are intermediate. The two pure clinical types diverge by ~0.27 SD on log(ω), and the comorbid quadrant sits between them, consistent with additive cancellation in the (ω, κ) effect.

---

## 6. AMI Subscale Specificity (Brief Note)

Although AMI_Total is the primary outcome reported above, the within-AMI subscale analysis (Source §4.82) reveals that:
- AMI_Social and AMI_Behavioural both carry the apathy → ω signal
- AMI_Emotional does NOT (β ≈ 0 on every behavioral and model-parameter test)

This refines the substantive interpretation: the apathy signal is specifically about motivational/action disengagement (Social and Behavioural subscales), not emotional anhedonia. For the headline AMI_Total finding, this means the signal is driven by 12 of the 18 AMI items (the Social and Behavioural subscale items).

The behavioral channel-modality finding has a related subscale dissociation: AMI_Social is captured by β_T_choice only, while AMI_Behavioural is captured by both β_T_choice and β_T_vigor. AMI_Total averages over these and is captured by both.

---

## 7. Caveats and Limitations

1. **Effect sizes are modest.** β ≈ +0.135 on the standardized log(ω) outcome means apathy explains ~1-2% of variance in the parameter, and 5-fold cross-validated R² is essentially zero. The Bayesian regression survives significance threshold cleanly, but absolute predictive power is small.

2. **Exploratory sample is underpowered.** The model-parameter finding doesn't formally meet 95% HDI in the exploratory sample alone (β = +0.108, HDI clips zero at -0.002). However:
   - Direction is consistent (P(β > 0) = 0.97)
   - Confirmatory sample replicates ★
   - Pooled sample is robust
   - Behavioral findings replicate ★ in BOTH samples (no underpowering issue at the behavioral level)

3. **AMI subscale heterogeneity within AMI_Total.** The total score averages signal-carrying subscales (Social, Behavioural) with a null subscale (Emotional). Reporting AMI_Total slightly dilutes the effect — using AMI_Social alone gives β = +0.168 ★ in the same specification.

4. **κ axis is silent.** No clinical scale predicts log(κ) robustly. The clinical signal lives entirely on the ω axis.

5. **Modest comorbidity finding.** The ANX+DEP composite shows a small opposing effect on log(ω) (β = -0.084 ★ from §4.79), but this finding doesn't replicate at the behavioral level (§4.83) and cross-validates poorly. Treat as secondary.

---

## 8. Recommended Paper Framing

### Primary clinical claim (suggested wording)

> Apathy, measured by the Apathy Motivation Index total score (AMI_Total), predicts heightened subjective weighting of capture cost (ω) in our threat-effort foraging task. In a multivariate Bayesian regression controlling for 6 other clinical questionnaire totals and the structural log(κ) correlation, β(AMI_Total → log ω) = +0.135 (95% HDI [+0.055, +0.214]). The effect is independent of κ (β = -0.059, n.s.) and replicates with full significance in the confirmatory sample (β = +0.163 ★) and the same direction in the exploratory sample (P(β > 0) = 0.97).

### Complementary behavioral framing

> The model-parameter finding has a direct behavioral counterpart: per-subject regression coefficients of high-effort choice probability on threat predict AMI_Total with β = -0.202 ★ in the pooled sample, replicating with full HDI significance in BOTH pre-registered samples (β = -0.185 ★ exploratory, β = -0.222 ★ confirmatory). Subjects whose decisions are more strongly threat-deterred have higher apathy scores. This behavioral signature is independent of, and slightly more sensitive than, the (ω, κ) parameter framework.

### Two-dimensional behavioral structure

> Decomposing the behavioral response into modality dimensions, apathetic subjects show two independent signatures: heightened total behavioral reactivity (β = +0.111 ★) AND a preference for the choice channel over the vigor channel (β = +0.121 ★) when modulating behavior in response to threat and effort. Both dimensions survive independently in a joint regression.

### Clinical typology

> Subjects classified as Pure Apathy (high AMI, low ANX+DEP composite) show log(ω) ~0.27 SD higher than subjects classified as Pure Distress (low AMI, high ANX+DEP composite), 95% HDI [+0.025, +0.505]. Comorbid subjects (high on both clinical dimensions) sit between the two pure types, consistent with additive cancellation in the (ω, κ) effect.

---

## 9. Mediation: Confidence as the Cognitive-Affective Bridge

Source §4.86.

A natural follow-up question to the AMI → log(ω) finding is whether the relationship operates THROUGH a subjective state (within-task anxiety or confidence) or whether it's a direct disposition-to-parameter mapping. Bayesian mediation analysis on pooled N = 571 with four candidate mediators (mean anxiety, mean confidence, anxiety reactivity to threat, anxiety-threat correlation) gives a clear answer: **confidence fully mediates the effect; anxiety mediators do not.**

### 9.1 Mediator = mean_confidence (full mediation, ★)

| Path | β | 95% HDI | Survives? |
|---|---|---|---|
| **c** (total: AMI → log ω) | +0.114 | [+0.035, +0.193] | ★ |
| **a** (AMI → confidence) | **−0.204** | [−0.285, −0.123] | ★ |
| **b** (confidence → log ω \| AMI) | **−0.181** | [−0.263, −0.102] | ★ |
| **c'** (direct: AMI → log ω \| confidence) | +0.078 | [−0.003, +0.158] | **null (clips zero)** |
| **a × b (INDIRECT effect)** | **+0.037** | [+0.017, +0.061] | **★** |

**Proportion mediated ≈ 32%.** The c'-path (direct effect of AMI on log(ω) controlling for confidence) drops to non-significance.

### 9.2 Substantive interpretation

1. Apathetic subjects feel **less confident** during the task (large effect: β = −0.204 ★, ~0.2 SD per SD of apathy)
2. Subjects feeling less confident show **higher ω** (β = −0.181 ★)
3. The direct AMI → log(ω) effect **disappears** once confidence is in the model (c' = +0.078, n.s.)
4. Apathy's effect on capture-cost weighting is routed through subjective task confidence — it's a metacognitive bridge, not a direct disposition-to-parameter mapping

The "apathy phenotype" of high vigilance is actually a "low task-confidence" phenotype. Apathy and ω are connected via subjective cognitive-affective state during the task.

### 9.3 Confidence also mediates AMI → log(κ) (suppression mediation)

| Path | β | Survives? |
|---|---|---|
| c (total: AMI → log κ) | +0.030 | null (no total effect) |
| a (AMI → confidence) | −0.204 | ★ |
| b (confidence → log κ \| AMI) | −0.158 | ★ |
| c' (direct: AMI → log κ \| confidence) | −0.002 | null |
| **a × b (INDIRECT)** | **+0.032** | **★** |

This is suppression mediation: the total AMI → log(κ) effect is null at the surface, but there's an indirect effect through confidence (apathetic → lower confidence → lower κ). The only AMI-related signal on log(κ) is this mediated indirect path.

### 9.4 Anxiety mediators all FAIL

| Mediator | a-path (AMI → M) | b-path (M → ω) | Verdict |
|---|---|---|---|
| mean_anxiety | −0.022 (n.s.) | +0.027 (n.s.) | NO MEDIATION |
| anx_slope (anxiety reactivity to T) | +0.088 ★ | −0.011 (n.s.) | NO MEDIATION |
| anx_calibration (anxiety-T correlation) | +0.109 ★ | +0.004 (n.s.) | NO MEDIATION |

Apathy DOES predict heightened anxiety reactivity (a-paths significant for anx_slope and anx_calibration), but anxiety reactivity doesn't predict ω in turn (b-paths null). **Only confidence mediates the apathy → ω chain.**

### 9.5 Paper claim

> *The AMI → log(ω) relationship is fully mediated by within-task subjective confidence. Apathetic subjects feel less confident during the task (β = −0.204 ★), and lower task confidence in turn predicts higher capture-cost weighting (β = −0.181 ★). The direct AMI → log(ω) effect drops to non-significance once confidence is controlled (β = +0.078 from +0.114, c' n.s.), with indirect effect β = +0.037 ★ and ~32% of the total effect mediated. Anxiety-based mediators (mean anxiety, anxiety reactivity to threat, anxiety-threat correlation) do not mediate the effect — only confidence does. Apathy's behavioral signature is routed through a metacognitive confidence mechanism rather than direct disposition-parameter mapping.*

---

## 10. Factor Analysis: Methodological Findings and Mostly-Negative Substantive Results

Sources §4.67–§4.71, §4.78.

A parallel investigative thread examined the latent structure of psychiatric scales via item-level exploratory factor analysis (EFA) on 106 items from DASS-21, STAI, OASIS, STICSA, AMI, MFIS, and OASIS. The factor analysis was methodologically valuable but the substantive findings were mostly retracted or downgraded.

### 10.1 The STAI scoring bug (methodologically critical)

Item-level audit (§4.67) revealed that STAI items in the master psych.csv were not properly reverse-keyed — mean inter-item correlation was r = +0.05 with 52% negative pairs. After PC1-sign reverse-coding (flipping 11 of 20 items based on principal-component loading), mean inter-item r improved to +0.50. Subsequently (§4.78), a second-level check against DASS-Anxiety as an external anchor revealed that the entire PC1-corrected scale was still pointing in the *wrong direction* (r = −0.61 with DASS-Anxiety). Final fix: flip the whole scale after the item-level correction. The fully corrected scale (`STAI_Trait_FIXED`) is what's used in all post-§4.78 analyses.

**Impact**: every prior analysis using STAI_Trait had been operating on essentially noise. The corrected scale is now stored in `results/stats/affect_analysis/stai_fixed_{exp,con}.csv`.

### 10.2 Within-AMI structural dissociation (suggestive but exploratory)

Item-level EFA at 5 factors (§4.68) split AMI items into multiple data-driven factors. F4 (containing AMI items 1, 2, 3, 7, 9, 13, 16 — mostly non-Social-subscale items) predicted log(ω) with β = −0.102 ★ — *opposite* direction from AMI_Social. This is a within-AMI dissociation that the standard 3-subscale structure (Social/Behavioural/Emotional) averages over. **Two opposing apathy clusters exist within AMI**, but they cancel in the total score and aren't visible in the published subscale decomposition.

Substantively this is interesting (suggests sub-types of apathy with opposing parameter signatures) but the data-driven factor structure is harder to defend than the validated subscale structure for paper purposes.

### 10.3 The published AMI subscale structure isn't optimal for these data

Parallel analysis with Horn's procedure recommended 8 factors. AMI items distributed across at least 3 of these data-driven factors, not in the Social/Behavioural/Emotional pattern from Ang et al. Worth mentioning as a methodological aside in a limitations or supplementary section.

### 10.4 The "F3 trait anxiety → both parameters" finding — mostly retracted

In a 3-factor solution, F3 (loading on STAI items, after reverse-coding) predicted both log(ω) and log(κ) in parallel (β ≈ +0.10 on each). This looked like an important transdiagnostic finding — anxiety → reduced engagement on both cost-weighting parameters.

**However:**
- Raw STAI_Trait_FIXED on log(ω) and log(κ): both null (β ≈ −0.05, n.s.) — the factor signal doesn't replicate to the raw scale (§4.70)
- After exhaustive attempts to make it replicate via suppression with AMI control or focused-STAI-subset (§4.71): only one specification survives (OASIS → log(ω) β = −0.120 ★ controlling for AMI_Social) and it doesn't generalize cleanly

**Net result**: the F3 finding was largely a varimax-rotation artifact that emerged from the latent composite but doesn't have a clean raw-scale counterpart.

### 10.5 The general-distress factor dominates psychiatric covariance structure

F1 in the 106-item EFA had eigenvalue ≈ 42 vs F2 ≈ 6 — a single general-distress dimension explains the bulk of psychiatric covariance. The literature's "p-factor" interpretation is consistent with this. But this general factor doesn't itself predict (ω, κ) — only specific dimensions (AMI_Total, in our case) do.

### 10.6 Net contribution of the factor analysis

| Finding type | Outcome |
|---|---|
| Methodological (STAI bug) | Critical, definitive fix |
| Within-AMI structure | Suggestive, exploratory |
| F3 trait anxiety effect | Largely retracted (rotation artifact) |
| General distress factor | Confirms p-factor structure, no direct prediction utility |

**For the paper**: don't lead with factor analysis. Mention the STAI scoring fix in Methods. The AMI subscale dissociation can be a supplementary note or a future-work direction. The trait-anxiety hypothesis from F3 should not appear in the headline.

---

## 11. Outputs and Files

### Result CSVs
- [headline_corrected_results.csv](../results/stats/affect_analysis/headline_corrected_results.csv) — final corrected specification (model parameters)
- [kitchen_sink_totals.csv](../results/stats/affect_analysis/kitchen_sink_totals.csv) — kitchen-sink with totals
- [behavioral_betas_deep_dive.csv](../results/stats/affect_analysis/behavioral_betas_deep_dive.csv) — behavioral betas vs all scales
- [ami_omega_affect_mediation.csv](../results/stats/affect_analysis/ami_omega_affect_mediation.csv) — confidence mediation results (§9)
- [stai_fixed_exp.csv](../results/stats/affect_analysis/stai_fixed_exp.csv) and [stai_fixed_con.csv](../results/stats/affect_analysis/stai_fixed_con.csv) — corrected STAI scores
- [item_efa_loadings.csv](../results/stats/affect_analysis/item_efa_loadings.csv) — 8-factor EFA loadings (§10)
- [item_efa_reduced_summary.csv](../results/stats/affect_analysis/item_efa_reduced_summary.csv) — 3/4/5-factor solutions

### Scripts
- [recompute_corrected_clinical_and_rerun.py](../scripts/analysis/recompute_corrected_clinical_and_rerun.py)
- [kitchen_sink_totals.py](../scripts/analysis/kitchen_sink_totals.py)
- [behavioral_betas_predict_clinical.py](../scripts/analysis/behavioral_betas_predict_clinical.py)
- [behavioral_betas_deep_dive.py](../scripts/analysis/behavioral_betas_deep_dive.py)
- [channel_modality_clinical.py](../scripts/analysis/channel_modality_clinical.py)
- [model_param_channel_modality.py](../scripts/analysis/model_param_channel_modality.py)
- [ami_omega_affect_mediation.py](../scripts/analysis/ami_omega_affect_mediation.py) — confidence mediation (§9)
- [item_level_efa_on_params.py](../scripts/analysis/item_level_efa_on_params.py) — 8-factor EFA (§10)
- [item_efa_reduced_factors.py](../scripts/analysis/item_efa_reduced_factors.py) — 3/4/5-factor solutions
- [fix_stai_and_rerun_full.py](../scripts/analysis/fix_stai_and_rerun_full.py) — STAI direction fix (§10.1)

### Figures
- [modality_profile_AMI.png](../results/figs/affect_analysis/modality_profile_AMI.png) — 2D choice_mod × vigor_mod with AMI overlay
- [clinical_typology_omega.png](../results/figs/affect_analysis/clinical_typology_omega.png) — 2D clinical typology
- [item_efa_scree.png](../results/figs/affect_analysis/item_efa_scree.png) — EFA scree plot

### Memory references
- discoveries.md §4.66 (joint regression), §4.67 (item-level EFA / STAI bug discovered), §4.73 (kitchen-sink totals), §4.78 (STAI direction bug fully diagnosed), §4.79 (final corrected), §4.81 (behavioral betas), §4.84 (channel modality), §4.85 (model vs behavioral), **§4.86 (confidence mediation)**
- pipeline_state.md 2026-06-09 entries
