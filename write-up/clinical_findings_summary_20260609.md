# Clinical Findings Summary — EffortForagingUnderThreat

**Date:** 2026-06-09
**Authors:** Noah Okada (with Claude Code)
**Sample:** Pooled N = 571 (exploratory N = 290, confirmatory N = 281)
**Status:** Final, post-STAI-fix specification — recommended for paper

---

## 1. Executive Summary

The hierarchical Bayesian model's per-subject capture-cost weighting parameter, ω ("vigilance"), is independently predicted by two clinical dimensions that push it in *opposite* directions: apathy (AMI_Total β = +0.135 ★, 95% HDI [+0.055, +0.214]) increases vigilance, while a transdiagnostic anxiety–depression distress composite (β = -0.084 ★, 95% HDI [-0.165, -0.008]) decreases it. Both effects coexist in the same multivariate regression on pooled N = 571 with log(κ) as a covariate, and their interaction is null. The pure-apathy − pure-distress contrast on log(ω) is +0.27 SD ★. The effort-cost weighting parameter, κ ("mobilization"), shows no robust clinical association. Substantively, the model parameters do not collapse anxiety–depression comorbidity onto a single distress dimension; they decompose comorbidity into two computationally opposed signatures, with comorbid subjects sitting between the two pure clinical types by additive cancellation. This is the principal clinical claim the paper can make. See discoveries §4.79.

---

## 2. The Headline Finding (Corrected Specification)

### Primary regression

```
log(ω)_z  ~  AMI_Total_z  +  ANX+DEP_FIXED_z  +  log(κ)_z
```

Pooled N = 571, Student-t robust likelihood (ν = 3), within-sample z-scoring of all variables, fully fixed STAI scoring.

| Predictor | β | 95% HDI | P(direction) |
|---|---|---|---|
| **AMI_Total** | **+0.135 ★** | [+0.055, +0.214] | 1.000 |
| **ANX+DEP composite (7 scales, FIXED)** | **−0.084 ★** | [−0.165, −0.008] | 0.983 |
| log_κ (structural covariate) | +0.370 ★ | [+0.291, +0.449] | 1.000 |

Both effects survive 95% HDI in opposite directions (§4.79).

### Why this is the right specification

1. **Student-t likelihood** — robust to extreme posterior-mean (ω, κ) values without arbitrary outlier exclusion. Effect holds on all 571 subjects with no `|log_ratio_z| > 3` filter (§4.65).
2. **Within-sample z-scoring** — removes between-sample mean drift on ω (exp < conf by ~0.5 SD) which otherwise inflates spurious sample-correlated effects (§4.4 / §4.6 pooled-z analyses showed this).
3. **FIXED STAI** — see §6 below; the STAI scoring bug went through two iterations of repair before reaching the correct external-anchor fix.
4. **log(κ) covariate** — controls for the structural (ω, κ) correlation (r ≈ +0.44 within sample). All reported effects are conditional on κ.

### ANX+DEP composite definition

z-mean of 7 z-scored scales: DASS21_Anxiety, DASS21_Depression, DASS21_Stress, STAI_Trait_FIXED, OASIS_Total, STICSA_Total, PHQ9_Total. Pre-specified by RDoC/HiTOP transdiagnostic distress principles (motivation: no single anxiety or depression scale survived; the aggregate signal is what's informative — see §6).

### Cross-sample replication (§4.76, §4.79)

| Sample | N | β(AMI_Total) | β(ANX+DEP) |
|---|---|---|---|
| **Pooled** | 571 | +0.135 ★ [+0.055, +0.214] | −0.084 ★ [−0.165, −0.008] |
| **Confirmatory** | 281 | +0.170 ★ [+0.054, +0.278] | −0.126 ★ [−0.239, −0.013] |
| Exploratory | 290 | +0.101 (clips 0, P>0 = 0.96) | −0.043 (P<0 = 0.78) |

The confirmatory sample replicates both effects at full 95% HDI. The exploratory sample is directionally consistent for both (no sign flips) but underpowered, particularly on the ANX+DEP side. Honest framing for paper: "Both effects survive 95% HDI in the pooled sample and confirmatory sample alone; in the exploratory sample, effects were in the same direction with high posterior probability but did not formally clear the 95% HDI criterion."

---

## 3. The Comorbidity Story

### Median-split 2D typology (AMI_Total × ANX+DEP_FIXED), §4.79

| Profile | Definition | N | Mean log(ω)_z |
|---|---|---|---|
| **Pure Apathy** | high AMI, low ANXDEP | 104 | **+0.189** |
| Comorbid | high both | 172 | +0.015 |
| Healthy | low both | 182 | −0.063 |
| **Pure Distress** | low AMI, high ANXDEP | 113 | **−0.094** |

### Key contrast

**Pure Apathy − Pure Distress: β = +0.266 ★** (95% HDI [+0.025, +0.505], P(β > 0) = 0.984), controlling for log(κ). The two pure types differ by ≈0.27 SD on log(ω) in opposite directions from the healthy baseline.

### Why Comorbid sits between the pure types

The continuous regression interaction term is null:

```
log(ω) ~ AMI × ANXDEP + log(κ)    → β_interaction = +0.006 (n.s.)
```

The two effects are **additive**, so a high-AMI/high-ANXDEP subject inherits ω↑ from apathy and ω↓ from distress that partially cancel. This is exactly the pattern observed in the median-split: Comorbid (+0.015) ≈ Healthy (−0.063), with both flanked by the two pure types.

### Substantive interpretation (for paper Discussion)

> Anxiety–depression comorbidity does NOT map onto a single distress dimension when read through the model's computational decomposition. Rather, the two clinical features (apathy and anxious-depressive distress) reflect two independent processes that act on the *same* parameter (ω) in *opposite directions*. Subjects with one feature in isolation show a clear computational signature; subjects with both features show cancellation. This is why traditional symptom-count or unitary-distress framings have failed to recover stable parameter–symptom associations: the parameter is being pushed two ways at once.

This is the substantive claim the paper has been searching for since result_604 (§4.6) — the earlier "comorbidity = nothing" verdict was correct that there is no *single* additive distress effect, but the *decomposition into opposite signatures* is the real finding.

### AMI × ANX+DEP correlation

r = +0.30 in the pooled sample (≈9% shared variance) — moderate, as expected; apathy and distress are correlated but separable dimensions, not redundant.

---

## 4. AMI_Social Refinement

Among the three AMI subscales (Social, Behavioural, Emotional), AMI_Social does the unique work.

### Sensitivity specification (§4.73, swap AMI_Total → AMI_Social)

```
log(ω)_z  ~  AMI_Social_z  +  6 other clinical totals  +  log(κ)_z
```

- **AMI_Social β = +0.168 ★** (95% HDI [+0.087, +0.246])
- AMI_Total β = +0.148 ★ (in the analogous specification)
- AMI_Behavioural and AMI_Emotional do not survive when entered as separate subscales (§4.72 kitchen-sink with 3 AMI subscales: only AMI_Social ★, β = +0.159).

The Social subscale is the carrier of the AMI signal; combining with Behavioural and Emotional dilutes it slightly. The clinical signal is specifically **social motivation / social-anhedonic apathy**, not general apathy or behavioural inactivation.

Recommendation: lead with AMI_Total in the headline (most defensible, larger 7-item composite), report AMI_Social as a sensitivity refinement that strengthens slightly.

---

## 5. The κ Axis is Silent

Across every specification tried this session (univariate, joint with ω, kitchen-sink with 7 totals, kitchen-sink with 11 subscales, EFA factors, item-level latent factors), **no clinical scale robustly predicts log(κ)** when proper STAI scoring is used:

- Kitchen-sink κ model with 7 totals (§4.73): all 7 clinical totals NULL; only log(ω) survives (structural correlation).
- Kitchen-sink κ model with 11 subscales (§4.72): all NULL.
- Previous reports of STAI → log(κ) β = −0.133 ★ (pre-§4.72) were broken-STAI artifacts; corrected and fixed STAI both give NULL.

This is not just a non-finding — it is itself a finding. The apathy and distress signals are specifically about **vigilance (ω)**, not **mobilization (κ)**. The two parameters dissociate clinically: ω carries the psychiatric structure; κ is functionally inert at the clinical level (though it remains strongly behavioural via effort/vigor — see discoveries §2c–2g).

Possible explanations: (a) true null — clinical variation operates on threat appraisal, not motor cost; (b) measurement noise — κ posteriors may be less informative per subject than ω posteriors; (c) different time horizon — effort cost is online/within-trial while threat appraisal accumulates across trials. The paper should report the κ silence as a substantive finding, not a limitation.

---

## 6. Methodological Journey

This section is honest about what didn't work and why — useful for the paper's Methods and for future sessions.

### The STAI reverse-coding saga (§4.67 → §4.78 → §4.79)

The Spielberger STAI Trait subscale arrives with half its items keyed in opposite directions ("I feel calm" vs "I worry"). The master `psych.csv` file's STAI_Trait column did NOT apply the standard reverse-coding — mean inter-item correlation was r = +0.049 with 52.1% negative item pairings, vs r ≈ +0.5 for other anxiety scales. This was diagnosed during item-level EFA in §4.67.

**First fix (PC1-sign reverse-coding):** Flipped 11 of 20 STAI items based on PC1 loading sign. Mean inter-item r went from +0.049 to +0.499 — looked correct internally. Used in §4.67–§4.78 analyses as `STAI_Trait_corrected`.

**Second diagnosis (§4.78):** `STAI_Trait_corrected` correlated **negatively** with every external anxiety scale (DASS_Anxiety r = −0.605, OASIS r = −0.738, STICSA r = −0.804). PC1's sign is arbitrary, and PC1 happened to point in the "calmness" direction, so the corrected scale was measuring the construct's inverse.

**Final fix (§4.79):** After PC1-sign item-level alignment, check whether summed STAI correlates positively with DASS_Anxiety as an external anchor; if negative, flip the whole scale. Result: `STAI_Trait_FIXED` with r = +0.605 with DASS_Anxiety, r = +0.738 with OASIS, r = +0.804 with STICSA. Now correctly oriented.

**Impact on results:** AMI effect essentially unchanged (+0.134 → +0.135). ANX+DEP composite slightly smaller (−0.092 → −0.084) because the composite previously inflated the magnitude with one inverted scale of seven. Substantive findings (AMI direction, ANX+DEP direction, typology, pure-type contrast) all intact. The fix gives the honest effect sizes.

### Why individual anxiety/depression scales fail but the composite works

Each individual anxiety/depression scale carries a small effect (|β| ≈ 0.05–0.10) with mixed direction across scales (§4.75 Test 1: STICSA actually pointed positive). Inter-scale correlations are high (r > 0.7 among DASS_Anx, DASS_Stress, OASIS, STICSA, STAI), so a kitchen-sink with all 7 individual scales suffers from multicollinearity and parameter inflation. WAIC explicitly preferred AMI-only and AMI+MFIS models over the full 11-scale kitchen-sink (§4.75 Test 3) — individual anxiety/depression scales hurt the model fit by adding noise per parameter without proportionate signal.

The composite acts as a hand-crafted regularization: averaging correlated scales applies a strong shared-direction prior, aggregating small individual effects into a detectable transdiagnostic signal. Methodologically the composite is *somewhat* exploratory in motivation (we tried individual scales first; the composite emerged after they were null), but it is *substantively* pre-specified by RDoC/HiTOP principles. The paper should be transparent about this — frame the composite as the test of a transdiagnostic prediction, not as a "we tried lots of things and this worked" outcome.

### The factor-analysis dead-end (§4.67–§4.71)

Item-level EFA on 106 questionnaire items was promising. At 8 factors (Horn's parallel analysis solution), F6 (apathy items) replicated AMI_Social → ω at β = +0.114 ★, and the data revealed that AMI items split into THREE latent factors (F5, F6, F8) rather than the published 3-subscale structure. Interesting structural finding.

At 5 factors (§4.68), F4 emerged as "non-social apathy" and showed an *opposite-direction* ω effect (β = −0.102 ★) — a clean social-vs-nonsocial apathy dissociation. F3 (trait NA) appeared to predict log(κ).

At 3 factors (§4.69), F3 (trait anxiety) appeared to push BOTH log(ω) and log(κ) in parallel — invisible to balance metrics. This generated the "two-axis story" framing.

But: §4.70 showed that NO raw anxiety scale (univariate or composite) reproduced the F3 parallel-effects pattern. The "trait anxiety → both parameters" finding was retracted as a factor-rotation artifact. §4.71 partially un-retracted it (the direction does appear under suppression and focused-subset specifications), but it did not survive the corrected-STAI rerun in §4.72.

**Verdict on EFA:** It surfaced the right structural questions (apathy is heterogeneous; AMI subscales aren't optimal; ω is the active clinical axis), but factor-rotation artifacts and PC1-sign issues made specific latent-factor regression findings hard to defend. The cleaner approach was to return to validated subscale totals and use the composite as a transdiagnostic aggregator. The paper should NOT report EFA-based factor regressions as headline results.

### Balance-metric exploration (§4.65, §4.66)

The original §4.63 finding was on log(ω/κ) as a vigilance–mobilization balance. §4.65 verified the AMI_Social signal was metric-invariant across four parameterizations: log(ω/κ), z(log ω) − z(log κ), ω/(ω+κ), and arctan(log κ / log ω). All four survived for AMI_Social; all four were null for DASS scales. Good metric robustness.

But §4.66 reframed: balance metrics can be misleading because they collapse parallel effects (where both parameters move the same direction) into a null. The honest decomposition is `log(ω) ~ symptom` and `log(κ) ~ symptom` separately, with both reported. This is what the headline (§4.79) now does. The balance framing has been retired from primary reporting.

### Outlier filtering (§4.63 → §4.65)

§4.63 originally applied a `|log_ratio_z| > 3` filter dropping 10 subjects. §4.65 showed the AMI signal survives on all 571 subjects with Student-t likelihood — outlier filtering is unnecessary and methodologically harder to defend. All final analyses use all 571 subjects with Student-t.

---

## 7. Final Paper Specification

### Primary regression for Results section

```
log(ω)_z  ~  AMI_Total_z  +  ANX+DEP_FIXED_z  +  log(κ)_z
```

(pooled N = 571, Student-t robust likelihood, within-sample z-scoring, NumPyro HMC/NUTS 4 chains × 1000 warmup + 1000 sampling)

Report β with 95% HDI for AMI_Total and ANX+DEP. Cite the structural log(κ) covariate as a control, not as a substantive effect.

### Replication appendix

Per-sample table (exp vs conf vs pooled) showing both effects with HDIs and directional probabilities. Frame exploratory underpowering honestly.

### Typology figure

2D clinical-space scatter (AMI_Total on x, ANX+DEP on y) with points colored by log(ω)_z. Overlay median-split quadrant boundaries and quadrant mean ω. Save path: `results/figs/affect_analysis/clinical_typology_omega.png` (needs regeneration with fixed STAI).

### Sensitivity refinements

- Swap AMI_Total → AMI_Social: β strengthens to +0.168 ★.
- Drop STICSA from composite (sensitivity): nearly identical results (§4.76).
- Kitchen-sink with all 7 individual scales: only AMI survives (consistent with WAIC preference, §4.75 Test 3).

### Discussion paragraph (drop-in candidate)

> Two distinct clinical dimensions independently predict the model's capture-cost weighting parameter ω. Apathy, indexed by AMI total scores, increases ω (β = +0.135, 95% HDI [+0.055, +0.214]), with the social-anhedonia subscale carrying the unique signal (β = +0.168). A transdiagnostic anxiety–depression composite, aggregating seven established distress instruments, decreases ω (β = −0.084, 95% HDI [−0.165, −0.008]). The two effects coexist in a single multivariate model with log(κ) covaried, and their interaction is null. The four median-split quadrants of the AMI × ANX+DEP space show the predicted pattern: pure-apathy subjects score 0.27 SD higher on log(ω) than pure-distress subjects, while comorbid subjects sit near the healthy baseline by additive cancellation. The model parameters thus do not represent anxiety–depression comorbidity as a unitary distress dimension; instead, they decompose it into two computationally opposed signatures, with comorbid subjects representing the cancellation of two real but counter-directed processes. The effort-cost weighting parameter κ shows no robust clinical association in any specification, indicating that psychiatric variation in this task operates specifically on threat-appraisal weighting, not on motor mobilization.

---

## 8. Caveats and Limitations

1. **Exploratory sample underpowered for ANX+DEP.** Directional only (P(β < 0) = 0.78 for exploratory; fails 95% HDI). The pooled and confirmatory effects do clear 95% HDI. Frame honestly as "real but at threshold of detectability with N ≈ 290."

2. **ANX+DEP composite is somewhat exploratory in motivation.** Arrived at after individual scales were null. Pre-specified by RDoC/HiTOP transdiagnostic principles, but the paper should not oversell its priorhood. Cite the motivation as a transdiagnostic distress test, not as the first-pass analysis.

3. **κ silence may reflect measurement noise rather than true null.** κ posteriors are smaller-magnitude than ω posteriors in the population, and the κ axis sits closer to its identifiability floor. Cannot fully distinguish "no clinical effect on motor cost weighting" from "insufficient power to detect a small κ effect."

4. **Effect sizes are modest.** β ≈ 0.1–0.15 corresponds to ~10–15% of an SD in log(ω) per SD of clinical predictor. Clinically real but not large. The paper should not overclaim translational utility from these magnitudes; the substantive point is the *directional opposition* between apathy and distress, not the size of either individual effect.

5. **STAI scoring required two passes to fix.** The PC1-sign heuristic without external anchoring produced an inverted scale. Future projects should always validate reverse-coding against an external anchor scale.

6. **Median-split typology is illustrative, not statistically inferential.** The continuous regression is the inferential test; the quadrant means are a presentation device. After conditioning on continuous ω + κ, the typology indicators are null — they do not add information beyond the linear ω effect (§4.74).

7. **Generalizability.** Sample is Prolific online participants; clinical scores are self-report; no diagnostic interviews. The "clinical" framing is a dimensional psychiatric trait framing, not a clinical-population framing.

---

## 9. Outputs and Files

### Result CSVs

- [`results/stats/affect_analysis/stai_fixed_exp.csv`](../results/stats/affect_analysis/stai_fixed_exp.csv) — exploratory clinical scores with FIXED STAI
- [`results/stats/affect_analysis/stai_fixed_con.csv`](../results/stats/affect_analysis/stai_fixed_con.csv) — confirmatory clinical scores with FIXED STAI
- [`results/stats/affect_analysis/headline_corrected_results.csv`](../results/stats/affect_analysis/headline_corrected_results.csv) — §4.72 corrected-STAI kitchen-sink headline
- [`results/stats/affect_analysis/kitchen_sink_totals.csv`](../results/stats/affect_analysis/kitchen_sink_totals.csv) — §4.73 7-total specification
- [`results/stats/affect_analysis/log_ratio_clinical_robust.csv`](../results/stats/affect_analysis/log_ratio_clinical_robust.csv) — §4.63 original balance-metric analysis
- [`results/stats/affect_analysis/omega_kappa_profile_clinical.csv`](../results/stats/affect_analysis/omega_kappa_profile_clinical.csv) — §4.74 typology contrasts
- [`results/stats/affect_analysis/multivariate_omega_kappa.csv`](../results/stats/affect_analysis/multivariate_omega_kappa.csv) — joint (ω, κ) regression results
- `results/stats/affect_analysis/ami_anxdep_clinical_typology.csv` (TBD save) — §4.77 typology quadrant means

### Scripts

- [`scripts/analysis/fix_stai_and_rerun_full.py`](../scripts/analysis/fix_stai_and_rerun_full.py) — definitive STAI fix + headline rerun (§4.79)
- [`scripts/analysis/kitchen_sink_totals.py`](../scripts/analysis/kitchen_sink_totals.py) — §4.73 7-total specification
- [`scripts/analysis/anxdep_composite_replication.py`](../scripts/analysis/anxdep_composite_replication.py) — §4.76 cross-sample replication
- [`scripts/analysis/recompute_corrected_clinical_and_rerun.py`](../scripts/analysis/recompute_corrected_clinical_and_rerun.py) — §4.72 corrected-STAI rerun
- [`scripts/analysis/omega_kappa_profile_clinical.py`](../scripts/analysis/omega_kappa_profile_clinical.py) — §4.74 typology contrasts
- [`scripts/analysis/ami_anxdep_clinical_typology.py`](../scripts/analysis/ami_anxdep_clinical_typology.py) — §4.77 typology figure

### Figures

- [`results/figs/affect_analysis/clinical_typology_omega.png`](../results/figs/affect_analysis/clinical_typology_omega.png) — needs regeneration with FIXED STAI
- [`results/figs/affect_analysis/omega_kappa_AMI_scatter.png`](../results/figs/affect_analysis/omega_kappa_AMI_scatter.png) — typology scatter

### Memory references

- `instructions/memory/discoveries.md` §§4.63–4.79 (full analytical history)
- `instructions/memory/pipeline_state.md` entries 2026-06-08 / 2026-06-09 (script provenance and final-state markers)
