---
result_id: 208
class: computational_model
title: ω and κ map onto survival, errors, vigor, and decision quality as preregistered
status: partial
prereg_h: [H4a, H4b, H4c, H4d, H4e]
internal_h: []
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/H4_profiles_optimality.ipynb]
scripts: []
outputs: [results/stats/confirmatory_hypothesis_results.csv, results/stats/joint_optimal/mcmc_m4_params.csv, results/stats/individual_diffs/profiles_exploratory.csv, results/stats/individual_diffs/profiles_confirmatory.csv]
figures: [results/figs/h4/h4_param_behavior_exploratory.pdf, results/figs/h4/h4_optimality_1d_exploratory.pdf]
created: 2026-05-27
last_run: 2026-05-27
---

# Result 208 — ω and κ map onto survival, errors, vigor, and decision quality as preregistered

## Overview

The preregistered H4 family asks whether the two per-subject parameters of the joint fitness model (ω, capture cost; κ, effort cost) predict five ecologically meaningful behavioral outcomes: escape rate on attack trials (H4a), the directional bias of suboptimal choices toward overcaution (H4b), mean vigor (H4c), the balance of effort-driven vs threat-driven avoidance as a predictor of overall optimality (H4d), and a model-consistency-to-earnings linkage (H4e). All five tests use Bayesian linear regressions on subject-level summaries, with the prereg's HDI-excludes-zero criterion. H4a–d pass in both samples with HDIs cleanly excluding zero in the predicted direction; H4e fails to replicate in the confirmatory sample despite passing in the exploratory sample. The result establishes that the two model parameters are interpretable individual-difference traits with the ecological meaning the prereg attributed to them, with H4e specifically isolated as a non-replicated indirect link.

## Hypothesis

**Statements (verbatim from prereg §H4):**
- **H4a.** Higher capture cost (ω) will predict higher escape rates on attack trials.
- **H4b.** Capture cost will predict the proportion of overcautious errors.
- **H4c.** Higher effort cost (κ) will predict lower pressing intensity.
- **H4d.** The balance between capture cost and effort cost will predict decision quality.
- **H4e.** Consistency with the joint fitness function will predict foraging earnings.

**Preregistered criterion (all five tests):** 95% HDI excludes zero in the predicted direction.

## Data Source

- **Samples:** Exploratory N = 290, confirmatory N = 281.
- **Input files:**
  - `results/stats/joint_optimal/{sample}/mcmc_m4_params.csv` — per-subject ω, κ posterior means.
  - `results/stats/individual_diffs/profiles_{sample}.csv` — per-subject behavioral outcomes (escape_rate, earnings, pct_opt, mean_vigor, overcaution_ratio, choice_consistency, intensity_deviation).
- **Unit of analysis:** Subject (one row per participant).
- **N entering each regression:** 290 / 281.

## Method

For each H4 sub-test, a Bayesian linear regression was fit with default weakly informative priors (Normal(0, σ) on coefficients, σ data-scaled).

**Regressions:**
- **H4a:** `escape_rate ~ omega_z + kappa_z`
- **H4b:** `overcaution_ratio ~ omega_z` (with overall overcaution percentage reported descriptively)
- **H4c:** `mean_vigor ~ kappa_z`
- **H4d:** `pct_optimal ~ angle_z` (angle = atan2(kappa_z, omega_z))
- **H4e:** `earnings ~ choice_consistency_z + intensity_deviation_z`

All predictors are z-scored within sample. ω and κ are log-transformed before z-scoring (per prereg Transformations section).

**Posterior sampling:** `bambi`, 4 chains × 2,000 draws + 1,000 tuning.

**Inference criterion:** 95% HDI excludes zero in the predicted direction.

**Notebook:** `notebooks/analysis/H4_profiles_optimality.ipynb`. Cells 3 (H4a), 5 (H4b), 7 (H4c), 9 (H4d), 11 (H4e). Validated 2026-05-27 against the cached `confirmatory_hypothesis_results.csv` and the H4 draft (`drafts/results_by_hypothesis/H4_profiles_optimality.md`).

## Result

**H4a — ω predicts escape rate:**

| Coefficient | Exploratory | Confirmatory |
|---|---|---|
| β(ω) | **+0.060** [+0.029, +0.093] | **+0.046** [+0.017, +0.075] |
| β(κ) | −0.003 [−0.033, +0.029] | +0.003 [−0.028, +0.030] |

ω predicts escape; κ does not. **PASS** in both samples.

**H4b — ω predicts overcaution:**

| Quantity | Exploratory | Confirmatory |
|---|---|---|
| % errors that are overcautious | 79% | 90% |
| β(ω → overcaution ratio) | **+0.177** [+0.163, +0.193] | **+0.123** [+0.109, +0.137] |

Overcaution dominates the error pool; ω drives it. **PASS** in both samples.

**H4c — κ predicts mean vigor:**

| Coefficient | Exploratory | Confirmatory |
|---|---|---|
| β(κ → mean vigor) | **−0.194** [−0.215, −0.173] | **−0.196** [−0.217, −0.176] |

Effect is large, precise, and nearly identical across samples. **PASS** in both samples.

**H4d — ω–κ angle predicts optimality:**

| Coefficient | Exploratory | Confirmatory |
|---|---|---|
| β(angle → pct optimal) | **−0.041** [−0.055, −0.026] | **−0.054** [−0.072, −0.036] |

Higher angle (more effort-driven relative to threat-driven avoidance) → lower decision quality. **PASS** in both samples.

**H4e — Model consistency → earnings:**

| Coefficient | Exploratory | Confirmatory |
|---|---|---|
| β(choice consistency → earnings) | +14.3 [+5.0, +23.2] | +8.4 [−2.3, +19.0] |
| β(intensity deviation → earnings) | −19.3 [−28.8, −9.4] | −4.1 [−14.6, +7.4] |

Both effects passed in exploratory. **Neither replicates** in confirmatory: both HDIs span zero. **FAIL** in confirmatory.

**Verdict on prereg criterion:** **PASS** for H4a, H4b, H4c, H4d in both samples. **FAIL** for H4e (confirmatory). Overall status: `partial`.

## Interpretation

The four direct parameter-to-outcome tests (H4a–d) replicate cleanly across two independent samples with HDIs that exclude zero in the predicted direction and effect-size estimates that are remarkably stable between samples (e.g., β(κ → vigor) = −0.194 expl vs −0.196 conf). The two parameters carry the ecological meaning the prereg attributed to them: ω is the capture-aversion trait that predicts who survives (H4a) and who errs on the overcautious side (H4b); κ is the effort-aversion trait that predicts who presses less (H4c); and the angle in (ω, κ) space — the relative balance of threat-driven vs effort-driven avoidance — predicts overall decision quality (H4d). The angle effect is small in absolute units but consistent across samples and aligns with a substantive theoretical claim: effort-driven avoidance is indiscriminate (avoid the hard option regardless of threat), while threat-driven avoidance is context-appropriate (avoid the hard option specifically when it is dangerous).

H4e tells a different story. The indirect link from model-consistency-to-earnings was significant in the exploratory sample but did not replicate. Both choice consistency and intensity deviation produced HDIs spanning zero in the confirmatory sample, with point estimates roughly half the exploratory magnitude. The failure is a clean replication-failure rather than a sign-flip, consistent with the exploratory finding being an upward-biased estimate of a smaller-than-claimed true effect. The substantive implication: model consistency does not translate into earnings as directly as H4e proposed. Subjects who deviate from model-predicted choices or intensities are not, in aggregate, earning less — possibly because the model's optimum is not the same as the task's reward-maximizing strategy under noise, or because individual-level deviations average out at the trial-aggregate earnings level.

The H4a–d results license the use of ω and κ as substantive individual-difference variables in downstream analyses, and the H4e failure should be noted as a place where the model-to-behavior bridge breaks down. Both findings together reinforce the prereg's framing of ω and κ as separable traits ([[result_204]]): the parameter-to-outcome mappings are direction-specific (ω to escape and overcaution; κ to vigor), not interchangeable.

## Caveats & Limitations

- **H4e failure is the only confirmatory replication failure in the prereg H1–H5 family.** It deserves a paragraph in the manuscript, framed as a clean null on an indirect linkage, not a contradiction of the underlying joint-model framework.
- **All H4 regressions use point-estimate posterior means of ω and κ as predictors,** ignoring posterior uncertainty in the parameters. A fully propagated Bayesian regression on the joint posterior would tighten or loosen these intervals; the cached results use the simpler approach (per the prereg).
- **`overcaution_ratio` is an empirical quantity computed from condition-cell expected rewards, not a model-derived quantity.** It depends on which cells are classified as "heavy is optimal" via the task's reward structure, not the fitted model. This makes H4b a model-to-behavior test rather than a self-consistency test.
- **Angle metric (H4d) compresses ω and κ into a single dimension.** This is the prereg specification, but it discards magnitude information. A 2D analysis (separate ω and κ effects) is in the related [[result_204]] and the exploratory choice-vigor coupling block (400s). Both the angle and the separate-parameter treatments rest on identifiable per-subject parameters: recovery is strong for *both* ω and κ (r ≈ 0.92 each, with calibrated credible intervals) under the production M4 + MCMC ([[result_205]]), so subject-level κ — not just its population distribution — is a reliable input here.
- **Pooled exploratory + confirmatory analyses are not reported here** because the prereg specifies sample-by-sample replication. Combined-sample regressions are documented in `instructions/memory/allocation_analysis.md` for downstream clinical analyses.

## Replication

**To regenerate this result from scratch:**

```bash
PYTHONPATH=notebooks/analysis \
  jupyter nbconvert --to notebook --execute \
  notebooks/analysis/H4_profiles_optimality.ipynb \
  --inplace --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=1800
```

**Expected runtime:** ~5–10 min per sample (Bayesian regressions with bambi).

**Expected outputs:**
- Stdout reports of each H4a–e regression with β posterior summaries and pass/fail.
- Figures regenerated at `results/figs/h4/`.

## References

**Related results:**
- [[result_201]] — Joint model M4 fit (source of ω, κ used here as predictors).
- [[result_204]] — M4 vs M3 (single-parameter), establishing ω and κ as separable.
- [[result_205]] — Parameter recovery: both ω and κ are well-recovered per-subject (r ≈ 0.92), licensing their use as individual-difference predictors here.
- [[result_402]] — β creates choice-vigor dissociation (extends H4 to a third parameter).
- [[result_502]] — Anxiety calibration as additional individual-difference predictor (H5a/b).

**Notebook / drafts:**
- `notebooks/analysis/H4_profiles_optimality.ipynb`
- `drafts/results_by_hypothesis/H4_profiles_optimality.md` — legacy bundled writeup.

**Literature:**
- Bednekoff, P. A. (2007). Foraging in the face of danger.
- Houston, A. I., & McNamara, J. M. (1999). Models of Adaptive Behaviour.
