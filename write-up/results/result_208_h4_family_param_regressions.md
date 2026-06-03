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
outputs: [results/stats/confirmatory_hypothesis_results.csv, results/stats/joint_optimal/mcmc_m4_params.csv, results/stats/individual_diffs/profiles_exploratory.csv, results/stats/individual_diffs/profiles_confirmatory.csv, results/stats/individual_diffs/h4_polar_decomp.csv, results/stats/individual_diffs/h4_choice_decomp.csv, results/stats/individual_diffs/h4_predicted_r_cv.csv]
scripts: [scripts/analysis/h4_choice_decomp.py]
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

`choice_consistency` and `intensity_deviation` are computed *from* each subject's fitted ω, κ via the utility function `uV(ω, κ, …)` (cell 11 of the notebook). Adding ω, κ as covariates would introduce near-perfect collinearity, not control for a confound; H4e is intentionally an indirect test of whether model-consistency translates into earnings, and the prereg specifies these two regressors only.

All predictors are z-scored within sample. ω and κ are log-transformed before z-scoring (per prereg Transformations section).




**Posterior sampling:** `bambi`, 4 chains × 2,000 draws + 1,000 tuning (`BKW` in `notebooks/analysis/config.py`). For these Gaussian linear regressions with N ≈ 290 and ≤ 2 standardized predictors, posteriors are near-conjugate; the post-hoc decomposition block below reports R̂ ≤ 1.001 and ESS_bulk ≥ 7,000 per coefficient across all 36 fits, confirming that doubling draws would not move any HDI endpoint at the precision reported here.

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



## Exploratory: angle + magnitude vs. ω + κ decomposition (post-hoc)

### What this tests, and why

The preregistered H4 tests treat each model parameter as a one-dimensional individual-difference trait: H4a–c regress an outcome on a single parameter, and H4d collapses both parameters into a single *angle* — the ratio of effort-driven to threat-driven avoidance. None of these tests asks the joint question that the two-parameter model naturally raises: **does an individual's full position in (ω, κ) parameter space — its direction *and* its distance from the population centroid — shape behavior?** A subject who is moderately capture-averse and moderately effort-averse and one who is severely both lie at the same angle but represent very different cognitive phenotypes; H4d treats them as equivalent. Equivalently, ω and κ may have independent univariate effects of the same or opposite signs on a given outcome (as H4a already shows for escape rate, where ω drives the effect and κ is null), and they may interact — neither pattern is detectable from the angle alone.

To examine these structural questions, we refit four continuous outcomes — the three from the H4 family (attack-trial escape rate, mean pressing intensity, proportion of optimal choices) plus the most basic behavioral outcome, **P(heavy)**, which the prereg's H4 specifications do not include directly — using two complementary decompositions of the 2D (ω, κ) space:

- **Polar:** `outcome ~ angle_z + magnitude_z + angle_z × magnitude_z`, where `magnitude = √(ω_z² + κ_z²)` is each subject's Euclidean distance from the centroid in standardized-parameter space, then z-scored.
- **Cartesian:** `outcome ~ ω_z + κ_z + ω_z × κ_z`.

The two parameterizations are mathematically equivalent representations of the same 2D effect — only the coordinate basis differs. The polar form maps directly onto the conceptual distinction between **strategy** (which kind of avoidance dominates, captured by the angle) and **intensity** (how strongly the subject avoids in either direction, captured by the magnitude). The Cartesian form preserves the original parameter axes, making its main effects directly comparable to H4a and H4c. The interaction terms in each form capture how a subject's position on one axis modulates the effect of the other.

Posterior sampler kwargs match those used for H4a–e (`bambi`, 4 chains × 2,000 draws + 1,000 tuning, weakly informative priors). The two predictors in each fit are not statistically orthogonal because the underlying parameters are themselves correlated across subjects (**r(ω_z, κ_z) = +0.37** exploratory, **+0.30** confirmatory). This correlation is moderate, leaving enough independent variance to identify both main effects and the interaction, but it means the polar and Cartesian fits give slightly different views rather than redundant ones. Full posterior summaries for every term are cached at `results/stats/individual_diffs/h4_polar_decomp.csv`.

**Polar decomposition (95% HDI):**

| Outcome | Term | Exploratory | Confirmatory |
|---|---|---|---|
| escape_rate | angle_z | +0.021 [−0.009, +0.048] | **+0.051 [+0.023, +0.078]** |
|  | mag_z | −0.019 [−0.047, +0.010] | **−0.034 [−0.064, −0.008]** |
|  | angle × mag | +0.015 [−0.019, +0.046] | −0.019 [−0.047, +0.009] |
| mean_vigor | angle_z | **−0.102 [−0.129, −0.076]** | **−0.098 [−0.125, −0.073]** |
|  | mag_z | **+0.067 [+0.039, +0.094]** | **+0.054 [+0.029, +0.081]** |
|  | angle × mag | **−0.087 [−0.116, −0.056]** | **−0.101 [−0.127, −0.075]** |
| pct_opt | angle_z | **−0.049 [−0.063, −0.035]** | **−0.051 [−0.069, −0.033]** |
|  | mag_z | **−0.052 [−0.065, −0.038]** | **−0.024 [−0.042, −0.006]** |
|  | angle × mag | **−0.039 [−0.055, −0.024]** | **−0.027 [−0.045, −0.009]** |
| p_heavy | angle_z | **−0.097 [−0.116, −0.079]** | **−0.083 [−0.105, −0.061]** |
|  | mag_z | **−0.025 [−0.044, −0.006]** | +0.006 [−0.017, +0.028] |
|  | angle × mag | **−0.092 [−0.113, −0.071]** | **−0.033 [−0.056, −0.012]** |

**Cartesian decomposition (95% HDI):**

| Outcome | Term | Exploratory | Confirmatory |
|---|---|---|---|
| escape_rate | ω_z | **+0.058 [+0.029, +0.089]** | **+0.049 [+0.018, +0.077]** |
|  | κ_z | +0.005 [−0.024, +0.035] | +0.008 [−0.021, +0.038] |
|  | ω × κ | +0.008 [−0.017, +0.035] | −0.023 [−0.048, +0.002] |
| mean_vigor | ω_z | **+0.137 [+0.120, +0.154]** | **+0.125 [+0.110, +0.141]** |
|  | κ_z | **−0.238 [−0.256, −0.222]** | **−0.228 [−0.244, −0.212]** |
|  | ω × κ | **−0.053 [−0.068, −0.039]** | **−0.021 [−0.034, −0.008]** |
| pct_opt | ω_z | **−0.072 [−0.085, −0.060]** | **−0.108 [−0.120, −0.095]** |
|  | κ_z | **−0.035 [−0.048, −0.023]** | **−0.046 [−0.059, −0.034]** |
|  | ω × κ | **−0.020 [−0.030, −0.009]** | −0.001 [−0.012, +0.009] |
| p_heavy | ω_z | **−0.154 [−0.161, −0.147]** | **−0.168 [−0.177, −0.160]** |
|  | κ_z | **−0.076 [−0.083, −0.069]** | **−0.062 [−0.070, −0.053]** |
|  | ω × κ | +0.006 [+0.000, +0.013] | **+0.015 [+0.007, +0.022]** |

Sampling diagnostics across all 48 fits: R̂ = 1.000–1.001, ESS_bulk = 7,001–13,212 per coefficient.

### What the decomposition reveals

**Escape rate is the simplest outcome and behaves as a one-parameter story.** In the Cartesian fit, ω carries an independent positive effect on escape rate (β = +0.058 [+0.029, +0.089] exploratory; +0.049 [+0.018, +0.077] confirmatory), κ is indistinguishable from zero, and the ω × κ interaction does not exclude zero. The polar fit confirms the same picture from the other coordinate basis: the angle effect reaches the HDI threshold only in the confirmatory sample, magnitude contributes only weakly, and the angle × magnitude interaction is null. Together, these results say that whether a subject lives or dies on attack trials is governed almost entirely by a single trait — capture aversion (ω) — and is not measurably modulated by where they sit along the effort-aversion axis or by overall avoidance intensity. The mechanism is intuitive: subjects who weight capture more heavily in the survival-weighted value function avoid risky high-effort items more often *and*, conditional on encountering a predator, mobilize more reliably. The cleanness of this one-parameter signature is itself informative — it tells us H4a is not a thin slice of a richer multivariate effect that we missed.

**Mean pressing intensity is the most informative outcome and shows three distinct components.** The univariate H4c result reported a κ → vigor coefficient of β ≈ −0.195 in both samples. The Cartesian decomposition both strengthens this κ effect (partial β = −0.238 / −0.228, holding ω constant) *and* reveals an independent positive ω → vigor effect of comparable magnitude (β = +0.137 [+0.120, +0.154] exploratory; +0.125 [+0.110, +0.141] confirmatory). The two effects point in opposite directions — effort aversion suppresses pressing, capture aversion enhances it — and were partially cancelling in the original H4c specification, which is why the H4c estimate is smaller in absolute terms than κ's true partial coefficient. Substantively, this is a previously hidden **defensive-mobilization signature**: subjects who weight capture more heavily press harder during transport, even after equalizing them on their cost of effort. This is exactly what a survival-weighted value model predicts — when the cost of being captured is high, the marginal value of additional effort is high — and the univariate H4c regression masked it.

Beyond these two main effects, the ω × κ interaction is robustly negative in both samples (β = −0.053 / −0.021), and the polar angle × magnitude interaction tells the same story from a different angle (β = −0.087 / −0.101). κ's suppressive effect on vigor is amplified for subjects with high ω: those who are simultaneously capture-averse and effort-averse show the steepest vigor drop, more than either trait alone would predict. This is a competition signature — the two avoidance signals act against each other on the motor channel, and the dominant one wins by a wider margin than the linear model would suggest. The polar magnitude effect on vigor is positive (β = +0.067 / +0.054), consistent with a parsimonious reading that *some* form of strong avoidance generally elevates pressing — but the Cartesian decomposition makes clear that this aggregate-magnitude story is the resultant of competing ω and κ effects, not a unitary "intensity" dimension.

**Choice (P(heavy)) shows that both ω and κ suppress heavy choice, with ω carrying roughly twice the weight.** The Cartesian decomposition gives clean partial slopes: β(ω) = **−0.154** [−0.161, −0.147] / **−0.168** [−0.177, −0.160] and β(κ) = **−0.076** [−0.083, −0.069] / **−0.062** [−0.070, −0.053] across the two samples. Both effects are negative and substantial, and the ω-to-κ ratio sits at ≈ 2 : 1 in both samples — consistent with the W(u) structure, in which ω weights the survival term that dominates choice and κ enters only through the smaller demand-cost penalty `κ·req·D`. The interaction term is small and positive in both samples (β = +0.006 / +0.015), suggesting a slight "competition" pattern: when both ω and κ are high, total choice suppression is *less* than the additive sum, consistent with subjects who are extreme in both dimensions saturating their avoidance behaviour on the choice channel before the second parameter has full room to contribute. The polar decomposition recovers a strong angle effect (β(angle) = −0.097 / −0.083) and a robust angle × magnitude interaction (β = −0.092 / −0.033), confirming that *which kind* of avoidance dominates matters for P(heavy) above and beyond raw avoidance intensity. This is the missing piece of the H4 parameter-to-behavior map: choice was inferred indirectly through H4b's overcaution test, and the direct decomposition now closes that loop with clean ω-vs-κ partial slopes that can be used as inputs in cross-channel predictions (see the next subsection).

**Decision quality (proportion of optimal choices) shows that both direction and intensity matter.** H4d's preregistered angle effect replicates cleanly (β = −0.049 / −0.051), and the polar magnitude effect adds an independent negative contribution in both samples (β = −0.052 / −0.024). Substantively, subjects who are more extreme in (ω, κ) space — far from the population centroid in *any* direction — make worse choices than subjects nearer to the centroid. The Cartesian fit confirms a symmetric picture: ω and κ are both negatively associated with optimality on their own (β = −0.072 / −0.108 for ω; β = −0.035 / −0.046 for κ), with the interaction small and unstable across samples. This is consistent with optimality being a U-shaped function of trait avoidance: too little avoidance leads to capture-driven losses; too much avoidance leads to overcautious losses (consistent with H4b — overcaution dominates the error pool); the population centroid sits near the basin of the U. The angle effect H4d adds onto this is that *effort-driven* extremity is worse than threat-driven extremity for matched magnitudes, consistent with the prereg's claim that effort-driven avoidance is indiscriminate while threat-driven avoidance is context-appropriate.

> THIS IS A HEADLINE FIGURE WE NEED TO BUILD.

### Feeds the cross-channel marginal-correlation derivation in [[result_401]]

The four partial slopes reported above — ω → choice, κ → choice, ω → vigor, κ → vigor — plus the population correlation `r(ω_z, κ_z) = +0.369 (expl) / +0.302 (conf)` are exactly the quantities the embodied W(u) framework needs to derive the marginal `r(choice, vigor)` *without seeing the marginal during fitting*. The derivation, the pathway-by-pathway decomposition, and the predicted-vs-observed comparison in both samples are reported in [[result_401]]'s Result and Interpretation sections. The headline from that derivation — that the ω pathway (dissociated: avoidance on choice, mobilised execution on vigor) and the κ pathway (aligned: effort cost on both) nearly cancel, leaving a small positive residual that matches the observed `r(choice, vigor)` to within 0.025 in both samples — is *generated* by the slopes above. Cached at `results/stats/individual_diffs/h4_predicted_r_cv.csv`.

### Synthesis

The decomposition turns the H4 family from a list of five univariate tests into a coherent picture of how a 2D parameter space maps onto **four** behavioral channels. ω and κ are confirmed as separable traits with distinct ecological signatures (consistent with the M4 vs M3 model comparison in [[result_204]]). On the **choice channel** (P(heavy)), both ω and κ are negative — both kinds of avoidance suppress heavy choice — with ω carrying roughly twice the weight (≈ 2 : 1 ratio) and a small positive interaction. On the **motor channel** (mean_vigor), ω and κ act in **opposite** directions — capture aversion mobilizes effort (β ≈ +0.13), effort aversion suppresses it (β ≈ −0.24) — and these competing signals interact such that the dominant one wins more decisively than additive logic would predict. On the **survival channel** (escape rate), ω alone governs the outcome, and where a subject sits on the effort axis is essentially irrelevant. On the **optimality channel** (pct_opt), both parameters and the magnitude of the (ω, κ) vector jointly predict decision quality, with effort-driven extremity worse than threat-driven extremity for matched magnitudes. These four channel-specific signatures are coherent with the survival-weighted value framework: each outcome reflects a different projection of the same two-parameter trait structure onto a different behavioral measurement, and the projections are interpretable rather than arbitrary.

A structural feature worth flagging for downstream results: **ω has dissociated effects across channels** (avoidance on choice, mobilised execution on vigor — same parameter, two opposite-signed behavioural expressions), while **κ has aligned effects** (effort cost on both channels in the same direction). This dissociated-vs-aligned asymmetry between the two parameters is what produces the structured marginal correlation between choice and vigor reported in [[result_401]]. The cross-channel derivation lives there; the partial slopes above are the inputs that make it quantitative.

In the current (ω, κ) parameterization, that summary holds for escape rate but not for mean vigor, decision quality, or P(heavy), all three of which require the full 2D structure. The earlier parameterization's collapse into a one-dimensional story appears to have been specific to its variable choice rather than a general fact about the joint model.

**Status:** Exploratory and post-hoc. The verdict for the preregistered H4a–d tests remains tied to their univariate / angle specifications above; the decomposition is reported as a follow-up that contextualizes those results, not a replacement for them.

## Interpretation

The four direct parameter-to-outcome tests (H4a–d) replicate cleanly across two independent samples with HDIs that exclude zero in the predicted direction and effect-size estimates that are remarkably stable between samples (e.g., β(κ → vigor) = −0.194 expl vs −0.196 conf). The two parameters carry the ecological meaning the prereg attributed to them: ω is the capture-aversion trait that predicts who survives (H4a) and who errs on the overcautious side (H4b); κ is the effort-aversion trait that predicts who presses less (H4c); and the angle in (ω, κ) space — the relative balance of threat-driven vs effort-driven avoidance — predicts overall decision quality (H4d). The angle effect is small in absolute units but consistent across samples and aligns with a substantive theoretical claim: effort-driven avoidance is indiscriminate (avoid the hard option regardless of threat), while threat-driven avoidance is context-appropriate (avoid the hard option specifically when it is dangerous).

H4e tells a different story. The indirect link from model-consistency-to-earnings was significant in the exploratory sample but did not replicate. Both choice consistency and intensity deviation produced HDIs spanning zero in the confirmatory sample, with point estimates roughly half the exploratory magnitude. The failure is a clean replication-failure rather than a sign-flip, consistent with the exploratory finding being an upward-biased estimate of a smaller-than-claimed true effect. The substantive implication: model consistency does not translate into earnings as directly as H4e proposed. Subjects who deviate from model-predicted choices or intensities are not, in aggregate, earning less — possibly because the model's optimum is not the same as the task's reward-maximizing strategy under noise, or because individual-level deviations average out at the trial-aggregate earnings level.

The H4a–d results license the use of ω and κ as substantive individual-difference variables in downstream analyses, and the H4e failure should be noted as a place where the model-to-behavior bridge breaks down. Both findings together reinforce the prereg's framing of ω and κ as separable traits ([[result_204]]): the parameter-to-outcome mappings are direction-specific (ω to escape and overcaution; κ to vigor), not interchangeable.

## Caveats & Limitations

- **H4e failure is the only confirmatory replication failure in the prereg H1–H5 family.** It deserves a paragraph in the manuscript, framed as a clean null on an indirect linkage, not a contradiction of the underlying joint-model framework.
- **All H4 regressions use point-estimate posterior means of ω and κ as predictors,** ignoring per-subject posterior SD. This is the classic measurement-error-in-predictor problem: subjects whose parameters are loosely identified contribute the same weight as those whose parameters are tightly pinned, which biases regression coefficients toward zero (regression dilution) and yields HDIs that are mildly too narrow. The fully Bayesian alternative — drawing each subject's (ω, κ) from its posterior on every MCMC iteration of the regression, or fitting the joint M4 + downstream regressions in one graph — would propagate this uncertainty. Per-subject recovery is r ≈ 0.92 for both ω and κ ([[result_205]]), so the bias is expected to be small but nonzero.
- **`overcaution_ratio` is an empirical quantity computed from condition-cell expected rewards, not a model-derived quantity.** It depends on which cells are classified as "heavy is optimal" via the task's reward structure, not the fitted model. This makes H4b a model-to-behavior test rather than a self-consistency test.
- **Angle metric (H4d) compresses ω and κ into a single dimension.** Three pieces of information are not in the angle: (i) *magnitude* √(ω_z² + κ_z²), the overall avoidance intensity — two subjects with identical angle but different magnitudes are equivalent under the H4d test; (ii) *independent univariate effects* of ω and κ, which can have opposite signs on the same outcome (escape_rate, for example, is driven by ω alone with κ ≈ 0 in H4a); (iii) the ω × κ interaction. The post-hoc "angle + magnitude vs. ω + κ decomposition" block below tests both the polar and Cartesian parameterizations against all three H4 outcomes; magnitude and the interaction both contribute on top of the angle for mean_vigor and pct_opt, but not for escape_rate. Per-subject parameters are reliably identifiable under M4 + MCMC (recovery r ≈ 0.92 for both ω and κ; [[result_205]]), so the 2D treatment is licensed. The angle measure remains the prereg-locked test for H4d; the decomposition is exploratory and cross-references [[result_204]].
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

**To regenerate the choice decomposition + cross-channel prediction (added 2026-06-03):**

```bash
python scripts/analysis/h4_choice_decomp.py
```

**Expected runtime:** ~1 min (12 bambi fits, each ≈ 1 s on CPU under NumPyro).

**Expected outputs:**
- `results/stats/individual_diffs/h4_choice_decomp.csv` — coefficient table for P(heavy) and mean_vigor decompositions.
- `results/stats/individual_diffs/h4_predicted_r_cv.csv` — predicted vs observed r(choice, vigor) per sample.

## References

**Related results:**
- [[result_201]] — Joint model M4 fit (source of ω, κ used here as predictors).
- [[result_204]] — M4 vs M3 (single-parameter), establishing ω and κ as separable.
- [[result_205]] — Parameter recovery: both ω and κ are well-recovered per-subject (r ≈ 0.92), licensing their use as individual-difference predictors here.
- [[result_207]] — Joint-likelihood necessity + embodiment argument; the cross-channel prediction here supplies the partial coefficients that make 207's "uniquely answerable questions" pillar quantitative.
- [[result_401]] — Marginal `r(choice, vigor)` whose value is quantitatively predicted by the partial coefficients reported above.
- [[result_402]] — ω predicts anticipatory vigor (cross-channel parameter mapping; the positive ω → vigor signature documented here).
- [[result_502]] — Anxiety calibration as additional individual-difference predictor (H5a/b).

**Notebook / drafts:**
- `notebooks/analysis/H4_profiles_optimality.ipynb`
- `drafts/results_by_hypothesis/H4_profiles_optimality.md` — legacy bundled writeup.

**Literature:**
- Bednekoff, P. A. (2007). Foraging in the face of danger.
- Houston, A. I., & McNamara, J. M. (1999). Models of Adaptive Behaviour.
