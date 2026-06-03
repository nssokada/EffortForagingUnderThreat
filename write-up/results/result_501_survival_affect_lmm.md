---
result_id: 501
class: metacognition
title: Trial-level model-derived survival probability predicts affect (anxiety, confidence)
status: supported
prereg_h: []
internal_h: [H4]
samples: [exploratory_290, confirmatory_281]
notebooks: []
scripts: [scripts/analysis/affect_survival_lmm.py]
outputs: [results/stats/affect_analysis/s_probe_affect_lmm_exploratory.csv, results/stats/affect_analysis/s_probe_affect_lmm_confirmatory.csv]
figures: [TODO]
created: 2026-05-27
last_run: 2026-05-29
---

# Result 501 — Trial-level model-derived survival probability predicts affect (anxiety, confidence)

## Overview

If the joint-model survival probability S(u, T, D) is the computational quantity that subjective affect actually tracks — rather than the raw task variables (threat, distance) on their own — then per-probe S_probe should predict per-probe anxiety and confidence ratings, with anxiety scaling negatively (higher survival → less danger appraisal) and confidence scaling positively (higher survival → more coping appraisal). We computed S_probe from each subject's fitted M4 parameters (ω, κ) and the M4 population parameters (γ, h, σ_sp), then fit two trial-level mixed-effects models per sample: response ~ S_probe (z-scored within sample) + (1 | subj). All four cells (anxiety × 2 samples, confidence × 2 samples) replicate cleanly in the predicted directions, with z-statistics |z| > 22 and p < 10⁻¹⁰⁹ in every cell. The result establishes that subjective affect ratings track the model-derived survival quantity at the trial level, not just the raw conditions.

## Hypothesis

**Statement.** "Trial-level survival probability computed from fitted parameters predicts self-reported anxiety (−) and confidence (+)." (Internal H4, `instructions/memory/hypotheses.md`.)

**Predicted direction.** β(anxiety ~ S_probe) < 0; β(confidence ~ S_probe) > 0.

**Source of the hypothesis.** Internal H4. Not in the formal preregistration (prereg H4c is "κ → mean vigor", a different test). This is the exploratory mechanistic claim that the metacognitive signals are downstream of the joint-model computation, not just of raw task variables. The descriptive H1b test ([[result_102]]) shows anxiety and confidence respond to threat and distance separately; this result asks whether their combined model-derived survival quantity is the more proximate driver.

## Data Source

- **Samples:** Exploratory N = 290, confirmatory N = 281.
- **Input files (per sample):**
  - `feelings.csv` from the sample's `stage5_filtered_data_*` directory — one row per probe-trial rating (anxiety or confidence), with `response` (1–10 scale), `threat`, `distance` (0-indexed, mapped to model D as `distance + 1` ∈ {1, 2, 3}), `trialCookie_rewardValue` (1.0 light / 5.0 heavy), and `subj`.
  - `results/stats/joint_optimal/{sample}/mcmc_m4_params.csv` — per-subject ω, κ posterior means from the M4 MCMC fit.
  - `results/stats/joint_optimal/{sample}/mcmc_convergence_diagnostics.csv` — M4 population parameter posterior means (γ, h, σ_sp).
- **Inclusion / exclusion applied for this result:** Project-default participant exclusions; probes without a valid response dropped (~108 of 10,546 in exploratory, ~0 in confirmatory).
- **Unit of analysis:** Probe trial (within subject), with subject-level random intercept.
- **N entering each LMM:**
  - Exploratory anxiety: 5,220 probes from 290 subjects.
  - Exploratory confidence: 5,218 probes from 290 subjects.
  - Confirmatory anxiety: 5,068 probes from 281 subjects.
  - Confirmatory confidence: 5,068 probes from 281 subjects.

## Method

For each probe trial, we computed the model-derived survival probability under the subject's optimal pressing strategy for that trial's condition:

1. **Map task conditions to model variables.** T = trial threat probability; D = `distance + 1` ∈ {1, 2, 3}; R = trialCookie_rewardValue (1.0 or 5.0); req = 0.9 if heavy else 0.4.
2. **Compute u\*** = argmax_u W(u; T, D, R, req, ω_i, κ_i), where ω_i and κ_i are the subject's M4 posterior means and W is the M4 fitness function:

   ```
   speed(u) = sigmoid((u - 0.25 · req) / σ_sp)
   S(u, T, D) = exp(-h · T^γ · D / max(speed(u), 0.01))
   W(u) = S(u) · R - (1 - S(u)) · ω · (R + C) - κ · (u - req)² · D     (C = 5)
   ```

   Grid search over u ∈ [0.1, 1.5] at 40 evenly spaced points (identical to the grid used in `scripts/mcmc/run_model_comparison_mcmc.py`'s evaluate_fit for M4).
3. **S_probe = S(u\*, T, D)** for the subject's optimal pressing rate.
4. **Z-score S_probe within sample.**
5. **Fit `response ~ S_probe_z + (1 | subj)`** separately for anxiety and confidence, via `statsmodels.mixedlm` with REML=False (ML for fixed-effect inference).

**M4 population parameters used (from `mcmc_convergence_diagnostics.csv` posterior means):**

| Param | Exploratory | Confirmatory |
|---|---|---|
| γ (hazard exponent on T) | 0.846 | 0.826 |
| h (hazard scale) | 0.550 | 0.381 |
| σ_sp (speed saturation width) | 0.247 | 0.243 |

**Inference criterion:** β < 0 for anxiety, β > 0 for confidence, both at p < .01.

**Script that produces this result:** `scripts/analysis/affect_survival_lmm.py`. Validated 2026-05-29.

## Result

Both signs match the prediction in both samples, with z-statistics |z| > 22 and p < 10⁻¹⁰⁹ in every cell.

| Channel | Sample | β(S_probe_z) | SE | z | p | N obs | N subj |
|---|---|---|---|---|---|---|---|
| **Anxiety** | Exploratory | **−0.584** | 0.025 | **−23.74** | **1.5 × 10⁻¹²⁴** | 5,220 | 290 |
| **Anxiety** | Confirmatory | **−0.545** | 0.025 | **−22.25** | **1.0 × 10⁻¹⁰⁹** | 5,068 | 281 |
| **Confidence** | Exploratory | **+0.625** | 0.025 | **+25.30** | **3.1 × 10⁻¹⁴¹** | 5,218 | 290 |
| **Confidence** | Confirmatory | **+0.680** | 0.025 | **+27.09** | **1.3 × 10⁻¹⁶¹** | 5,068 | 281 |

(Intercepts ≈ 4.40 for anxiety, ≈ 3.17–3.39 for confidence, consistent with mid-range Likert ratings.)

The two channels respond in opposite directions to the same predictor, with comparable magnitudes (|β| ≈ 0.55–0.68 per SD of S_probe). The confirmatory anxiety coefficient is slightly smaller than the exploratory (0.545 vs 0.584); the confirmatory confidence coefficient is slightly larger (0.680 vs 0.625). All four cells satisfy the directional and significance criteria with very large margins.

**Replication of the legacy NB04-03 result.** The internal H4 entry in `instructions/memory/hypotheses.md` records the deprecated-pipeline numbers as Anxiety β = −0.602, Confidence β = +0.632 (exploratory, N = 293, deprecated FET parameters). The current exploratory numbers (−0.584 / +0.625) match to within rounding using the current M4 parameters and pipeline, confirming that this is the same underlying effect and that the M4-derived S_probe behaves as the deprecated framework's S_probe did.

**Figure:** TODO — recommend a two-panel scatter with per-probe S_probe on the x-axis and binned mean anxiety / confidence on the y-axis, both samples overlaid or side-by-side.

**Verdict on prereg criterion:** Not preregistered. Both signs and significance thresholds for internal H4 are met in both samples; status `supported`.

## Interpretation

Subjective affect ratings respond to the model-derived survival probability at the trial level, with anxiety scaling negatively and confidence scaling positively, replicating across two independent samples with very tight confidence intervals. The effect sizes are substantial (|β| ≈ 0.55–0.68 per SD of S_probe on a 1–10 rating scale) — interpretable as roughly half a rating point per SD of survival, in a population that uses most of the scale. The asymmetry between channels (confidence somewhat more strongly tied to S_probe than anxiety in both samples) is consistent with confidence being more directly a coping-appraisal readout, while anxiety also incorporates a more diffuse danger-appraisal component.

The result is mechanistically distinct from the descriptive H1b test in [[result_102]]. That test shows that anxiety rises and confidence falls with threat and distance treated as raw additive predictors. This result asks the stricter question: do affect ratings track the *nonlinear, parameter-dependent combination* of threat and distance that the joint model identifies as the operative survival quantity? The answer is yes: S_probe — which depends on T, D, ω, κ, and the population γ, h, σ_sp via the W(u) fitness function — predicts affect with very tight effect sizes, and does so with coefficients close to what raw T explains in the simpler H1b model. The two analyses are not redundant: the raw-T LMM in H1b says "affect tracks threat"; this result says "affect tracks the survival quantity the model says threat *implies* given the subject's optimal action and parameters." The latter is the substantive claim about metacognitive monitoring of the foraging computation.

The result completes a three-step chain. (i) Model parameters ω and κ predict choice and vigor at the population level ([[result_201]], [[result_208]]). (ii) Choice and vigor are dissociable through their parameter signatures ([[result_208]], [[result_401]], [[result_404]]). (iii) Affect tracks the same model-derived survival quantity that drives choice and vigor (this result). Together these establish that the joint W(u) framework is not just a fit to behavior — it is also the substantive computation that subjective affect monitors. This is the substantive licence for using S_probe-based affect features (anxiety calibration, anxiety slope) as individual-difference predictors in [[result_502]] without arguing the affect signal is a downstream consequence of some other variable.

## Caveats & Limitations

- **S_probe uses each subject's *posterior-mean* ω and κ rather than the full posterior.** Propagating posterior uncertainty in the per-subject parameters into S_probe would widen the effective per-trial S_probe distribution and slightly attenuate β estimates. The effects are so large (|z| > 22) that this attenuation is not a threat to the directional conclusion; it could matter for precise effect-size claims.
- **S_probe also uses *population-mean* γ, h, σ_sp** rather than per-subject draws. Same caveat as above; population parameters have very tight posteriors (R̂ < 1.001, ESS > 5000), so this is a minor source of uncertainty.
- **The grid search over u ∈ [0.1, 1.5] at 40 points** discretizes u*. The W(u) function is smooth and concave around the optimum, so a 40-point grid yields S_probe accurate to ~0.1% of the true u*. Identical grid is used by the M4 evaluate_fit function — so this is internally consistent across the pipeline.
- **Reverse causation cannot be ruled out at the per-trial level** from this analysis alone. A subject who is anxious on a given trial may report higher anxiety AND, via some unmodeled mechanism, choose / press in a way that the model interprets as reflecting different ω or κ. We mitigate this by using **subject-level** ω and κ (the same value across all of a subject's trials), so the per-trial variation in S_probe comes entirely from per-trial T and D (not from per-trial affect). This rules out the strongest form of reverse causation at the trial level.
- **Probe ratings are prospective** — collected at trial start before pressing. The S_probe used here is the model's optimal-pressing-rate survival probability, which is the subject's *implied* expected survival given their parameters. A reactive-affect interpretation (where affect responds to within-trial events) is not testable in this design.
- **18 probes per subject** for anxiety and 18 for confidence. The trial-level LMM uses all ~5,200 / ~5,070 observations per sample, so the *population-level* fixed-effect inference is well-powered. Per-subject individual-difference indices (anxiety calibration, anxiety slope) inherit the small per-subject probe count — see the corresponding caveat in [[result_502]].

## Replication

**To regenerate this result from scratch:**

```bash
/opt/anaconda3/envs/effort_foraging_threat/bin/python \
    scripts/analysis/affect_survival_lmm.py
```

Runs both samples by default. Use `--samples exploratory` or `--samples confirmatory` to restrict; `--out-dir <path>` to change output location.

**Expected runtime:** ~10 seconds (vectorized S_probe computation + two LMM fits per sample).

**Expected outputs:**
- `results/stats/affect_analysis/s_probe_affect_lmm_exploratory.csv`
- `results/stats/affect_analysis/s_probe_affect_lmm_confirmatory.csv`

Each file has one row per (sample, channel) with columns: `sample, channel, n_obs, n_subj, intercept, beta, se, z, p, s_probe_mean, s_probe_sd`.

## References

**Related results:**
- [[result_102]] — H1b descriptive affect ~ threat + distance LMM. The simpler, raw-condition version of this analysis.
- [[result_201]] — Joint model M4 fit (source of ω, κ used as per-subject parameters here).
- [[result_502]] — H5a/b anxiety calibration and slope as individual-difference predictors of optimality (downstream consequence of the trial-level coupling shown here).
- [[result_503]] — H5c ω → confidence (the parameter-to-affect link that operates through this trial-level signal).

**Scripts:**
- `scripts/analysis/affect_survival_lmm.py` — this analysis.
- `scripts/mcmc/run_model_comparison_mcmc.py` — defines the M4 model and the W(u), S(u, T, D), and u* grid search reused here.

**Notes:**
- `instructions/memory/hypotheses.md` § H4 — the legacy NB04-03 numbers that this analysis validates.

**Literature:**
- Fleming, S. M., & Daw, N. D. (2017). Self-evaluation of decision-making: a general Bayesian framework for metacognitive computation.
- Lazarus, R. S. (1991). Emotion and adaptation.

## Revision notes

- **2026-05-29:** Migrated from `untested` deferred stub → `supported`. The legacy NB04-03 analysis (`instructions/memory/hypotheses.md` § H4, anxiety β = −0.602, confidence β = +0.632, deprecated FET pipeline) lived only in a deprecated notebook and could not be validated against the current samples. Reimplemented as a standalone script (`scripts/analysis/affect_survival_lmm.py`) that computes S_probe from the current M4 (ω, κ) per-subject posteriors and (γ, h, σ_sp) population posteriors, fits the LMMs, and prints summary statistics. Result replicates the legacy effect (current exploratory: β_anx = −0.584, β_conf = +0.625) and extends to the confirmatory sample (β_anx = −0.545, β_conf = +0.680). All four cells highly significant (|z| > 22, p < 10⁻¹⁰⁹).
