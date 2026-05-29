---
result_id: 502
class: metacognition
title: Anxiety calibration improves optimality beyond ω and κ; anxiety slope predicts choice shifting
status: supported
prereg_h: [H5a, H5b]
internal_h: []
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/H5_metacognitive.ipynb]
scripts: []
outputs: [results/stats/confirmatory_hypothesis_results.csv, results/stats/affect_analysis/anx_params.csv, results/stats/affect_analysis/metacognitive_accuracy.csv]
figures: [results/figs/h5/h5_panel_e_exploratory.pdf]
created: 2026-05-27
last_run: 2026-05-27
---

# Result 502 — Anxiety calibration improves optimality beyond ω and κ; anxiety slope predicts choice shifting

## Overview

If anxiety is a metacognitive monitor of the survival computation, then two properties of a subject's anxiety profile should carry behavioral information beyond what the joint-model parameters (ω, κ) capture. Calibration — how accurately anxiety tracks threat condition by condition — should index the *quality* of the monitoring signal. Slope — how strongly anxiety responds to a unit change in threat — should index the *reactivity* of the monitoring signal and predict adaptive choice shifting. We tested both via Bayesian regressions on subject-level summaries, using the prereg's LOO-based and HDI-based criteria. Calibration improves model fit for three foraging outcomes (optimality, escape, earnings) in both samples; slope predicts choice shift across threat levels with HDIs excluding zero in both samples.

## Hypothesis

**Statements (verbatim from prereg §H5):**
- **H5a.** Anxiety calibration (how well anxiety tracks threat) will predict foraging optimality beyond the model parameters.
- **H5b.** Anxiety reactivity (slope on threat) will predict adaptive choice shifting across threat levels.

**Preregistered criteria:**
- **H5a:** LOO-CV comparing `pct_optimal ~ ω_z + κ_z` (base) vs `+ calibration_z` (full); calibration improves fit if ΔELPD > 0 with SE excluding zero. Escape rate and earnings tested as supporting outcomes.
- **H5b:** 95% HDI on β(anxiety slope → choice shift) excludes zero in the positive direction.

## Data Source

- **Samples:** Exploratory N = 290, confirmatory N = 281.
- **Input files:**
  - `results/stats/affect_analysis/anx_params.csv` — per-subject anxiety regression on threat/distance/cookie (intercept, b_threat, b_distance, b_cookie, R²).
  - `results/stats/affect_analysis/metacognitive_accuracy.csv` — per-subject correlation between anxiety rating and S_probe (anx_accuracy column), and between confidence and S_probe/EV.
  - `results/stats/joint_optimal/{sample}/mcmc_m4_params.csv` — ω, κ used as baseline predictors.
  - `results/stats/individual_diffs/profiles_{sample}.csv` — subject-level outcomes (pct_opt, escape_rate, earnings, choice_shift).
- **Unit of analysis:** Subject.

## Method

**H5a (calibration → optimality):**

For each of three outcomes (pct_optimal, escape_rate, earnings), fit base and full Bayesian regressions and compare via LOO-CV:

```
base: outcome ~ omega_z + kappa_z
full: outcome ~ omega_z + kappa_z + calibration_z
```

Calibration is operationalized as the per-subject Pearson r between anxiety rating and threat (or, in robustness checks, between anxiety rating and S_probe). Higher r = better calibration = anxiety tracks objective danger more accurately.

**H5b (slope → choice shift):**

Per-subject anxiety slope on threat is taken from `anx_params.b_threat` (the threat coefficient from a per-subject regression of anxiety on threat/distance/cookie). Choice shift is `P(heavy at T = 0.1) − P(heavy at T = 0.9)` per subject. Regression:

```
choice_shift ~ anxiety_slope_z
```

**Posterior sampling:** `bambi`, 4 chains × 2,000 draws + 1,000 tuning.

**Inference criteria:** ΔELPD > 0 with SE excluding zero (H5a); 95% HDI excludes zero (H5b).

**Notebook:** `notebooks/analysis/H5_metacognitive.ipynb`, cells 5 (H5a) and 7 (H5b).

## Result

**H5a — Calibration improves fit for all three outcomes:**

| Outcome | Exploratory ΔELPD | Confirmatory ΔELPD | Pass |
|---|---|---|---|
| pct_optimal | > 0, SE excl 0 | > 0, SE excl 0 | ✓ |
| escape_rate | > 0, SE excl 0 | > 0, SE excl 0 | ✓ |
| earnings | > 0, SE excl 0 | > 0, SE excl 0 | ✓ |
| **Outcomes improved** | **3/3** | **3/3** | **PASS** |

**H5b — Anxiety slope predicts choice shift:**

| Coefficient | Exploratory | Confirmatory |
|---|---|---|
| β(anxiety slope → choice shift) | **+0.123** [HDI excludes 0] | **+0.099** [+0.065, +0.134] |

**PASS** in both samples.

**Verdict on prereg criterion:** **PASS** for both H5a and H5b in both samples.

## Interpretation

The two metacognitive properties of anxiety — calibration (signal quality) and slope (reactivity) — carry behavioral information beyond what the joint-model parameters (ω, κ) contain. Calibration improves LOO predictive fit for *all three* of the prereg's supporting outcomes (optimality, escape, earnings) in *both* samples, satisfying the prereg's most stringent variant of the test. This is a non-trivial finding: ω and κ together describe each subject's full position in the joint-model parameter space, so any additional predictor must add information not already encoded by the joint computation. Calibration does this, suggesting that the metacognitive monitor reports something about how accurately the subject's internal model tracks task structure — independent of the first-order parameter values themselves.

Slope adds a complementary individual-difference signal. Subjects whose anxiety responds more strongly to threat shift their choices more across threat levels: a +0.123 (exploratory) / +0.099 (confirmatory) standardized regression coefficient with HDIs cleanly excluding zero. The substantive reading is that anxiety reactivity drives the adaptive component of the avoidance channel — it determines *how much* a subject adjusts their strategy as the environment changes, not just whether they have an absolute level of caution. Calibration is about accuracy; slope is about responsiveness; the two are approximately orthogonal in the data (calibration is a correlation, slope is a regression coefficient; one can be high with the other low).

Together with the appraisal dissociation in [[result_503]] and the error-type result in [[result_504]], the H5 family establishes a three-channel architecture of metacognitive monitoring: confidence (level / coping appraisal), anxiety calibration (accuracy), and anxiety slope (reactivity). Each channel predicts a different aspect of foraging performance, and none of the three is reducible to ω or κ alone. The computation governs *what* the subject does; the metacognitive layer governs *how wisely* and *how adaptively* they do it.

## Caveats & Limitations

- **Probe count per subject is small (~18 anxiety probes).** Per-subject calibration and slope estimates inherit substantial sampling error, as the prereg already acknowledges. The H5a LOO improvement and H5b slope effect both survive this noise in both samples, but split-half reliability of these indices should be reported as supplementary (see prereg Other Planned Analyses §4).
- **Calibration is defined as Pearson r between anxiety rating and threat.** Alternative operationalizations — anxiety vs S_probe (model-derived survival), anxiety vs the joint (T, D) condition — would yield slightly different per-subject indices. The prereg specifies the threat-only version.
- **The LOO improvement in H5a is small in ΔELPD magnitude** (single-digit ΔELPD typical, vs hundreds for model comparison in [[result_202]]). This is appropriate — calibration is one subject-level summary statistic, not an alternative full model — but the *direction* and *consistency* across outcomes and samples is the key finding.
- **The anxiety slope index is uncorrected for the threat-distance correlation in choice trials,** which is essentially zero in the balanced design but could matter for downstream uses.
- **Pooled and split-half reliability analyses are deferred to supplementary.** This result reports only the preregistered confirmatory tests.

## Replication

**To regenerate this result from scratch:**

```bash
PYTHONPATH=notebooks/analysis \
  jupyter nbconvert --to notebook --execute \
  notebooks/analysis/H5_metacognitive.ipynb \
  --inplace --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=1800
```

**Expected runtime:** ~10–20 min per sample (LOO over 3 outcome models × 2 versions each, plus the H5b regression).

**Expected outputs:**
- Cell 5 stdout reports ΔELPD and SE for each outcome.
- Cell 7 stdout reports β(slope → choice shift) with HDI.
- `results/figs/h5/` updated.

## References

**Related results:**
- [[result_501]] — Trial-level survival → affect LMM (the input that licenses calibration as a meaningful subject-level index). *Deferred: no current canonical notebook.*
- [[result_503]] — ω → confidence vs anxiety (H5c, the appraisal dissociation).
- [[result_504]] — Confidence → error type (H5d).
- [[result_208]] — H4 family parameter regressions (the baseline regressions H5a improves upon).

**Notebook / drafts:**
- `notebooks/analysis/H5_metacognitive.ipynb`
- `drafts/results_by_hypothesis/H5_metacognitive.md`

**Literature:**
- Fleming, S. M., & Daw, N. D. (2017). Self-evaluation of decision-making: a general Bayesian framework for metacognitive computation.
- Lazarus, R. S. (1991). Emotion and adaptation.
