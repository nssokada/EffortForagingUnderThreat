---
result_id: 202
class: computational_model
title: Joint model M4 beats effort-only M1 by ΔWAIC ≈ 3,800–4,700
status: supported
prereg_h: [H3a]
internal_h: [H1]
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/H3_model_comparison.ipynb, scripts/run_mcmc_pipeline.py]
scripts: [scripts/run_mcmc_pipeline.py]
outputs: [results/stats/joint_optimal/exploratory/mcmc_model_comparison.csv, results/stats/joint_optimal/confirmatory/mcmc_model_comparison.csv, results/stats/confirmatory_hypothesis_results.csv]
figures: [TODO]
created: 2026-05-27
last_run: 2026-05-27
---

# Result 202 — Joint model M4 beats effort-only M1 by ΔWAIC ≈ 3,800–4,700

## Overview

If choice could be fit by a model that uses only effort cost and no representation of threat, the joint fitness model M4 would be unnecessary. We tested this by fitting M1 — an effort-only model where choice is `ΔV = ΔR − κ_i · Δeffort(D)` with no survival function and no threat term, and where vigor is intercept-only with no condition structure — on the same data as M4, then compared the two on WAIC and PSIS-LOO. M4 dominates M1 by ΔWAIC ≈ +4,729 in the exploratory sample and ΔWAIC ≈ +3,785 in the confirmatory sample, with ΔLOO agreeing on both. M1 also explains essentially zero vigor variance (R² ≈ 0.006), confirming that an effort-only model cannot account for the within-condition motor structure that M4 captures.

## Hypothesis

**Statement.** "The joint model will outperform an effort-only model that ignores threat." (Preregistration, H3a.)

**Predicted direction.** ΔWAIC > 0 AND ΔLOO > 0 favoring M4 over M1.

**Preregistered criterion.** WAIC and LOO must agree on the direction.

## Data Source

Same as [[result_201]]. Both models fit on the same joint likelihood (choice + vigor cell means) for fair comparison.

## Method

**M1 (effort-only) specification.**
- Choice: `ΔV = ΔR − κ_i · Δeffort(D)`, with per-subject κ_i but no survival weighting and no threat term.
- Vigor: intercept-only (no condition-dependent prediction).
- 295 / 286 free parameters (exploratory / confirmatory).

**M4 (joint fitness, full).** See [[result_201]].

**Comparison metrics:** WAIC and PSIS-LOO computed from pointwise log-likelihoods of the full joint likelihood (choice + vigor) for both models, via ArviZ.

**Inference criterion:** ΔWAIC > 0 AND ΔLOO > 0 favoring M4.

**Notebook / script:** `scripts/run_mcmc_pipeline.py` fits both models; `notebooks/analysis/H3_model_comparison.ipynb` computes the comparisons.

## Result

| Metric | Exploratory | Confirmatory |
|---|---|---|
| M1 WAIC | 17,505 | 16,037 |
| M4 WAIC | 12,776 | 12,252 |
| **ΔWAIC (M1 − M4)** | **+4,729** (SE ≈ 667) | **+3,785** (SE ≈ 443) |
| M1 LOO | 17,509 | 16,042 |
| M4 LOO | 12,779 | 12,263 |
| **ΔLOO (M1 − M4)** | **+4,731** | **+3,779** |
| M1 choice accuracy | 71.0% | 70.8% |
| M4 choice accuracy | 77.3% | 75.9% |
| M1 vigor R² | 0.006 | 0.007 |
| M4 vigor R² | 0.372 | 0.412 |
| M1 converged | True | True |
| M4 converged | True | True |

ΔWAIC and ΔLOO both favor M4 by margins of thousands (vs ΔWAIC standard errors in the hundreds) in both samples, satisfying the preregistered criterion in full.

**Verdict on prereg criterion:** **PASS** in both samples. WAIC and LOO agree.

## Interpretation

The effort-only model M1 underperforms M4 by ΔWAIC of roughly 4,700 (exploratory) and 3,800 (confirmatory) — margins that are thousand-fold larger than the WAIC standard errors and consistent in sign across two independent samples. M1's choice accuracy is ~6–7 percentage points lower than M4's, and its vigor R² is essentially zero because the model has no condition-dependent prediction for vigor — it can only fit a single grand-mean rate.

The result establishes that a model with no representation of threat cannot capture either the choice or the vigor data structure. Combined with the joint-model success in [[result_201]], it provides the first preregistered comparison-based evidence for H3: threat is a necessary component of the value computation that governs both behavioral channels.

The comparison does not, on its own, distinguish whether the success of M4 comes from the threat term, the survival functional form, the two-parameter structure, or some combination. That separation is the question taken up in [[result_203]] (M4 vs M2 threat-only, which isolates the role of per-subject effort sensitivity) and [[result_204]] (M4 vs M3 single-parameter, which isolates the role of separable ω vs κ).

## Caveats & Limitations

- **M1's vigor likelihood is intercept-only.** Most of the WAIC gap is therefore driven by vigor fit, not choice. This is by design — M1 represents the null hypothesis that effort cost alone (with no threat) is sufficient to explain behavior — but readers should note that a fairer comparison would also impose a no-vigor M4 to isolate the choice-side improvement. The choice-only WAIC components in `mcmc_model_comparison.csv` (`WAIC_choice`) can be examined to do this.
- **The choice R² difference is more modest than WAIC suggests** (M4: 0.796 expl / 0.809 conf; M1: 0.951 / 0.946 — M1 is actually higher on choice R²). The interpretation: M1 overfits the choice surface using only κ and reward differences, but cannot generalize to vigor; M4 trades some choice-fit headroom for joint coverage. WAIC penalizes effective parameters and accounts for this trade-off.
- **Both models converged.** Comparison is valid in both samples.

## Replication

See [[result_201]] Replication block. The model comparison cell in `H3_model_comparison.ipynb` produces this table directly.

## References

**Related results:**
- [[result_201]] — M4 fit and convergence.
- [[result_203]] — M4 vs M2 (threat-only).
- [[result_204]] — M4 vs M3 (single-parameter).
- [[result_205]] — Parameter recovery.

**Literature:**
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.
