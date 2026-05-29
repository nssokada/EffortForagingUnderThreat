---
result_id: 201
class: computational_model
title: Joint fitness model M4 jointly fits choice and vigor with two per-subject parameters
status: supported
prereg_h: [H3]
internal_h: [H1]
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/00_modeling.ipynb, notebooks/analysis/H3_model_comparison.ipynb, scripts/run_mcmc_pipeline.py]
scripts: [scripts/run_mcmc_pipeline.py]
outputs: [results/stats/joint_optimal/exploratory/mcmc_model_comparison.csv, results/stats/joint_optimal/confirmatory/mcmc_model_comparison.csv, results/stats/joint_optimal/exploratory/mcmc_convergence_diagnostics.csv, results/stats/joint_optimal/confirmatory/mcmc_convergence_diagnostics.csv, results/stats/joint_optimal/mcmc_m4_params.csv]
figures: [results/figs/h3/param_estimates_vertical_exploratory.pdf, results/figs/h3/ppc_subject_vigor_exploratory.pdf]
created: 2026-05-27
last_run: 2026-05-27
---

# Result 201 — Joint fitness model M4 jointly fits choice and vigor with two per-subject parameters

## Overview

The preregistered theoretical model is a survival-weighted fitness function W(u) that takes a pressing rate u and returns a scalar combining the rewarded value of safe return, the cost of capture, and a quadratic deviation cost on motor effort. We fit a hierarchical Bayesian implementation in which each subject has two free parameters — ω (avoidance sensitivity, the subjective cost of capture) and κ (activation intensity, the subjective cost of effort) — that together govern both the choice between heavy and light cookies and the optimal pressing rate during execution. The model achieves choice accuracy ≈ 76% and vigor R² ≈ 0.37–0.41 in two independent samples, with full convergence on all population and subject-level parameters. This result documents the fit and convergence of M4; results 202–204 compare it against simpler alternatives.

## Hypothesis

**Statement.** "A joint fitness model with two per-subject parameters (subjective cost of capture, subjective cost of effort) will outperform simpler alternatives." (Preregistration, H3 umbrella.)

**Predicted direction.** WAIC and LOO should both favor M4 over each alternative.

**Preregistered criterion.** WAIC and LOO must agree (ΔWAIC > 0 AND ΔLOO > 0) on each pairwise comparison.

**Source of the hypothesis.** Preregistration §H3 (`write-up/preregistration.md`, lines 232–326). Motivated by optimal foraging theory (Bednekoff 2007; Brown 1999): if survival probability and effort cost combine into a single subjective value signal that governs both decision and execution, the same parameter pair should fit choice and vigor data simultaneously without separate equations.

## Data Source

- **Samples:** Exploratory N = 290, confirmatory N = 281 (same exclusions as result_101).
- **Input file (both samples):** `behavior_rich.csv` for choice trials and trial-level pressing rates. Per-subject condition-cell means computed for the vigor likelihood.
- **Inclusion / exclusion applied for this result:** Project-default participant exclusions. Choice trials enter the choice likelihood; condition-cell means (subject × threat × distance × cookie) enter the vigor likelihood.
- **Unit of analysis:** Choice likelihood is at the trial level; vigor likelihood is at the cell-mean level (≈18 cells per subject).
- **N entering the model:** ~13,050 choice trials × ~5,215 vigor cells (exploratory); ~12,645 × ~5,058 (confirmatory).

## Method

The joint fitness function predicts both choice and pressing rate from the same parameter pair. Choice arises from a softmax comparison of two cookies' optimized fitness values; vigor arises from the rate u* that maximizes the fitness function in the chosen condition.

**Fitness function:**
$$W(u) = S(u) \cdot R - (1 - S(u)) \cdot \omega \cdot (R + C) - \kappa \cdot (u - \text{req})^2 \cdot D$$

where S(u, T, D) = exp(−h · T^γ · D / speed(u)), speed(u) is a sigmoid saturating above the required rate, R is the cookie reward, C = 5 is the capture penalty, req is the cookie-specific required rate (0.9 heavy, 0.4 light), and D is distance.

**Choice prediction:** For each cookie j, V_j = max_u W_j(u) − κ · req_j · D_j (optimized fitness minus the total demand cost). Then P(heavy) = σ((V_H − V_L) / τ), where τ is a population noise parameter.

**Vigor prediction:** For the chosen cookie, u* = argmax_u W(u). Observed cell-mean rates are modeled as Normal(u* + b_cookie · is_heavy, σ_v / √n_trials), with the √n_trials weighting reflecting the per-cell precision.

**Per-subject parameters** (hierarchical, non-centered):
- ω_i = exp(m_ω + s_ω · z_i), where m_ω ~ N(0, 1), s_ω ~ HalfN(1.0), z_i ~ N(0, 1)
- κ_i = exp(m_κ + s_κ · z_i), where m_κ ~ N(−1, 1), s_κ ~ HalfN(0.5), z_i ~ N(0, 1)

**Population parameters** (weakly informative log-scale priors): γ, h, σ_sp, τ, σ_v, b_cookie.

**Inference:** NumPyro NUTS, 4 chains × 2,000 warmup + 4,000 samples (16,000 post-warmup draws total), target_accept = 0.95, max_tree_depth = 10. Convergence requirement: R̂ < 1.01 and bulk ESS > 400 for all parameters.

**Software:** NumPyro / JAX, ArviZ for WAIC / PSIS-LOO. Environment: `effort_foraging_threat` (Python 3.11).

**Notebook / script:** The MCMC is run by `scripts/run_mcmc_pipeline.py` (designed for GPU execution) and the fitted output is read by `notebooks/analysis/H3_model_comparison.ipynb`. Per-subject parameters are saved to `results/stats/joint_optimal/{sample}/mcmc_m4_params.csv` and population diagnostics to `mcmc_convergence_diagnostics.csv`.

## Result

M4 converged on all parameters in both samples and explains choice and vigor jointly with two per-subject parameters.

**Fit quality (M4):**

| Metric | Exploratory (N=290) | Confirmatory (N=281) |
|---|---|---|
| Total free parameters | 590 (2 per subject × 290 + ~10 population) | 572 (2 per subject × 281 + ~10 population) |
| WAIC (joint, lower = better) | 12,776 | 12,252 |
| p_WAIC (effective parameters) | 467 | 415 |
| LOO | 12,779 | 12,263 |
| Choice accuracy | 77.3% | 75.9% |
| Choice R² | 0.796 | 0.809 |
| Vigor R² | 0.372 | 0.412 |
| Pareto-k > 0.7 (% bad LOO) | 6.8% | 8.3% |
| MCMC wall time | ~35 min | ~33 min |
| Converged | True | True |

**Convergence (M4 selected population parameters):**

| Parameter | Exploratory R̂ (ESS) | Confirmatory R̂ (ESS) |
|---|---|---|
| γ (hazard exponent) | 1.000 (16,000) | 1.000 (16,000) |
| hazard h | 1.000 (16,000) | 1.000 (16,000) |
| σ_v (vigor noise) | 1.000 (16,000) | 1.000 (16,000) |
| τ (population noise) | < 1.001 | < 1.001 |
| m_ω, m_κ (group means) | < 1.001 (4,500–6,300) | < 1.001 (5,400–6,400) |
| s_ω, s_κ (group SDs) | < 1.001 (5,600–6,000) | < 1.001 (5,400–5,800) |
| ω_i (max across subjects) | 1.002 (2,506) | 1.000 (8,975) |
| κ_i (max across subjects) | 1.001 (3,531) | 1.000 (16,000) |

All R̂ < 1.01 in both samples, all ESS > 400. M4 meets the preregistered convergence criterion in full.

**Population-level posterior means (exploratory / confirmatory):**

| Parameter | Exploratory | Confirmatory |
|---|---|---|
| γ (hazard exponent on T) | 0.847 | 0.827 |
| h (hazard scale) | 0.551 | 0.382 |
| m_ω (log-mean of ω) | 0.078 | 0.552 |
| s_ω (log-SD of ω) | 0.928 | 0.959 |
| m_κ (log-mean of κ) | −1.549 | −1.587 |
| s_κ (log-SD of κ) | 1.429 | 1.369 |
| b_cookie (vigor heavy − light intercept) | −0.206 | −0.202 |
| σ_v (vigor noise) | 0.454 | 0.384 |
| τ (choice noise) | 0.667 | 0.926 |

**Per-subject parameter ranges:** ω_i typically ∈ [0.3, 30] with subject-mean ≈ 2 (right-skewed); κ_i typically ∈ [0.05, 5] with subject-mean ≈ 0.5. Distributions consistent across samples.

**Figures (paper-grade):**

- `results/figs/h3/param_estimates_vertical_exploratory.pdf` — per-subject posterior means and 89% HDIs for ω and κ.
- `results/figs/h3/ppc_subject_vigor_exploratory.pdf` — posterior predictive checks: model-predicted vs observed cell-mean rates.

**Verdict on prereg criterion:** M4 fit successfully with full convergence in both samples. The fit is the precondition for the model-comparison tests in [[result_202]], [[result_203]], and [[result_204]].

## Interpretation

The joint fitness model M4 fits both the choice surface and the within-subject vigor cell-means with two free parameters per subject, achieving choice accuracy near 76% and explaining roughly 80% of choice-variance and 37–41% of vigor-variance across two independent samples. Convergence is clean: every population-level parameter and every subject-level parameter has R̂ < 1.01 and ESS in the thousands, satisfying the preregistered convergence criterion in full. The model has 572–590 free parameters fit jointly to ≈18,000 likelihood contributions per sample (choice trials + condition cells), which is a non-trivial inferential burden — the convergence and the comparable fit quality across two independent samples together indicate that the joint constraint does not break the inferential pipeline.

Two structural features of the fit are worth flagging. First, ω and κ are estimated on log scales with population log-means near 0 (ω) and −1.5 (κ), with substantial subject-level SDs (s_ω ≈ 0.93–0.96, s_κ ≈ 1.37–1.43). On the natural scale, this places ω near 1 with a heavy right tail and κ near 0.2 with a similar right tail — distributions that are consistent with the prereg's expectation that capture aversion and effort sensitivity vary substantially across people. Second, the posterior predictive vigor R² (0.37–0.41) is markedly lower than the choice R² (≈0.80), reflecting the relative noise levels of the two likelihoods: choice is a 1-bit decision per trial with strong condition structure, while vigor is a continuous cell-mean with motor noise that the model does not model trial-by-trial.

The fit on its own does not establish that M4 is the *right* model — only that it can be fit reliably with full convergence. The substantive comparison against simpler alternatives (effort-only, threat-only, single-parameter) is the question taken up in [[result_202]], [[result_203]], and [[result_204]], all of which use the same MCMC machinery and the same likelihood components, ensuring fair comparison.

## Caveats & Limitations

- **Vigor R² of 0.37–0.41 is moderate.** The remaining variance is partly motor noise (which is structural, not signal) and partly model misspecification of the speed-saturation function or the quadratic-deviation cost. PPC plots show no systematic miscalibration but do show heavy tails. Result 205 (parameter recovery) constrains how much of the residual is identifiable signal vs noise.
- **Pareto-k diagnostics flag 6.8–8.3% of observations as influential (k > 0.7).** This is within tolerance for PSIS-LOO but means LOO is approximate at the upper edge. Both WAIC and LOO are reported in result 202–204 to triangulate.
- **τ (population choice noise) differs substantially between samples** (0.67 expl vs 0.93 conf), suggesting that the confirmatory sample exhibits slightly noisier choice. This does not invalidate the joint fit but is a quantity worth understanding before pooling samples for downstream regressions.
- **The fitness function fixes the survival functional form to S(u, T, D) = exp(−h·T^γ·D/speed(u)).** Alternative survival kernels (hyperbolic, Weibull) were explored in earlier model selection and are documented in `instructions/memory/joint_model_development.md`; the exponential form is the prereg's pre-committed choice.
- **MCMC runtime is non-trivial (~35 min per sample on CPU).** GPU execution via `scripts/run_mcmc_pipeline.py` is the canonical pipeline. Reproducing the fit from scratch is therefore not as cheap as the H1 family.

## Replication

**To regenerate this result from scratch:**

```bash
# Fit the full MCMC pipeline (CPU ~35 min/sample, GPU faster)
python scripts/run_mcmc_pipeline.py --sample exploratory
python scripts/run_mcmc_pipeline.py --sample confirmatory

# Or re-execute the model-comparison notebook (reads cached fits and re-computes WAIC/LOO):
PYTHONPATH=notebooks/analysis \
  jupyter nbconvert --to notebook --execute \
  notebooks/analysis/H3_model_comparison.ipynb \
  --inplace --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=3600
```

**Expected runtime:** ~35 min per sample for MCMC fit; ~2 min for WAIC/LOO recomputation from cached posteriors.

**Expected outputs:**
- `results/stats/joint_optimal/{exploratory,confirmatory}/mcmc_m4_params.csv` — per-subject ω, κ posterior means.
- `results/stats/joint_optimal/{exploratory,confirmatory}/mcmc_convergence_diagnostics.csv` — R̂ and ESS for all parameters.
- `results/stats/joint_optimal/{exploratory,confirmatory}/mcmc_model_comparison.csv` — WAIC, LOO, choice/vigor fit metrics for M1–M4 and M3b.
- `results/figs/h3/` — population parameter plots and PPCs.

## References

**Related results:**
- [[result_202]] — M4 vs M1 (effort-only) model comparison.
- [[result_203]] — M4 vs M2 (threat-only) model comparison.
- [[result_204]] — M4 vs M3 (single-parameter θ = ω = κ) model comparison.
- [[result_205]] — Parameter recovery: how well ω and κ are recovered from synthetic data.
- [[result_208]] — H4 family Bayesian regressions on ω and κ as individual-difference predictors.
- [[result_501]] — Model-derived survival probability predicts affect (internal H4).

**Notebooks / scripts:**
- `scripts/run_mcmc_pipeline.py` — canonical MCMC pipeline.
- `notebooks/analysis/00_modeling.ipynb` — model development and sanity checks.
- `notebooks/analysis/H3_model_comparison.ipynb` — model comparison and PPCs.
- `instructions/memory/joint_model_development.md` — full development history and alternative model variants.

**Literature:**
- Bednekoff, P. A. (2007). Foraging in the face of danger.
- Brown, J. S. (1999). Vigilance, patch use and habitat selection: foraging under predation risk.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.
