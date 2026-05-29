---
result_id: 203
class: computational_model
title: Joint model M4 beats threat-only M2 by ΔWAIC ≈ 1,600–2,000
status: supported
prereg_h: [H3b]
internal_h: []
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/H3_model_comparison.ipynb, scripts/run_mcmc_pipeline.py]
scripts: [scripts/run_mcmc_pipeline.py]
outputs: [results/stats/joint_optimal/exploratory/mcmc_model_comparison.csv, results/stats/joint_optimal/confirmatory/mcmc_model_comparison.csv]
figures: [TODO]
created: 2026-05-27
last_run: 2026-05-27
---

# Result 203 — Joint model M4 beats threat-only M2 by ΔWAIC ≈ 1,600–2,000

## Overview

Does individual variation in effort sensitivity matter, beyond per-subject threat sensitivity? We tested this by fitting M2 — a threat-only model that has per-subject ω but a single population κ shared across all subjects — and comparing it to M4 (which has per-subject ω AND per-subject κ). M4 outperforms M2 by ΔWAIC ≈ +1,966 in the exploratory sample and ≈ +1,621 in the confirmatory sample, with ΔLOO agreeing. The result establishes that individual differences in effort sensitivity are a separate, identifiable source of behavioral variation that the model needs to represent.

## Hypothesis

**Statement.** "The joint model will outperform a threat-only model that lacks individual effort sensitivity." (Preregistration, H3b.)

**Predicted direction.** ΔWAIC > 0 AND ΔLOO > 0 favoring M4 over M2.

**Preregistered criterion.** WAIC and LOO must agree.

## Data Source

Same as [[result_201]]. Joint likelihood comparison.

## Method

**M2 (threat-only) specification.**
- Same fitness function W(u) as M4 — survival weighting, fitness-derived choice and vigor.
- Per-subject ω_i, but κ is a single population parameter (no subject-level variation in effort sensitivity).
- 299 / 290 free parameters.

**M4 (joint fitness, full).** See [[result_201]].

**Comparison metrics:** WAIC and PSIS-LOO from the joint likelihood, computed via ArviZ.

## Result

| Metric | Exploratory | Confirmatory |
|---|---|---|
| M2 WAIC | 14,742 | 13,873 |
| M4 WAIC | 12,776 | 12,252 |
| **ΔWAIC (M2 − M4)** | **+1,966** (SE ≈ 669) | **+1,621** (SE ≈ 449) |
| M2 LOO | 14,745 | 13,881 |
| M4 LOO | 12,779 | 12,263 |
| **ΔLOO (M2 − M4)** | **+1,967** | **+1,618** |
| M2 choice accuracy | 78.9% | 77.8% |
| M4 choice accuracy | 77.3% | 75.9% |
| M2 vigor R² | 0.013 | 0.012 |
| M4 vigor R² | 0.372 | 0.412 |
| M2 Pareto-k > 0.7 (% bad) | 1.09% | 0.97% |
| M4 Pareto-k > 0.7 (% bad) | 0.68% | 0.83% |
| M2 converged | True | True |

**Verdict on prereg criterion:** **PASS** in both samples.

## Interpretation

M2 captures choice behavior nearly as well as M4 (M2 choice accuracy is in fact slightly higher in both samples — 78.9% vs 77.3% expl, 77.8% vs 75.9% conf) but fails almost completely on the vigor side (M2 vigor R² ≈ 0.012 vs M4's 0.37–0.41). The WAIC gap is therefore driven by vigor: when all subjects are forced to share a single κ, the model cannot account for the individual differences in pressing rate that show up clearly in the per-subject cell-mean data.

The result is a direct test of whether κ varies meaningfully across subjects. If individual differences in effort sensitivity were small or absent, M2 (one population κ) would suffice and M4's per-subject κ_i would be overfitting. Instead, allowing κ to vary across subjects improves joint fit by ΔWAIC ≈ 1,600–2,000 — a margin many standard errors wide and consistent across samples. The conclusion is that κ is a stable subject-level trait, identifiable from the joint behavioral data, and necessary for explaining individual variation in motor execution.

This complements [[result_202]] (which showed threat is necessary) by showing that effort sensitivity is also necessary as a per-subject parameter. The two preregistered comparisons together establish that the joint model's two-parameter structure (per-subject ω and per-subject κ) is the minimal architecture that fits the data well, motivating the H4 individual-difference regressions in [[result_208]] that use these parameters as predictors of behavior.

## Caveats & Limitations

- **Most of M2's deficit is on vigor, not choice.** As with [[result_202]], readers should note that the choice-only WAIC components show a much smaller M2-vs-M4 gap; the WAIC headline is driven by vigor fit. This is appropriate: the prereg's H3 model comparison is over the joint likelihood, which is the whole point of a "joint" fitness model. But it does mean M2 is essentially a "choice model that ignores vigor," not a "model that doesn't explain choice."
- **M2's choice accuracy slightly exceeds M4's.** This reflects a real trade-off: with no per-subject κ, M2's population κ can be tuned to fit choice optimally without the constraint of also fitting vigor. M4 commits κ to do double duty (choice AND vigor) and pays a small choice-fit cost. The joint WAIC framing penalizes M2 for failing the vigor likelihood, which is the correct accounting.
- **M2 has slightly worse Pareto-k diagnostics than M4.** Not enough to invalidate LOO, but worth noting for methodological transparency.

## Replication

See [[result_201]] Replication block.

## References

**Related results:**
- [[result_201]] — M4 fit and convergence.
- [[result_202]] — M4 vs M1 (effort-only).
- [[result_204]] — M4 vs M3 (single-parameter).
- [[result_208]] — H4 family Bayesian regressions on ω and κ.

**Literature:** Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.
