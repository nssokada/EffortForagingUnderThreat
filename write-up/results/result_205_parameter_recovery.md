---
result_id: 205
class: computational_model
title: ω is well-recovered in synthetic data; κ recovery depends on population spread
status: partial
prereg_h: [H3]
internal_h: []
samples: [synthetic]
notebooks: [scripts/run_mcmc_pipeline.py]
scripts: [scripts/run_mcmc_pipeline.py]
outputs: [results/stats/joint_optimal/param_recovery_v8c.csv]
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 205 — ω is well-recovered in synthetic data; κ recovery depends on population spread

## Overview

The preregistration commits to parameter recovery as a check on identifiability before claiming substantive interpretation of subject-level ω and κ. We simulated 500 synthetic subjects from three population distributions (the empirical fitted distribution, a wide-spread distribution, and a correlated-parameter distribution), fit M4 to each synthetic dataset, and correlated recovered subject-level posterior means with true parameters. ω is well-recovered across all three scenarios (Pearson r ≈ 0.89–0.99). κ recovery is excellent only in the wide-spread scenario (r ≈ 0.97); in the empirical-distribution and correlated scenarios κ recovery is essentially zero (r ≈ −0.07 and −0.06). This is a substantive limitation: the empirical distribution of κ in real subjects is narrow enough that posterior means barely track true values, even though the population-level κ structure is correctly recovered.

## Hypothesis

**Statement.** "We will simulate 500 synthetic subjects from known ω and κ values, fit the model to the simulated data, and correlate recovered with true parameters to verify identifiability." (Prereg §H3 / Parameter recovery.)

## Result

| Scenario | r(ω) | r(κ) | r(baseline) | γ recovery | hazard recovery |
|---|---|---|---|---|---|
| **Fitted distribution** (matches empirical) | 0.892 | **−0.074** | 0.983 | 0.844 | 1.152 |
| **Wide spread** | 0.989 | **+0.970** | 0.983 | 0.819 | 0.890 |
| **Correlated** (ω and κ co-vary) | 0.893 | **−0.060** | 0.982 | 0.856 | 0.851 |

(`r_omega` and `r_kappa` are Pearson correlations between true and recovered subject-level posterior means.)

## Interpretation

The recovery profile is asymmetric in a way that affects interpretive claims. ω, which enters the survival function and is heavily constrained by the choice likelihood, is well-recovered across all three population scenarios — the choice data carries enough subject-by-subject signal to estimate ω individually. κ, which enters both the choice demand-cost term and the vigor optimum, is well-recovered only when subjects are spread wide enough across the parameter space to give the model leverage. At the empirical level of variation, κ recovery is essentially zero: posterior means for κ at the subject level do not reliably track the true generating values.

This has direct implications for downstream individual-difference analyses. Per-subject ω can be used as an interpretable quantity in [[result_208]] (H4 family) and downstream — recovery is excellent. Per-subject κ should be interpreted more cautiously: the population-level distribution is well-estimated, and group-level effects involving κ (e.g., "high-κ subjects have lower mean vigor") will reflect the underlying truth on average, but a *specific subject's* κ posterior mean is unreliable as an estimate of their true parameter at the empirical spread level. The H4c result (β(κ → mean vigor) ≈ −0.20 in both samples) is on solid ground because it is a population-level slope; an interpretation that ranks subjects by their estimated κ would not be.

The wide-spread scenario shows that the model itself is capable of recovering κ — the limitation is the spread of κ values in real subjects, not the model architecture. This is a feature of the data, not a bug in the inference.

## Caveats & Limitations

- **Recovery test uses only three synthetic scenarios.** A more complete test would sweep over population standard deviations to find the threshold at which κ becomes recoverable.
- **The "Fitted distribution" scenario simulates from M4's fitted population posteriors,** so it represents the most realistic test. The κ recovery failure there is therefore the most consequential finding.
- **ω-only individual-difference analyses are most defensible.** Analyses combining ω and κ at the subject level (e.g., the H4d angle test) should be reported with this caveat.

## Replication

`scripts/run_mcmc_pipeline.py --recovery-mode` regenerates the recovery table; expected output at `results/stats/joint_optimal/param_recovery_v8c.csv`. Runtime: ~2 hours for 500 subjects × 3 scenarios.

## References

- [[result_201]] — M4 fit (the model whose recovery is tested here).
- [[result_208]] — H4 individual-difference regressions (where this caveat matters).
- `instructions/memory/joint_model_development.md` — full development history including earlier recovery attempts.
