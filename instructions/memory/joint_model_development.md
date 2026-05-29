---
name: Joint Model Development
description: Development history of joint choice-vigor models — independent Bayesian pipeline (primary) + correlated LKJ model (robustness), λ sensitivity discovery
type: project
---

## Current architecture (2026-03-22)

**Primary: Independent Bayesian pipeline**
```
Choice SVI (AutoNormal, 15k steps) → λ=13.9, k_i, β_i
Vigor SVI  (AutoNormal, 10k steps, λ fixed) → α_i, δ_i
Correlate posterior means → r(log(β), δ) = +0.55
```
Script: inline in conversation. To be formalized in `scripts/run_mcmc_pipeline.py` (MCMC version ready).

**Robustness: Joint correlated model**
```
[log(k), log(β), α, δ] ~ MVN(μ, Σ), Ω ~ LKJCholesky(η=2)
```
Script: `scripts/run_joint_correlated.py`. Confirms all 6 pairwise CIs exclude zero.

## λ sensitivity (key finding, 2026-03-22)

The joint model's LKJ ρ estimates depend on fixed λ:
- λ = 13.8 → ρ(β,δ) = +0.75
- λ = 15.1 → ρ(β,δ) = +0.30
- λ free (informative prior) → λ pushes to 35, ρ(β,δ) = +0.75

**Cause:** Vigor is more threat-driven than distance-driven. Large λ compresses S toward (1−T), changing how variance is attributed to β vs other parameters.

**Resolution:** Independent Bayesian r = +0.55 is the headline (robust to λ). Joint model provides directional confirmation. Paper documents this transparently.

## Development history

1. **v1 (independent priors, AutoNormal):** σ_δ collapsed. β unidentified (single S per trial — β cancels in ΔSV).
2. **v2 (LKJ + AutoMultivariateNormal, option-specific S):** β identified but λ inflated to 50.
3. **v3 (λ fixed from choice-only at 15.1):** ρ(β,δ) = +0.30. Worked but λ sensitivity discovered.
4. **v4 (independent Bayesian pipeline):** Separate choice + vigor models, correlate posterior means. r(β,δ) = +0.55. Robust, clean, no shared structure to inflate correlations.

## Parameter recovery — MCMC, production M4 (2026-05-28)

**Result (result_205, now `status: supported`):** Both ω and κ are well-recovered
under the production M4 + NUTS at the empirical spread (σ_log κ = 1.36).
N=500 synthetic subjects: **r_ω = 0.924, r_κ = 0.918**; 80% coverage 77%/81%,
95% coverage 92%/96%; max R-hat = 1.0013, min ESS = 4567. σ_log κ recovered as
1.41 (true 1.36). Script: `scripts/modeling/joint_optimal/param_recovery_m4_mcmc.py`;
outputs `results/stats/joint_optimal/param_recovery_m4_mcmc*`; fig
`results/figs/paper/fig_recovery_m4_mcmc.pdf`.

**Why this overturned the earlier "κ unrecoverable" claim:** the prior recovery
harness (`param_recovery_v8c*.py`) used a model that DIVERGED from the production
`make_m4` (in `model_comparison_cm.py`): it (a) dropped the `−κ·req·D` total-demand
choice term — κ's main identifiability channel — and (b) added a per-subject vigor
`baseline` intercept that production M4 does not have. That model fails to converge
even under MCMC (max R-hat = 6.9; baseline was the worst param). SVI/AutoNormal
compounded it with over-confident posteriors. Root cause = model spec, not inference.
**Lesson:** recovery harnesses must import the EXACT production model
(`_cm.make_m4`, `_cm.eu_sat`), not re-implement it. Real-data M4 converges at
max R-hat = 1.0016 (`exploratory/mcmc_convergence_diagnostics.csv`).

**Downstream:** the per-subject-κ caveat is lifted; result_208 (H4 family) κ
predictor and the H4d ω–κ angle test no longer need the "interpret κ cautiously"
hedge. result_208 prose still carries that hedge — flagged for update, not yet edited.

## Key outputs

| File | Content |
|---|---|
| `independent_bayesian_params.csv` | Per-subject k, β, α, δ from independent Bayesian models |
| `independent_bayesian_correlations.csv` | Cross-domain r values |
| `joint_correlated_correlations.csv` | LKJ ρ posteriors (from latest joint fit) |
| `joint_correlated_omega_samples.csv` | 4000 posterior samples of correlation matrix |
| `mcmc_*.csv` | (Future) MCMC results from GPU run |

**How to apply:** Lead with independent Bayesian r for the coupling claim. Use joint model for structural confirmation. Document λ sensitivity in Methods/Supplement. For confirmatory sample, use same pipeline.
