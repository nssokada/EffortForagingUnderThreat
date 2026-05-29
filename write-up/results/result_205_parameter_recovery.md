---
result_id: 205
class: computational_model
title: Both ω and κ are well-recovered under the paper's M4 + MCMC; the earlier κ-recovery failure was a recovery-harness model-specification artifact
status: supported
prereg_h: [H3]
internal_h: []
samples: [synthetic]
notebooks: [scripts/modeling/joint_optimal/param_recovery_m4_mcmc.py]
scripts: [scripts/modeling/joint_optimal/param_recovery_m4_mcmc.py]
outputs: [results/stats/joint_optimal/param_recovery_m4_mcmc_summary.csv, results/stats/joint_optimal/param_recovery_m4_mcmc.csv, results/stats/joint_optimal/param_recovery_m4_mcmc_population.csv, results/stats/joint_optimal/param_recovery_m4_mcmc_diagnostics.csv]
figures: [results/figs/paper/fig_recovery_m4_mcmc.pdf]
created: 2026-05-28
last_run: 2026-05-29
---

# Result 205 — Both ω and κ are well-recovered under the paper's M4 + MCMC

## Overview

The preregistration commits to parameter recovery as an identifiability check before interpreting subject-level ω and κ. We simulated 500 synthetic subjects from the **empirical** joint distribution of (ω, κ) — drawing log-parameters from a bivariate normal matched to the fitted M4 posterior (μ, σ, and the ω–κ correlation), generated choice and cell-mean vigor data **from the production M4 generative model**, refit with the **identical production M4 model and inference** (NUTS, 4 chains × 2000 warmup + 4000 samples, SVI warm-start, target_accept 0.95), and correlated recovered subject-level posterior means with true values. **Both parameters are well-recovered:** ω at Pearson r = 0.924 and κ at r = 0.918, both with well-calibrated credible intervals (80% coverage 77–81%; 95% coverage 92–96%). The fit converged cleanly (max R-hat = 1.0013, min ESS = 4567).

This **supersedes** the earlier conclusion (that κ was "essentially unrecoverable at the empirical spread," r_κ ≈ −0.07). That failure was a **model-specification artifact in the prior recovery harness**, not a property of κ in the real data: the old harness (`param_recovery_v8c*.py`) used a model that (a) dropped the `−κ·req·D` total-demand choice term — κ's main identifiability channel — and (b) added a per-subject vigor baseline intercept that the production M4 does not contain. That model fails to converge even under MCMC (max R-hat = 6.9). With the **actual production M4**, κ recovers.

## Hypothesis

**Statement.** "We will simulate 500 synthetic subjects from known ω and κ values, fit the model to the simulated data, and correlate recovered with true parameters to verify identifiability." (Prereg §H3 / Parameter recovery.)

## Result

**Recovery under the production M4 model + MCMC (N = 500, empirical spread σ_log κ = 1.36):**

| Parameter | r (true vs rec) | ρ (Spearman) | RMSE (log) | 80% coverage | 95% coverage | Top-decile overlap |
|---|---|---|---|---|---|---|
| **ω** | 0.924 | 0.940 | 0.373 | 77.2% | 92.4% | 78% |
| **κ** | 0.918 | 0.925 | 0.549 | 81.4% | 95.6% | 82% |

Cross-talk r(true ω, recovered κ) = +0.374 and r(true κ, recovered ω) = +0.388 — these track the **true** built-in ω–κ correlation (r(log ω, log κ) = 0.369 in the empirical distribution), i.e. they reflect genuine parameter covariation, not estimation bleed.

**Population-level recovery:**

| Population parameter | True | Recovered |
|---|---|---|
| μ_log ω | +0.078 | +0.164 |
| σ_log ω | 0.871 | 0.874 |
| μ_log κ | −1.555 | −1.721 |
| **σ_log κ** | **1.359** | **1.410** |
| γ (hazard nonlinearity) | 0.847 | 0.866 |
| hazard | 0.551 | 0.512 |

Convergence: **max R-hat = 1.0013, min ESS = 4567 (PASS).**

**Why the earlier result said κ was unrecoverable — model comparison at the same empirical spread:**

| Recovery harness | Model | Inference | r_κ | Converged? |
|---|---|---|---|---|
| `param_recovery_v8c.py` (old) | v8c (no `−κ·req·D` choice term; extra baseline) | SVI / AutoNormal | **−0.07** | n/a (SVI) |
| `param_recovery_v8c_mcmc.py` | v8c (same degenerate model) | NUTS | 0.85 | ✗ **max R-hat = 6.9** |
| **`param_recovery_m4_mcmc.py`** | **production M4** | **NUTS** | **0.918** | ✅ **max R-hat = 1.001** |

The decisive control: the **production M4 fits the real exploratory data with max R-hat = 1.0016** ([[result_201]]). The degenerate v8c model cannot converge on synthetic data drawn at the same spread; the production model can, and recovers κ.

## Interpretation

The earlier "κ recovery is essentially zero at the empirical spread" finding was an artifact of the recovery code, not a limit on κ in the data. Two specification differences in the old harness destroyed κ's identifiability:

1. **Dropped choice demand term.** In production M4, κ enters the *choice* likelihood through `V = max_u W(u) − κ·req·D` (the total-demand cost), which gives κ strong subject-by-subject leverage from the choice data. The old recovery model omitted this term, leaving κ identified only through the near-flat vigor optimum — so its per-subject posteriors collapsed toward the prior (95% CrI widths ≈ 5.5 in log units).
2. **Spurious per-subject baseline.** The old model added a free per-subject vigor intercept that production M4 does not have. This intercept forms an unidentified ridge with the vigor optimum u*(κ), and was the single worst-converging parameter (R-hat 6.9 under MCMC).

SVI/AutoNormal additionally produced over-confident posteriors on that mis-specified model (coverage far below nominal), compounding the apparent failure — but the root cause was the model, since the same model fails under exact MCMC too.

**With the production M4 + MCMC, both ω and κ are interpretable subject-level quantities.** Recovery is strong (r ≈ 0.92 for both) and, critically, the credible intervals are **well-calibrated** (80%/95% coverage at 77–96%), so per-subject posterior means *and their uncertainty* can be used in downstream individual-difference analyses. The population-level structure — including the wide κ spread (σ_log κ recovered as 1.41 vs true 1.36) — is recovered accurately.

This **removes the per-subject-κ caveat** that previously qualified downstream work. The H4 family ([[result_208]]) uses κ as a predictor (e.g. H4c: β(κ → mean vigor) ≈ −0.20 in both samples); that and the H4d ω–κ angle test rest on κ being an identifiable per-subject trait — which this recovery now establishes. The angle test in particular (which combines per-subject ω and κ) no longer needs the earlier "interpret with caution" hedge.

## Caveats & Limitations

- **Single spread tested (the empirical one).** Recovery is demonstrated at σ_log κ = 1.36, the value fitted in real subjects — the most decision-relevant case. We did not sweep σ; the earlier (degenerate-model) "wide spread vs fitted" contrast is no longer the operative comparison.
- **True parameters are over-dispersed slightly.** The synthetic σ_log κ uses the SD of the fitted *posterior means* (1.36), close to the hierarchical σ_κ (≈1.43); this is mild and conservative.
- **Cross-talk equals the true correlation.** The +0.37/+0.39 cross-parameter correlations reflect the genuine ω–κ covariation built into the empirical distribution, not recovery error.
- **Recovery uses the production model for both generation and inference** (self-consistency / well-specified case). It does not test robustness to generative model misspecification — but it is exactly the identifiability claim the prereg makes.

## Replication

```bash
python scripts/modeling/joint_optimal/param_recovery_m4_mcmc.py \
    --n_subj 500 --num_warmup 2000 --num_samples 4000 \
    --num_chains 4 --target_accept 0.95 --seed 42 --sample exploratory
```
Outputs `results/stats/joint_optimal/param_recovery_m4_mcmc{,_summary,_population,_diagnostics}.csv` and `_samples.npz`. GPU run time ≈ 60 min (NVIDIA RTX 4080, JAX `chain_method='vectorized'`). Verify `convergence_passed = True` in `_summary.csv` before interpreting.

## References

- [[result_201]] — M4 fit on real data (the model whose recovery is tested here; real-data max R-hat = 1.0016).
- [[result_208]] — H4 individual-difference regressions (where κ is used as a per-subject predictor; the earlier κ caveat is now lifted).
- `instructions/memory/joint_model_development.md` — development history, including the diagnosis of the old recovery-harness model mismatch.

## Revision notes

- **2026-05-29:** Conclusion reversed and `status: partial → supported`. The prior version (SVI, `param_recovery_v8c.py`) reported κ "essentially unrecoverable at the empirical spread" (r_κ ≈ −0.07) and attributed this to a narrow κ distribution. Re-running recovery with the **exact production M4 model + MCMC** (`param_recovery_m4_mcmc.py`, N=500) gives **r_κ = 0.918 with calibrated coverage** (max R-hat = 1.0013). The earlier failure was traced to a **model-specification artifact** in the old recovery harness (dropped `−κ·req·D` choice term; spurious per-subject vigor baseline) — a model that does not converge even under MCMC (R-hat = 6.9) — compounded by SVI/AutoNormal over-confidence. The empirical κ spread is in fact wide (σ_log κ = 1.36, recovered as 1.41). Per-subject-κ caveat lifted; `[[result_208]]` H4d/κ hedges flagged for follow-up update.
