# Pipeline State

Current execution status of each notebook and script in the analysis pipeline.
Last updated: 2026-03-20.

---

## Preprocessing (`notebooks/01_preprocessing/`)

| Notebook | Status | Output |
|----------|--------|--------|
| `01_run_pipeline.ipynb` | ✅ Complete | `data/exploratory_350/processed/stage{1-5}_*/` |
| `02_data_prep.ipynb` | ✅ Complete | Various |
| `03_data_prep_stage1_analysis_table.ipynb` | ✅ Complete | `analysis_table.parquet` (deprecated for vigor) |
| `04_behavior_overview.ipynb` | ✅ Complete | `results/figs/behavior/fig{1-5}_*.{pdf,png}` |

**Active stage5 output:** `data/exploratory_350/processed/stage5_filtered_data_20260320_191950/`
- `behavior.csv` — N=293 trials
- `psych.csv` — psychiatric battery (all subscales scored), N=293 subjects
- `feelings.csv` — 10,546 rows, 293 subjects (5,274 anxiety + 5,272 confidence)
- `subject_mapping.csv` — participantID → subj integer

---

## Choice Modeling (`notebooks/02_choice_modeling/`)

| Notebook | Status | Notes |
|----------|--------|-------|
| `01_fit_compare_ppc.ipynb` | ✅ Complete | FETExponentialBias fit (superseded by L3_add) |
| `02_parameter_recovery.ipynb` | ⚠️ Not run | Needs to run against L3_add fit |
| `03_unified_model_comparison.ipynb` | ✅ Complete | **11-model SVI comparison. Winner: L4a_add (α in survival, additive effort, hyperbolic kernel).** Saved: `unified_model_comparison.csv`, `unified_3param_clean.csv` |
| `scripts/run_unified_model_comparison.py` | ✅ Complete | Standalone re-run on new data path (stage5_20260320_191950). Results consistent with NB03. |

**Current winning model: L4a_add** (by ELBO and BIC)
```
SV = R·S - k·E - β·(1-S)
S = (1-T) + T/(1+λ·D/α)
```
Note: L3_add (no α) is still primary for subject-level parameter extraction (unified_3param_clean.csv) since α comes from vigor independently. L4a_add wins by 15.7 ELBO over L3_add.

- k, β per-subject; λ, τ population-level
- α (from vigor HBM) enters survival kernel — marginal gain (+15.7 ELBO vs L3_add)
- Additive >> multiplicative (+158 ELBO)
- Hyperbolic >> exponential (+190 ELBO vs L3_survival)

**Key model comparison findings (2026-03-20 re-run, N=293, 13185 trials):**
- L4a_add: ELBO=−6259.7, BIC=18135.6 (best)
- L3_add:  ELBO=−6275.4, BIC=18167.1 (primary parameter source)
- Per-subject z hurts (−112 ELBO) — not needed
- α in effort only (L4c): hurts (−24 ELBO vs L3_add)
- α in effort+survival (L4d): hurts (−2.6 ELBO vs L3_add)
- k-β r=−0.138 (p=0.018), k-α r=−0.052 (p=0.37), β-α r=+0.264 (p<0.001)

---

## Vigor Data Prep

| Script | Status | Output |
|--------|--------|--------|
| `scripts/vigor_data_prep.py` | ✅ Complete | `data/exploratory_350/processed/vigor_prep/` |

**vigor_prep contents:**
- `keypress_events.parquet` — 899,936 rows (one per keypress)
- `trial_events.parquet` — 23,733 rows (one per trial)
- `effort_ts.parquet` — 293 rows (calibrationMax)
- `subject_mapping.csv` — 293 rows

---

## Vigor Analysis (`notebooks/03_vigor_analysis/`)

| Notebook | Status | Key Output | Notes |
|----------|--------|------------|-------|
| `01_single_trial_visualization.ipynb` | ✅ Fixed | — | Column harmonization done |
| `02_kernel_smoothing.ipynb` | ✅ Complete | `smoothed_vigor_ts.parquet` (48.2 MB) | EVAL_HZ=20 |
| `03_tonic_phasic_decomposition.ipynb` | ✅ Fixed | — | Column harmonization done |
| `04_phase_extraction.ipynb` | ✅ Complete | `phase_vigor_metrics.parquet`, `phase_trial_metrics.parquet` | |
| `05_subject_features.ipynb` | ✅ Complete | `subject_vigor_table.csv` | |
| `06_choice_vigor_mapping.ipynb` | ✅ Complete | `results/choice_vigor_mapping_results.csv` | |
| `07_clinical_prediction.ipynb` | ✅ Unblocked | — | Factor scores now available from NB06-psych |
| `08_parameter_dissociation.ipynb` | ✅ Complete | `results/tables/table_s2_parameter_dissociation.csv/.tex` | |
| `09_final_stats.ipynb` | ✅ Complete | `results/step1_modelfree_results.csv` | |
| `10_pls_vigor_params.ipynb` | ✅ Complete | `results/stats/pls_vigor_params_results.csv` | PLS + trial-level LMM |
| `11_vigor_ode.ipynb` | ✅ Dead end | — | ODE kinetics degenerate, no new findings |
| `12_imminence_diagnostics.ipynb` | ✅ Complete | — | Phase-based encounter diagnostics |
| `13_encounter_vigor_counts.ipynb` | ✅ Complete | — | Encounter-centered count-based vigor |
| `14_choice_vigor_dissociation.ipynb` | ✅ Complete | `results/figs/fig_*.png` | 6-figure dissociation visualization |
| `15_dissociation_formal_tests.ipynb` | ✅ Complete | — | Phase 0-6 statistical pipeline |
| `16_bayesian_vigor_model.ipynb` | ✅ Complete | `vigor_hbm_posteriors.csv`, `vigor_hbm_population.csv`, `vigor_hbm_idata.nc` | **Two-window HBM: α (pre-enc) + ρ (terminal)** |

**Vigor model (final) — re-run 2026-03-20 via scripts/run_vigor_hbm.py:**
```
pre_enc_rate  ~ Normal(α_i, σ_pre)                     # [enc-2, enc], vigor_norm
terminal_rate ~ Normal(γ_i + ρ_i·attack, σ_term)       # [trialEnd-2, trialEnd], vigor_norm
```
Data source: `smoothed_vigor_ts.parquet` (mean vigor_norm per window), N=293, 23,554 trials.
- μ_α=0.315, SB=0.964, shrinkage=89%, max Rhat=1.008
- μ_ρ=0.067, P(>0)=1.0, SB=0.635, shrinkage=37%, max Rhat=1.006
- α-ρ: r=+0.016, p=0.78 (independent)
- 0 divergences. idata.nc saved (549 MB).

---

## Psychological Analysis (`notebooks/04_psych_analysis/`)

| Notebook | Status | Notes |
|----------|--------|-------|
| `01_bayesian_mental_health_regressions.ipynb` | ⚠️ Unknown | Not checked recently |
| `02_psychological_analysis.ipynb` | ⚠️ Unknown | Not checked recently |
| `03_affect_survival.ipynb` | ✅ Complete (re-run 2026-03-20) | S_probe (L3_add, λ=2.0) → anxiety/confidence LMM; state-trait decomposition |
| `04_anxiety_vigor_coupling.ipynb` | ✅ Complete | Anxiety → vigor coupling NULL at all levels |
| `05_metacognitive_calibration.ipynb` | ✅ Complete | Probe-trial linkage, S_probe→ratings, k→calibration |
| `06_factor_analysis.ipynb` | ✅ Complete (re-run 2026-03-20) | 3-factor EFA (distress/fatigue/apathy), α→F3(apathy) R²=0.123, t=−6.11 |
| `07_pls_params_mental_health.ipynb` | ✅ Complete | PLS 5 params→MH+affect, CV R²=0.039, perm p<0.001 |
| `08_mixture_model_subtypes.ipynb` | ✅ Complete | GMM k=3; coupled/decoupled hypothesis NULL |

---

## Publication Figures (`notebooks/05_figures/`)

| Notebook | Status | Notes |
|----------|--------|-------|
| `01_publication_figures.ipynb` | ⚠️ Needs update | Will need rerun after draft rewrite |

---

## Results Files

**`results/stats/` (key files):**
- `unified_model_comparison.csv` ✅ (12-model SVI comparison)
- `unified_3param_clean.csv` ✅ (L3_add subject parameters: k, β)
- `vigor_hbm_posteriors.csv` ✅ (per-subject α, ρ, γ with posterior SDs; re-run 2026-03-20 via smoothed_vigor_ts)
- `vigor_hbm_population.csv` ✅ (population hyperparameters + split-half reliability)
- `affect_lmm_results.csv` ✅ (re-run 2026-03-20, L3_add S_probe)
- `affect_trait_scores.csv` ✅ (re-run 2026-03-20, per-subject mean affect + k/β)
- `affect_vigor_cross_domain.csv` ✅ (all n.s.)
- `psych_factor_scores.csv` ✅ (re-run 2026-03-20, 3-factor EFA, N=291)
- `psych_factor_loadings.csv` ✅ (re-run 2026-03-20)
- `psych_params_to_factors.csv` ✅ (re-run 2026-03-20, 3-param + 4-param OLS)
- `choice_vigor_dissociation_results.csv` ✅ (2026-03-20, 20-row stats table: correlations, ANOVAs, t-tests)
- `choice_vigor_dissociation_subjects.csv` ✅ (2026-03-20, N=293 subject-level data with quadrant labels)
- `pls_mh_*.csv` ✅ (PLS params→MH)
- `joint_correlated_correlations.csv` ✅ (2026-03-21, LKJ ρ posteriors for all 6 param pairs)
- `joint_correlated_subjects.csv` ✅ (2026-03-21, per-subject k, β, α, δ from joint model)
- `joint_correlated_population.csv` ✅ (2026-03-21, population hyperparameters + ELBO)
- `joint_correlated_omega_samples.csv` ✅ (2026-03-21, 4000 posterior samples of correlation matrix)

**EVC+gamma parameter recovery (2026-03-26):**
- `evc_parameter_recovery.csv` ✅ (5 synthetic datasets × 50 subjects; c_death r=0.946, epsilon r=0.926, c_effort r=0.04 NOT recoverable, gamma=0.262 vs true 0.283)

**Superseded (keep for reference):**
- `FET_Exp_Bias_*.csv` — old model, replaced by L3_add
- `joint_model_*.csv` — old joint model (independent priors, σ_δ collapsed), replaced by joint_correlated_*

**`results/model_fits/exploratory/`:**
- `vigor_hbm_idata.nc` ✅ (full MCMC trace, 549 MB, re-run 2026-03-20 via smoothed_vigor_ts)
- `FET_Exp_Bias_fit.pkl` — superseded

**`results/tables/`:**
- `table_s2_parameter_dissociation.csv/.tex` ✅
