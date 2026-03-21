# Pipeline State

Current execution status of each notebook and script in the analysis pipeline.
Last updated: 2026-03-18.

---

## Preprocessing (`notebooks/01_preprocessing/`)

| Notebook | Status | Output |
|----------|--------|--------|
| `01_run_pipeline.ipynb` | ✅ Complete | `data/exploratory_350/processed/stage{1-5}_*/` |
| `02_data_prep.ipynb` | ✅ Complete | Various |
| `03_data_prep_stage1_analysis_table.ipynb` | ✅ Complete | `analysis_table.parquet` (deprecated for vigor) |
| `04_behavior_overview.ipynb` | ✅ Complete | `results/figs/behavior/fig{1-5}_*.{pdf,png}` |

**Active stage5 output:** `data/exploratory_350/processed/stage5_filtered_data_20260317_094210/`
- `behavior.csv` — N=293, 13,185 trials
- `psych.csv` — psychiatric battery (all subscales scored)
- `subject_mapping.csv` — participantID → subj integer

---

## Choice Modeling (`notebooks/02_choice_modeling/`)

| Notebook | Status | Notes |
|----------|--------|-------|
| `01_fit_compare_ppc.ipynb` | ✅ Complete | Full plotter version; N=270 GPU fit used for WAIC table |
| `02_parameter_recovery.ipynb` | ⚠️ Not run | Needs to run against N=293 fit |
| `03_unified_model_comparison.ipynb` | ✅ Complete | 12-model SVI comparison. Winner: L4a_add (additive effort, hyperbolic escape, unified α). Saved: `unified_model_comparison.csv`, `unified_3param_clean.csv` |

**Scripts:**
| Script | Status | Output |
|--------|--------|--------|
| `scripts/run_fit_best_model.py` | ✅ Complete | `results/model_fits/exploratory/FET_Exp_Bias_fit.pkl` (217 MB) |
| `scripts/run_ppc_analysis.py` | ✅ Complete | `results/stats/FET_Exp_Bias_*.csv` |

**Fit quality (N=293):** WAIC=12,063, R²=0.454, AUC=0.912, Accuracy=82.5%, ECE=0.023

---

## Vigor Data Prep

| Script | Status | Output |
|--------|--------|--------|
| `scripts/vigor_data_prep.py` | ✅ Complete | `data/exploratory_350/processed/vigor_prep/` |

**vigor_prep contents:**
- `keypress_events.parquet` — 899,936 rows (one per keypress)
- `trial_events.parquet` — 23,733 rows (one per trial); columns include `effort_L`, `calibrationMax`
- `effort_ts.parquet` — 293 rows (calibrationMax)
- `subject_mapping.csv` — 293 rows

---

## Vigor Analysis (`notebooks/03_vigor_analysis/`)

| Notebook | Status | Key Output | Notes |
|----------|--------|------------|-------|
| `01_single_trial_visualization.ipynb` | ✅ Fixed | — | Column harmonization + f_max_i merge added this session |
| `02_kernel_smoothing.ipynb` | ✅ Complete | `smoothed_vigor_ts.parquet` (48.2 MB, 3,988,277 rows), `demand_curves.parquet` | EVAL_HZ=20 restored this session |
| `03_tonic_phasic_decomposition.ipynb` | ✅ Fixed | — | Column harmonization + c_it optional added this session |
| `04_phase_extraction.ipynb` | ✅ Complete | `phase_vigor_metrics.parquet`, `phase_trial_metrics.parquet`, `encounter_phase_ts.parquet`, `terminal_phase_ts.parquet` | |
| `05_subject_features.ipynb` | ✅ Complete | `subject_vigor_table.csv` | |
| `06_choice_vigor_mapping.ipynb` | ✅ Complete | `results/choice_vigor_mapping_results.csv` | |
| `07_clinical_prediction.ipynb` | ❌ Blocked | — | Needs `modeling_factor_param.csv` (EFA of psych battery) |
| `08_parameter_dissociation.ipynb` | ✅ Complete | `results/tables/table_s2_parameter_dissociation.csv/.tex` | |
| `09_final_stats.ipynb` | ✅ Complete | `results/step1_modelfree_results.csv` | |
| `10_pls_vigor_params.ipynb` | ✅ Complete | `results/stats/pls_vigor_params_results.csv` | New this session; PLS + trial-level LMM |
| `11_vigor_ode.ipynb` | ✅ Run (dead end) | `results/stats/vigor_ode_params.csv`, `vigor_ode_correlations.csv` | ODE kinetics degenerate, no new findings |
| `12_imminence_diagnostics.ipynb` | ✅ Complete | — | Phase-based encounter diagnostics |
| `13_encounter_vigor_counts.ipynb` | ✅ Complete | — | Encounter-centered count-based vigor |
| `14_choice_vigor_dissociation.ipynb` | ✅ Complete | `results/figs/fig_*.png` | 6-figure dissociation visualization |
| `15_dissociation_formal_tests.ipynb` | ✅ Complete | — | Phase 0-6 statistical pipeline |
| `16_bayesian_vigor_model.ipynb` | ✅ Complete | `results/stats/vigor_hbm_posteriors.csv`, `vigor_hbm_population.csv`, `results/model_fits/exploratory/vigor_hbm_idata.nc` | Two-window HBM for α (pre-enc) and ρ (terminal) via NumPyro |

**smoothed_vigor_ts.parquet columns (key):**
- `subj`, `trial`, `t` (trial-relative seconds), `vigor_norm` (v_t), `vigor_resid`
- `isAttackTrial` (encounter flag), `encounterTime` (ALL trials — scheduled predator time)
- `threat` (0.1/0.5/0.9), `choice` (0/1), `distance_H` (1/2/3), `effort_H` (0.6/0.8/1.0)
- `startDistance` (5/7/9 — predator start distance)

---

## Psychological Analysis (`notebooks/04_psych_analysis/`)

| Notebook | Status | Notes |
|----------|--------|-------|
| `01_bayesian_mental_health_regressions.ipynb` | ⚠️ Unknown | Not checked this session |
| `02_psychological_analysis.ipynb` | ⚠️ Unknown | Not checked this session |
| `03_affect_survival.ipynb` | ✅ Complete | S_probe → anxiety/confidence LMM; state-trait decomposition; cross-domain vigor×affect (all n.s.) |
| `04_anxiety_vigor_coupling.ipynb` | ✅ Complete | Anxiety → vigor coupling at trial level — NULL at all levels (concurrent, residual, predictive) |
| `05_metacognitive_calibration.ipynb` | ✅ Complete | Probe-trial linkage, S_probe→ratings, k→calibration |
| `06_factor_analysis.ipynb` | ✅ Complete | 3-factor EFA (distress/fatigue/apathy), α→apathy R²=0.155 |
| `07_pls_params_mental_health.ipynb` | ✅ Complete | PLS 5 params→MH+affect, CV R²=0.039, perm p<0.001 |

---

## Publication Figures (`notebooks/05_figures/`)

| Notebook | Status | Notes |
|----------|--------|-------|
| `01_publication_figures.ipynb` | ⚠️ Unknown | Not checked this session |

---

## Results Files

**`results/stats/`:**
- `FET_Exp_Bias_waic.csv` ✅
- `FET_Exp_Bias_predictions.csv` ✅
- `FET_Exp_Bias_subject_metrics.csv` ✅
- `FET_Exp_Bias_population_params.csv` ✅
- `FET_Exp_Bias_k_params.csv` ✅ (column: `subject`, not `subj`)
- `FET_Exp_Bias_z_params.csv` ✅ (column: `subject`, not `subj`)
- `FET_Exp_Bias_beta_params.csv` ✅ (column: `subject`, not `subj`)
- `modeling_factor_param.csv` ❌ MISSING (blocks NB07)
- `pls_vigor_params_results.csv` ✅
- `vigor_ode_params.csv` ✅ (exploratory only — dead end)
- `vigor_ode_correlations.csv` ✅ (exploratory only — dead end)
- `vigor_hbm_posteriors.csv` ✅ (NB16: per-subject α, ρ Bayesian posteriors + choice params)
- `vigor_hbm_population.csv` ✅ (NB16: population-level hyperparameters)
- `affect_lmm_results.csv` ✅ (NB12)
- `affect_threat_slopes.csv` ✅ (NB12)
- `affect_vigor_cross_domain.csv` ✅ (NB12 — all n.s.)
- `affect_trait_scores.csv` ✅ (NB12)

**`results/tables/`:**
- `table_s2_parameter_dissociation.csv` ✅
- `table_s2_parameter_dissociation.tex` ✅
