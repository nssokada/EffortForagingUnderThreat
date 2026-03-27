# Plan 1: Repository Cleanup

**Goal:** Organize the repository around the EVC+gamma direction. Move all deprecated files to an `_deprecated/` folder organized by the project/idea they belong to. Keep the active codebase clean and navigable.

---

## Cleanup Rules

1. **Do NOT delete anything** — move to `_deprecated/{category}/`
2. Preserve git history (just `git mv`)
3. Keep all preprocessing infrastructure (it's shared across directions)
4. Keep the active EVC+gamma model and its direct dependencies

---

## KEEP (Do Not Touch)

### Core Model

- `scripts/modeling/oc_evc_gamma.py` — THE current model
- `scripts/modeling/optimal_control.py` — Low-level utility functions (survival, tier EU)
- `scripts/modeling/__init__.py`

### Preprocessing (all)

- `scripts/preprocessing/` — entire directory (shared infrastructure)
- `notebooks/01_preprocessing/` — all notebooks

### Data

- `data/` — all raw and processed data

### Current Drafts

- `drafts/draft002/oc_story.md` — current narrative
- `drafts/draft002/paper_outline.md` — current outline
- `drafts/prereg_aspredicted.md` — preregistration template

### Active Plotting

- `scripts/plotting/plotter.py` — utility library
- `scripts/plotting/plot_pareto_optimal.py` — Pareto figure

### Active Results

- `results/stats/oc_evc_gamma_params.csv` — current model params
- `results/stats/oc_3param_eps_params.csv` — comparison model (no gamma)
- `results/stats/psych_factor_loadings.csv` — factor analysis
- `results/stats/psych_factor_scores.csv` — factor scores
- `results/figs/paper/` — publication figures

### Instructions & Memory

- `instructions/CLAUDE.md`
- `instructions/memory/` — all memory files (update after cleanup)

### Vigor Pipeline (core notebooks)

- `notebooks/03_vigor_analysis/01-05` — core vigor preprocessing
- `notebooks/03_vigor_analysis/09_final_stats.ipynb`

### Psych & Metacognition

- `notebooks/04_psych_analysis/03_affect_survival.ipynb`
- `notebooks/04_psych_analysis/05_metacognitive_calibration.ipynb`
- `notebooks/04_psych_analysis/06_factor_analysis.ipynb`

---

## DEPRECATE → `_deprecated/`

### `_deprecated/fet_models/` — Old FET Descriptive Models

**What these were:** The original choice model family (FETExponential, FETHyperbolic, etc.) with parameters z (hazard sensitivity), k (effort discounting), β (threat bias). Superseded by the mechanistic EVC framework.

Files to move:

- `scripts/modeling/base_model.py`
- `scripts/modeling/models.py`
- `scripts/modeling/fitter.py`
- `scripts/modeling/ppc.py`
- `scripts/modeling/oc_model.py` (2-param OC, pre-epsilon)
- `scripts/run_fit_best_model.py`
- `scripts/run_unified_model_comparison.py`
- `scripts/run_effort_scaled_comparison.py`
- `scripts/run_ppc_analysis.py`
- `scripts/run_mcmc_pipeline.py`
- `notebooks/02_choice_modeling/` — all 3 notebooks
- `results/stats/FET_Exp_Bias_*.csv` (7 files)
- `results/stats/effort_scaled_comparison.csv`
- `results/model_fits/` — old MCMC fit pickle

### `_deprecated/joint_coupling_models/` — Old Joint/Coupling Pipeline

**What these were:** The approach of fitting choice (k, β) and vigor (α, δ) in separate models, then testing coupling. Superseded by unified EVC which predicts both from shared params.

Files to move:

- `scripts/run_joint_correlated.py`
- `scripts/run_joint_choice_vigor_model.py`
- `scripts/run_vigor_hbm.py`
- `scripts/run_choice_vigor_dissociation.py`
- `scripts/analysis/run_h2_tests.py`
- `scripts/analysis/run_h2_coupling_tests.py`
- `scripts/plotting/plot_h2_coupling.py`
- `scripts/plotting/plot_h2_coupling_v5.py`
- `scripts/plotting/plot_allocation_space.py`
- `notebooks/03_vigor_analysis/08_parameter_dissociation.ipynb`
- `notebooks/03_vigor_analysis/14_choice_vigor_dissociation.ipynb`
- `notebooks/03_vigor_analysis/15_dissociation_formal_tests.ipynb`
- `notebooks/03_vigor_analysis/16_bayesian_vigor_model.ipynb`
- `notebooks/06_paper_pipeline/04_coherent_shift.py`
- `notebooks/06_paper_pipeline/05_joint_model.py`
- `notebooks/06_paper_pipeline/06_fig5_joint_model.ipynb`
- `notebooks/06_paper_pipeline/07_fig4_coherent_shift.ipynb`
- `results/stats/independent_bayesian_*.csv`
- `results/stats/joint_correlated_*.csv`
- `results/stats/joint_model_*.csv`
- `results/stats/joint_choice_vigor_*.csv`
- `results/stats/mcmc_*.csv`
- `results/stats/choice_vigor_dissociation_*.csv`

### `_deprecated/root_experiments/` — Root-Level Exploration Scripts

**What these were:** One-off experiments exploring different vigor functions (power, sigmoid, log). Dead ends.

Files to move:

- `run_v8_log_vigor.py`
- `run_v9_power_vs_sigmoid.py`
- `sequential_fit.py`

### `_deprecated/old_drafts/` — Superseded Paper Drafts

**What these were:** Earlier paper framings before the EVC pivot.

Files to move:

- `drafts/main.md`
- `drafts/paper.md`
- `drafts/discovery_results.md`
- `drafts/discovery_results_with_figs.html`
- `drafts/preregistration.md` (old H1-H7 prereg for k,β,α,δ model)
- `drafts/prereg_working.html`
- `drafts/prereg_with_figs.html`
- `drafts/EffortThreat Draft_22326.pdf`
- `drafts/EXAMPLE_discovery.docx`

### `_deprecated/old_vigor_analysis/` — Dead-End Vigor Explorations

Files to move:

- `notebooks/03_vigor_analysis/11_vigor_ode.ipynb` (explicitly marked dead end)
- `notebooks/06_paper_pipeline/11_vigor_timecourse_v1.ipynb`
- `notebooks/06_paper_pipeline/11_vigor_timecourse_v2.ipynb`

### `_deprecated/old_plotting/` — Superseded Plot Scripts

Files to move:

- `scripts/plotting/plot_h1_figure_v2.py`
- `scripts/plotting/plot_h5_figure.py`
- `scripts/plotting/plot_h6_figure.py`
- `scripts/plotting/plot_h7_figure.py`

---

## UNCERTAIN (Needs Noah's Decision)

These files might be useful depending on final paper scope:

| File                                                                | Question                                                      |
| ------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `scripts/plotting/plot_h1_figure.py`, `plot_h1_panels.py`           | Keep if basic behavioral effects (H1) are still in the paper? |
| `scripts/plotting/plot_h4_figure.py`                                | Keep if clinical results section uses this format?            |
| `scripts/analysis/run_h1_tests.py`, `run_h1_lmm_tests.py`           | Keep if model-free behavioral tests are in the paper?         |
| `scripts/analysis/run_h3_h4_tests.py`, `run_h3_optimality_tests.py` | Keep for optimality analysis?                                 | --> DROP                                                                                |
| `scripts/run_affect_survival.py`                                    | Keep if affect analyses use this pipeline?                    | --> need new version                                                                    |
| `scripts/run_factor_analysis.py`                                    | Keep — factor analysis is still potentially relevant          |
| `notebooks/03_vigor_analysis/06-07, 10, 12-13`                      | Keep if vigor sub-analyses are in the paper?                  | --> drop and rewrite using the properly handled vigor the way our EVC model handles it. |
| `notebooks/04_psych_analysis/01-02, 04, 07-08`                      | Keep if corresponding analyses are in the paper?              | --> drop we'll use new psych                                                            |
| `notebooks/06_paper_pipeline/01-03, 08-09, 10-11`                   | Keep if these pipeline steps feed current figures?            | -->drop this is old                                                                     |
| `drafts/nature_comms_proposal.md`                                   | Keep for reference?                                           | --> drop                                                                                |
| `drafts/presentation.html`                                          | Keep for talks?                                               | --> drop                                                                                |
| `results/stats/affect_*.csv`, `anxiety_vigor_*.csv`                 | Keep if affect results are in paper?                          | --> will need to recalculate                                                            |
| `results/stats/h*_results.json`                                     | Keep if H-test results are referenced?                        | --> will need to redo.                                                                  |

---

## Post-Cleanup Actions

1. Update `instructions/CLAUDE.md` — remove references to deprecated files/notebooks
2. Update `instructions/memory/pipeline_state.md` — mark deprecated notebooks
3. Update `instructions/memory/MEMORY.md` — reflect new organization
4. Create `_deprecated/README.md` explaining why files were moved and when
