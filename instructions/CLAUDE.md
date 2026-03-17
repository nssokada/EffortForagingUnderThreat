# EffortForagingUnderThreat — Project Guide for Claude

## Paper

**Title:** A Common Computational Structure Integrates Effort and Threat Across Decision, Emotion, and Action
**Authors:** Noah Okada, Ketika Garg, Toby Wise, Dean Mobbs
**Status:** Pre-publication (draft in `drafts/`)

**Core claim:** Humans integrate energetic cost and exposure-dependent threat into a unified survival-weighted subjective value signal that governs choice behavior, subjective affect (anxiety/confidence), and defensive action vigor. A hierarchical Bayesian model with exponential effort discounting + survival function + threat bias term best explains behavior across two large online samples (exploratory N=293, confirmatory N=350).

---

## Repository Structure

```
EffortForagingUnderThreat/
├── instructions/
│   └── CLAUDE.md               ← this file
├── data/
│   ├── exploratory_350/
│   │   ├── raw/                ← raw JSON per participant (Prolific)
│   │   └── processed/          ← pipeline outputs
│   └── confirmatory_350/
│       ├── raw/                ← to be added
│       └── processed/
├── scripts/
│   ├── preprocessing/          ← 5-stage preprocessing pipeline
│   │   ├── config.py           ← paths, exclusion thresholds
│   │   ├── pipeline.py         ← orchestrates all stages
│   │   ├── stage1_raw_processing.py
│   │   ├── stage2_trial_processing.py   ← trial-level features, vigor extraction
│   │   ├── stage3_subjective_processing.py  ← anxiety/confidence probes
│   │   ├── stage4_mental_health.py      ← questionnaire scoring (DASS-21, PHQ-9, OASIS, STAI-T, AMI, MFIS)
│   │   ├── stage5_filtering.py          ← exclusion criteria
│   │   └── utils.py
│   ├── modeling/               ← hierarchical Bayesian choice model library
│   │   ├── base_model.py       ← abstract base class
│   │   ├── models.py           ← model variants (effort-only, threat-only, combined, bias-extended)
│   │   ├── fitter.py           ← NumPyro HMC/NUTS fitting
│   │   └── ppc.py              ← posterior predictive checks
│   ├── analysis/
│   │   └── bayesian_regression.py  ← Bayesian regressions (model params × psych measures)
│   └── plotting/
│       └── plotter.py          ← publication-level figures
├── notebooks/
│   ├── 01_preprocessing/
│   │   ├── 01_run_pipeline.ipynb
│   │   ├── 02_data_prep.ipynb
│   │   └── 03_data_prep_stage1_analysis_table.ipynb
│   ├── 02_choice_modeling/
│   │   ├── 01_fit_compare_ppc.ipynb        ← fit models, WAIC comparison, PPC
│   │   ├── 02_fit_compare_ppc_with_plotter.ipynb
│   │   └── 03_parameter_recovery.ipynb
│   ├── 03_vigor_analysis/
│   │   ├── 01_single_trial_visualization.ipynb
│   │   ├── 02_kernel_smoothing.ipynb
│   │   ├── 03_tonic_phasic_decomposition.ipynb
│   │   ├── 04_phase_extraction.ipynb
│   │   ├── 05_subject_features.ipynb
│   │   ├── 06_choice_vigor_mapping.ipynb
│   │   ├── 07_clinical_prediction.ipynb
│   │   ├── 08_parameter_dissociation.ipynb
│   │   └── 09_final_stats.ipynb
│   ├── 04_psych_analysis/
│   │   ├── 01_bayesian_mental_health_regressions.ipynb
│   │   └── 02_psychological_analysis.ipynb
│   └── 05_figures/
│       └── 01_publication_figures.ipynb
├── results/
│   ├── model_fits/             ← NumPyro fit outputs (fit_params.json per run)
│   │   ├── v1_initial/
│   │   ├── v2_new/
│   │   ├── v3_new_priors_diffform/
│   │   ├── v4_new_priors_diffform2/
│   │   └── v5_parallel/
│   └── stats/                  ← statistical output tables
├── figs/                       ← all figures (intermediate + publication)
└── drafts/
    └── EffortThreat Draft_22326.pdf
```

---

## Analysis Pipeline

Run each stage in order. Both exploratory and confirmatory samples use the same pipeline.

### Step 1 — Preprocessing (`notebooks/01_preprocessing/`)
- **Input:** `data/{sample}/raw/*.json`
- **Stages:** raw parsing → trial features → subjective probes → mental health scoring → exclusions
- **Output:** `data/{sample}/processed/` (trial-level and subject-level tables)
- **Key exclusions:** incomplete sessions, implausible keypress rates, invalid predator dynamics, escape rate < 35%

### Step 2 — Choice Modeling (`notebooks/02_choice_modeling/`)
- **Input:** processed trial data
- **Model:** Hierarchical Bayesian (NumPyro, HMC/NUTS)
  - Survival function: `S = exp(-T · D^z)` with subject-level `z` (hazard sensitivity)
  - Effort discounting: `R_eff = R · f(E; k)` with subject-level `k`; best form = exponential
  - Subjective value: `SV = R_eff · S - (1-S) · C`
  - Choice rule: softmax on `SV_H - SV_L`, with optional bias term `β · T`
- **Model variants:** 4 effort-only × 4 discount forms, threat-only, 4 combined, bias-extended
- **Comparison:** WAIC (lower = better); winning model = exponential + survival + bias
- **Output:** `results/model_fits/`, posterior parameters per subject

### Step 3 — Vigor Analysis (`notebooks/03_vigor_analysis/`)
- **Input:** processed trial data (press-rate timeseries)
- **Key analyses:**
  - Kernel smoothing of press-rate signal
  - Tonic/phasic decomposition
  - Phase extraction (pre-encounter vs. post-encounter)
  - Subject-level vigor features
  - Choice–vigor coupling (survival × z interaction)
  - Clinical prediction (model params → vigor)
  - Parameter dissociation
- **Output:** `results/stats/`, `figs/`

### Step 4 — Psychological Analysis (`notebooks/04_psych_analysis/`)
- **Input:** processed questionnaire data + fitted model parameters
- **Key analyses:**
  - Survival probability → trial-level anxiety/confidence (mixed-effects)
  - Model params (z, k, β) × DASS-21/PHQ-9/OASIS/STAI-T/AMI/MFIS
  - Individual differences in emotional reactivity
- **Output:** `results/stats/`, `figs/`

### Step 5 — Publication Figures (`notebooks/05_figures/`)
- **Input:** all processed data + model results
- Uses `scripts/plotting/plotter.py`
- **Output:** `figs/`

---

## Key Model Parameters
| Param | Name | Meaning |
|---|---|---|
| `z` | Hazard sensitivity | How steeply distance scales perceived danger |
| `k` | Effort discounting | How strongly effort reduces reward value |
| `β` | Threat bias | Residual threat sensitivity beyond expected value |
| `τ` | Inverse temperature | Choice stochasticity (population-level) |

---

## Data Notes
- Two samples: **exploratory** (N=350 recruited, N=293 after exclusions) and **confirmatory** (N=350, to be added)
- Raw data: one JSON per participant from Prolific
- Questionnaires: DASS-21, PHQ-9, OASIS, STAI-T, AMI, MFIS — z-scored before analysis
- Task: virtual foraging arena (Unity/WebGL), 3 blocks × 27 trials, predator dynamics

## Modeling Stack
- Inference: NumPyro (HMC/NUTS), 4 chains, 1000 warmup + 1000 sampling iterations
- Target acceptance: 0.95, max tree depth: 10
- Model comparison: WAIC (pointwise log-likelihoods)
