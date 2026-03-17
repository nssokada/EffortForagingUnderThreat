# A Common Computational Structure Integrates Effort and Threat Across Decision, Emotion, and Action

**Authors:** Noah Okada, Ketika Garg, Toby Wise, Dean Mobbs

This repository contains the analysis code for the paper. We developed a controlled foraging paradigm in which participants chose between options varying in reward, energetic effort, and exposure to threat. Using hierarchical Bayesian modeling, we show that a survival-weighted subjective value computation — integrating effort discounting and exposure-dependent danger — explains foraging choices and generalizes to predict trial-level subjective affect and defensive action vigor.

---

## Repository Structure

```
├── scripts/
│   ├── preprocessing/      # 5-stage data pipeline (raw JSON → analysis-ready tables)
│   ├── modeling/           # Hierarchical Bayesian choice model library (NumPyro/JAX)
│   ├── analysis/           # Bayesian regression scripts (model params × psych measures)
│   └── plotting/           # Publication-level figure utilities
├── notebooks/
│   ├── 01_preprocessing/   # Run preprocessing pipeline
│   ├── 02_choice_modeling/ # Fit models, WAIC comparison, PPC, parameter recovery
│   ├── 03_vigor_analysis/  # Vigor signal processing and statistical analyses
│   ├── 04_psych_analysis/  # Psychiatric symptom regressions
│   └── 05_figures/         # Publication figures
├── data/
│   ├── exploratory_350/    # N=350 exploratory sample (N=293 after exclusions)
│   │   ├── raw/            # Raw JSON files (one per participant, from Prolific)
│   │   └── processed/      # Pipeline outputs
│   └── confirmatory_350/   # N=350 confirmatory sample (to be added)
│       ├── raw/
│       └── processed/
├── results/
│   ├── model_fits/         # Saved NumPyro posterior samples
│   └── stats/              # Statistical output tables
├── figs/                   # All figures
├── drafts/                 # Paper drafts
├── instructions/
│   └── CLAUDE.md           # Full pipeline documentation (for AI-assisted development)
└── environment.yml         # Conda environment specification
```

---

## Setup

**1. Create the conda environment**

```bash
conda env create -f environment.yml
conda activate effort_foraging_threat
```

**2. Register the Jupyter kernel**

```bash
python -m ipykernel install --user \
  --name effort_foraging_threat \
  --display-name "Python (effort_foraging_threat)"
```

**3. Launch JupyterLab**

```bash
jupyter lab
```

---

## Analysis Pipeline

Run notebooks in order. Each stage depends on outputs from the previous.

### Stage 1 — Preprocessing (`notebooks/01_preprocessing/`)

Converts raw per-participant JSON files into structured analysis tables.

| Notebook | Description |
|---|---|
| `01_run_pipeline.ipynb` | Runs the full 5-stage preprocessing pipeline |
| `02_data_prep.ipynb` | Exploratory data preparation |
| `03_data_prep_stage1_analysis_table.ipynb` | Constructs the stage-1 analysis table |

Pipeline stages (in `scripts/preprocessing/`):
- **Stage 1** — Raw JSON parsing and trial extraction
- **Stage 2** — Trial-level feature computation and vigor signal extraction
- **Stage 3** — Subjective report processing (anxiety / confidence probes)
- **Stage 4** — Psychiatric questionnaire scoring (DASS-21, PHQ-9, OASIS, STAI-T, AMI, MFIS)
- **Stage 5** — Participant exclusion (incomplete sessions, keypress rate, predator dynamics, escape rate < 35%)

### Stage 2 — Choice Modeling (`notebooks/02_choice_modeling/`)

Fits a family of hierarchical Bayesian models and compares them via WAIC.

| Notebook | Description |
|---|---|
| `01_fit_compare_ppc.ipynb` | Model fitting, WAIC comparison, posterior predictive checks |
| `02_fit_compare_ppc_with_plotter.ipynb` | Same with publication-ready figure generation |
| `03_parameter_recovery.ipynb` | Parameter recovery analyses |

**Model library** (`scripts/modeling/`):

| Model | Description |
|---|---|
| `FETExponential` | Effort × threat, exponential discounting |
| `FETHyperbolic` | Effort × threat, hyperbolic discounting |
| `FETQuadratic` | Effort × threat, quadratic discounting |
| `FETLinear` | Effort × threat, linear discounting |
| `FETExponentialBias` | Exponential + threat-induced choice bias β *(winning model)* |
| `ThreatOnly` | Threat only (effort ablation) |
| `EffortOnly` | Effort only (threat ablation), 4 discount forms |

**Key parameters:**

| Parameter | Name | Interpretation |
|---|---|---|
| `z` | Hazard sensitivity | Nonlinearity of distance-dependent danger |
| `k` | Effort discounting | How strongly effort reduces reward value |
| `β` | Threat bias | Residual threat sensitivity beyond expected value |
| `τ` | Inverse temperature | Choice stochasticity (population-level) |

Inference: HMC/NUTS in NumPyro — 4 chains, 1,000 warmup + 1,000 sampling iterations, target acceptance 0.95.

### Stage 3 — Vigor Analysis (`notebooks/03_vigor_analysis/`)

Processes the keypressing vigor signal and relates it to model-derived survival estimates.

| Notebook | Description |
|---|---|
| `01_single_trial_visualization.ipynb` | Single-trial vigor trace inspection |
| `02_kernel_smoothing.ipynb` | Kernel smoothing of press-rate signal |
| `03_tonic_phasic_decomposition.ipynb` | Decompose vigor into tonic and phasic components |
| `04_phase_extraction.ipynb` | Extract pre- and post-encounter phase features |
| `05_subject_features.ipynb` | Compute subject-level vigor summaries |
| `06_choice_vigor_mapping.ipynb` | Map model-derived survival to trial-level vigor |
| `07_clinical_prediction.ipynb` | Predict vigor from model parameters |
| `08_parameter_dissociation.ipynb` | Dissociate effects of z, k, β on vigor |
| `09_final_stats.ipynb` | Final statistical tests for paper |

### Stage 4 — Psychological Analysis (`notebooks/04_psych_analysis/`)

Links model parameters and survival estimates to psychiatric symptom dimensions.

| Notebook | Description |
|---|---|
| `01_bayesian_mental_health_regressions.ipynb` | Bayesian regressions: model params × DASS-21/PHQ-9/OASIS/STAI-T/AMI/MFIS |
| `02_psychological_analysis.ipynb` | Survival estimates → trial-level anxiety and confidence |

### Stage 5 — Publication Figures (`notebooks/05_figures/`)

Generates all publication-ready figures using `scripts/plotting/plotter.py`.

---

## Data

Raw data consists of one JSON file per participant, collected via Prolific on a desktop browser. The task was implemented in Unity (2022.3.4f1) and deployed via WebGL.

- **Exploratory sample:** N=350 recruited, **N=293** after pre-registered exclusions
- **Confirmatory sample:** N=350 (to be added)

Raw data files are not included in this repository. Contact the corresponding author for access.

---

## Citation

> Okada, N., Garg, K., Wise, T., & Mobbs, D. (in prep). *A common computational structure integrates effort and threat across decision, emotion, and action.*
