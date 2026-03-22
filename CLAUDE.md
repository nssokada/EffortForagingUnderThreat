# EffortForagingUnderThreat — Project Guide for Claude

## Working Protocol

Follow this protocol on every prompt, without exception.

### Step 1 — Read memory before doing anything
At the start of each conversation or task, read the memory index and all relevant files:

1. Read `instructions/memory/MEMORY.md` (index of all memory files)
2. Read whichever specific files are relevant to the task:
   - `active_issues.md` — what is currently broken or blocked
   - `pipeline_state.md` — what has been run and what outputs exist
   - `discoveries.md` — what the data actually shows
   - `task_design.md` — task mechanics, variable meanings, model structure
   - `open_questions.md` — unresolved questions
   - `session_history.md` — what was done in recent sessions

Do not rely on conversational memory alone. Memory files are authoritative — they may have been updated between sessions.

### Step 2 — Make a plan and present it
Before taking any action, write out a numbered plan and present it to the user. Include:
- What you intend to do and in what order
- What files you will read or modify
- Any assumptions or risks you are flagging

Wait for confirmation unless the task is clearly a single, scoped action (e.g., "fix this cell").

## Autonomous Mode
When the prompt contains the word AUTORUN, skip Step 2 (plan confirmation) 
and execute directly. Still read memory and still update memory files after.

### Step 3 — Execute with memory re-reads
Before executing each non-trivial step in the plan:
- Re-read any memory files directly relevant to that step
- Check whether the step is consistent with known pipeline state and discoveries
- Flag any conflicts before proceeding

### Why this matters
This project spans many sessions, many notebooks, and findings that evolve. Acting on stale context produces errors that are hard to debug. Reading memory is fast; fixing broken notebooks is slow.

---

## Paper

**Title:** A Common Computational Structure Integrates Effort and Threat Across Decision, Emotion, and Action
**Authors:** Noah Okada, Ketika Garg, Toby Wise, Dean Mobbs
**Status:** Pre-publication (draft in `drafts/main.md`)
**Target:** Nature Communications

**Core claim:** Humans integrate energetic cost and exposure-dependent threat into a unified survival-weighted subjective value signal that governs choice behavior, subjective affect (anxiety/confidence), and defensive action vigor. A hierarchical Bayesian model with exponential effort discounting + survival function + threat bias term best explains behavior across two large online samples (exploratory N=293, confirmatory N=350).

---

## Memory Files

All memory lives in `instructions/memory/`. Read `MEMORY.md` for the index.

| File | What it contains |
|------|-----------------|
| `MEMORY.md` | Index — start here every session |
| `active_issues.md` | Blocking bugs, missing outputs, tech debt — check before any notebook work |
| `pipeline_state.md` | Execution status of every notebook and script; which outputs exist |
| `discoveries.md` | All empirical findings: model fits, vigor effects, affect results, null results |
| `task_design.md` | Full task mechanics, reward structure, predator dynamics, probe design, variable mappings |
| `open_questions.md` | Unresolved theoretical, methodological, and statistical questions |
| `session_history.md` | Chronological log of what was done each session |
| `project_goal.md` | Submission target, core claim, what remains to be done |

---

## Repository Structure

```
EffortForagingUnderThreat/
├── instructions/
│   ├── CLAUDE.md                    ← this file
│   └── memory/                      ← persistent memory across sessions
│       ├── MEMORY.md
│       ├── active_issues.md
│       ├── discoveries.md
│       ├── open_questions.md
│       ├── pipeline_state.md
│       ├── project_goal.md
│       ├── session_history.md
│       └── task_design.md
├── data/
│   ├── exploratory_350/
│   │   ├── raw/                     ← raw JSON per participant (Prolific)
│   │   └── processed/
│   │       ├── stage5_filtered_data_20260317_094210/  ← ACTIVE (N=293)
│   │       │   ├── behavior.csv     ← 13,185 trials
│   │       │   ├── psych.csv        ← psychiatric battery scores
│   │       │   ├── feelings.csv     ← probe ratings
│   │       │   └── subject_mapping.csv
│   │       └── vigor_processed/     ← vigor pipeline outputs
│   │           ├── smoothed_vigor_ts.parquet
│   │           ├── phase_trial_metrics.parquet
│   │           └── subject_vigor_table.csv
│   └── confirmatory_350/
│       └── raw/                     ← not yet preprocessed
├── scripts/
│   ├── preprocessing/               ← 5-stage preprocessing pipeline
│   ├── modeling/                    ← hierarchical Bayesian model library
│   │   ├── base_model.py
│   │   ├── models.py
│   │   ├── fitter.py
│   │   └── ppc.py
│   ├── analysis/
│   │   └── bayesian_regression.py
│   └── plotting/
│       └── plotter.py
├── notebooks/
│   ├── 01_preprocessing/
│   │   ├── 01_run_pipeline.ipynb    ✅
│   │   └── 02_data_prep.ipynb       ✅
│   ├── 02_choice_modeling/
│   │   ├── 01_fit_compare_ppc.ipynb ✅
│   │   └── 02_parameter_recovery.ipynb  ⚠️ not run on N=293
│   ├── 03_vigor_analysis/
│   │   ├── 01_single_trial_visualization.ipynb  ✅
│   │   ├── 02_kernel_smoothing.ipynb            ✅
│   │   ├── 03_tonic_phasic_decomposition.ipynb  ✅
│   │   ├── 04_phase_extraction.ipynb            ✅
│   │   ├── 05_subject_features.ipynb            ✅
│   │   ├── 06_choice_vigor_mapping.ipynb        ✅
│   │   ├── 07_clinical_prediction.ipynb         ❌ blocked (needs EFA output)
│   │   ├── 08_parameter_dissociation.ipynb      ✅
│   │   ├── 09_final_stats.ipynb                 ✅
│   │   ├── 10_pls_vigor_params.ipynb            ✅
│   │   └── 11_vigor_ode.ipynb                   ✅ (dead end — do not use)
│   ├── 04_psych_analysis/
│   │   ├── 01_bayesian_mental_health_regressions.ipynb  ⚠️ status unknown
│   │   ├── 02_psychological_analysis.ipynb              ⚠️ status unknown
│   │   ├── 03_affect_survival.ipynb                     ✅ (NB12)
│   │   └── 04_anxiety_vigor_coupling.ipynb              ✅ (NB13)
│   └── 05_figures/
│       └── 01_publication_figures.ipynb  ⚠️ status unknown
├── results/
│   ├── model_fits/exploratory/
│   │   └── FET_Exp_Bias_fit.pkl     ← 217 MB, N=293 MCMC fit
│   └── stats/                       ← statistical output CSVs (see pipeline_state.md)
└── drafts/
    ├── main.md                      ← working paper draft
    └── prereg.md
```

---

## Analysis Pipeline

### Step 1 — Preprocessing
- **Script:** `scripts/preprocessing/pipeline.py`
- **Output:** `data/exploratory_350/processed/stage5_filtered_data_*/`
- **Key files:** `behavior.csv`, `psych.csv`, `feelings.csv`
- **Status:** ✅ Complete for exploratory sample (N=293)

### Step 2 — Choice Modeling
- **Model:** FETExponentialBias — exponential discounting + survival function + threat bias
- **Inference:** NumPyro HMC/NUTS, 4 chains × 1000 warmup + 1000 samples, target_accept=0.95
- **Fit quality (N=293):** WAIC=12,063, R²=0.454, AUC=0.912, Accuracy=82.5%
- **Subject parameters:** z (hazard sensitivity), κ (effort discounting), β (threat bias)
- **Output:** `results/stats/FET_Exp_Bias_*.csv`

### Step 3 — Vigor Analysis
- **Input:** keypress timeseries → kernel-smoothed → phase-extracted
- **Key output:** `vigor_processed/smoothed_vigor_ts.parquet` (3.9M rows, 20Hz)
- **Phase metrics:** `phase_trial_metrics.parquet` (trial = global event index 0–80)
- **Alignment note:** `trial` in vigor files = global event index, same as `trialNumber` in feelings.csv

### Step 4 — Psychological Analysis
- **NB12:** S_probe → anxiety/confidence LMM; state-trait decomposition; cross-domain correlations
- **NB13:** Anxiety × vigor coupling; functional regression (params → vigor(t)); all affect→vigor null

---

## Key Model Parameters

| Param | Name | Meaning |
|---|---|---|
| `z` | Hazard sensitivity | Nonlinearity of distance → danger scaling; z>1 = convex, z<1 = concave |
| `k` / `κ` | Effort discounting | How strongly effort cost reduces reward value |
| `β` | Threat bias | Residual threat aversion beyond survival-weighted EV |
| `τ` | Inverse temperature | Choice stochasticity (population-level only) |

**Parameter files:** `results/stats/FET_Exp_Bias_{z,k,beta}_params.csv`
**Column note:** These files use `subject` (not `subj`) and `mean` for the posterior mean. Rename on load.

---

## Task Design Summary

See `instructions/memory/task_design.md` for full details. Key facts:

- **Arena:** Circular, top-down view. Safe zone at center. Cookies at radial distances.
- **Choice:** Click to select cookie, then press S+D+F repeatedly to transport back to center
- **Effort:** Press rate → movement speed (≥100%/≥50%/≥25%/<25% = full/half/quarter/zero speed)
- **Rewards:** R_H = 5 pts (fixed), R_L = 1 pt (fixed). Capture penalty C = 5 pts + lose current cookie.
- **Threat:** Probability T ∈ {0.1, 0.5, 0.9}. On attack trials: predator appears at encounterTime (slow approach, 0.5 units/sec), then strikes at Gaussian-distributed time (4× player speed).
- **Distance:** D ∈ {1, 2, 3} = {5, 7, 9} game units. Drives both effort duration AND survival probability.
- **Probes:** Forced-choice trials (identical options). Rating collected after click, before pressing (game pauses). Prospective judgment about current trial demands.
- **Event stream:** 3 blocks × 27 trials = 81 total events (0–80). 15 regular + 12 probe per block = 45 behavioral + 36 probe total. This is why `trial` index goes 0–80.

---

## Data Notes

- **`trial` indexing:** Global event-stream index (0–80), shared across `phase_trial_metrics.parquet`, `smoothed_vigor_ts.parquet`, and `feelings.csv` (`trialNumber`)
- **`encounterTime`:** Set for ALL trials (attack and non-attack). On non-attack trials = scheduled but never fired.
- **Probe schedules:** 7 unique probe schedules across subjects. Always align per-subject.
- **Questionnaires:** DASS-21, PHQ-9, OASIS, STAI-T, AMI, MFIS, STICSA. Z-scored before analysis.
- **Conda env:** `effort_foraging_threat`. Python at `/opt/anaconda3/envs/effort_foraging_threat/bin/python3.11`. Kernel name for nbconvert: `python3`.

## Modeling Stack

- Inference: NumPyro (HMC/NUTS), 4 chains, 1000 warmup + 1000 sampling iterations
- Target acceptance: 0.95, max tree depth: 10
- Model comparison: WAIC (pointwise log-likelihoods)
