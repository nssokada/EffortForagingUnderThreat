# Data & Directory Overview

## Repository Structure

```
EffortForagingUnderThreat/
├── data/                          689 MB — raw + processed data
├── results/                        40 MB — stats, figures, tables
├── drafts/                        1.4 MB — paper drafts, prereg, hypotheses
├── scripts/                       1.3 MB — modeling, preprocessing, analysis
├── notebooks/                     9.3 MB — analysis pipeline (Jupyter + Python)
├── plans/                          76 KB — analysis plans
├── instructions/memory/                  — persistent memory for Claude
├── MODEL.md                              — current model (3-param v2) documentation
├── MODEL_2plus2.md                       — superseded model documentation
├── MODELING_DECISIONS.md                 — 15 modeling decisions with rationale
├── CLAUDE.md                             — project instructions
└── survival_landscape_plan.md            — survival function upgrade plan
```

---

## Data Files

### Primary Data (`data/exploratory_350/processed/stage5_filtered_data_20260320_191950/`)

All data is post-QC (5-stage filtering pipeline). N=293 subjects from 350 recruited.

#### `behavior.csv` — 13,185 rows × 10 cols (678 KB)
The clean choice-trial dataset. One row per free-choice trial (45 per subject).

| Column | Type | Description |
|--------|------|-------------|
| participantID | string | Prolific ID |
| trial | int | Trial index (0-80, shared with vigor data) |
| threat | float | Threat probability: 0.1, 0.5, or 0.9 |
| effort_L | float | Low-effort cookie demand (always 0.4) |
| effort_H | float | High-effort cookie demand: 0.6, 0.8, or 1.0 |
| distance_L | int | Low-effort distance (always 1) |
| distance_H | int | High-effort distance: 1, 2, or 3 |
| choice | int | 1 = chose heavy, 0 = chose light |
| outcome | int | 0 = escaped, 1 = captured |
| subj | int | Subject index (1-293) |

**Design:** 3 threat × 3 distance × 5 repetitions = 45 trials. Perfectly balanced: 5 trials per cell per subject. Distance confounded with effort: D=1→E=0.6, D=2→E=0.8, D=3→E=1.0.

#### `behavior_rich.csv` — 23,733 rows × 85 cols (395 MB)
ALL trial events (choice + probe), with raw keypress data and timing. One row per event.

**Key columns beyond behavior.csv:**

| Column | Description |
|--------|-------------|
| type | 1=free choice, 5=probe(heavy assigned), 6=probe(light assigned) |
| trialCookie_weight | 3.0=heavy, 1.0=light (what cookie was chosen/assigned) |
| calibrationMax | Subject's max presses per 10s (individual calibration) |
| alignedEffortRate | JSON array of keypress timestamps (raw data for vigor) |
| isAttackTrial | 1=predator attacked, 0=no attack |
| trialEndState | "escaped", "captured", or "free" (incomplete) |
| encounterTime | When predator appeared (seconds from trial start) |
| strike_time | When predator struck (actual, not scheduled) |
| predatorAttackTime | Scheduled strike time (fixed per distance: 5, 7, 10s) |
| trialEscapeTime | When participant reached safety (if escaped) |
| trialCaptureTime | When capture occurred (if captured) |
| startDistance | Starting distance in game units (5, 7, or 9) |
| mean_preEncounter_effort | Pre-encounter mean press rate |
| mean_postEncounter_effort | Post-encounter mean press rate |

**Trial types:** 45 free choice (type=1) + 36 probes (type=5,6) = 81 per subject. Probes assign identical cookies (50/50 heavy/light) with affect ratings.

#### `feelings.csv` — 10,546 rows × 16 cols (1.5 MB)
Probe trial affect ratings. One row per rating.

| Column | Description |
|--------|-------------|
| questionLabel | "anxiety" or "confidence" |
| response | 0-7 Likert scale |
| threat | Threat probability on this probe |
| trialNumber | Global event index (matches trial in behavior_rich) |
| trialCookie_weight | 3.0=heavy, 1.0=light |

**Structure:** 18 anxiety + 18 confidence ratings per subject. Collected after cookie selection, before pressing begins (prospective judgment).

#### `psych.csv` — 293 rows × 27 cols (69 KB)
Psychiatric battery scores. One row per subject.

| Instrument | Columns | What it measures |
|-----------|---------|-----------------|
| DASS-21 | DASS21_Stress, _Anxiety, _Depression, _Total | Depression, anxiety, stress |
| PHQ-9 | PHQ9_Total | Depression severity |
| OASIS | OASIS_Total | Anxiety severity |
| STAI | STAI_State, STAI_Trait | State and trait anxiety |
| AMI | AMI_Behavioural, _Social, _Emotional, _Total | Apathy |
| MFIS | MFIS_Physical, _Cognitive, _Psychosocial, _Total | Fatigue |
| STICSA | STICSA_Total | Cognitive/somatic anxiety |

Each also has an _RT column (response time). All subscale scores z-scored before analysis.

#### `subject_mapping.csv` — 293 rows
Maps participantID (Prolific) → subj (integer index 1-293).

#### `participant_qc.csv` — 341 rows
QC flags for all 341 pre-exclusion participants (293 kept + 48 excluded). Columns: kept, present_in_behavior/behavior_rich/subjective/mental_health, escape_rate.

---

### Vigor Timeseries (`data/exploratory_350/processed/vigor_processed/`)

#### `smoothed_vigor_ts.parquet` — ~3.9M rows × 19 cols (48 MB)
20 Hz kernel-smoothed vigor timeseries. One row per timepoint per trial.

| Column | Description |
|--------|-------------|
| subj | Subject index |
| trial | Global event index (0-80) |
| t | Time within trial (seconds) |
| r_hat | Smoothed press rate (kernel estimate) |
| vigor_norm | Normalized vigor (proportion of calibrated max) |
| vigor_resid | Cookie-type-centered residual vigor |
| threat | Threat probability |
| isAttackTrial | Whether predator attacked |
| encounterTime | Predator appearance time |

**Note:** 20 Hz is oversampled relative to the ~5 Hz raw keypress rate. Each trial has ~100-400 timepoints depending on duration. Used primarily for encounter dynamics (within-trial alignment to predator appearance).

---

### Confirmatory Sample (`data/confirmatory_350/raw/`)
N=350 raw JSON files from Prolific. Collected but NOT preprocessed or analyzed. Identical task design to exploratory.

---

## Results Files

### Model Parameters (`results/stats/`)

**Current model (3-param v2: k + β + cd):**
- `oc_evc_3param_v2_params.csv` — Per-subject k, β, cd (N=293)
- `oc_evc_3param_v2_population.csv` — τ, p_esc, σ_motor, ce_vigor, σ_v
- `oc_evc_3param_v2_conditions.csv` — Predicted P(heavy) per condition
- `oc_evc_3param_v2_recovery.csv` — Parameter recovery (150 simulated subjects)
- `oc_evc_3param_v2_recovery_summary.csv` — Recovery r values

**Branch B (frac_full + Gaussian CDF survival):**
- `branchB_params.csv` — Per-subject k, β, cd (N=293)
- `branchB_population.csv` — α, τ, σ_v, v_full, remaining_frac, buffer, p_floor
- `branchB_recovery.csv` — Recovery (150 simulated subjects)

**Superseded 2+2 model (ce + cd with γ, ε):**
- `oc_evc_final_params.csv` — Per-subject ce, cd (45-trial)
- `oc_evc_final_81_params.csv` — Per-subject ce, cd (81-trial)
- `oc_evc_final_population.csv` / `*_81_population.csv` — γ, ε, τ, etc.
- `evc_model_comparison_final.csv` — 7-model comparison (BIC, r²)

**Earlier model iterations** (24 specifications tested):
- `oc_evc_gamma_*.csv`, `oc_evc_lqr_*.csv`, `oc_evc_theta_*.csv`, etc.

### Pipeline Results
- `3param_v2_pipeline_results.csv` — All stats from the 3-param pipeline
- `3param_v2_routes.csv` — 6×6 correlation matrix (predictors × outcomes)
- `3param_v2_optimal_policy.csv` — EV-optimal choice per condition
- `3param_v2_deviations.csv` — Per-subject deviation metrics
- `deviation_param_associations.csv` — ce/cd → deviation stats (2+2 model)
- `residual_suboptimality.csv` — Affect → residual overcaution (2+2 model)
- `critical_checks.csv` — Press rate × survival, heuristic comparison, etc.
- `evc_vigor_dynamics.csv` — Encounter dynamics stats
- `evc_bayesian_clinical.csv` — Bayesian regression results (bambi/PyMC)
- `evc_ml_clinical.csv` — Machine learning clinical prediction (null)
- `four_routes.csv` — Independence matrix (2+2 model)
- `optimal_policy.csv` — Per-condition optimal (2+2 model)
- `per_subject_deviations.csv` — Per-subject deviations (2+2 model)
- `exposure_time_params.csv` / `*_population.csv` — Exposure-time model params

### Figures (`results/figs/paper/`)
~40 figures including:
- `fig_3param_pipeline.png` — 6-panel figure for 3-param model
- `fig_deviations.png` — ce → overcaution, cd → vigor gap
- `fig_residual_affect.png` — Affect → residual suboptimality
- `fig_critical_checks.png` — Press rate × survival, heuristic comparison
- `fig_bayesian_clinical.png` — Clinical associations
- Various earlier figures from superseded models

---

## Drafts

| Draft | Title / Focus | Model |
|-------|--------------|-------|
| `draft011/paper.md` | **CURRENT**: Three separable cost signals | 3-param v2 (k, β, cd) |
| `draft010/paper.md` | Integrating effort and threat | 2+2 (ce, cd, γ, ε) |
| `draft009/paper.md` | With reviewer revisions | 2+2 |
| `draft005-008/` | Iterative revisions | 2+2 |
| `draft004/paper.md` | First full draft | 2+2 |
| `draft003/` | EVC-LQR exploration | Earlier |
| `draft002/` | Paper outline | Earlier |
| `prereg_evc_aspredicted.md` | **CURRENT** preregistration (H1-H6) | 3-param v2 |
| `discovery_results_evc.md` | All H1-H4 detailed results | 2+2 |
| `hypotheses/H1-H5.md` | Per-hypothesis writeups | 2+2 |

---

## Scripts

### Modeling (`scripts/modeling/`)
| File | Description |
|------|-------------|
| `evc_3param_v2.py` | **CURRENT** model (k + β + cd, no γ/ε) |
| `evc_3param_v2_recovery.py` | Recovery for current model |
| `evc_branchB.py` | Branch B (frac_full + Gaussian CDF survival) |
| `evc_branchB_recovery.py` | Recovery for Branch B |
| `evc_final_2plus2.py` | Superseded 2+2 model |
| `evc_3param.py` | Failed 3-param with S in choice |
| `evc_mcmc.py` | MCMC model for validation |
| `evc_model_comparison.py` | 7-model comparison harness |
| Others | 24 earlier model specifications |

### Pipeline (`notebooks/07_evc_pipeline/`)
| File | Description |
|------|-------------|
| `18_3param_pipeline.py` | **CURRENT** full pipeline (3-param v2) |
| `19_exposure_time_model.py` | Exposure-time survival exploration |
| `14_optimal_policy.py` | Optimal policy derivation (2+2) |
| `15_deviation_analysis.py` | Parameter → deviation (2+2) |
| `16_residual_suboptimality.py` | Affect + residual analysis (2+2) |
| `17_critical_checks.py` | Press rate × survival, heuristics |
| `13_vigor_dynamics.py` | Encounter dynamics |
| `11_bayesian_clinical.py` | Bayesian clinical regressions |
| `12_ml_clinical.py` | ML clinical prediction |
| Others | PPC, profiles, affect analyses |

### Preprocessing (`scripts/preprocessing/`)
5-stage pipeline: raw JSON → trial processing → subjective → mental health → filtering

### MCMC (`scripts/mcmc/`)
NUTS validation scripts (4 chains). Ready for GPU but not run on this machine.

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Exploratory sample | N=293 (from 350 recruited, 83.7% retention) |
| Confirmatory sample | N=350 (collected, not analyzed) |
| Choice trials per subject | 45 (5 per condition × 9 conditions) |
| Probe trials per subject | 36 (18 anxiety + 18 confidence) |
| Total events per subject | 81 (3 blocks × 27) |
| Conditions | 3 threat × 3 distance (confounded with effort) |
| Keypress rate | ~4-5 Hz (variable) |
| Smoothed vigor rate | 20 Hz (oversampled) |
| Psychiatric instruments | 7 (DASS-21, PHQ-9, OASIS, STAI, AMI, MFIS, STICSA) |
| Model specifications tested | 24+ |
| Paper drafts | 11 |
