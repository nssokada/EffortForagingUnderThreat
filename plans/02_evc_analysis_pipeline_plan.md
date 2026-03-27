# Plan 2: EVC Analysis Pipeline

**Goal:** Build a complete, organized analysis pipeline for the EVC+gamma paper, from model validation through preregistration and confirmatory replication.

**Paper:** "Expected Value of Control in Foraging Under Threat"
**Target:** Nature Communications
**Samples:** Exploratory N=293 (analyzed), Confirmatory N=350 (collected, not analyzed)

---

## Paper Story (5 Results Sections)

- **R1:** The EVC model captures foraging choice under threat
- **R2:** The same computation predicts motor vigor + probability weighting (γ)
- **R3:** Where humans sit relative to optimal (Pareto analysis)
- **R4:** Efficacy (ε) separates adaptive vs maladaptive threat sensitivity
- **R5:** Metacognitive miscalibration — confidence tracks choice, not vigor

---

## Phase 1: Model Validation (CRITICAL — Do First)

### 1.1 Parameter Recovery Simulation

**Priority:** Critical
**Script:** `scripts/analysis/evc_parameter_recovery.py`
**What:**

1. Draw 50 synthetic subjects from fitted population distribution
2. Simulate choice + vigor data using EVC+gamma generative model (45 trials/subject, full task design)
3. Re-fit EVC+gamma to synthetic data
4. Repeat for 10 synthetic datasets
   **Metrics:**

- Correlation between true and recovered params (target: r > 0.7 for each)
- Bias (mean recovery error)
- Coverage (fraction of 95% CIs containing truth)
- Recovery of population gamma
  **Output:** `results/stats/evc_parameter_recovery.csv`, `results/figs/paper/fig_s_parameter_recovery.png`
  **Key risk:** cd×ε correlation (r=+0.66) suggests potential identifiability issue. If recovery r < 0.6 for either, must restructure model.
  **Dependencies:** None

### 1.2 Posterior Predictive Checks

**Priority:** Critical
**Notebook:** `notebooks/07_evc_pipeline/02_ppc.ipynb`
**What:**

1. Sample 500 posterior draws from fitted guide
2. Generate predicted choices and vigor for all 13,185 trials
3. Plot:
   - Choice PPC: P(heavy) across 9 conditions (3T × 3D), predicted vs observed + 95% CIs
   - Vigor PPC: mean excess effort across 6 conditions (3T × 2 cookie), predicted vs observed
   - Individual PPC: predicted vs observed P(heavy) per subject (scatter)
   - Residual analysis by condition
     **Output:** `results/figs/paper/fig_ppc_evc.png` (main), `results/figs/paper/fig_s_ppc_detail.png` (supplement)
     **Dependencies:** None (uses existing fit)

### 1.3 Identifiability Analysis

**Priority:** Critical
**Notebook:** `notebooks/07_evc_pipeline/03_identifiability.ipynb`
**What:**

1. Pairwise posterior correlations for (c_effort, c_death, ε) — both raw and log space
2. Profile likelihood: fix one param at different values, refit, plot ELBO surface
3. Test: does cd-ε tradeoff affect behavioral predictions? (Fix cd high + ε low vs cd low + ε high, compare predicted behavior)
   **Output:** `results/stats/evc_identifiability.csv`, correlation matrices
   **Key question:** Is cd×ε = +0.66 a structural problem or a meaningful correlation?
   **Dependencies:** None

---

## Phase 2: Model Comparison Within EVC Family (CRITICAL)

### 2.1 Define and Fit EVC Variants

**Priority:** Critical
**Script:** `scripts/modeling/evc_model_comparison.py`

| Model             | Subject params       | Pop params               | Tests                                  |
| ----------------- | -------------------- | ------------------------ | -------------------------------------- |
| EVC_base          | c_effort, c_death    | tau, p_esc, σ_v, σ_motor | Baseline: no ε, no γ                   |
| EVC_eps           | + epsilon            | same                     | Does individual ε help?                |
| EVC_gamma         | + epsilon            | + gamma                  | Does population γ help? (current best) |
| EVC_gamma_ind     | + epsilon, + gamma_i | same                     | Should γ be per-subject?               |
| EVC_linear_effort | same as gamma        | same                     | Linear (not quadratic) u cost          |
| EVC_exp_survival  | same as gamma        | same                     | Exponential (not sigmoid) survival     |

Fit all with identical settings: SVI, 35k steps, Adam lr=0.002.

**Output:** `results/stats/evc_model_comparison.csv` (model, ELBO, BIC, n_params, choice_acc, vigor_r², gamma)
**Dependencies:** None

### 2.2 Model Comparison Figure

**Priority:** High
**Notebook:** `notebooks/07_evc_pipeline/04_model_comparison.ipynb`
**Output:** `results/figs/paper/fig_model_comparison.png`, `results/tables/table_model_comparison.csv`
**Dependencies:** Step 2.1

---

## Phase 3: Behavioral Analysis (HIGH)

### 3.1 Parameter Profile / Quadrant Analysis

**Priority:** High
**Notebook:** `notebooks/07_evc_pipeline/05_behavioral_profiles.ipynb`
**What:**

1. Log-space parameters: log(ce), log(cd), log(ε)
2. 2D scatter: log(cd) vs log(ε), colored by escape rate, earnings, OASIS
3. Median-split quadrants:
   - High cd + High ε = Adaptive vigilance
   - High cd + Low ε = Anxious helplessness
   - Low cd + High ε = Bold + vigorous
   - Low cd + Low ε = Disengaged
4. Compare quadrants on: earnings, escape rate, OASIS, AMI, PHQ-9
   **Input:** `oc_evc_gamma_params.csv`, `behavior.csv`, `psych.csv`
   **Output:** `results/stats/evc_quadrant_analysis.csv`, `results/figs/paper/fig_quadrants.png`
   **Dependencies:** Phase 1.3 (confirm identifiability)

### 3.2 Optimality / Pareto Analysis

**Priority:** High
**Script:** `scripts/plotting/plot_pareto_optimal.py` (exists, may need updates)
**What:**

1. Compute optimal choice and vigor per condition
2. Per-subject optimality deviation
3. Which parameter predicts deviation? (Hypothesis: low ε → worst vigor deviation)
4. Pareto frontier: earnings vs effort cost
   **Input:** `oc_evc_gamma_params.csv`, `behavior.csv`
   **Output:** `results/figs/paper/fig_pareto_optimal.png` (exists), `results/stats/evc_optimality.csv`
   **Dependencies:** Phase 2 (final model)

### 3.3 Simpson's Paradox with EVC Predictions

**Priority:** Medium
**Notebook:** `notebooks/07_evc_pipeline/06_simpsons_paradox.ipynb`
**What:** Show the EVC model naturally predicts the Simpson's paradox — flat marginal vigor but robust conditional vigor. The model explains WHY: choice reallocation shifts the composition.
**Output:** `results/figs/paper/fig_simpsons_evc.png`
**Dependencies:** Phase 1.2 (PPC predictions). (I DONT THINK THE SIMPSONS PARADOX IS A CORE FINDING)

---

## Phase 4: Affect and Metacognition (HIGH)

### 4.1 EVC-Derived Survival Predicts Affect

**Priority:** High
**Notebook:** `notebooks/07_evc_pipeline/07_affect.ipynb`
**What:**

1. Compute S from EVC for probe trials: S = (1 − T^γ) + ε_i × T^γ × p_esc
2. LMMs: `anxiety ~ S_z + (1 + S_z | subject)`, `confidence ~ S_z + (1 + S_z | subject)`
3. Test ε moderation: `anxiety ~ S_z * ε_z + (1 + S_z | subject)`
   **Input:** `feelings.csv`, `oc_evc_gamma_params.csv`
   **Output:** `results/stats/evc_affect_lmm.csv`
   **Dependencies:** Phase 2 (final model)

### 4.2 Metacognitive Miscalibration

**Priority:** High
**Notebook:** `notebooks/07_evc_pipeline/08_metacognition.ipynb`
**What:**

1. Per-subject choice quality (EV-optimality) and vigor quality (excess effort → survival)
2. Correlate confidence with choice quality vs vigor quality
3. **Prediction:** Confidence tracks choice (wrong channel), not vigor (right channel)
4. 2×2: {high/low choice} × {high/low vigor} — compare confidence
5. Do EVC params predict this? c_effort → confidence (via choice), ε → confidence (via vigor)?
   **Input:** `feelings.csv`, `behavior.csv`, `oc_evc_gamma_params.csv`
   **Output:** `results/stats/evc_metacognition.csv`, `results/figs/paper/fig_metacognition.png`
   **Dependencies:** Step 4.1

### 4.3 Calibration vs Discrepancy Double Dissociation

**Priority:** High
**Notebook:** `notebooks/07_evc_pipeline/09_double_dissociation.ipynb`
**What:**

1. **Calibration:** Within-subject r(anxiety, 1−S) — how well anxiety tracks danger
2. **Discrepancy:** Mean anxiety residual after regressing out S — excess anxiety
3. Calibration → performance (earnings, escapes) but NOT clinical
4. Discrepancy → clinical (OASIS, STAI, DASS) but NOT performance
5. Test orthogonality: calibration × discrepancy r ≈ 0
   **Input:** `feelings.csv`, `psych.csv`, `oc_evc_gamma_params.csv`, `behavior.csv`
   **Output:** `results/stats/evc_double_dissociation.csv`, `results/figs/paper/fig_double_dissociation.png`
   **Dependencies:** Steps 4.1, 4.2

---

## Phase 5: Clinical Analysis (HIGH)

### 5.1 Log-Space Clinical Correlations

**Priority:** High
**Notebook:** `notebooks/07_evc_pipeline/10_clinical.ipynb`
**What:**

1. Compute log(ce), log(cd), log(ε)
2. Correlate with all z-scored psychiatric subscales
3. FDR correction across tests within each parameter
4. Report: r, p_uncorrected, p_FDR, 95% CI, BF10
   **Expected:**

- log(ε) → OASIS: r ≈ −0.11 (marginal)
- log(cd) → AMI_Emotional: r ≈ +0.12 (check in log space)
- log(ce): likely null
  **Input:** `oc_evc_gamma_params.csv`, `psych.csv`
  **Output:** `results/stats/evc_clinical_log.csv`
  **Dependencies:** Phase 1.3

### 5.2 cd × ε Interaction

**Priority:** High
**Same notebook as 5.1**
**What:**

1. Multiple regression: OASIS ~ log(cd) + log(ε) + log(cd):log(ε)
2. Repeat for all clinical measures
3. **Hypothesis:** The interaction captures adaptive vs maladaptive threat — high cd + low ε predicts highest anxiety
4. Simple slopes / Johnson-Neyman plots
   **Output:** `results/stats/evc_interaction.csv`, `results/figs/paper/fig_interaction.png`
   **Dependencies:** Step 5.1

### 5.3 Factor Score Prediction

**Priority:** Medium
**Same notebook as 5.1**
**What:**

1. Regress 3 factor scores on log(ce), log(cd), log(ε), and interactions
2. PLS: 3 EVC params → 3 factors. CV R², permutation p.
   **Expected:** Small effects (R² < 0.05), interaction may help
   **Input:** `oc_evc_gamma_params.csv`, `psych_factor_scores.csv`
   **Output:** `results/stats/evc_factor_prediction.csv`
   **Dependencies:** Step 5.1

### 5.4 Effect Size Honesty

**Priority:** Medium
**Same notebook**
**What:**

1. Bayesian equivalence tests (ROPE −0.1 to 0.1)
2. BF10 for significant effects, BF01 for null effects
3. Context: compare to published comp psych effect sizes
4. Frame: "ε captures perceived controllability — a computational dimension with theoretical links to anxiety, not a clinical prediction tool"
   **Output:** Paragraph for discussion section
   **Dependencies:** Steps 5.1-5.3

---

## Phase 6: Preregistration (CRITICAL — After All Exploratory)

### 6.1 Write EVC Preregistration

**Priority:** Critical
**Output:** `drafts/preregistration_evc.md`
**Hypotheses to register:**

| H   | Test                                                 | Exploratory result | Threshold           |
| --- | ---------------------------------------------------- | ------------------ | ------------------- |
| H1a | Threat reduces heavy choice (LMM)                    | β < 0, t > 10      | p < 0.01            |
| H1b | Threat increases vigor conditioned on choice         | β > 0              | p < 0.05            |
| H1c | S predicts anxiety (−) and confidence (+)            | β = −1.84, +2.00   | \|t\| > 3           |
| H2a | EVC+γ choice accuracy > 75%                          | 81%                | > 75%               |
| H2b | EVC+γ vigor r² > 0.30                                | 0.504              | > 0.30              |
| H2c | EVC+γ outperforms EVC_base                           | ΔBIC ≈ 972         | ΔBIC > 100          |
| H3  | Parameter recovery r > 0.7                           | TBD                | r > 0.7             |
| H4a | γ < 1 (probability compression)                      | 0.283              | γ < 1               |
| H4b | cd×ε quadrant: adaptive vigilance has highest escape | TBD                | p < 0.05            |
| H5a | log(ε) → OASIS r < 0                                 | r = −0.11          | one-tailed p < 0.05 |
| H5b | log(cd) → AMI_Emotional r > 0                        | r ≈ +0.12          | one-tailed p < 0.05 |
| H5c | cd×ε interaction → clinical anxiety                  | TBD                | two-tailed p < 0.05 |
| H6a | Confidence tracks choice > vigor                     | TBD                | Δr > 0              |
| H7a | Calibration → performance, not clinical              | TBD                | p < 0.05 / p > 0.1  |
| H7b | Discrepancy → clinical, not performance              | TBD                | p < 0.05 / p > 0.1  |

**Exploratory (not registered):** Factor prediction, PLS, sensitization, apathy paradox, specific interaction patterns.

**Dependencies:** All of Phases 1-5

### 6.2 Lock Code

**What:** Tag repository as `v1.0-preregistration`, freeze analysis scripts
**Dependencies:** Step 6.1

---

## Phase 7: Confirmatory Pipeline (HIGH — Blocked on Data)

**Status:** Raw data directory `data/confirmatory_350/raw/` exists but is empty. Data must be transferred first.

### 7.1 Preprocess Confirmatory

**Script:** `scripts/preprocessing/pipeline.py` with confirmatory path
**Output:** `data/confirmatory_350/processed/stage5_*/`

### 7.2 Compute behavior_rich.csv

**What:** Enrich with alignedEffortRate, calibrationMax, etc.

### 7.3 Fit EVC+γ

**Script:** `scripts/modeling/oc_evc_gamma.py` with confirmatory data
**Output:** `results/stats/confirmatory_evc_gamma_params.csv`

### 7.4 Run All Registered Tests

**Script:** Automated script executing all H1-H7 tests from locked code
**Output:** `results/stats/confirmatory_hypothesis_tests.csv`

### 7.5 Replication Assessment

**Notebook:** Compare exploratory vs confirmatory effect sizes, compute replication BFs
**Output:** `results/figs/paper/fig_replication.png`

---

## Phase 8: Publication Figures

| Fig | Content                                                            | Phase | Status         |
| --- | ------------------------------------------------------------------ | ----- | -------------- |
| 1   | Task schematic + EVC equations + model comparison                  | 2     | Needs creation |
| 2   | Choice fit (9 conditions) + vigor fit (scatter, r²=0.50)           | 1.2   | Needs creation |
| 3   | γ probability weighting (T vs T^γ) + vigor calibration improvement | 2     | Needs creation |
| 4   | cd×ε quadrant: behavioral profiles + escape rates                  | 3.1   | Needs creation |
| 5   | Pareto optimal ("Where should you be?")                            | 3.2   | **Done**       |
| 6   | Metacognition: confidence tracks choice not vigor                  | 4.2   | Needs creation |
| 7   | Double dissociation: calibration→performance, discrepancy→clinical | 4.3   | Needs creation |
| S1  | Parameter recovery                                                 | 1.1   | Needs creation |
| S2  | Full PPC panels                                                    | 1.2   | Needs creation |
| S3  | Clinical correlation forest plots                                  | 5.1   | Needs creation |
| S4  | Simpson's paradox with EVC predictions                             | 3.3   | Needs creation |
| S5  | Confirmatory replication panels                                    | 7.5   | Blocked        |

---

## Execution Schedule

```
Week 1 (Validation — all independent, run in parallel):
  ├── 1.1 Parameter recovery
  ├── 1.2 PPC
  ├── 1.3 Identifiability
  └── 2.1 Model comparison (6 variants)

Week 2 (Core analyses — some dependencies):
  ├── 2.2 Model comparison figure (← 2.1)
  ├── 3.1 Behavioral profiles / quadrants (← 1.3)
  ├── 4.1 Affect-EVC LMMs (← 2.1)
  └── 5.1 Clinical correlations log-space (← 1.3)

Week 3 (Downstream analyses):
  ├── 3.2 Optimality / Pareto (← 2.1)
  ├── 3.3 Simpson's paradox (← 1.2)
  ├── 4.2 Metacognition (← 4.1)
  ├── 4.3 Double dissociation (← 4.1, 4.2)
  ├── 5.2 cd×ε interaction (← 5.1)
  └── 5.3-5.4 Factor prediction + effect sizes (← 5.1)

Week 4 (Preregistration):
  ├── 6.1 Write preregistration (← ALL exploratory)
  ├── 6.2 Lock code
  └── 8.* All figures

Week 5+ (Confirmatory — when data available):
  ├── 7.1-7.3 Preprocess + fit
  ├── 7.4 Hypothesis tests
  └── 7.5 Replication assessment
```

---

## Key Decision Points

1. **After Phase 1.3:** If cd×ε identifiability fails (recovery r < 0.6), must restructure — either fix ε at population level or add regularization. Blocks all clinical interpretation.

2. **After Phase 2.1:** If individual γ significantly outperforms population γ, adds a 4th subject param. Changes prereg and clinical analyses.

3. **After Phase 5:** If clinical effects don't survive FDR, pivot framing to "computational dimensions orthogonal to symptom clusters" rather than "params predict symptoms."

4. **Before Phase 7:** Confirmatory raw data must be transferred to repo. Confirm location with collaborators.

---

## New Directory Structure (Post-Cleanup)

```
notebooks/
  07_evc_pipeline/
    01_fit_evc_gamma.ipynb         ← fit the model (or reference script)
    02_ppc.ipynb                   ← posterior predictive checks
    03_identifiability.ipynb       ← parameter identifiability
    04_model_comparison.ipynb      ← EVC variant comparison
    05_behavioral_profiles.ipynb   ← quadrant analysis
    06_simpsons_paradox.ipynb      ← Simpson's with EVC
    07_affect.ipynb                ← S → anxiety/confidence LMMs
    08_metacognition.ipynb         ← confidence tracks choice not vigor
    09_double_dissociation.ipynb   ← calibration vs discrepancy
    10_clinical.ipynb              ← clinical correlations + interactions + factors

scripts/
  modeling/
    oc_evc_gamma.py               ← THE model
    evc_model_comparison.py       ← fit all EVC variants
    optimal_control.py            ← low-level utilities
  analysis/
    evc_parameter_recovery.py     ← parameter recovery simulation
  plotting/
    plot_pareto_optimal.py        ← Pareto figure
    plotter.py                    ← utility library

plans/
  01_repo_cleanup_plan.md         ← this plan (cleanup)
  02_evc_analysis_pipeline_plan.md ← this plan (analysis)
```
