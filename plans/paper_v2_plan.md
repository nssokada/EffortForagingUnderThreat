# Plan: Paper v2 — Effort-Threat Integration in Human Foraging

## Core Story

Humans integrate effort cost and predation risk into a unified computation that governs both what they choose and how vigorously they act. A joint model with two separable parameters explains structured deviations from optimal foraging policy, and the relationship between affect and this computation reveals additional functional costs that the parameters alone cannot account for.

---

## Phase 1: New Analyses

### 1.1 Optimal Policy Derivation
**Script:** `notebooks/07_evc_pipeline/14_optimal_policy.py`

**What to compute:**
- For each of 9 conditions (3T × 3D), compute the EV-maximizing choice using empirical conditional survival rates
- For each condition × cookie type, compute the optimal press rate from the model's vigor EU at the population parameters
- Define "subjective optimal" (under γ=0.21) vs "objective optimal" (under γ=1)
- Per-subject: classify each trial as optimal, overcautious, or overrisky

**Output:**
- `results/stats/optimal_policy.csv` — per-condition optimal choice + press rate
- `results/stats/per_subject_deviations.csv` — per-subject overcautious rate, overrisky rate, vigor gap, earnings loss

### 1.2 Parameters Drive Specific Deviations (R4 analyses)
**Script:** `notebooks/07_evc_pipeline/15_deviation_analysis.py`

**What to compute:**
- log(ce) → overcautious error rate (Pearson r, controlling for cd)
- log(cd) → survival rate, vigor gap (Pearson r, controlling for ce)
- Partial correlations: does ce predict overcaution INDEPENDENTLY of cd, and vice versa?
- Multiple regression: deviation_type ~ log(ce) + log(cd) — unique variance explained by each
- Optimal surface under γ=0.21 vs γ=1: how much "suboptimality" disappears when we account for probability weighting?
- Behavioral profiles (ce×cd quadrants) → earnings, overcaution rate, vigor gap

**Output:**
- `results/stats/deviation_param_associations.csv`
- `results/figs/paper/fig_deviations.png` — Panel A: optimal policy surface, Panel B: ce→overcaution scatter, Panel C: cd→vigor gap scatter, Panel D: γ shifts the optimal surface

### 1.3 Residual Suboptimality and Affect (R6 analyses)
**Script:** `notebooks/07_evc_pipeline/16_residual_suboptimality.py`

**What to compute:**
- Per-subject residual choice suboptimality: regress overcautious rate on log(ce) + log(cd), take residuals
- Per-subject residual vigor suboptimality: regress vigor gap on log(ce) + log(cd), take residuals
- Discrepancy → residual overcaution (Pearson r, and hierarchical regression ΔR²)
- Discrepancy → excess vigor on low-threat trials (compute per-subject excess vigor at T=0.1, control for cd)
- Calibration → policy alignment: per-subject |actual_choice - model_predicted_choice| mean, correlated with calibration
- Calibration → residual choice accuracy (incremental over ce, cd)
- Formal test: do discrepancy and calibration explain UNIQUE variance in suboptimality beyond ce and cd?
- Four-route independence: correlation matrix of (ce, cd, calibration, discrepancy, reactivity) × (overcaution, vigor gap, policy alignment, residual overcaution, task engagement)

**Output:**
- `results/stats/residual_suboptimality.csv`
- `results/stats/four_routes.csv` — correlation matrix
- `results/figs/paper/fig_residual_affect.png` — Panel A: discrepancy→residual overcaution, Panel B: calibration→policy alignment, Panel C: four-route independence matrix

### 1.4 Update Encounter Dynamics for New Framing
**Script:** Update existing `13_vigor_dynamics.py`

**Additional analyses:**
- Reactivity → overall task engagement (total earnings, total heavy choices, overall vigor)
- Show reactivity is independent of calibration, discrepancy, and ce
- Reactivity → apathy (AMI), incremental over ce + cd (already done, verify)

**Output:** Update `results/stats/evc_vigor_dynamics.csv`

---

## Phase 2: New Draft

### Structure: `drafts/draft010/paper.md`

**Title:** "Integrating effort and threat in human foraging: a unified computation of choice and vigor under predation risk"

**Abstract:** ~250 words. Lead with the integration problem. Model captures choice + vigor. Parameters drive specific deviations from optimal. Affect adds functional costs beyond what computation explains.

**Introduction** (4-5 paragraphs):
1. Foraging under threat: the optimization problem (Lima & Dill, Bednekoff, McNamara & Houston)
2. Three isolated literatures: effort discounting, defensive behavior/threat imminence, motor vigor
3. The gap: no joint model of choice + vigor, no framework for quantifying deviation from optimal
4. EVC/optimal control as candidate architecture (Shenhav, Todorov)
5. Present study: task + model + optimal policy + affect

**Results:**
- R1: Task captures foraging-under-threat trade-off (behavioral effects, Simpson's paradox note)
- R2: Joint choice-vigor model (static fit, model comparison, recovery, probability weighting)
- R3: Encounter dynamics extend cd to real-time defense (strategic vs reactive)
- R4: Parameters drive specific deviations from optimal policy (ce→overcaution, cd→underinvestment, γ shifts surface)
- R5: Survival computation tracks moment-to-moment affect (LMMs, convergent validity)
- R6: Affective bias drives residual suboptimality (discrepancy→excess caution, calibration→policy alignment, four dissociable routes)

**Discussion** (5-6 paragraphs):
1. Summary: unified computation spans choice, vigor, and encounter dynamics
2. Foraging theory contribution: optimal policy deviations explained by two parameters + probability weighting
3. Strategic vs reactive defense: ce→choice, cd→tonic+phasic vigor, bridging the imminence continuum
4. Affect as functional information: calibration helps, miscalibration costs — grounded in task outcomes not questionnaires
5. Clinical implications: modest, honest, await combined sample + clinical populations
6. Limitations: population ε, distance confound, online sample, static model

**Methods:** Full detail, same as draft009 + new analysis descriptions

**Figures:**
1. Task design + behavioral effects
2. Model fit + recovery + model comparison table
3. Encounter dynamics (encounter-aligned timeseries, cd correlation, threat independence)
4. **NEW: Optimal policy + deviations** (optimal surface, ce→overcaution, cd→vigor gap, γ shift)
5. **NEW: Affect + residual suboptimality** (affect tracks S, discrepancy→residual overcaution, calibration→alignment, four routes)

**Supplementary:**
- Full model comparison table (7 models)
- Behavioral profiles
- Simpson's paradox
- MCMC validation
- Distance gradient failure with population ce
- Clinical questionnaire correlations (convergent validity + param associations)

---

## Phase 3: Updated Preregistration

### File: `drafts/prereg_v2_aspredicted.md`

**Title:** "Integrating effort and threat in human foraging: a unified computation of choice and vigor under predation risk"

**Hypotheses:**

**H1: Behavioral effects** (unchanged)
- a. Choice decreases with threat and distance
- b. Vigor increases with threat (conditioned on choice)
- c. Model-derived S predicts anxiety and confidence

**H2: Joint model fit** (unchanged)
- a. Per-subject choice r² > 0.85
- b. Trial-level vigor r² > 0.30
- c. Distance gradient captured
- d. Parameter recovery r > 0.70
- e. Parameter independence |r| < 0.25

**H3: Parameters drive specific deviations from optimal**
- a. log(ce) predicts overcautious error rate (r > 0, p < .01)
- b. log(cd) predicts survival rate (r > 0, p < .01)
- c. ce and cd explain unique variance in deviation types (partial R² each > 0.05)
- d. Behavioral profiles: Vigilant (low ce, high cd) earns most

**H4: Encounter dynamics**
- a. Encounter reactivity is trait-stable (cross-block r > 0.50)
- b. Reactivity correlates with log(cd) (r > 0, p < .05)
- c. Reactivity is NOT threat-modulated (ANOVA p > .10)

**H5: Affect tracks computation and drives functional consequences**
- a. Calibration and discrepancy are orthogonal (|r| < 0.15)
- b. Calibration predicts policy alignment (r > 0, p < .05), controlling for ce + cd
- c. Discrepancy predicts residual overcaution (r > 0, p < .05), controlling for ce + cd
- d. Discrepancy predicts at least 2 clinical symptom measures at p < .05 (exploratory: STAI, OASIS, STICSA, DASS, PHQ-9)

**Exploratory:**
- Combined sample (N≈580) tests param→clinical associations at adequate power
- Machine learning prediction of clinical outcomes
- Factor analysis of psychiatric battery
- Encounter reactivity → apathy

**Sample Size:**
N = 350 confirmatory (recruited). Combined N ≈ 580.
- H1-H2: power > 0.95 at N=280 for all tests
- H3: power > 0.95 for r=0.20 at N=280
- H4: power > 0.99 for r=0.47 at N=280
- H5b-c: power depends on incremental effect size; if ΔR²=0.03 (from exploratory), power ≈ 80% at N=280
- Exploratory clinical: combined N≈580 provides 83% power for r=0.12

---

## Phase 4: Execution Order

```
Step 1: Run Phase 1 analyses (14, 15, 16 scripts)
        → produces new results CSVs and figures
        ↓
Step 2: Verify all new statistics
        ↓
Step 3: Write draft010 following the arc above
        ↓
Step 4: Write prereg_v2
        ↓
Step 5: Review cycle (editor + 3 reviewers)
        ↓
Step 6: Revise → draft011 (submission-ready)
        ↓
Step 7: Run confirmatory sample when data available
```

---

## Key Differences from Previous Drafts

| Aspect | Draft009 | Paper v2 |
|--------|---------|----------|
| Core story | Metacognitive bias bridges to clinical | Unified effort-threat computation, deviations from optimal |
| Lead finding | Discrepancy → STAI | Model explains choice + vigor jointly |
| Optimal policy | Mentioned in passing | Central result (R4) |
| Affect framing | "Metacognitive miscalibration" | "Affective calibration drives functional costs within the task" |
| Clinical | Headline finding | Converging evidence, awaits combined sample |
| ce role | Effort cost | Drives FOREGONE OPPORTUNITY (specific deviation type) |
| cd role | Capture aversion | Bridges TONIC + PHASIC defense (strategic-reactive continuum) |
| Discrepancy role | Predicts clinical anxiety | Drives RESIDUAL SUBOPTIMALITY the model can't explain |
| Title emphasis | Metacognition → clinical | Effort-threat integration in foraging |

---

## What This Buys Us With Reviewers

1. **R1 (computational):** The model is now serving foraging theory, not just fitting data. Optimal policy + deviations is the core contribution.
2. **R2 (clinical):** Clinical findings are grounded in TASK OUTCOMES (wasted effort, foregone reward), not just questionnaire correlations. Much more defensible.
3. **R3 (ecology):** The foraging theory engagement is genuine — optimal policy, deviation types, probability weighting shifting the surface. Not just ecological window dressing.
4. **All reviewers:** The "metacognition" terminological concern disappears. We're talking about affect and its functional consequences, not metacognition.
