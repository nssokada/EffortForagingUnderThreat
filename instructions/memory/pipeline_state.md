# Pipeline State

Current execution status of each notebook and script in the analysis pipeline.
Last updated: 2026-06-03 (H4 choice decomposition + parameter-mediated r(choice, vigor) prediction).

---

## 2026-06-04 — result_604 Stage 3: HiTOP-style factor analysis confirms clinical null

**Script:** `scripts/analysis/embodied_clinical_factor_analysis.py` ✅

**Method:** Parallel analysis (Horn's, 500 perms) + EFA with varimax rotation on N = 568 pooled. Tested whether (ω, κ) predicts factor scores rather than individual subscales (addresses the HiTOP / p-factor concern).

**Parallel analysis:** 2 factors retained. F1 eigenvalue = 8.52 (dominant — consistent with p-factor literature). F2 eigenvalue = 1.42 (modest). Subsequent eigenvalues drop below random 95% threshold.

**Factor structure (varimax):**
- **F1 (general internalising distress)**: positive loadings on DASS21_Anxiety (+0.86), STICSA (+0.86), DASS21_Stress (+0.82), OASIS (+0.77), PHQ9 (+0.76), DASS21_Depression (+0.72), STAI_State (+0.71). STAI_Trait anomalously loads opposite (−0.59).
- **F2 (apathy/fatigue/anhedonia)**: negative loadings on MFIS_Psychosocial (−0.75), MFIS_Cognitive (−0.72), MFIS_Physical (−0.69), AMI_Behavioural (−0.57), DASS21_Depression (−0.47), PHQ9 (−0.46).

**Structure matches HiTOP's internalising distress + somatic-form distinction. Theoretically sensible.**

**(ω, κ) → factor scores: ALL NULL.**
- F1: β(ω) = −0.073 [−0.160, +0.016] marginal wrong-direction; β(κ) = +0.027; β(ω×κ) = +0.029
- F2: β(ω) = −0.003; β(κ) = −0.016; β(ω×κ) = +0.033

**Verdict across three analyses:** Stage 1 (subscales, pooled-z): 1 hit in interpretable direction. Stage 2 (comorbidity groups): all null. Stage 3 (factor analysis): all null. **The clinical decomposition is genuinely absent in this data, not just underpowered or hidden by comorbidity confound.**

**Outputs:** `results/stats/clinical/parallel_analysis.csv`, `factor_loadings.csv`, `factor_scores.csv`, `factor_param_regressions.csv`.

---

## 2026-06-04 — result_604 verification: analysis correct, but cross-sample heterogeneous + result_602 doesn't replicate at cell-mean vigor

**Script:** `scripts/analysis/verify_clinical_decomp.py` ✅

**Triggered by:** Skepticism about the result_604 Stage 1 finding (ω → AMI Social/Emotional/Total positive, β ≈ +0.10–0.14, 95% HDI excludes zero). Question: was the analysis right, or was there a scoring/sign error?

**Verdict: the analysis was correct.** The bambi regression coefficients (β ≈ +0.11 on ω → AMI_Total) are properly within-sample partial standardised slopes; the raw pooled Pearson (r = +0.062, p = 0.14) appears smaller because pooled-raw is contaminated by between-sample heterogeneity in both AMI and ω means. Within-sample z-scoring (which result_604 uses) is the correct way to handle this.

**But verification surfaced two real concerns:**

1. **Cross-sample heterogeneity in the AMI → ω signal.** Within-sample raw Pearson:
   - Exploratory: r(AMI_Total, log_ω) = +0.076 (weak, would not pass HDI test alone)
   - Confirmatory: r(AMI_Total, log_ω) = +0.146 (moderate)
   - The pooled β ≈ +0.114 is the *average* of these. The "ω → social/emotional apathy" headline is driven primarily by confirmatory. Exploratory shows same-sign but much smaller effects. Not a strong "both samples" replication.

2. **The legacy result_602 (AMI → vigor) does NOT replicate at the master-table mean_vigor metric.** r(AMI_Total, mean_vigor):
   - Exploratory: +0.012 (p = 0.84, NULL)
   - Confirmatory: −0.029 (p = 0.63, NULL)
   - Result_602's claim "AMI apathy tracks vigor (negative)" relied on a different vigor operationalisation (pre-encounter capacity-normalised). At the cell-mean aggregate used by result_208 / result_401 / result_604, AMI and vigor are uncorrelated. The κ → apathy chain (κ → vigor → AMI) breaks at the second link.

**Parameter distributions verified reasonable** in both samples (ω log-normal-ish, κ log-normal-ish, r(log_ω, log_κ) = +0.369 / +0.302 matching result_208).

**No item-level AMI data** available in psych.csv (only subscales + total). Cannot rule out scoring-direction issues by inspecting items. r(AMI, mean_vigor) ≈ 0 in both samples is *consistent with* either standard scoring (AMI ↑ = apathy ↑, behavioural correlate just absent) or reverse-scoring; without items, the direction is determined by convention (result_602 framing assumes standard).

**Implications for the paper:**
- The result_604 headline "ω → social/emotional apathy" survives mathematically but with the honest caveat that confirmatory drives most of the signal
- The result_602 finding (κ → apathy via vigor) does not hold at the metric consistent with the rest of the paper
- The clinical story stays narrow: a small ω → social/emotional apathy effect, replicated in direction but heterogeneous in magnitude. Not strong enough to anchor Frame A.

---

## 2026-06-03 — H4 choice decomposition + cross-channel r(choice, vigor) prediction (result_208 update)

**Script:** `scripts/analysis/h4_choice_decomp.py` ✅ run on both samples.

**Outputs:**
- `results/stats/individual_diffs/h4_choice_decomp.csv` ✅ (32 coefficient rows: P(heavy) + mean_vigor × polar + Cartesian + main-effects × both samples)
- `results/stats/individual_diffs/h4_predicted_r_cv.csv` ✅ (2 rows: predicted vs observed r(choice, vigor) per sample)

**Key findings:**

Choice partial coefficients (Cartesian with interaction):
- Exploratory: β(ω → P(heavy)) = −0.154 [−0.161, −0.147], β(κ → P(heavy)) = −0.076 [−0.083, −0.069]
- Confirmatory: β(ω → P(heavy)) = −0.168 [−0.177, −0.160], β(κ → P(heavy)) = −0.062 [−0.070, −0.053]
- Both samples: ω ≈ 2× stronger than κ on choice; small positive ω × κ interaction (+0.006 / +0.015)

Predicted vs observed r(choice, vigor) — embodied W(u) framework prediction:
- Exploratory: predicted = +0.143, observed = +0.150 (p=0.011) — match within 0.007
- Confirmatory: predicted = +0.052, observed = +0.077 (p=0.201) — match within 0.025

Pathway decomposition (both samples):
- ω-pathway Cov (β_ωc·β_ωv): consistently ≈ −0.020 (negative; ω lowers choice but raises vigor)
- κ-pathway Cov (β_κc·β_κv): +0.018 / +0.014 (positive; κ lowers both)
- Cross term (driven by r(ω,κ) ≈ +0.30–0.37): +0.010 / +0.010
- Near-cancellation between channel pathways; small positive residual from cross term explains observed r

**Implication:** Marginal r(choice, vigor) is *quantitatively* predicted by the embodied two-parameter framework, not just qualitatively consistent. Anchors the embodiment argument for [[result_401]] rewrite (Phase 2, deferred).

**Sampling diagnostics:** R̂ = 1.000 across all 12 fits; ESS_bulk ≥ 7,001. Each fit ≈ 1 s wall time (bambi/NumPyro).

**Note on operationalization:** `mean_vigor` here is the M4 cell-mean aggregate, NOT the legacy "pre-encounter capacity-normalized + choice-ratio-adjusted" metric used in legacy H29 (which reported r ≈ −0.018). Different operationalization → different sign on marginal r. The current operationalization is the one that matches 208's vigor partial coefficients, so it is the consistent metric for the cross-channel prediction.

---

## 2026-05-28/29 — M1 effort-kernel discount-form selection (result_206)

**Script:** `scripts/modeling/joint_optimal/m1_effort_kernels.py` ✅ run on both samples.

**Outputs (Axis B — discount function, headline):**
- `results/stats/joint_optimal/m1_effort_kernels_exploratory.csv` ✅
- `results/stats/joint_optimal/m1_effort_kernels_confirmatory.csv` ✅

**Outputs (Axis A — effort exponent robustness):**
- `results/stats/joint_optimal/m1_effort_exponent_exploratory.csv` ✅
- `results/stats/joint_optimal/m1_effort_exponent_confirmatory.csv` ✅

**Figures:** `results/figs/paper/fig_s_m1_effort_kernels_{exploratory,confirmatory}.{pdf,png}` ✅

**Result:** Linear discount on `E = req²·D` wins decisively in BOTH samples. ΔBIC = 0 / 266 / 333 / 861 (linear / exp / quad / hyp) in exploratory; ΔBIC = 0 / 273 / 386 / 952 (linear / quad / exp / hyp) in confirmatory. Subject-level choice R² ≈ 0.95 for linear in both samples. Free-power optimum p̂ ≈ 5.2–5.4 in both (uninterpretable high; M1's p=2 is the principled near-optimum). **result_206 status upgraded `supported_exploratory → supported`.** Runtime: ~2 min per sample on CPU.

### Bug discovered + fixed: `data/model_input/` was mislabeled

While running the confirmatory M1 sweep, discovered that `data/model_input/` — which the prior result_206 frontmatter labeled "exploratory, N=281" — in fact contained the **confirmatory sample's data** (281 subjects, 12,645 choice trials, p_heavy=0.414, matching `stage5_filtered_data_20260403_142413/behavior_rich.csv`). True exploratory has 293 subjects, 13,185 trials, p_heavy=0.431.

**Fix:** Generated explicit per-sample model_input snapshots:
- `data/model_input_exploratory/` — true exploratory (N=293, 13,185 trials, 3,935 vigor cells). Built 2026-05-29 from `data/exploratory_350/processed/stage5_filtered_data_20260403_133425/` via `prepare_model_input.py`.
- `data/model_input_confirmatory/` — confirmatory (N=281, 12,645 trials, 3,822 vigor cells). Built 2026-05-29 from `stage5_filtered_data_20260403_142413/`.

`data/model_input/` left in place as the confirmatory snapshot for backward compatibility. **All new code should reference the explicit dirs, not `data/model_input/`.**

The cached MCMC fits in `results/stats/joint_optimal/exploratory/mcmc_m4_params.csv` and `results/stats/joint_optimal/confirmatory/mcmc_m4_params.csv` use a separate data-loading path (`scripts/run_mcmc_pipeline.py`) and are NOT affected — their subject counts (290 expl, 281 conf) match the actual sample they purport to be.

---

## 2026-06-02 — H2 vigor dynamics (result_104) blocked by two bugs

**Status:** Deferred. Notebook execution attempted; cannot ship a valid both-sample result_104 yet.

**Investigation:**

1. Patched a missing `outputs: []` / `execution_count: null` on the "H2 Summary" cell of `notebooks/analysis/H2_vigor_dynamics.ipynb` (via a sub-agent). This fixed `nbformat.validator.NotebookValidationError`. No source-code change.
2. Re-executing the notebook end-to-end fails on the H2c GAM cell with `LinAlgError: Singular matrix` — the cell tries to fit a MixedLM with `K = min(K_SPLINE, 4)` cubic-spline basis on a `t_epoch` column with only 4 unique values, which is rank-deficient. The cell's own comment notes that the original analysis used the raw `alignedEffortRate` timecourse, not 4 discrete epochs. This is a substantive analysis bug, not an environment issue.
3. Extracted cells 1 + 3 + 5 (imports, H2a paired-t, H2b encounter spike) as a standalone Python script and ran them. Output for the two samples is **byte-for-byte identical**: Heavy Δ=+0.0349 / t=7.72 / p=1.89e-13 / d=+0.454 and Light Δ=+0.0541 / t=12.95 / p=1.45e-30 / d=+0.762 in both samples; H2b mean spike +0.0358 / t=11.01 / p=8.73e-24 / d=+0.647 in both. The H2b numbers match `confirmatory_hypothesis_results.csv` exactly, indicating `vigor_metrics` is loaded from a single confirmatory source regardless of which sample is requested.

**Diagnosis of the data-loading bug.** `len(d['vigor_metrics'])` is `93960` for both "exploratory" and "confirmatory" in `load_both()` output, while `len(d['trials'])` correctly differs (23,490 expl vs 22,761 conf). Same shape of bug as the `data/model_input/` mislabel resolved 2026-05-29 — a data-loading path returning one sample's data under both labels.

**Fix needed (next session):**
- Trace `vigor_metrics` source inside `notebooks/analysis/load_data.py` and fix sample dispatch.
- Decide the H2c GAM specification (restore raw `alignedEffortRate` timecourse with K=10 matching prereg, or accept degenerate epoch-level with K ≤ 3 and document deviation).
- Re-execute H2 notebook, T3-validate confirmatory against the cached CSV (d=0.647, GAM enc χ²=1024.8, GAM threat χ²=114.8), write up result_104 as a full lab report.

**At-risk results.** Any result that reads `d['vigor_metrics']` is at the same risk. Trial-level paths (`d['vigor']`, `d['vigor_valid']`) used by H1 (results 101–103) and H8 (result 402) are NOT affected.

---

## 2026-05-29 — Trial-level affect ~ S_probe LMM (result_501)

**Script:** `scripts/analysis/affect_survival_lmm.py` ✅ run on both samples.

**Outputs:**
- `results/stats/affect_analysis/s_probe_affect_lmm_exploratory.csv` ✅
- `results/stats/affect_analysis/s_probe_affect_lmm_confirmatory.csv` ✅

**Approach.** For each probe trial: compute u* = argmax_u W(u; T, D, ω, κ) using the subject's fitted M4 (ω, κ) and the M4 posterior-mean population params (γ, h, σ_sp). S_probe = S(u*, T, D). Z-score within sample. Fit `response ~ S_probe_z + (1|subj)` separately for anxiety and confidence using `statsmodels.mixedlm` (ML).

**Result — both signs as predicted (higher survival → less anxiety, more confidence), replicates across both samples:**

| Sample | Channel | β(S_probe_z) | SE | z | p | N obs | N subj |
|---|---|---|---|---|---|---|---|
| Exploratory | Anxiety | −0.584 | 0.025 | −23.74 | 1.5e-124 | 5,220 | 290 |
| Exploratory | Confidence | +0.625 | 0.025 | +25.30 | 3.1e-141 | 5,218 | 290 |
| Confirmatory | Anxiety | −0.545 | 0.025 | −22.25 | 1.0e-109 | 5,068 | 281 |
| Confirmatory | Confidence | +0.680 | 0.025 | +27.09 | 1.3e-161 | 5,068 | 281 |

**Validates the legacy NB04-03 numbers from `instructions/memory/hypotheses.md` § H4** (anxiety β = −0.602, confidence β = +0.632 on the older N=293 exploratory). Current exploratory (β = −0.584 / +0.625) matches to within rounding — confirms the M4-derived S_probe behaves like the deprecated framework's S_probe.

**Population params used (from M4 mcmc_convergence_diagnostics.csv posterior means):**

- Exploratory: γ = 0.846, h = 0.550, σ_sp = 0.247
- Confirmatory: γ = 0.826, h = 0.381, σ_sp = 0.243

**Implication:** result_501 upgraded from `untested` (deferred stub) → `supported`. The trial-level affect-survival coupling is a robust population-level effect that operates through the model-derived survival quantity, not just through raw threat/distance. Mechanistically distinct from the threat-only LMMs in [[result_102]] because S_probe is a model-derived nonlinear function of (T, D, ω, κ) rather than the raw conditions.

---

## Historical / older state below — last refreshed 2026-03-20

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

**EVC Option 2 parameter recovery (2026-03-27):**
- `evc_option2_recovery.csv` ✅ (5 datasets × 50 subj; ce r=0.941 PASS, cd r=0.917 PASS, eps r=-0.025 FAIL — no individual variance, gamma=0.274 vs true 0.210 slight positive bias)
- `fig_s_option2_recovery.png` ✅ (3-panel scatter: ce, cd, eps true vs recovered)
- Script: `scripts/analysis/evc_option2_recovery.py`

**EVC-LQR full pipeline (2026-03-27):**
- `evc_lqr_recovery.csv` ✅ (5 datasets × 50 subj; cd r=0.888, eps r=0.933, gamma 0.314 vs true 0.318)
- `evc_lqr_ppc.csv` ✅ (Choice acc=75.4%, AUC=0.819, subj choice r=0.901, vigor r=0.510, subj vigor r=0.717)
- `evc_lqr_clinical.csv` ✅ (No FDR survivors; best uncorrected: cd→AMI_Emotional r=0.121 p=0.039)
- `evc_lqr_clinical_interactions.csv` ✅ (No significant cd×eps interactions)
- `evc_lqr_clinical_factors.csv` ✅ (F1/F2/F3 all null)
- `evc_lqr_affect.csv` ✅ (Anxiety beta=-0.786 t=-13.09; Confidence beta=0.848 t=13.40)
- `evc_lqr_metacognition.csv` ✅ (Conf-CQ r=0.012 null; Conf-SR r=-0.048 null; Steiger z=0.82 ns)
- `evc_lqr_dissociation.csv` ✅ (Partial dissociation: cal→CQ r=0.239, disc→STAI-State r=0.308)
- `evc_lqr_profiles.csv` ✅ (4 quadrants; P(heavy) R²=0.877; Helpless archetype lowest earnings)

**Figures (2026-03-27):**
- `fig_s_lqr_recovery.png` ✅ (2-panel scatter: cd and eps recovery)
- `fig_ppc_lqr.png` ✅ (6-panel PPC)
- `fig_s_lqr_clinical.png` ✅ (Forest plot)
- `fig_lqr_metacognition.png` ✅ (4-panel metacognition)
- `fig_lqr_quadrants.png` ✅ (4-panel profiles)

**Draft:**
- `drafts/draft003/evc_lqr_paper.md` ✅ (Full paper + critical review)

**DEFINITIVE EVC 2+2 model (2026-03-27) — population epsilon:**
- Model: `scripts/modeling/evc_final_2plus2.py` ✅
- `oc_evc_final_params.csv` ✅ (N=293, per-subject ce and cd)
- `oc_evc_final_population.csv` ✅ (epsilon=0.098, gamma=0.210, ce_vigor=0.003, tau=0.476)
- **Fit:** BIC=17,768, Choice acc=79.3%, Subj choice r²=0.951, Vigor r²=0.511, Subj vigor r²=0.687
- `evc_final_recovery.csv` ✅ (3 datasets×50 subj; ce r=0.916 PASS, cd r=0.943 PASS, gamma PASS)
- `evc_final_ppc.csv` ✅ (Choice acc=79.3%, AUC=0.876, subj choice r=0.976, vigor r=0.722, subj vigor r=0.836)
- `evc_final_affect.csv` ✅ (Anxiety beta=-0.557 t=-14.04; Confidence beta=0.575 t=13.48)
- `evc_final_metacognition.csv` ✅ (Conf-CQ r=-0.081 null; Conf-SR r=-0.048 null; Steiger z=-0.50 ns)
- `evc_final_dissociation.csv` ✅ (cal→CQ r=0.230 p<.001, disc→STAI-State r=0.327 p<.0001)
- `evc_final_clinical.csv` ✅ (No FDR survivors; no significant interactions)
- `evc_final_clinical_factors.csv` ✅ (F1/F2/F3 all null)
- `evc_final_profiles.csv` ✅ (4 quadrants: Cautious/Lazy/Vigilant/Bold; P(heavy) R²=0.953)
- Figures: fig_s_final_recovery.png, fig_ppc_final.png, fig_s_final_clinical.png, fig_final_metacognition.png, fig_final_quadrants.png

**Superseded (keep for reference):**
- `FET_Exp_Bias_*.csv` — old model, replaced by L3_add
- `joint_model_*.csv` — old joint model (independent priors, σ_δ collapsed), replaced by joint_correlated_*

**`results/model_fits/exploratory/`:**
- `vigor_hbm_idata.nc` ✅ (full MCMC trace, 549 MB, re-run 2026-03-20 via smoothed_vigor_ts)
- `FET_Exp_Bias_fit.pkl` — superseded

**`results/tables/`:**
- `table_s2_parameter_dissociation.csv/.tex` ✅
