# Session History

Chronological log of work sessions. Most recent first.

---

## Session 2026-03-24

### Prereg rewrite (via Discord with Noah)

**Hypothesis numbering overhaul:**
- Switched from old detailed prereg (H1-H6) to simple prereg numbering (H1-H7 in 4 sections)
- H1 = threat shifts behavior (new), H2 = coupling (new), H3 = optimality (new), H4 = choice model (old H1+H2), H5 = vigor (old H3), H6 = cross-model coupling (old H5), H7 = metacognition (old H6)
- Updated `preregistration.md`, `hypotheses.md`, `MEMORY.md`

**AsPredicted format prereg:**
- Rewrote full prereg in AsPredicted template format (`prereg_aspredicted.md`)
- Sections: Hypothesis, DV, Conditions, Analyses, Exclusions, Sample Size, Other

**H1 analysis decisions (confirmed with Noah):**
- Switched from ANOVAs to LMMs throughout H1 for consistency
- H1a: logistic LMM on trial-level choice + monotonicity via all-pairwise adjacent t-tests (p < 0.01)
- H1b: linear LMM with effort_chosen_z covariate (constant-demand control built in) — NEEDS VERIFICATION on exploratory data
- H1c: linear LMMs for anxiety and confidence with threat + distance
- Justification: Barr et al. 2013, Jaeger 2008, consistent with H4c approach

**Figures:**
- Wrote `scripts/plotting/plot_h1_figure.py` (3-panel: choice/vigor/affect by distance × threat)
- Cannot run — devcontainer lacks scientific Python
- Updated Dockerfile to include miniconda + effort_foraging_threat env — needs rebuild

**Blocked on:**
- Devcontainer rebuild (for running any Python analysis)
- H1b verification (must confirm before submitting prereg)
- Continuing H2-H7 walkthrough with Noah

---

## Session 2026-03-20

### Completed

**NB16 — Bayesian Hierarchical Vigor Model (`16_bayesian_vigor_model.ipynb`)**
- Built and iterated through 3 model versions:
  1. Terminal-only: ρ great (SB=0.76) but α was terminal idling (r=−0.56 with pre-enc)
  2. Enc-aligned two-window [enc-2,enc]+[enc,enc+2]: α great but ρ terrible (SB=0.28, attack effect too small in first 2s post-enc)
  3. **Final: separate windows** — pre-enc [enc-2,enc] for α, terminal [trialEnd-2,trialEnd] for ρ with nuisance γ
- NumPyro NUTS, 4 chains × 2000, 0 divergences, 58s wall time
- μ_α=0.519 (52% capacity), μ_ρ=0.526 (53% capacity boost), P(μ_ρ>0)=1.0
- Bayes-OLS: α r=1.000, ρ r=0.991. Shrinkage: α 2.1%, ρ 16.8%
- Split-half: α SB=0.925, ρ SB=0.762
- α-ρ: r=−0.237 (ceiling effect, not artifact)
- 5-param correlations confirmed: choice-vigor cross-correlations near zero (only k-α reaches significance at r=−0.196)
- Saved: `vigor_hbm_posteriors.csv`, `vigor_hbm_population.csv`, `vigor_hbm_idata.nc`

**NB06-psych — Factor Analysis of Psychiatric Battery (`06_factor_analysis.ipynb`)**
- 14 subscales (excluding totals), z-scored, KMO=0.931
- Parallel analysis suggested 2 factors; 3-factor solution used (theoretical: distress vs fatigue vs apathy)
- F1 (37%): General distress (STICSA, DASS, OASIS, PHQ9, STAI_State)
- F2 (20%): Fatigue (MFIS subscales)
- F3 (12%): Apathy/amotivation (AMI subscales, DASS_Dep, STAI_Trait)
- 5 params → factors: only α → F3 apathy (R²=0.155, p=3×10⁻⁹). Nothing else significant.
- Saved: `psych_factor_loadings.csv`, `psych_params_to_factors.csv`, `psych_factor_scores.csv`

**NB07-psych — PLS: 5 Params → Mental Health + Affect (`07_pls_params_mental_health.ipynb`)**
- X: {k, z, β, α, ρ}; Y: 3 psychiatric factors + mean anx/conf + threat sensitivity of anx/conf
- Overall: R²=0.073, perm p=0.0000, CV R²=0.039
- Comp 1 (r=0.538): α+low k → better anxiety calibration + more apathy + lower mean anxiety
- Comp 2 (r=0.300): z+k+β → lower confidence + more apathy
- Comp 3 (r=0.228): ρ → barely predicts anything
- Per-Y: anx threat sens R²=0.145, apathy R²=0.130 best predicted
- Saved: pls_mh_x_weights.csv, pls_mh_y_loadings.csv, pls_mh_cv_results.csv

**NB08-psych — Mixture Model / Clustering (`08_mixture_model_subtypes.ipynb`)**
- GMM with log-transformed choice params: BIC selects k=3
- 3 clusters: Vigorous-Engaged (n=120, best escape 55%), Avoidant (n=44), Ambitious-Weak (n=127, worst escape 22%)
- Coupled vs decoupled hypothesis: NULL. No subgroup structure. β doesn't gate coupling (r=0.05).
- Continuous analysis more appropriate than clustering (silhouette=0.17)

**Unified 3-Parameter Model (SVI)**
- Replaced per-subject z with α in survival function: S = exp(-λ·T·(D/α)^z), z and λ population-level
- SVI comparison: unified model more parsimonious (saves 293 z_i params, BIC-favored)
- k_unified correlates r=0.857 with original k. β_unified changes meaning (r=-0.22 with original β — now purer threat bias after α handles distance sensitivity)
- Head-to-head: unified 3-param matches or beats original on choice (R²=0.88 vs 0.83), escape (0.73 vs 0.72), conf miscalibration (0.45 vs 0.43), apathy (0.15 vs 0.14)
- k-α independence confirmed: r=0.006. The dissociation holds even when α enters the choice model.

**Continuous interaction analysis**
- k×α, β×α, k×β interactions on all outcomes
- Escape/earnings: purely additive, no interactions (p>0.26)
- Conf miscalibration: k×α interaction significant (p=0.006, ΔR²=0.03)
- Apathy: purely α-driven, no interactions (p=0.69)

**Mental health → behavioral profiles (predictive direction)**
- MH predicts vigor (62%, AUC=0.675) — AMI drives it
- MH predicts HL vs LH (61%, AUC=0.645) — AMI→LH, trait anxiety→HL
- MH does NOT predict choice (49%) or coupled/decoupled (51%)
- PHQ-9 shows quadrant effect (FDR p=0.043) — high α → more depression

**Model Comparison from First Principles (NB03-choice)**
- Rebuilt model comparison from scratch: 12 models tested via SVI
- Corrected S formulation: S = (1-T) + T·f(D/α) separates attack prob from escape prob (old model conflated them)
- Corrected β formulation: SV = R·S - k·E - β·(1-S), β IS the subjective capture cost (old model had β outside cost term)
- Additive effort (R·S - k·E) >> multiplicative (R·exp(-k·E)·S) by +158 ELBO. Solves k-β identifiability (r goes from +0.45 to -0.11)
- Hyperbolic escape kernel >> exponential by +207 ELBO
- α in survival helps (+16 ELBO), α in effort hurts (−294 ELBO)
- Per-subject z hurts (−112 ELBO) — not needed
- Winner: L4a_add: SV = R·S - k·E - β·(1-S), S = (1-T) + T/(1+λD/α) by ELBO, but see below re: α in S
- Parameters: k-β r=-0.11, k-α r=-0.08, β-α r=+0.14 — all essentially independent
- HOWEVER: α in survival function is degenerate — λ→∞ makes f(D/α)→0, so S≈(1-T). α effectively drops out.
- L3_add (no α in choice) is the honest model: SV = R·S - k·E - β·(1-S), S = (1-T) + T/(1+λD)

**Deep dive: what IS α?**
- NOT motor ability: capacity→α r=+0.03, CalMax→α r=+0.10, onset rate→escape r=-0.04
- NOT task engagement: controlling for questionnaire RT, choice entropy, affect variability changes nothing
- IS fraction of capacity deployed: α = mean pressing rate / 95th percentile capacity
- Stable across trials (SB=0.925), doesn't adjust for threat/distance/choice (onset rates flat)
- Speed tier structure: within a tier, pressing faster doesn't help → no incentive to adjust
- Predicts escape (r=+0.84), AMI apathy (r=+0.34), anxiety calibration (r=+0.26), mean anxiety (r=-0.16)
- All survive engagement controls
- Dynamic vigor: people do NOT adjust pressing strategically. Pre-enc "choice effect" was window-timing confound. Onset rates flat across all conditions after removing mechanical demand.
- Effort × distance perfectly confounded in design (E=0.6/D=1, E=0.8/D=2, E=1.0/D=3) — only 3 difficulty × 3 threat = 9 unique conditions

### Decisions
- Two separate windows (pre-enc + terminal) with separate likelihoods for vigor HBM
- **L3_add is the choice model**: SV = R·S - k·E - β·(1-S), S = (1-T) + T/(1+λD). α does NOT enter.
- **Additive effort** — pressing cost is physical, doesn't scale with reward. Solves k-β identifiability.
- **Hyperbolic escape kernel** — fits +207 ELBO over exponential
- **Must separate attack prob from escape prob**: S = (1-T) + T·f(D)
- **α = fraction of capacity deployed** — not motor ability, not engagement, not strategic. A stable default motor setting.
- α predicts escape, apathy, anxiety calibration but is invisible to the choice system
- AMI scoring confirmed correct: HIGH AMI = MORE APATHETIC
- Coupled/decoupled subtype hypothesis rejected
- Outcome=1 means CAPTURED in stage2 trials

---

## Session 2026-03-18 (continued — second half)

### Completed

**NB12 — Affect × Survival (`03_affect_survival.ipynb`) — completed and extended**
- Core LMMs already run (previous half); added three new sections:
- **Section 7: State-trait decomposition** — between-subjects OLS `trait ~ z + κ + β`; within-subjects LMM `state ~ p_threat_z + dist_safety_z`
  - Trait confidence: z β=−0.719 (p=0.044*), κ β=−0.163 (p=0.010*); adj R²=0.036
  - Trait anxiety: κ β=+0.146 (p=0.019*) only; z n.s. (p=0.097); adj R²=0.020
  - State (phasic) responses robust (β≈±0.575–0.586) and parameter-independent
- **Section 8: Cross-domain vigor × affect correlations** — 15 pairs (5 vigor × 3 affect); all null (max r=+0.124, FDR p=0.196)
- Saved: `affect_lmm_results.csv`, `affect_threat_slopes.csv`, `affect_vigor_cross_domain.csv`, `affect_trait_scores.csv`

**Terminology correction**
- Renamed `trait_anx/conf` → `mean_task_anx/conf` throughout (mean probe rating, NOT trait anxiety)
- Real trait anxiety = `STAI_Trait` from psych.csv (available but not previously used)

**NB13 — Anxiety × Vigor Coupling (`04_anxiety_vigor_coupling.ipynb`) — built and run**
- Key data structure discovery: `feelings.trialNumber` = global event-stream index (0–80), same as `phase_trial_metrics.trial` and `smoothed_vigor_ts.trial`; 45 behavioral + 36 probe events = 81 total per subject
- 7 unique probe schedules across subjects — alignment done per-subject
- **Phase-specific LMMs** (4 DVs: onset_slope, onset_mean, enc_spike, term_mean; N≈3,100 probe-trial pairs): ALL null across all phases and all affect types (FDR p_fdr > 0.67 everywhere)
- **Residual affect** (anxiety beyond threat+distance): also null for all phases
- **PLS** (subject-level means, N=281): r_obs=0.196 p_perm=0.033 — but CV R²=−0.071 (overfits, no generalizable structure)
- **Functional regression — model params → vigor(t)**: at each 0.1s bin × 3 alignment windows:
  - z_z: ramps from β≈0 → +0.065 by t=0.75s in onset window; positive throughout encounter; REVERSES to β≈−0.02 in terminal
  - κ_z: globally suppresses pressing across ALL phases (onset β≈−0.04 to −0.08; encounter β≈−0.02 to −0.05; terminal β≈−0.02 to −0.03)
  - β_z: modest positive boost in onset and encounter phases
- **Functional regression — anxiety → vigor(t)**: null at every time bin across all windows (max uncorrected β≈0.009)
- Saved: `vigor_affect_phase_lmm.csv`, `vigor_param_functional.csv`, `vigor_anxiety_functional.csv`

**Scientific conclusions (two-system architecture, fully supported)**
- Affect → vigor: COMPLETE NULL at every level — phase metrics, residual affect, PLS, and time-resolved functional regression
- Affect and vigor are parallel outputs of the same threat computation — not serially linked
- Serial architecture (threat → anxiety → vigor) REJECTED; parallel architecture (threat → anxiety AND threat → vigor) SUPPORTED
- Model params have rich temporal dissociation: z = onset mobilizer; κ = global chronic suppressor; β = modest onset/encounter boost

**Memory files updated**
- `discoveries.md`: added full NB12 state-trait results, cross-domain null, NB13 phase + functional results
- `pipeline_state.md`: NB12 extended to complete; NB13 added ✅
- `active_issues.md`: draft update note revised with state-trait findings
- `open_questions.md`: affect question closed (now answered)
- `session_history.md`: this entry

---

## Session 2026-03-18

### Completed

**Vigor notebook fixes**
- Fixed NB01 (`01_single_trial_visualization.ipynb`): column harmonization dict (_rename), merged `effort_L` and `f_max_i` from `trial_events.parquet`; made `c_it`/`d_t` optional in plot functions
- Fixed NB03 (`03_tonic_phasic_decomposition.ipynb`): same harmonization pattern; made `c_it` optional in `compute_trial_summaries`
- Restored NB02 to EVAL_HZ=20 (was temporarily 10 due to disk space); reran full downstream chain NB04→NB09 at 20Hz; `smoothed_vigor_ts.parquet` now 48.2 MB, 3,988,277 rows

**Analysis design**
- Created `instructions/vigor_params_analysis_design.md`: documented problem (N=293, X=3 params, Y=7 resid vigor features), pros/cons for PCA/CCA/LASSO/PLS/Bayesian, chose PLS as primary
- Key decisions: subject-level means, residual stream, permutation test for component significance, explicit null test on reactive/terminal loadings

**NB10 — PLS vigor × params (`10_pls_vigor_params.ipynb`)**
- PLSCanonical(n_components=3, scale=False) on z-scored X (z, κ, β) and Y (7 resid vigor features)
- Permutation test (N=5000): Component 1 significant
- Bootstrap CIs with sign alignment (per-component correlation check to flip sign)
- Trial-level LMM: S_trial → terminal mean β = −0.011 (FDR-surviving); z_i interaction marginal
- Fixed: KeyError 'threat_c' by only merging novel columns from beh

**NB11 — ODE vigor dynamics (`11_vigor_ode.ipynb`) — EXPLORATORY DEAD END**
- Built leaky-integrator analysis: encounter-aligned epochs, exponential rise fit per subject
- Key design clarification: `encounter_time` exists for ALL trials (scheduled predator time), not just attack trials — enables exact attack vs. no-attack contrast using same t_rel reference frame
- Results: `v_tonic_mean` ~ κ r=−0.20 (p_fdr=0.011), `v_amplitude` ~ z r=−0.16 (p_fdr=0.042)
- ODE kinetics (α) degenerate: median α=0.06/s (16s time constant), no asymptote visible in 10s window
- Conclusion: NB11 replicates existing pipeline findings, does not add new information; confirmed dead end

**Scientific framing discussion**
- Identified paper's critical gap: three-column structure (Choice ✅, Affect ❌ NEVER COMPUTED, Vigor ⚠️ sprawled across 10 notebooks)
- Agreed on three clean vigor results for paper: (1) κ → chronic tonic pressing, (2) S_trial → terminal persistence, (3) reactive spike dissociated from model parameters
- Discussed whether current vigor features are right — concluded they are, ODE approach confirmed it

**Data structure clarification**
- `encounter_time` is set for ALL trials (attack and non-attack) — represents scheduled predator appearance time
- `isAttackTrial` / `encounter` flag distinguishes whether predator actually appeared
- `startDistance` (5, 7, 9) = predator starting distance; `distance_H` (1, 2, 3) = cookie distance for high-effort option

### Blocked / Not Completed
- Affect analysis (S_trial → anxiety/confidence LMM): CRITICAL, never computed, blocks paper's core claim
- NB07 (`07_clinical_prediction.ipynb`): still blocked on `modeling_factor_param.csv` (EFA of psych battery)
- Parameter recovery (`02_parameter_recovery.ipynb`): not run against N=293
- Full 7-model WAIC comparison on N=293: not run
- Confirmatory sample (N=350): not started

---

## Session 2026-03-17

### Completed

**Choice modeling**
- Removed stale `from fet_models.ppc import compare_models, compute_waic` in NB01 cell 14 (already imported via `from modeling.ppc import ...` in cell 2)
- Consolidated `notebooks/02_choice_modeling/`: deleted `01_fit_compare_ppc.ipynb` (plain, 39 cells), renamed `02_fit_compare_ppc_with_plotter.ipynb` → `01_fit_compare_ppc.ipynb`, renamed `03_parameter_recovery.ipynb` → `02_parameter_recovery.ipynb`
- Fit FETExponentialBias on N=293 with full MCMC settings (2000w/4000s/4 chains, target_accept=0.90) via `scripts/run_fit_best_model.py` — saved to `results/model_fits/exploratory/FET_Exp_Bias_fit.pkl` (~217 MB), ran in ~6 min
- Ran `scripts/run_ppc_analysis.py`: WAIC=12,063 (SE=121), McFadden R²=0.454, AUC=0.912, Accuracy=82.5%, ECE=0.023
- Saved to `results/stats/`: `FET_Exp_Bias_waic.csv`, `FET_Exp_Bias_predictions.csv`, `FET_Exp_Bias_subject_metrics.csv`, `FET_Exp_Bias_population_params.csv`, `FET_Exp_Bias_{k,z,beta}_params.csv`

**Vigor pipeline setup**
- Created `scripts/vigor_data_prep.py`: converts `stage2_trial_processing_*/processed_trials.pkl` into NB02-compatible parquet files
  - `keypress_events.parquet` (899,936 rows — one per keypress, with effort-onset-relative timestamps)
  - `trial_events.parquet` (23,733 rows — trial metadata with effort-onset-relative encounter/escape/capture times)
  - `effort_ts.parquet` (293 rows — participantID + calibrationMax)
  - All saved to `data/exploratory_350/processed/vigor_prep/`

**Disk space management**
- Ran out of space during NB02 (232 MB free → OSError); resolved by:
  1. Deleting `data/exploratory_350/processed/stage1_raw_processing_20260317_093304/` (2.7 GB, aborted run)
  2. Reducing EVAL_HZ 20→10 in NB02
  3. Adding float32 casting + zstd compression on parquet saves
- Disk now has 259 GB free (user cleared space manually)

**Vigor notebooks fixed and run (NB02→NB09, except NB01, NB03, NB07)**
- NB02 (`02_kernel_smoothing.ipynb`): updated paths, loads from vigor_prep, saves `smoothed_vigor_ts.parquet` (23 MB) and `demand_curves.parquet`
- NB04 (`04_phase_extraction.ipynb`): path updates; added new cell computing `_resid` and `_norm` DV variants, saves `phase_vigor_metrics.parquet`
- NB05 (`05_subject_features.ipynb`): path updates; fixed read-only parquet array bug (`idx = cell.index.values.copy()`); saves `subject_vigor_table.csv`
- NB06 (`06_choice_vigor_mapping.ipynb`): path updates; saves `results/choice_vigor_mapping_results.csv`
- NB08 (`08_parameter_dissociation.ipynb`): path updates; added column harmonization; merged `subject_vigor_table.csv` for z_z/kappa_z/beta_z; fixed statsmodels IndexError (dropna on predictors + reset_index); fixed undefined `comparison` variable; saves table_s2
- NB09 (`09_final_stats.ipynb`): path updates; added column harmonization + param merge; saves `results/step1_modelfree_results.csv`

**Memory system created**
- `instructions/memory/active_issues.md` — blocking issues and tech debt
- `instructions/memory/discoveries.md` — key empirical findings
- `instructions/memory/open_questions.md` — unresolved questions
- `instructions/memory/session_history.md` — this file
- `instructions/memory/pipeline_state.md` — pipeline execution status
- `.claude/commands/update-memory.md` — slash command for session summary → memory update

### Blocked / Not Completed
- NB01 (`01_single_trial_visualization.ipynb`): partial fix (paths, trial_type, subj→subj_id rename) but `v_t`, `f_max_i`, `encounter_time`, `escape_time` etc. still broken (~13 cells need full column remap)
- NB03 (`03_tonic_phasic_decomposition.ipynb`): same status as NB01
- NB07 (`07_clinical_prediction.ipynb`): blocked on `modeling_factor_param.csv` (needs EFA of psych battery)
- NB02 parameter recovery: `02_parameter_recovery.ipynb` not run against N=293 fit
- Full 7-model WAIC comparison on N=293: only FETExponentialBias fitted

---

## Session 2026-03-16 (previous)

*Earlier session — model comparison on N=270, initial vigor notebook structure, preprocessing pipeline.*

*(Details not captured — see git log for file-level history)*
