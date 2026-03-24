---
name: Next Steps
description: Priority action items for Nature Comms submission — confirmatory sample, remaining figures, final polish
type: project
---

# Next Steps (as of 2026-03-24)

## Priority 1: Run confirmatory sample (N=350)
- [ ] Preprocess through stages 1-5 (`scripts/preprocessing/pipeline.py`)
- [ ] Fit choice model (M5) → k, β, λ
- [ ] Fit vigor HBM (λ fixed from choice) → α, δ
- [ ] Compute cross-domain correlations + posterior bootstrap
- [ ] Run affect LMMs (S → anxiety/confidence)
- [ ] Compute optimality metrics
- [ ] Compute metacognitive calibration slopes
- [ ] Test all H1-H7 against preregistered criteria
- [ ] Script: `scripts/run_mcmc_pipeline.py --platform cuda`
- [ ] Data: `data/confirmatory_350/raw/` (collected, not preprocessed)

## Priority 2: Remaining figures
- [ ] **Fig 1** — Task schematic + 5-model comparison bar chart + choice by threat × distance + parameter distributions (NEEDS UNITY SCREENSHOT)
- [ ] **Fig 2** — Vigor: δ distribution / excess by threat / within-choice control (partly covered by fig_vigor_timecourse but needs standalone version)
- [ ] **Fig 3** — Already generated (fig4_coherent_shift.png) but needs optimality panel added
- [ ] **Fig 4** — Already generated (fig_s_metacognition.png)
- [ ] Add optimality panel to Fig 3 or as standalone supplementary figure

## Priority 3: Tables
- [ ] Supplementary Table 1: Full 5-model comparison (ELBO, BIC, accuracy, params)
- [ ] Supplementary Table 2: Population parameters from MCMC (k, β, α, δ, λ, τ with CIs)
- [ ] Supplementary Table 3: Cross-domain correlation matrix (MCMC point, bootstrap CI)
- [ ] Supplementary Table 4: Clinical profile (psychiatric measure means/SDs)
- [ ] Supplementary Table 5: Parameter recovery results

## Priority 4: Paper polish
- [ ] Update abstract with optimality finding
- [ ] Ensure all figure legends match current figures
- [ ] Add paragraph on ecological limitations (Reviewer 2 concern)
- [ ] Trim MH section to one paragraph + supplement
- [ ] Add confirmatory results when available
- [ ] Complete references (currently 20, likely need 30-40)
- [ ] Format for Nature Comms (Methods after Discussion, Extended Data)

## Priority 5: Preregistration
- [ ] Finalize simple_prereg.md → submit to AsPredicted
- [ ] Timestamp before opening confirmatory data
- [ ] Reference DOI in paper

## Already done
- ✅ MCMC validation (choice + vigor converged, joint didn't)
- ✅ Posterior bootstrap CIs for coupling
- ✅ PPCs and parameter recovery
- ✅ Bayesian MH analysis with ROPE
- ✅ Metacognitive calibration (corrected distance)
- ✅ Optimality analysis
- ✅ Vigor time course figure
- ✅ All numbers verified against MCMC
- ✅ λ sensitivity documented (Supp Note 1)
- ✅ Joint model documented (Supp Note 2)
- ✅ Discovery document with behavioral descriptives
- ✅ HTML presentation and prereg with embedded figures
