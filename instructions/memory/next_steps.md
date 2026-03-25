---
name: Next Steps
description: Priority action items — binary-E model validation with parquet data is BLOCKING, then prereg update, remaining hypotheses, confirmatory sample
type: project
---

# Next Steps (as of 2026-03-24, end of session)

## Priority 0: BLOCKERS

- [ ] **Install pyarrow** to read `smoothed_vigor_ts.parquet` — needed to validate new models with actual vigor data
- [ ] **Verify binary-E choice model + effort-controlled vigor model with parquet data** — current results used behavior_rich proxy. Must confirm +88 ELBO, λ=0.8, and β-δ coupling hold with actual timeseries data
- [ ] **Rerun full MCMC pipeline** with new model specification (binary E, effort-controlled vigor)

## Priority 0.5: Prereg finalization

- [x] Rewrote prereg in AsPredicted format (`drafts/prereg_aspredicted.md`)
- [x] Updated detailed prereg (`drafts/preregistration.md`) to H1-H7 simple numbering
- [x] H1 analyses updated: binary E in H1a `(1|subj)`, threat×dist interaction in H1b, DV definition fixed
- [x] H1 tested and passes all criteria
- [x] H2 tested (Pearson r confirmed, r=-0.50, split-half robust)
- [x] H3 tested (empirical escape rates, reallocation→earnings r=0.58, 94% too-cautious)
- [ ] **UPDATE PREREG for new model specification:** Binary E choice model replaces graded E. Need to update H4 (model comparison) and H5 (vigor HBM) to reflect: (1) binary effort, (2) effort-controlled vigor model with γ·E_chosen, (3) λ=0.8
- [ ] **Walk through H4-H7** with updated model specification
- [ ] **Add allocation dimension hypothesis** — angle vs magnitude predicting outcomes, possibly as new H or extension of H3/H6
- [ ] After all hypotheses reviewed: regenerate AsPredicted
- [ ] Submit to AsPredicted before opening confirmatory data

## Priority 1: Generate H1 figure
Redesigned as 3 standalone panels (script: `scripts/plotting/plot_h1_panels.py`).
All use within-subject SEM (Cousineau-Morey), grouped bar charts, plotter.py styling.

- [x] **Panel A — Choice bars:** `fig_h1a_choice.pdf` — grouped bars, distance × threat. Done.
- [ ] **Panel B — Vigor timecourse:** `fig_h1b_vigor_timecourse.pdf` — excess effort over 0–2.5s from trial start, 3 threat lines with SEM ribbons. **BLOCKED: needs pyarrow** to read `smoothed_vigor_ts.parquet`. Code is ready in `plot_h1_panels.py` (currently skipped). Install pyarrow then rerun.
- [x] **Panel C — Affect bars:** `fig_h1c_affect.pdf` — side-by-side anxiety + confidence, grouped bars by distance × threat. Done.

**Why timecourse for Panel B:** Mean excess effort difference is tiny (~0.06 at near, ~0 at far). The timecourse shows the separation band between threat lines over 2.5s which is visually compelling. Bar chart of means would not be.

Old versions (kept for reference): `fig_h1_threat_shifts.pdf` (v1, 1×3 lines), `fig_h1_threat_shifts_v2.pdf` (v2, 2×2 with difference scores).

## Priority 2: Remaining paper figures
- [ ] **Fig 1** — Task schematic + 5-model comparison + choice by T×D + param distributions (NEEDS UNITY SCREENSHOT)
- [ ] **Fig 2** — Vigor: δ distribution / excess by threat / within-choice control
- [ ] **Fig 3** — Already have fig4_coherent_shift.pdf, needs optimality panel
- [ ] **Fig 4** — Already have fig_s_metacognition.pdf

## Priority 3: Run confirmatory sample (N=350)
- [ ] Preprocess through stages 1-5
- [ ] Fit choice model (M5) → k, β, λ
- [ ] Fit vigor HBM (λ fixed from choice) → α, δ
- [ ] Compute cross-domain correlations + posterior bootstrap
- [ ] Run affect LMMs
- [ ] Test all H1-H7 against preregistered criteria
- [ ] Data: `data/confirmatory_350/raw/` (collected, not preprocessed)

## Priority 4: Tables
- [ ] Supp Table 1: Full 5-model comparison
- [ ] Supp Table 2: Population parameters from MCMC
- [ ] Supp Table 3: Cross-domain correlation matrix
- [ ] Supp Table 4: Clinical profile
- [ ] Supp Table 5: Parameter recovery

## Priority 5: Paper polish
- [ ] Update abstract with optimality finding
- [ ] Figure legends match current figures
- [ ] Ecological limitations paragraph
- [ ] Trim MH section
- [ ] Complete references (20 → 30-40)
- [ ] Format for Nature Comms

## Key files for next session
- `drafts/prereg_aspredicted.md` — AsPredicted format prereg (WORKING COPY)
- `drafts/preregistration.md` — detailed prereg (updated to H1-H7)
- `drafts/simple_prereg.md` — authoritative hypothesis numbering
- `scripts/plotting/plot_h1_figure.py` — ready to run, needs conda
- `.devcontainer/Dockerfile` — updated with conda, needs rebuild

## Already done
- ✅ MCMC complete — choice, vigor, and joint models all run. Outputs in `results/stats/mcmc_*.csv`
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
- ✅ Prereg rewritten in AsPredicted format
- ✅ H1 analyses switched from ANOVA to LMMs (logistic choice, linear vigor/affect)
- ✅ Dockerfile updated with conda/scientific Python
