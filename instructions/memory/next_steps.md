---
name: Next Steps
description: Priority action items for Nature Comms submission — prereg finalization, figures, confirmatory sample. MCMC already complete.
type: project
---

# Next Steps (as of 2026-03-24, end of session)

## Priority 0: BLOCKERS before preregistration submission
- [ ] **H1b verification:** Run the H1b LMM on exploratory data: `excess_effort ~ threat_z + dist_z + effort_chosen_z + (1 | subject)`. Need to confirm β(threat) > 0 — that threat increases pressing beyond what demand explains. If it fails, fall back to terminal vigor (F=35) operationalization. Script ready: `scripts/plotting/plot_h1_figure.py` (also needs to be run).
- [ ] **Rebuild devcontainer** with conda/scientific Python (Dockerfile updated, needs rebuild via "Dev Containers: Rebuild Container")

## Priority 0.5: Prereg finalization (in progress)
- [x] Rewrote prereg in AsPredicted format (`drafts/prereg_aspredicted.md`)
- [x] Updated detailed prereg (`drafts/preregistration.md`) to H1-H7 simple numbering
- [x] Updated `instructions/memory/hypotheses.md` with concordance
- [x] H1 analyses updated to LMMs throughout (logistic for choice, linear for vigor/affect)
- [ ] **Resume walkthrough at H2** — H1 is finalized (LMMs, all-pairwise monotonicity). H2 text looks good but Noah hasn't confirmed yet. Need to walk through H3-H7 with same level of detail.
- [ ] **Key decisions still needed from Noah:**
  - H2: Confirm Pearson r on shift scores is the right test (vs Spearman)
  - H3: How exactly to compute optimal policy / reallocation index — needs precise specification
  - H4-H7: Walk through each, same as we did for H1
- [ ] After all hypotheses reviewed: regenerate the HTML version and AsPredicted .md
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
