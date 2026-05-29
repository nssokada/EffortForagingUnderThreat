---
result_id: 104
class: behavioral_effects
title: Vigor timecourse splines distinguish encounter and threat effects (H2 family)
status: untested
prereg_h: [H2a, H2b]
internal_h: []
samples: []
notebooks: [notebooks/analysis/H2_vigor_dynamics.ipynb]
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: null
---

# Result 104 — Vigor timecourse splines distinguish encounter and threat effects (H2 family)

> **⚠️ Deferred — `H2_vigor_dynamics.ipynb` has only its import cell executed (1/5 code cells run).** Re-execution required before this result can be validated.

## Overview (planned)

The H2 family of preregistered tests examines vigor dynamics across the threat-imminence continuum: (H2a) whether predator encounter triggers a rapid motor spike; (H2b) whether the temporal shape of vigor differs by encounter status and threat level. The analysis uses GAMs with natural cubic splines (K = 10) on epoch-level vigor data, with mixed-effect random intercepts by subject, and likelihood ratio tests for smooth-by-condition interactions.

## What needs to happen

1. Execute `notebooks/analysis/H2_vigor_dynamics.ipynb` end-to-end via the PYTHONPATH-nbconvert recipe.
2. Validate H2a encounter spike (paired t against zero) and H2b GAM LRTs for encounter status × time and threat × time interactions.
3. Populate this file with frontmatter (samples, outputs, figures) and Result/Interpretation sections.

A relevant cached number from `results/stats/confirmatory_hypothesis_results.csv`:

- H2a encounter spike: d = 0.647, p = 8.73e-24 (confirmatory)
- H2b GAM encounter LRT: χ² = 1024.8, p ≈ 0 (confirmatory)
- H2b GAM threat LRT: χ² = 114.8, p = 2.55e-15 (confirmatory)

Exploratory equivalents will appear once the notebook is executed.

## References

- `notebooks/analysis/H2_vigor_dynamics.ipynb`
- `instructions/memory/hypotheses.md` § H11 (encounter spike, refuted in deprecated framework but H2a as cached suggests survives in current framework — needs re-check)
- [[result_103]] — H1c vigor under threat (the static counterpart of these dynamics).
- [[result_307]] — Phase dissociation by parameters.
