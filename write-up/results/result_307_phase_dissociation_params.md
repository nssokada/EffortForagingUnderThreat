---
result_id: 307
class: vigor_dynamics
title: Model parameters dissociate across temporal phases of vigor (z, k, β framework)
status: supported_exploratory
prereg_h: []
internal_h: [H7]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 307 — Model parameters dissociate across temporal phases of vigor (deprecated z, k, β framework)

> **⚠️ Exploratory, deprecated three-parameter (z, k, β) framework.** Re-analysis with current (ω, κ) parameters required. Numbers from `instructions/memory/hypotheses.md` § H7.

## Result (legacy)

| Param | Onset (pre-encounter) | Encounter spike | Terminal slope |
|---|---|---|---|
| z (hazard sensitivity) | +0.13 to +0.22 | −0.12 | −0.19 |
| k (effort cost) | −0.07 to −0.22 (global suppressor) | strongest at terminal | — |
| β (threat bias) | +0.18 (onset slope) | +0.14 (post-encounter) | — |

Effect sizes are small (R² = 0.024–0.062 across phases), but the dissociation pattern is consistent: each parameter has a distinct phase-specific vigor signature.

## Interpretation

z, k, β each project onto a different phase of the within-trial vigor timecourse — but with effect sizes in the 2–6% R² range. The pattern is suggestive of three orthogonal motor regulation channels but is not strong enough to anchor a paper claim on its own. The current M4 framework with (ω, κ) does not include β, and the z analog (M4's γ × hazard) is a population-level parameter, not subject-level. Migration is non-trivial and may shift conclusions.

Results 208 and 401 ([[result_208]], [[result_401]]) address the analogous question (cross-channel ω → vigor) in the current framework, and the results there are stronger and cleaner. Result 307 should be treated as a documentation of the deprecated approach, with the H8 framework being the live test.

## References

- `instructions/memory/hypotheses.md` § H7, § H14 ("two-system architecture, reframed").
- [[result_208]], [[result_401]] — Cross-channel ω → vigor (M4-era analog).
