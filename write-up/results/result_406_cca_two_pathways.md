---
result_id: 406
class: choice_vigor_coupling
title: CCA recovers two independent parameter → behavior pathways (deprecated z, k, β)
status: supported_exploratory
prereg_h: []
internal_h: [H35]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 406 — CCA recovers two independent parameter → behavior pathways

> **⚠️ Exploratory, deprecated (z, k, β) framework.** Re-analysis with current (ω, κ) needed. Numbers from `instructions/memory/hypotheses.md` § H35.

## Result (legacy)

Canonical correlation analysis on (z, k, β) → (choice, vigor) yields two significant canonical dimensions:

- **Dim 1:** r = 0.909, p = 10⁻¹¹³ → maps almost exclusively to Choice.
- **Dim 2:** r = 0.289, p = 5 × 10⁻⁷ → maps almost exclusively to Vigor.

MANOVA: Wilks' λ significant for all three params (all p ≈ 0).

## Interpretation

The CCA solution finds two orthogonal pathways: one carries (z, k, β) jointly into choice, the other carries them into vigor. The two dimensions are nearly orthogonal in canonical-loading space, with the choice dimension explaining the bulk of the canonical variance and the vigor dimension a smaller but cleanly separable component. This is the multivariate confirmation of the parameter → behavior dissociation that appears piece-by-piece in [[result_402]] and [[result_404]].

Under the current M4 framework with only (ω, κ), the analog CCA would yield at most two dimensions, with similar interpretation: one mostly choice-loaded, one mostly vigor-loaded.

## References

- `instructions/memory/hypotheses.md` § H35.
- [[result_402]] — Cross-channel ω → vigor.
- [[result_404]] — Threat reverses coupling.
