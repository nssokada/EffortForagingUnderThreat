---
result_id: 603
class: individual_differences
title: STAI-Trait scoring bug — identified and corrected (methodological note)
status: retracted
prereg_h: []
internal_h: []
samples: [exploratory_290, confirmatory_281]
notebooks: []
scripts: [scripts/preprocessing/]
outputs: []
figures: []
created: 2026-05-28
last_run: 2026-05-28
---

# Result 603 — STAI-Trait scoring bug — identified and corrected (methodological note)

> **Methodological / supplementary.** Documents a scoring error discovered and fixed during clinical-analysis development.

## The bug

The STAI-Trait scoring originally applied the State-form reverse-coding to the Trait items. State and Trait subscales of the STAI have *different* reverse-coding patterns — applying the wrong set inverts the signal on the affected items, producing trait scores that are partially anti-correlated with what they should be.

## The fix

Reverse-coding was corrected to use the Trait-form key. Re-scored values are now used throughout.

## Residual concerns

Even after the fix:

- STAI-T standard deviation in this sample is low (≈ 5.8) compared to published norms.
- STAI-T correlates *negatively* with other distress measures (DASS-21 Anxiety, OASIS, PHQ-9) in this sample, which is unexpected.

The second issue suggests something beyond the scoring fix may still be problematic — possibly a presentation-order or instruction issue. STAI-T results should be reported with this caveat or excluded from primary clinical analyses.

## Impact on prior results

The STAI-T-relevant cells in [[result_601]] and the H13 internal sweep are computed with the corrected scoring. No published claims rest on STAI-T as the primary outcome; the variable appears in transdiagnostic correlation sweeps only, where the broader null pattern absorbs the STAI-T-specific concern.

## References

- `instructions/memory/hypotheses.md` § H13 (where the bug-and-fix history is also documented).
- [[result_601]] — Clinical null sweep (includes corrected STAI-T).
