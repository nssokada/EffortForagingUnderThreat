---
result_id: 407
class: choice_vigor_coupling
title: Encounter spike was a demand-driven artifact + threat confound (refuted)
status: refuted_exploratory
prereg_h: []
internal_h: [H11]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 407 — Encounter spike was a demand-driven artifact + threat confound (refuted)

> **Kept for transparency.** This result is documented to record the path from an initial positive finding to its eventual refutation. Numbers from `instructions/memory/hypotheses.md` § H11.

## Overview

An early exploratory analysis reported a robust "encounter spike" — a phasic vigor increase at predator encounter — that appeared specific to attack trials and that we hoped would index a clean reactive imminence signal. On follow-up: the spike vanishes when vigor is residualized against task demand (vigor_resid), and the attack-vs-non-attack contrast in encounter spike is non-significant after controlling for threat level. The original positive finding was driven by the threat confound (high-threat conditions have more attacks AND more vigor, for unrelated reasons).

## Result

- **Raw vigor encounter spike:** appeared significant in initial analysis.
- **After residualizing against demand (vigor_resid):** p = 0.644, NULL.
- **Attack vs non-attack after threat control:** p = 0.126, n.s.

## Interpretation

The encounter spike is not a real phasic signal — it is what you get when you fail to control for two confounds simultaneously (motor demand structure across trial types, threat-attack covariation). The lesson is methodological: defensive-vigor measures need both within-trial demand controls and between-trial threat controls before any "reactive" interpretation is licensed.

The clean reactive signal that survives these controls is the *attack-driven encounter transition* documented in [[result_306]] (post-encounter rates rise by +0.033, t = 6.95, p = 2e-11), not the original encounter-spike claim.

## References

- `instructions/memory/hypotheses.md` § H11 (refuted).
- [[result_306]] — The cleaned-up replacement (attack-driven transition).
- `instructions/memory/hypotheses.md` § H14 (the "two-system architecture, reframed" entry that documents this lesson).
