---
result_id: 306
class: vigor_dynamics
title: Attack triggers larger post-encounter vigor transition (within threat level)
status: supported_exploratory
prereg_h: []
internal_h: [H27]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 306 — Attack triggers larger post-encounter vigor transition (within threat level)

> **⚠️ Exploratory. Numbers from `instructions/memory/hypotheses.md` § H27.**

## Overview

When the predator actually appears (attack trials), post-encounter pressing increases relative to pre-encounter beyond what threat level alone predicts. Within-threat-level attack vs non-attack contrast:

- Post-encounter mean: diff = +0.033, t = 6.95, p = 2 × 10⁻¹¹
- Transition (post − pre): diff = +0.042, t = 5.29, p = 2 × 10⁻⁷
- Pre-encounter: diff = −0.009, t = −1.01, p = 0.31 (correctly null — no foreknowledge of attack)

## Interpretation

This is the clean reactive imminence signal: when a real predator appears, subjects accelerate their pressing in a manner that is not predicted by the trial's threat level alone. The pre-encounter null is reassuring: subjects do not have foreknowledge of whether a trial will be attacked, so pre-encounter rates should not differ between (attack, non-attack) at fixed threat — and they don't. The post-encounter and transition signals are the genuine within-trial reactive response.

This complements [[result_305]] (threat modulates the transition between-subject-aggregate) by showing the within-threat attack-driven version. The two together establish that the encounter transition is sensitive both to *prospective* threat (T) and to *acute* presence of the predator.

## References

- `instructions/memory/hypotheses.md` § H27.
- [[result_305]] — Threat-modulated transition.
- [[result_407]] — Encounter spike artifact (the deprecated spike-based result this supersedes).
