---
result_id: 305
class: vigor_dynamics
title: Encounter transition magnitude scales with threat (corrected encounter-time frame)
status: supported_exploratory
prereg_h: []
internal_h: [H26]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 305 — Encounter transition magnitude scales with threat (corrected encounter-time frame)

> **⚠️ Exploratory.** Corrected-frame result; the wrong-frame version was retracted (`instructions/memory/hypotheses.md` § H25). Numbers from § H26.

## Overview

After correcting an encounter-time frame alignment bug, the pre→post encounter transition in vigor scales with threat probability. Higher threat → larger positive transition. Low: −0.013, Med: +0.049, High: +0.064. Between-subject ANOVA: F = 7.84, p = 0.0004. Post-encounter pressing also threat-modulated (F = 6.88, p = 0.001).

## Note on the frame correction

The wrong-frame version (F = 0.91, p = 0.40) showed no effect. The corrected frame uses trial-start-relative encounterTime alignment rather than the prior absolute-time alignment. This was a methodological fix that revealed real signal previously masked by misalignment.

## References

- `instructions/memory/hypotheses.md` § H25 (retraction), § H26 (corrected result).
- [[result_306]] — Attack-driven encounter transition (the within-threat companion).
