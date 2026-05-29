---
result_id: 302
class: vigor_dynamics
title: Model-derived survival predicts trial-level vigor at terminal phase only (partial)
status: partial
prereg_h: []
internal_h: [H6]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 302 — Model-derived survival predicts trial-level vigor at terminal phase only (partial)

> **⚠️ Exploratory, partial. Numbers from `instructions/memory/hypotheses.md` § H6.** Uses deprecated FET parameters; current-model replication required.

## Overview

If lower survival probability drives higher motor output trial-by-trial, we should see S_trial enter negatively in a trial-level LMM for vigor. Internal H6 tested this with phase-specific LMMs and found a robust effect only at the **terminal mean** phase (S_trial β = −0.011, p_fdr = 0.0002). Onset, encounter spike, and other phases were null. Between-subject ANOVAs at the threat-grouping level were all non-significant (all p > 0.20).

## Result (legacy, internal H6)

- **Terminal mean vigor ~ S_trial + effort + (1|subj):** β(S_trial) = −0.011, p_fdr = 0.0002. ✓
- Other phases (onset, encounter spike, post-encounter): all non-significant after FDR.
- Between-subject ANOVA of threat on any phase metric: all p > 0.20.

## Interpretation

The survival-vigor coupling is real but localized: only the terminal-phase pressing rate (where escape is most imminent) responds to model-derived survival probability at the trial level. The other phases show no such relationship. Combined with the H1c finding that threat raises mean vigor ([[result_103]]) and the H8 finding that ω predicts anticipatory vigor cross-channel ([[result_402]]), this suggests the survival → vigor relationship is mediated more by *stable individual differences* (ω predicts level) than by *trial-by-trial state* (only terminal phase tracks S_trial).

The phase-specific localization motivates the broader "individual differences in pressing style" framing for the vigor results (`instructions/memory/hypotheses.md` § H15) and warns against over-interpreting any single trial-level "survival drives vigor" story.

## References

- [[result_103]] — H1c mean vigor under threat.
- [[result_402]] — Cross-channel ω → vigor (the individual-difference companion).
- [[result_307]] — Phase dissociation by parameters.
- `instructions/memory/hypotheses.md` § H6, § H15.
