---
result_id: 309
class: vigor_dynamics
title: ODE-fit motor kinetics α — parameter is degenerate (dead end)
status: dead-end
prereg_h: []
internal_h: [H12]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: []
created: 2026-05-28
last_run: 2026-05-28
---

# Result 309 — ODE-fit motor kinetics α — parameter is degenerate (dead end)

> **Dead end — preserved for transparency.** Documents an analytical direction that did not yield interpretable results.

## What was tried

Fit an exponential-rise ODE to the encounter-window vigor timecourse per subject, parameterized by a time-constant α intended to capture individual differences in motor mobilization speed.

## Why it died

α is degenerate — the parameter is not reliably identifiable from the available trial counts per subject (~27 attack trials per subject). Multiple very different α values produce indistinguishable fits to subject-level data.

## Lesson

The trial structure of this task (encounter window is ~2s, ~5 Hz pressing → ~10 data points per trial × ~27 trials = ~270 observations per subject) is insufficient for a continuous ODE-based parameterization of within-trial dynamics. The corresponding individual differences are better captured by event-aligned count-based features (see [[result_303]] for the PLS approach that worked).

## References

- `instructions/memory/hypotheses.md` § H12 (dead end entry).
- [[result_303]] — PLS encounter window features (the working alternative).
