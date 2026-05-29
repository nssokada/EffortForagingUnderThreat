---
result_id: 602
class: individual_differences
title: AMI apathy tracks vigor, not choice — "adaptive apathy"
status: supported_exploratory
prereg_h: []
internal_h: [H39]
samples: [exploratory_290]
notebooks: [notebooks/analysis/H10_mediation.ipynb]
scripts: []
outputs: [results/stats/avoid_activate/h10_mediation_main.csv, results/stats/avoid_activate/h10_mediation_specificity.csv]
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 602 — AMI apathy tracks vigor, not choice — "adaptive apathy"

## Overview

The Apathy Motivation Index (AMI) is the one clinical measure in this project's battery that shows a robust task-behavior relationship. AMI total scores correlate positively with mean vigor (β = +0.311, R² = 0.093 in exploratory) — counterintuitively, *higher* self-reported apathy predicts *higher* pressing rate. The choice channel shows no relationship (β = −0.056, n.s.). The effect survives FDR correction and aligns with the off-diagonal-group result in [[result_405]]: LH subjects (chose easy, pressed hard) report more apathy AND perform better on attack trials.

## Hypothesis

**Statement.** "Self-reported apathy (AMI) correlates with vigor (positive — high pressers report more apathy) but not choice." (Internal H39.)

## Result (legacy, internal H39)

| Regression | β | R² | FDR |
|---|---|---|---|
| AMI ~ vigor + choice + interaction | β(vigor) = **+0.311** | **0.093** | ✓ survives |
| | β(choice) = −0.056 | | n.s. |

H10 mediation analyses (`H10_mediation.ipynb`) develop this further by testing whether confidence mediates (ω, κ) → AMI; that work is documented separately.

## Interpretation

The "adaptive apathy" pattern is striking and clinically interesting. Subjects who score high on self-reported apathy (don't want to do things) actually press *harder* during the task — they execute well when they do engage. This is the LH quadrant phenotype from [[result_405]] rendered as a continuous relationship with a clinical scale: people who "feel apathetic" by self-report are not behaviorally apathetic; they are selective about when to engage but commit fully when they do.

The choice null is also informative. AMI does not predict cookie selection (whether subjects go for the heavy cookie) — only execution intensity. This dissociation is consistent with apathy being primarily a motivational *engagement* phenomenon rather than a *decision* phenomenon: apathic subjects can decide normally, but their engagement profile differs.

The result has implications for affect-mediated psychopathology accounts: the link from joint-model parameters to clinical scales runs through confidence and engagement, not through the parameters directly ([[result_601]] establishes the parameter-level null). The H10 mediation work shows that confidence specifically mediates ω/κ → AMI, completing a three-step bridge from foraging computation to clinical relevance.

## Caveats & Limitations

- **Status: `supported_exploratory`.** Confirmatory replication required.
- **AMI total** is the strongest hit; sub-scales (AMI-Social, AMI-Behavioural, AMI-Emotional) show weaker patterns and are reported in the H10 mediation analyses.
- **The "adaptive apathy" framing is counter to clinical intuition** (apathy → less doing) but consistent with the task data. Worth a deliberate paragraph in any clinical-implications section to avoid reader confusion.
- **Pooled samples (~580 subjects total) are used in the H10 mediation work** but the H39 univariate result is exploratory-sample only and should be re-tested in confirmatory.

## References

- `instructions/memory/hypotheses.md` § H39.
- `notebooks/analysis/H10_mediation.ipynb` — affect-mediated paths from parameters to AMI.
- [[result_405]] — Off-diagonal LH group reports more apathy and performs better.
- [[result_601]] — Psych symptoms × params null (the context that makes this AMI hit notable).
