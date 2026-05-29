---
result_id: 304
class: vigor_dynamics
title: Distance modulation of pre-encounter pressing is the single strongest vigor feature → parameter link
status: supported_exploratory
prereg_h: []
internal_h: [H24]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 304 — Distance modulation of pre-encounter pressing is the single strongest vigor feature → parameter link

> **⚠️ Exploratory, deprecated framework (z, k, β).** Re-analysis with current (ω, κ) needed. Numbers from `instructions/memory/hypotheses.md` § H24.

## Overview

Of the 12 encounter-window vigor features used in [[result_303]]'s PLS, the single strongest individual predictor of model parameters is the per-subject regression slope of pre-encounter pressing rate on distance. Subjects who modulate their pressing rate strongly with distance (pressing harder at far distances during the pre-encounter window) are the subjects with low k — those who are *least* effort-discounting. The bivariate correlation r = −0.435 between dist_pre and k is the strongest single vigor-feature-to-parameter link in the project.

## Hypothesis

**Statement.** "Individual differences in how much people modulate pressing rate by distance in the pre-encounter window is the strongest single predictor of choice model parameters — particularly k (effort discounting)." (Internal H24.)

## Result (legacy, from internal H24)

| Vigor feature × parameter | r |
|---|---|
| **dist_pre × k** | **−0.435** (strongest) |
| dist_pre × z | −0.270 |
| dist_pre × β | −0.212 |
| dist_trans × k | +0.407 (flipped: high-k subjects show bigger distance-dependent encounter transitions) |

**PLS Component 1 loadings:** dist_pre (+0.668) and dist_trans (−0.658) dominate.

## Interpretation

A subject's k governs effort discounting in choice — high k means effort cost weighs heavily, so they avoid heavy cookies. The vigor signature of high k is *uniform* pressing across distances: a high-k subject does not waste energy by modulating pressing rate based on how far they have to travel. Low-k subjects, by contrast, scale their pre-encounter pressing rate with distance, suggesting they treat distance as a meaningful task variable that warrants motor adjustment.

The sign-flipped result on dist_trans (the encounter transition magnitude) is informative: high-k subjects show *larger* distance-dependent transitions at encounter, even though they show smaller distance-dependent pre-encounter modulation. This suggests two complementary regulation strategies: low-k subjects gradually scale pressing during the pre-encounter window (continuous regulation), while high-k subjects hold a more uniform pressing rate and rely on the encounter event to trigger an adjustment (event-driven regulation).

PLS Component 1 captures this dist_pre vs dist_trans contrast with nearly equal-magnitude opposite-sign loadings, suggesting these two features index complementary aspects of the same underlying regulation strategy.

## Caveats & Limitations

- **Status: `supported_exploratory`.** Re-run with (ω, κ) needed; given κ is the M4 analog of k, the κ link is expected to replicate.
- **Bivariate correlations** are sensitive to outliers; the PLS analysis in [[result_303]] is the multivariate confirmation.
- **Pre-encounter vs encounter-window framing** depends on the corrected encounter-time alignment (see retraction in `instructions/memory/hypotheses.md` § H25 for the earlier mis-frame).

## Replication

See [[result_303]]. Same migration steps.

## References

**Related results:**
- [[result_303]] — Multivariate PLS context.
- [[result_402]] — Cross-channel ω/κ → vigor.

**Source:**
- `instructions/memory/hypotheses.md` § H24.
