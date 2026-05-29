---
result_id: 308
class: vigor_dynamics
title: Trial-level pressing rate predicts escape on attack trials beyond choice, threat, and distance
status: supported_exploratory
prereg_h: []
internal_h: [H37]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 308 — Trial-level pressing rate predicts escape on attack trials beyond choice, threat, and distance

> **⚠️ Exploratory, deprecated.** Numbers below are from `instructions/memory/hypotheses.md` § H37 (last validated 2026-03-24). Migration to a current notebook required before manuscript inclusion.

## Overview

Result 403 showed at the between-subject level that vigor dominates choice as a predictor of escape rate. The within-subject (trial-level) version of the same question asks: holding subject, choice, threat, and distance constant, does pressing rate on a given trial predict whether that trial ends in escape? We fit a trial-level LMM on all attack trials in the exploratory sample, with subject random intercept and pre-encounter pressing rate as the focal predictor. Vigor enters with β = +0.091, p ≈ 10⁻⁷⁷ on N ≈ 10,257 attack trials, and adding vigor to a choice-only baseline lowers AIC by 341 — a decisive within-subject confirmation of the survival-relevance of motor execution.

## Hypothesis

**Statement.** "Pre-encounter pressing rate predicts escape on attack trials, controlling for choice, threat, and distance." (Internal H37.)

**Predicted direction.** β(vigor) > 0 in a trial-level LMM.

## Data Source (legacy)

- **Sample:** N = 290 exploratory.
- **Unit of analysis:** Attack trial (N ≈ 10,257).

## Method (legacy)

Trial-level logistic mixed-effects model:

```
escaped ~ vigor_z + choice_z + threat_z + dist_z + (1|subj)
```

`vigor_z` is the z-scored pre-encounter pressing rate; `choice_z` is the binary cookie choice (heavy = 1, light = 0); threat and distance are the trial's condition values, z-scored.

## Result (legacy, from internal H37)

| Coefficient | β | p |
|---|---|---|
| Vigor | **+0.091** | **10⁻⁷⁷** |
| Choice | **−0.177** | ≈ 0 (choosing hard *hurts* escape) |
| Threat | (sign as expected, negative) | — |
| Distance | (sign as expected, negative) | — |

**Δ AIC for adding vigor to choice-only model:** 341.

**N attack trials:** ≈ 10,257 (across 290 subjects).

**Verdict:** Vigor predicts escape beyond choice/threat/distance at the trial level. Effect is small per trial but extremely reliable.

## Interpretation

The trial-level test cleanly reproduces the between-subject finding in [[result_403]] without any of the cross-subject confounds. Holding the same subject's choice, threat, and distance constant, the trials on which they pressed harder were more likely to end in escape — by a small but extraordinarily reliable margin (β = +0.091 standardized, p ≈ 10⁻⁷⁷). The ΔAIC of 341 for adding vigor to a choice-only model is decisive by any standard model-comparison criterion.

The within-subject framing is important because it rules out the possibility that the between-subject vigor → escape result is a confounded individual-difference effect (e.g., motor skill correlates with general task engagement, which correlates with both vigor and survival). Here, we ask the question "on the same subject's trials, does the trial-by-trial variation in pressing rate predict the trial-by-trial variation in escape outcome?" The answer is yes, in the predicted direction, with overwhelming statistical evidence.

Combined with [[result_402]] (ω predicts vigor) and [[result_403]] (vigor dominates escape between-subject), this completes a chain: subjects who internalize capture as costly (high ω) press harder (cross-channel correlation), and pressing harder yields better trial-level escape outcomes (this result). The chain links the joint-model parameter ω to a directly survival-relevant behavioral consequence via the vigor channel — a stronger story than the model's choice-only predictions.

## Caveats & Limitations

- **Status: `supported_exploratory`.** Confirmatory replication required.
- **β = +0.091 standardized is small per trial,** but the extraordinarily large N and the structural relationship between pressing rate and movement speed make this a real and consequential effect at the population level.
- **The "choosing hard hurts escape" result (β_choice = −0.177)** is mechanically correct: heavy cookies are farther from safety. The result does not mean choice causes failure — it means that at fixed vigor, the heavy-cookie path is harder. Vigor is what compensates.
- **No condition × vigor interactions tested.** The vigor effect on escape may be larger at high threat or far distance; that decomposition is left for [[result_307]] (phase dissociation) and the joint model PPCs.

## Replication

Migration needed. Steps:

1. Load `behavior_rich.csv` (both samples), filter to attack trials.
2. Z-score pre-encounter pressing rate per subject.
3. Fit `escaped ~ vigor_z + choice_z + threat_z + dist_z + (1|subj)` via `statsmodels` (or pymc / bambi for Bayesian version).
4. Compute ΔAIC vs choice-only baseline.
5. Update file with both samples' results.

## References

**Related results:**
- [[result_402]] — ω → anticipatory vigor (the upstream parameter link).
- [[result_403]] — Vigor dominates escape (between-subject).
- [[result_307]] — Phase dissociation by parameters (the temporal decomposition).
- [[result_201]] — Joint model M4 (mechanism via S(u, T, D)).

**Source:**
- `instructions/memory/hypotheses.md` § H37 (last validated 2026-03-24).
