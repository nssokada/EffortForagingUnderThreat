---
result_id: 506
class: metacognition
title: Confidence miscalibration tracks the choice–vigor dissociation
status: supported_exploratory
prereg_h: []
internal_h: [H38]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 506 — Confidence miscalibration tracks the choice–vigor dissociation

> **⚠️ Exploratory, deprecated framework.** Numbers from `instructions/memory/hypotheses.md` § H38. The choice-vigor quadrant analysis is model-free (uses raw p_heavy and mean vigor) and should replicate trivially; the confidence-miscalibration link depends on per-subject confidence ratings which are intact in current data.

## Overview

The choice-vigor coupling reversal in [[result_404]] creates four quadrants of subjects: HH (chose hard, pressed hard), HL (chose hard, pressed easy), LH (chose easy, pressed hard), LL (chose easy, pressed easy). These quadrants differ markedly in their confidence-versus-escape calibration: HL subjects are overconfident (confidence higher than their escape rate would warrant by +0.98 SD) and LH subjects are underconfident (confidence lower than warranted by −1.18 SD). A continuous regression of confidence-bias on z-scored choice and vigor recovers the same pattern: choice positively drives miscalibration (β = +0.423), vigor negatively drives it (β = −0.783), with R² = 0.415. The result is the strongest single affect finding in the project and ties the choice-vigor dissociation work to the metacognitive monitoring framework.

## Hypothesis

**Statement.** "The choice-vigor quadrants predict confidence miscalibration (confidence relative to actual escape rate)." (Internal H38.)

## Result (legacy, from internal H38)

**Quadrant ANOVA on confidence bias (conf_z − escape_z):**
- F = 50.2, p = 10⁻²⁶
- HL bias = **+0.98** (overconfident)
- LH bias = **−1.18** (underconfident)

**Continuous regression** (confidence bias ~ choice_z + vigor_z):
- R² = **0.415**
- β(choice → overconfidence) = **+0.423**
- β(vigor → accurate calibration) = **−0.783** (negative means vigor reduces overconfidence — high-vigor subjects are accurately calibrated or underconfident)

## Interpretation

This is the strongest single affect finding in the project. The choice-vigor coupling reversal in [[result_404]] is not just a behavioral quirk — it creates systematic and large metacognitive errors. Subjects in the HL quadrant (chose the hard cookie despite pressing softly) are overconfident: their confidence ratings are nearly one standard deviation higher than their actual escape rate would warrant. LH subjects (chose easy but pressed hard) are underconfident by an even larger margin. The continuous regression confirms the quadrant pattern with a clean R² = 0.415, meaning that nearly half the between-subject variance in confidence miscalibration is explained by just two behavioral summaries.

The signs are diagnostic. Choice drives confidence in a *self-belief* direction — subjects who choose the heavy cookie are people who *believe* they can handle it, and that belief inflates their confidence ratings independent of their actual vigor or escape rate. Vigor drives confidence in a *reality-check* direction — subjects who actually press hard are more often correct about their capabilities, so they don't need to inflate confidence ratings. The two channels operate at cross-purposes when they disagree: HL subjects let their (over-optimistic) choice drive their confidence; LH subjects let their (under-acknowledged) vigor drive their accurate-to-pessimistic confidence.

Substantively, this links the joint-model behavioral framework to the metacognitive layer that the H5 family explores. Confidence is not just a readout of ω (as in [[result_503]]); it is also a readout of the *alignment between intention and execution*. When that alignment is broken (HL or LH), confidence becomes miscalibrated in a predictable direction. The finding deserves a dedicated panel in any choice-vigor-coupling figure and connects naturally to clinical work on metacognitive deficits in apathy and anxiety.

## Caveats & Limitations

- **Status: `supported_exploratory`.** Confirmatory replication required.
- **Confidence bias is defined as z(confidence) − z(escape rate),** which assumes both quantities are appropriately scaled. The R² = 0.415 result is large for individual-difference work in psychology.
- **HL and LH groups are formed by median splits on choice and vigor;** the continuous regression is the inferential anchor.
- **The H5c result in [[result_503]] showed ω → confidence directly,** and the result here shows choice and vigor → confidence-miscalibration. The two are not contradictory: ω drives mean confidence level, while the choice-vigor coupling drives *bias* relative to actual ability.

## Replication

Migration needed.

1. Compute per-subject p_heavy, mean_vigor, mean_confidence, escape_rate from `profiles_{sample}.csv`.
2. Construct quadrants (median splits) and compute confidence bias = z(confidence) − z(escape rate).
3. Run ANOVA on confidence bias by quadrant.
4. Run continuous regression: `bias ~ choice_z + vigor_z`.
5. Update file with both samples.

## References

**Related results:**
- [[result_403]] — Vigor dominates escape (the survival-outcome anchor that licenses the "actually warranted" half of the calibration metric).
- [[result_404]] — Threat reverses choice-vigor coupling (the source of the quadrants).
- [[result_405]] — Off-diagonal groups differ in confidence, calibration, and apathy.
- [[result_503]] — ω → confidence (mean confidence, complementary).

**Source:**
- `instructions/memory/hypotheses.md` § H38.
