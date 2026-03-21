---
name: Project Goal
description: Target journal, core scientific claim, current status, and what remains for submission to Nature Communications
type: project
---

**Goal:** Submit "A Common Computational Structure Integrates Effort and Threat Across Decision, Emotion, and Action" (Okada, Garg, Wise, Mobbs) to **Nature Communications**.

**Why:** This paper addresses a fundamental question — how do humans balance energetic effort and threat in a unified framework when making decisions, and how does this process relate to subjective emotions and trait-level individual differences? Nature Communications is the target venue.

## Central question
How do humans integrate energetic cost and exposure-dependent threat when making foraging decisions? And how does this computational process connect to subjective affect (anxiety, confidence) and stable trait variables (psychiatric symptom dimensions)?

## Current working framework
A survival-weighted, effort-discounted subjective value model best explains choice behavior. The paper has three pillars:

1. **Choice** — model comparison (WAIC) favors combined effort–threat model with bias term. Strong (R²=0.45, AUC=0.91).
2. **Affect** — model-derived survival predicts trial-level anxiety (−) and confidence (+). Strong within-subject effects. Parameter moderation is limited: z → chronic confidence deficit; k → trait anxiety/confidence; moderation interactions marginal/n.s. after FDR.
3. **Vigor** — **the story is about individual differences, not real-time tracking.** Stable person-level differences in pressing style (ICC up to 0.74) are linked to choice parameters: z → anticipatory preparation, k → global motor suppression, β → onset ramp. But threat does not modulate vigor at the group level, the encounter spike is demand-driven, and the attack contrast is confounded with threat. One genuine reactive effect: terminal sprint on attack trials (generic, parameter-independent).

## Key revision needed for the draft
The current draft (v2) overclaims on vigor. It frames the result as "the same computational structure governs choice, affect, and vigor" — implying a shared real-time survival computation across domains. The data actually shows:
- **Choice ↔ Affect:** Survival computation predicts trial-level affect. Solid.
- **Choice → Vigor (individual differences):** People who value threat/effort differently in choices also press differently. Trait-level correspondence, not shared real-time computation.
- **Choice → Vigor (real-time):** Only terminal survival → terminal vigor survives (small trial-level effect). No encounter spike, no real-time tracking.
- **Affect → Vigor:** Complete null at every level.

The vigor section needs reframing from "common computation" to "individual differences in valuation manifest in execution strategy."

## Current status (as of 2026-03-18)
- **Exploratory sample (N=293):** Core analyses complete. Vigor diagnostics (NB12) have clarified the story.
- **Confirmatory sample (N=350):** Data not yet added to repo; pipeline ready.
- **Draft:** Version 2 (2026-02-23) — needs vigor section rewrite; references empty; figure numbers placeholder.
- **Dead ends closed:** ODE vigor, continuous temporal alignment, encounter spike, attack contrast (see hypotheses.md).
- **Open:** Psychiatric battery factor analysis (blocks H13); confirmatory replication.

## What remains for submission
- **Reframe vigor section** of the draft around individual differences
- Run confirmatory sample through full pipeline
- Psychiatric battery factor analysis → model param associations
- Finalize figures, references, code/data availability
- Ensure all results tables and stats are final

**How to apply:** All work should be evaluated against: "Does this bring the manuscript closer to Nature Communications submission?" The vigor story needs honest reframing. Prioritize analyses that strengthen the choice→affect link and the individual-differences story for vigor. Be skeptical of claims about real-time survival tracking in vigor.
