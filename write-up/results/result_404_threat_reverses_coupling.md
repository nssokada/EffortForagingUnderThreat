---
result_id: 404
class: choice_vigor_coupling
title: Threat reverses the sign of the choice–vigor correlation across subjects
status: supported_exploratory
prereg_h: []
internal_h: [H32, H36]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 404 — Threat reverses the sign of the choice–vigor correlation across subjects

> **⚠️ Exploratory, deprecated framework.** Originally derived from a model with the deprecated three-parameter (z, k, β) FET architecture. The choice-vigor coupling phenomenon itself is model-free (it correlates raw per-subject p_heavy and per-subject mean vigor within each threat level) and should replicate trivially with current data, but the canonical cell lives in `notebooks/_deprecated/joint_coupling_models/`. Numbers below are from `instructions/memory/hypotheses.md` § H32 + H36 (last validated 2026-03-24). Recommend migration before manuscript inclusion.

## Overview

The two behavioral channels — choice (p_heavy) and execution (mean vigor) — are approximately uncorrelated across subjects when collapsed across all conditions (see internal H29). But this aggregate null hides a striking interaction with threat: at low threat the two are positively correlated (subjects who choose hard also press hard), at intermediate threat they are uncorrelated, and at high threat they are *negatively* correlated (the harder-pressing subjects choose easier). We tested this per-threat-level correlation in the exploratory sample (legacy), plus a cross-level LMM confirming the threat × choice interaction on vigor.

## Hypothesis

**Statement.** "At low threat, choice and vigor align (people who choose hard press hard). At high threat, they decouple (people who choose easy press hard)." (Internal H32, with H36 as the LMM confirmation.)

**Predicted direction.** Per-threat correlations: r(T = 0.1) > 0, r(T = 0.5) ≈ 0, r(T = 0.9) < 0. Cross-level LMM: choice × threat interaction β < 0.

## Data Source (legacy)

- **Sample:** N = 290 exploratory.
- **Unit of analysis:** Subject × threat level for the correlation; trial-level (with subject random effect) for the LMM.

## Method (legacy)

**Per-threat correlation:** Within each threat level, compute subject-level p_heavy (proportion of heavy choices at that threat) and subject-level mean vigor (mean normalized pressing rate at that threat). Pearson r across subjects.

**Cross-level LMM (internal H36):** `vigor_trial ~ choice_subj_z × threat_z + dist_z + (1|subj)`.

**Fisher z-test:** Compare the per-threat correlations against the null of equal r across threat levels.

## Result (legacy, from internal H32 + H36)

**Per-threat correlation between p_heavy and mean vigor (across subjects):**

| Threat | r | p |
|---|---|---|
| 0.1 | **+0.196** | 0.001 |
| 0.5 | +0.013 | n.s. |
| 0.9 | **−0.219** | < 0.001 |

**Cross-level LMM** (H36):
- choice × threat interaction: β = −0.022, z = −3.54, p = 0.0004
- Survives addition of random slopes: p = 0.002
- Fisher z-test on per-threat correlations: z = 5.07, p < 0.0001

**Verdict:** Threat reverses the sign of the choice-vigor correlation across subjects. The reversal is detectable both as a per-threat correlation and as a continuous threat × choice interaction in a trial-level LMM.

## Interpretation

The aggregate null on choice-vigor correlation (`r ≈ −0.02` across all conditions, internal H29) is hiding a real threat-modulated structure. Under low threat, subjects who choose hard also press hard — they are the "high-effort generalists." Under high threat, the population reorganizes: the conservative-but-capable subgroup (chose easy, pressed hard) becomes salient and pushes the correlation negative. The intermediate threat level is the crossover where the two populations balance.

The substantive reading connects to the trait dispositions identified by the joint model. The high-threat condition is where the cost of capture (ω) and the cost of effort (κ) trade off most sharply: a high-ω subject avoids the heavy cookie (choice goes down) but compensates by pressing harder when they do go for it (vigor stays high). At low threat, ω has little bite, so neither parameter creates the conservative-but-capable strategy and the correlation reflects the residual "generalist" pattern. The LMM confirms this with a continuous interaction term that survives random slope variation — it is not driven by a small subgroup of subjects with extreme values.

This result is the empirical anchor for framing choice and vigor as *complementary* survival strategies rather than redundant ones. The threat-modulated reversal would not arise if choice and vigor were simply two readouts of a single "behavioral activation" trait; it requires that the two channels respond to threat by different routes and that those routes can favor opposite behaviors depending on conditions. It also licenses the metacognitive miscalibration result in [[result_506]]: subjects in the off-diagonal quadrants (HL, LH) are visible specifically *because* the threat reversal creates these quadrants in the first place.

## Caveats & Limitations

- **Status: `supported_exploratory`.** Confirmatory replication required. The basic correlation analysis is straightforward to re-run on `behavior_rich.csv` + `profiles_{sample}.csv`.
- **Per-threat correlations use ~290 subjects per cell;** the LMM uses ~13,000 trials × 290 subjects. Both are well-powered.
- **The reversal is between-subject;** within-subject correlations between trial-level choice and trial-level vigor would test a different question (whether the same subject's vigor varies with their choice on individual trials).
- **The β = −0.022 interaction is small in raw units** but the z-statistic (−3.54) is robust. The substantive magnitude is best seen in the per-threat correlations.

## Replication

**Migration needed.** Steps:

1. Compute per-subject p_heavy and mean vigor at each of the three threat levels.
2. Pearson correlate p_heavy and vigor across subjects within each threat level.
3. Fit `vigor_trial ~ choice_subj_z × threat_z + dist_z + (1|subj)`.
4. Run on both samples; update this file with the validated numbers.

## References

**Related results:**
- [[result_401]] — Choice and vigor are independent dimensions in the aggregate (internal H29, the null this result contextualizes).
- [[result_402]] — Cross-channel ω → vigor (the parameter-level dissociation).
- [[result_403]] — Vigor dominates escape over choice.
- [[result_405]] — Off-diagonal groups differ in metacognition (the consequences of the reversal-induced quadrants).
- [[result_506]] — Confidence miscalibration tracks the dissociation.

**Source:**
- `instructions/memory/hypotheses.md` § H32 + H36 (last validated 2026-03-24).
