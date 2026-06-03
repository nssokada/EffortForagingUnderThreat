---
result_id: 403
class: choice_vigor_coupling
title: Pressing intensity dominates escape outcomes far more than cookie choice
status: supported_exploratory
prereg_h: []
internal_h: [H31]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 403 — Pressing intensity dominates escape outcomes far more than cookie choice

> **⚠️ Exploratory, deprecated notebook.** This result was first established in the earlier project structure using the deprecated three-parameter FET model (z, k, β). The behavioral analysis (subject-level vigor + choice regressed on escape rate) is straightforward and does not depend on the deprecated model parameters, but the canonical cell that produced these numbers lives in `notebooks/_deprecated/joint_coupling_models/` and has not been re-run on the current samples. Numbers reported below are from `instructions/memory/hypotheses.md` § H31 (last validated 2026-03-24). Recommend migration to a current notebook cell in `H8_avoid_activate.ipynb` or `H2_vigor_dynamics.ipynb` and re-execution on both samples before manuscript inclusion.

## Overview

Does cookie choice or pressing intensity matter more for surviving a predator attack? Subjects vary substantially in both dimensions, and at the trial level the heavy cookie is structurally farther from safety than the light cookie. We regressed per-subject escape rate on per-subject z-scored choice tendency (p_heavy) and z-scored mean vigor, with their interaction. Vigor dominates: standardized β(vigor) ≈ +0.80 vs β(choice) ≈ −0.16, with the model explaining R² ≈ 0.66 of between-subject variance in escape rate. Choosing the heavy cookie actually *hurts* survival on attack trials (the negative choice coefficient), because heavy is at the far distance and gives the predator more time to catch up. The result reframes the "what predicts surviving" question away from the decision and toward execution.

## Hypothesis

**Statement.** "How hard you press predicts escape from predators far better than what you chose." (Internal H31, `instructions/memory/hypotheses.md`.)

**Predicted direction.** β(vigor) > 0 and dominant; β(choice) likely negative (choosing hard → farther from safety → harder to escape).

## Data Source (legacy)

- **Sample:** N = 290 exploratory (per legacy run).
- **Inputs:** subject-level p_heavy, mean vigor (capacity-normalized), escape rate on attack trials.
- **Unit of analysis:** Subject.

## Method (legacy)

OLS regression: `escape_rate ~ choice_z + vigor_z + choice_z × vigor_z`.

## Result (legacy, from internal H31)

| Coefficient | Standardized β |
|---|---|
| Vigor | **+0.795** |
| Choice | **−0.160** |
| Choice × Vigor | (sign not reported) |
| **R²** | **0.66** |

**Per-quadrant escape rates** (HL = high choice / low vigor; LH = low choice / high vigor; etc.):

| Quadrant | Escape rate |
|---|---|
| HH (chose hard, pressed hard) | 53% |
| LH (chose easy, pressed hard) | 60% |
| HL (chose hard, pressed easy) | 19% |
| LL (chose easy, pressed easy) | 25% |

**Verdict:** Vigor dominates choice as a predictor of escape; choosing hard *reduces* escape rate at fixed vigor.

## Interpretation

The decision and the execution channel predict survival on opposite directions and very different magnitudes. Pressing harder protects you; choosing the heavy cookie hurts you (at fixed vigor) because heavy is at the far distance. The R² of 0.66 explained by two simple subject-level variables is large by individual-difference standards, suggesting that escape outcomes are dominated by *what subjects do during execution*, not by *which patch they entered*. The quadrant table sharpens the story: LH subjects (chose easy, pressed hard) achieve the highest escape rate, 60%, despite earning less reward per trial; HL subjects (chose hard, pressed easy) achieve the lowest, 19%, because they walked into the dangerous patch and then didn't compensate motorically.

This result has implications for how the paper frames the choice-vigor coupling. A naive reading would treat choice as the primary behavioral output and vigor as a secondary motor parameter — but the survival data inverts that ordering. Vigor is where the action is when the question is "what predicts trial-level success." Choice determines the option entered; vigor determines what happens once you are in that option. The two channels are dissociable in their parameter signatures ([[result_208]], [[result_401]]) and in their downstream consequences for survival (this result), and both dissociations are robust at the between-subject level.

The finding also licenses framing the metacognitive miscalibration in [[result_506]]: subjects whose choice and vigor are *out of alignment* (HL and LH quadrants) make systematic confidence errors that track the dissociation. This connects the choice-vigor coupling work to the H5 metacognition family.

## Caveats & Limitations

- **Status: `supported_exploratory`.** Numbers are from the exploratory sample only; the deprecated notebook has not been re-run on the confirmatory sample. Migration is the recommended next step before manuscript inclusion.
- **The negative β(choice) is partly mechanical.** Heavy cookies are at greater distance D, which directly reduces survival via the survival function S(u, T, D) ∝ exp(−h·T^γ·D/speed(u)). This is not a "choice causes failure" claim; it is "choosing heavy puts you in a harder-to-escape position that your vigor must overcome."
- **R² = 0.66 is between-subject, not within-subject.** A trial-level analysis (within subject, controlling for condition) is reported separately in [[result_308]] (also exploratory/deprecated).
- **The quadrant table uses median splits** and is therefore descriptive; the regression result is the inferential anchor.

## Replication

**Currently not reproducible from the active notebook tree.** To migrate:

1. Add a cell to `notebooks/analysis/H8_avoid_activate.ipynb` (or a new `H8b_escape_outcomes.ipynb`) that:
   - Loads per-subject p_heavy, mean_vigor, escape_rate from `profiles_{sample}.csv` or computes them from `behavior_rich.csv`.
   - Fits the OLS regression and prints standardized betas + R².
   - Constructs the quadrant table.
2. Run on both samples and update this file with the validated numbers.

## References

**Related results:**
- [[result_208]], [[result_401]] — Channel-specific parameter slopes and marginal coupling (the parameter-level dissociation).
- [[result_404]] — Threat reverses choice-vigor coupling (the threat-modulation companion).
- [[result_308]] — Trial-level vigor predicts escape (the within-subject version).
- [[result_506]] — Confidence miscalibration tracks the choice-vigor dissociation.

**Source:**
- `instructions/memory/hypotheses.md` § H31 (last validated 2026-03-24).
