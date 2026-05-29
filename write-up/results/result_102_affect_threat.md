---
result_id: 102
class: behavioral_effects
title: Anxiety rises and confidence falls with threat and distance
status: supported
prereg_h: [H1b]
internal_h: []
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/H1_adaptive_shifts.ipynb]
scripts: []
outputs: [results/stats/confirmatory_hypothesis_results.csv]
figures: [TODO]
created: 2026-05-23
last_run: 2026-05-27
---

# Result 102 — Anxiety rises and confidence falls with threat and distance

## Overview

Do subjective affect ratings track the same task variables that shift choice? On forced-choice probe trials we collected prospective ratings of anxiety ("how anxious are you about being captured?") and confidence ("how confident are you in reaching safety?") on a 1–10 scale, then fit linear mixed-effects models with z-scored threat and distance as fixed effects and a random threat slope per subject. In both samples, anxiety rises and confidence falls with both threat probability and distance, with effect sizes on threat that mirror those on choice (result_101) and effect sizes on distance roughly half as large. The finding establishes that affect ratings track the same exposure-and-effort gradient that suppresses heavy-cookie choice.

## Hypothesis

**Statement.** "Anxiety will increase with threat and distance. Confidence will decrease with threat and distance." (Preregistration, H1b.)

**Predicted direction.** β(threat) > 0 for anxiety; β(threat) < 0 for confidence. (Distance predicted same-signed.)

**Preregistered criterion.** |t| > 3 on each main effect.

**Source of the hypothesis.** Preregistration §H1b (`write-up/preregistration.md`, lines 18–22). Motivated by the prediction that subjective affect monitors the same survival-related quantities that govern choice: anxiety as a danger appraisal that should track exposure to attack, confidence as a coping appraisal that should fall as the foraging attempt becomes more precarious.

> **Matches preregistration (updated 2026-05-27).** Earlier runs of this result fit `response ~ T_z` only, omitting the preregistered distance predictor. The notebook now fits the full preregistered model `response ~ threat_z + dist_z + (1 + threat_z | subject)`. Both threat and distance effects are reported below, so all four H1b sub-predictions (threat and distance, for anxiety and confidence) are directly evaluated.

## Data Source

- **Samples:** Exploratory N = 290, data dir `data/exploratory_350/processed/stage5_filtered_data_20260403_133425`. Confirmatory N = 281, data dir `data/confirmatory_350/processed/stage5_filtered_data_20260403_142413`.
- **Input file (both samples):** `feelings.csv` per sample directory — one row per probe-trial rating, with `response` (1–10 scale), `threat`, `subj`, and `questionLabel` ∈ {"anxiety", "confidence"}.
- **Inclusion / exclusion applied for this result:** Project-default participant exclusions only. All probe-trial ratings retained (no probe-level filtering).
- **Unit of analysis:** Probe trial (within subject), with subject-level random intercept and random slope on threat.
- **N entering the model:** ~18 anxiety probes × N + ~18 confidence probes × N per sample — approximately 5,220 anxiety + 5,220 confidence ratings (exploratory); ~5,058 + ~5,058 (confirmatory). Counts may be slightly lower for participants with missed probes.

## Method

For each sample and each affect channel (anxiety, confidence) separately, a linear mixed-effects model was fit on probe-trial response ratings as a function of z-scored threat probability and z-scored distance, with a random intercept and a random slope on threat at the subject level. Both predictors are standardized within sample; `dist_z` is the z-score of the probe `distance` column (0, 1, 2 → D = 1, 2, 3). The prereg's directional criterion is that each fixed-effect coefficient (threat and distance) should be positive for anxiety and negative for confidence, with |t| > 3.

**Model specification (per affect channel):**

```python
smf.mixedlm(
    "response ~ T_z + dist_z",   # threat and distance — both preregistered fixed effects
    data=df,                     # df is anxiety probes OR confidence probes
    groups=df['subj'],
    re_formula="~T_z"            # random intercept + random threat slope
).fit(reml=False)
```

**Software / packages:**
- `statsmodels` (mixedlm with REML=False for fixed-effect inference)
- `scipy.stats.zscore` for predictor standardization
- Environment: `effort_foraging_threat` (Python 3.11)

**On the random-slope LMM choice.** Allowing the threat slope to vary by subject (`re_formula="~T_z"`) absorbs individual differences in how reactive each person's affect is to threat. Without random slopes, the model would assume every subject responds to threat with the same magnitude, and within-subject variance from heterogeneous reactivity would inflate the residual term and shrink the fixed-effect t-statistic. The random-slope specification therefore gives a stricter test of the population-level main effect: it asks whether the *average* slope across subjects is reliably different from zero, accounting for subject-level slope variability. `reml=False` (ML rather than REML) is used because the inferential target here is a fixed-effect coefficient, not a variance component, and ML is unbiased for fixed-effect tests in standard mixed-effects software.

**Inference criterion:** |t| > 3 on the threat fixed effect, with the predicted sign per affect channel.

**Notebook that produces this result:** `notebooks/analysis/H1_adaptive_shifts.ipynb`, cell 5 (the H1b LMM fit). Validated 2026-05-23: re-executed end-to-end, outputs match the values below.

## Result

Anxiety rises with both threat and distance in both samples; confidence falls with both. All eight tests (threat + distance, for anxiety + confidence, in two samples) pass the prereg criterion (|t| > 3 with the predicted sign), with large z-statistics throughout.

| Affect | Predictor | Exploratory (N=290) | Confirmatory (N=281) |
|---|---|---|---|
| **Anxiety** | threat | β = **+0.580** (z = +14.67, **p = 1.1 × 10⁻⁴⁸**) | β = **+0.534** (z = +12.51, **p = 6.2 × 10⁻³⁶**) |
| **Anxiety** | distance | β = **+0.230** (z = +9.93, **p = 3.1 × 10⁻²³**) | β = **+0.276** (z = +12.22, **p = 2.4 × 10⁻³⁴**) |
| **Confidence** | threat | β = **−0.582** (z = −13.72, **p = 7.8 × 10⁻⁴³**) | β = **−0.671** (z = −15.25, **p = 1.7 × 10⁻⁵²**) |
| **Confidence** | distance | β = **−0.295** (z = −12.84, **p = 1.0 × 10⁻³⁷**) | β = **−0.260** (z = −11.25, **p = 2.4 × 10⁻²⁹**) |

(β is the population-level fixed-effect coefficient on the z-scored predictor; z is the Wald z-statistic from the LMM.)

The threat coefficients are essentially identical to the earlier threat-only model (anxiety +0.580, confidence −0.582 in exploratory), indicating that threat and distance are near-orthogonal in the balanced probe design — adding distance does not alter the threat estimate.

**Figure:** TODO — no dedicated H1b affect-by-threat figure was found in `results/figs/paper/` or `results/stats/avoidance_activation/`. The closest is `results/figs/paper/fig_h1c_affect.pdf`, which is mis-labeled "H1c" but appears to depict affect (anxiety/confidence) as a function of condition; needs verification. Recommend producing a clean two-panel plot: mean rating × threat level, one panel per affect channel, both samples overlaid or side-by-side.

**Verdict on prereg criterion:** **PASS** on all four sub-predictions (threat and distance, for anxiety and confidence) in both samples. Anxiety coefficients are positive and confidence coefficients negative, with |z| ≫ 3 throughout.

## Interpretation

Subjective affect tracks both task variables in the predicted directions, with replication across two independent samples. Each one-SD increase in threat probability raises anxiety by ~0.55 rating points and lowers confidence by ~0.58 points (averaging across samples); each one-SD increase in distance raises anxiety by ~0.25 points and lowers confidence by ~0.28 points. The threat coefficients are roughly twice the magnitude of the distance coefficients, and the two are essentially uncorrelated in the balanced probe design — the threat estimates are unchanged from a threat-only specification.

The pattern is what risk-sensitive foraging theory predicts of a monitoring signal: anxiety as a danger appraisal that tracks the probability of attack and the duration of exposure, and confidence as the inverse coping appraisal. Two features of the data are notable. First, anxiety and confidence are not redundant — they respond in opposite directions to the same predictors, with similar magnitudes, suggesting two complementary readouts rather than a single bipolar dimension. Second, both predictors that govern choice also govern affect: this rules out the simplest alternative in which subjective reports reflect only one task variable (e.g., only the visually salient threat number on screen) and not the integrated demand of the foraging attempt.

The result establishes that affect ratings respond to the same gradient that suppresses high-effort choice in [[result_101]], but it does not adjudicate whether the affect signal *drives* choice, merely *accompanies* it, or is a downstream consequence of the same value computation. That distinction is addressed by results that test whether trial-level affect predicts trial-level vigor ([[result_103]] and the H9 null in `instructions/memory/hypotheses.md`), and whether model-derived survival probability predicts affect at the trial level beyond what raw threat and distance explain ([[result_501]]).

## Caveats & Limitations

- **Distance omission resolved (2026-05-27).** A prior version of this result fit only `response ~ T_z`, omitting the preregistered `dist_z` fixed effect, so the prereg's distance prediction was untested. This was resolved by refitting the full preregistered model `response ~ threat_z + dist_z + (1 + threat_z | subject)` (resolution (i) of the three previously documented options). Distance passes in the predicted direction in both samples and both channels (see Result table), and the threat estimates are unchanged. H1b is now evaluated exactly as preregistered.
- **Probe count per subject is small (~18 per affect channel).** Individual-difference indices derived from these probes — anxiety calibration, anxiety slope, mean confidence — will have substantial sampling error, as the prereg already acknowledges. The population-level fixed-effect tests above are not affected by this (large total N), but downstream individual-difference analyses ([[result_502]], [[result_503]]) inherit the probe-count limitation.
- **Confidence is rated on a 1–10 Likert scale, not a continuous measure.** Treating Likert ratings as approximately Gaussian for LMM inference is conventional but not strictly justified at the boundary cells. The very large effect sizes here make the conclusion robust to this approximation.
- **Anxiety and confidence are not fit jointly.** Each is fit as a separate LMM with its own random-effect structure. A joint bivariate LMM would allow estimation of the within-subject correlation between anxiety and confidence ratings, which is reported descriptively elsewhere but not formally inferred here.
- **Probes are prospective ratings.** Ratings are collected at trial start, before the participant has begun pressing or learned the trial outcome. Anxiety here is therefore an *anticipatory* state, not a reactive one. Reactive (within-trial) affect is not observed in this task design.

## Replication

**To regenerate this result from scratch:**

```bash
PYTHONPATH=notebooks/analysis \
  jupyter nbconvert --to notebook --execute \
  notebooks/analysis/H1_adaptive_shifts.ipynb \
  --inplace --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=600
```

Run from the project root. `PYTHONPATH` is required so the notebook's local imports (`config`, `load_data`) resolve when the kernel is launched by `nbconvert` from outside `notebooks/analysis/`. On systems where the relative `PYTHONPATH` does not resolve correctly, use the absolute path: `PYTHONPATH=$(pwd)/notebooks/analysis`.

**Expected runtime:** ~30 s on CPU (same notebook as result_101; both run in one pass).

**Expected outputs:**
- Cell 5 stdout in `notebooks/analysis/H1_adaptive_shifts.ipynb` reports anxiety and confidence β/z/p for both samples.
- No standalone CSV is currently saved; numbers appear in confirmatory column of `results/stats/confirmatory_hypothesis_results.csv`.

## References

**Related results:**
- [[result_101]] — Threat probability and distance both reduce P(heavy). Same notebook, the choice-side companion to this affect-side result.
- [[result_103]] — Within-cookie vigor under threat. Completes the H1 triad (choice + affect + vigor).
- [[result_501]] — Model-derived survival probability predicts affect ratings (prereg H4c). The mechanistic version of this descriptive H1b test.
- [[result_502]] — Per-subject anxiety calibration and slope as individual-difference predictors of foraging optimality (prereg H5a/b).
- [[result_503]] — ω predicts confidence but not anxiety (prereg H5c, ROPE test).

**Notebook:**
- `notebooks/analysis/H1_adaptive_shifts.ipynb` — produces this and all other H1 results.

**Literature:**
- Mobbs, D., et al. (multiple) — defensive cascade and anxiety as a danger-monitoring signal.
- LeDoux, J. (2014). Coming to terms with fear.
