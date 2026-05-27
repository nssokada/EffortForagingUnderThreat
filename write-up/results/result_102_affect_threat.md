---
result_id: 102
class: behavioral_effects
title: Anxiety rises and confidence falls with threat probability
status: supported
prereg_h: [H1b]
internal_h: []
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/H1_adaptive_shifts.ipynb]
scripts: []
outputs: [results/stats/confirmatory_hypothesis_results.csv]
figures: [TODO]
created: 2026-05-23
last_run: 2026-05-23
---

# Result 102 — Anxiety rises and confidence falls with threat probability

## Overview

<TODO Stage 2>

## Hypothesis

**Statement.** "Anxiety will increase with threat and distance. Confidence will decrease with threat and distance." (Preregistration, H1b.)

**Predicted direction.** β(threat) > 0 for anxiety; β(threat) < 0 for confidence. (Distance predicted same-signed.)

**Preregistered criterion.** |t| > 3 on each main effect.

**Source of the hypothesis.** Preregistration §H1b (`write-up/preregistration.md`, lines 18–22). Motivated by the prediction that subjective affect monitors the same survival-related quantities that govern choice: anxiety as a danger appraisal that should track exposure to attack, confidence as a coping appraisal that should fall as the foraging attempt becomes more precarious.

> **Implementation deviation from prereg** — see Caveats. The analysis as preregistered specifies `response ~ threat_z + dist_z + (1 + threat_z | subject)`. The notebook actually fits `response ~ T_z + (1 + T_z | subject)`, omitting the distance predictor. The result below reports the as-run model; the prereg's distance prediction is not directly evaluated here.

## Data Source

- **Samples:** Exploratory N = 290, data dir `data/exploratory_350/processed/stage5_filtered_data_20260403_133425`. Confirmatory N = 281, data dir `data/confirmatory_350/processed/stage5_filtered_data_20260403_142413`.
- **Input file (both samples):** `feelings.csv` per sample directory — one row per probe-trial rating, with `response` (1–10 scale), `threat`, `subj`, and `questionLabel` ∈ {"anxiety", "confidence"}.
- **Inclusion / exclusion applied for this result:** Project-default participant exclusions only. All probe-trial ratings retained (no probe-level filtering).
- **Unit of analysis:** Probe trial (within subject), with subject-level random intercept and random slope on threat.
- **N entering the model:** ~18 anxiety probes × N + ~18 confidence probes × N per sample — approximately 5,220 anxiety + 5,220 confidence ratings (exploratory); ~5,058 + ~5,058 (confirmatory). Counts may be slightly lower for participants with missed probes.

## Method

For each sample and each affect channel (anxiety, confidence) separately, a linear mixed-effects model was fit on probe-trial response ratings as a function of z-scored threat probability, with a random intercept and random slope on threat at the subject level. The threat predictor is standardized within sample. The prereg's directional criterion is that the fixed-effect threat coefficient should be positive for anxiety and negative for confidence, with |t| > 3.

**Model specification (per affect channel):**

```python
smf.mixedlm(
    "response ~ T_z",
    data=df,             # df is anxiety probes OR confidence probes
    groups=df['subj'],
    re_formula="~T_z"
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

Anxiety rises with threat in both samples; confidence falls. All four tests pass the prereg criterion (|t| > 3 with the predicted sign), with very large z-statistics.

| Affect | Exploratory (N=290) | Confirmatory (N=281) |
|---|---|---|
| **Anxiety** | β = **+0.580** (z = +14.67, **p = 1.1 × 10⁻⁴⁸**) | β = **+0.534** (z = +12.52, **p = 5.6 × 10⁻³⁶**) |
| **Confidence** | β = **−0.582** (z = −13.72, **p = 8.0 × 10⁻⁴³**) | β = **−0.671** (z = −15.25, **p = 1.6 × 10⁻⁵²**) |

(β here is the population-level fixed-effect coefficient on z-scored threat; z is the Wald z-statistic from the LMM.)

**Figure:** TODO — no dedicated H1b affect-by-threat figure was found in `results/figs/paper/` or `results/stats/avoidance_activation/`. The closest is `results/figs/paper/fig_h1c_affect.pdf`, which is mis-labeled "H1c" but appears to depict affect (anxiety/confidence) as a function of condition; needs verification. Recommend producing a clean two-panel plot: mean rating × threat level, one panel per affect channel, both samples overlaid or side-by-side.

**Verdict on prereg criterion:** **PASS** for both affect channels in both samples. The anxiety slope is positive and the confidence slope is negative, with |z| ≫ 3 throughout.

## Interpretation

<TODO Stage 2/3 — paper-grade.>

## Caveats & Limitations

- **Implementation deviates from preregistration in omitting distance.** The prereg specifies `response ~ threat_z + dist_z + (1 + threat_z | subject)`. The notebook fits only `response ~ T_z`. The prereg's distance prediction (anxiety should also rise with distance, confidence should also fall) is therefore untested by this result. Three resolutions are possible: (i) refit with `dist_z` added and report both effects; (ii) treat the threat-only model as the official H1b test and document the deviation in the manuscript; (iii) report the distance effect as a separate exploratory result. The choice has implications for what the manuscript can claim about prereg H1b and should be resolved before publication.
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
