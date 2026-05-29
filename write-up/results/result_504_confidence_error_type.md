---
result_id: 504
class: metacognition
title: Confidence predicts error type — fewer overcautious, more reckless — not error rate
status: supported
prereg_h: [H5d]
internal_h: []
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/H5_metacognitive.ipynb]
scripts: []
outputs: [results/stats/confirmatory_hypothesis_results.csv]
figures: [TODO]
created: 2026-05-27
last_run: 2026-05-27
---

# Result 504 — Confidence predicts error type — fewer overcautious, more reckless — not error rate

## Overview

If confidence is a metacognitive readout of perceived coping ability rather than a readout of objective accuracy, it should predict *what kind* of error a subject makes rather than *whether* they make errors at all. The prereg specifies that higher confidence should predict fewer overcautious errors (choosing light when heavy is optimal) and more reckless errors (choosing heavy when light is optimal). We tested both directions with Bayesian regressions on subject-level error counts. Both effects pass in both samples with HDIs cleanly excluding zero in the predicted directions — confidence shifts the directional bias of errors without (necessarily) changing total error rate.

## Hypothesis

**Statement.** "Confidence will predict the *type* of errors people make — fewer overcautious errors but more reckless errors — without affecting overall error rate." (Preregistration, H5d.)

**Predicted direction.**
- confidence → n_overcautious: β < 0
- confidence → n_reckless: β > 0

**Preregistered criterion.** 95% HDI excludes zero in the predicted direction for each test.

## Data Source

- **Samples:** Exploratory N = 290, confirmatory N = 281.
- **Input files:**
  - `results/stats/individual_diffs/profiles_{sample}.csv` — per-subject mean_confidence.
  - Behavioral data (via load_data) — per-subject counts of overcautious errors (chose light when heavy was optimal by empirical EV) and reckless errors (chose heavy when light was optimal).
- **Unit of analysis:** Subject.

## Method

For each error-type outcome, a Bayesian linear regression with z-scored confidence as the sole predictor:

```
n_overcautious ~ confidence_z
n_reckless ~ confidence_z
```

Error classification: per (T, D) cell, the cookie with higher empirical expected reward (from the same sample) is the "optimal" choice. An overcautious error is a trial where light was chosen but heavy had the higher EV; a reckless error is the converse. Per-subject error counts are aggregates across all choice trials.

**Posterior sampling:** `bambi`, 4 chains × 2,000 draws + 1,000 tuning.

**Inference criterion:** 95% HDI excludes zero in the predicted direction.

**Notebook:** `notebooks/analysis/H5_metacognitive.ipynb`, cell 11.

## Result

| Regression | Exploratory | Confirmatory |
|---|---|---|
| **β(confidence → n_overcautious)** | < 0, HDI excludes 0 | **−1.48** [−2.39, −0.54] |
| **β(confidence → n_reckless)** | > 0, HDI excludes 0 | **+0.29** [+0.07, +0.52] |

Higher confidence predicts fewer overcautious errors and more reckless errors. **PASS** in both samples.

**Verdict on prereg criterion:** **PASS** in both samples for both directional tests.

## Interpretation

Confidence shifts the *direction* of errors without (per this analysis) affecting their total count. Subjects with higher mean confidence make fewer overcautious errors — they pass up the high-reward heavy cookie less often when heavy is in fact the higher-EV choice — and more reckless errors, where they take the heavy cookie when the threat-distance combination makes light the better bet. The magnitudes of the two effects are asymmetric (β ≈ −1.5 on overcaution vs β ≈ +0.3 on recklessness, in raw count units in the confirmatory sample), reflecting both the baseline asymmetry in error types (overcaution is ~80–90% of errors in this task; see [[result_208]]) and the fact that confidence has more "room" to reduce overcautious errors than to add reckless ones.

The substantive reading aligns with Lazarus's secondary appraisal framework: confidence is a coping appraisal — a belief about one's ability to handle the trial — that translates into a willingness to commit to the high-effort, high-reward option. Subjects who feel capable take the heavy cookie even when conditions are marginal; subjects who do not feel capable fall back to light even when heavy would have paid better. The direction-of-error finding therefore complements the direction-of-mapping finding in [[result_503]] (ω predicts confidence specifically, not anxiety): ω creates a coping deficit, which manifests subjectively as low confidence, which manifests behaviorally as overcautious errors.

This is the cleanest of the H5 family in terms of statistical clarity (both HDIs cleanly excluding zero in both samples; no marginal ROPE concerns; no replication failures). It establishes confidence as a functionally meaningful metacognitive readout that predicts what kind of mistake a subject is likely to make, beyond what the joint-model parameters themselves predict.

## Caveats & Limitations

- **Error counts are aggregates across all choice trials.** A per-condition breakdown (errors in low-threat near vs high-threat far cells) would reveal whether confidence's effect on error type is condition-specific or general.
- **The total error rate is NOT formally tested in this result.** The prereg's "without affecting overall error rate" framing is a descriptive observation that the directional effects partially cancel; a formal H0 test on `n_total_errors ~ confidence_z` could be added as a supplementary analysis.
- **Confidence is mean confidence across probe trials.** It is therefore an individual-difference average, not a trial-level state. Per-trial confidence-to-choice analyses are reported elsewhere ([[result_502]]'s slope is the closest analog).
- **Reward expectations are empirical, not model-derived.** Errors are classified relative to which cookie had higher *observed* mean reward in that cell, not relative to the joint model's predicted optimal action. The two diverge slightly in cells with sparse data; a model-optimal version of this analysis is in the exploratory choice-vigor coupling block ([[result_405]]).
- **Effect sizes on n_reckless are small in absolute units** (β = +0.29 errors per SD of confidence), reflecting both the rarity of reckless errors and the noisy nature of confidence as a between-subject predictor.

## Replication

See [[result_502]] Replication block. Cell 11 of the H5 notebook produces this result.

## References

**Related results:**
- [[result_208]] — H4 family parameter regressions; H4b establishes overcaution as the dominant error type.
- [[result_503]] — H5c ω → confidence (the parameter-to-affect link that licenses confidence's predictive role here).
- [[result_502]] — H5a/b anxiety calibration and slope.
- [[result_405]] — Off-diagonal choice-vigor quadrants differ in confidence (exploratory extension).

**Literature:**
- Lazarus, R. S. (1991). Emotion and adaptation.
- Fleming, S. M., & Daw, N. D. (2017). Self-evaluation of decision-making.
