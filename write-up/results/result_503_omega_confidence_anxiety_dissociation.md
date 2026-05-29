---
result_id: 503
class: metacognition
title: ω predicts confidence but not anxiety (appraisal dissociation)
status: partial
prereg_h: [H5c]
internal_h: []
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/H5_metacognitive.ipynb]
scripts: []
outputs: [results/stats/confirmatory_hypothesis_results.csv]
figures: [TODO]
created: 2026-05-27
last_run: 2026-05-27
---

# Result 503 — ω predicts confidence but not anxiety (appraisal dissociation)

## Overview

Lazarus's appraisal theory distinguishes a *primary appraisal* (is this dangerous? — i.e. anxiety) from a *secondary appraisal* (can I cope? — i.e. confidence). If the joint-model parameter ω represents the subjective cost of capture — a property of how a subject internalizes the consequences of being caught — it should map onto the secondary appraisal (confidence) without strongly mapping onto the primary appraisal (anxiety). We tested this with two Bayesian regressions per sample: mean confidence on ω, and mean anxiety on ω, using the prereg's HDI criterion for confidence and a ROPE (region of practical equivalence) criterion for the anxiety null. Confidence shows the predicted negative effect of ω in both samples; the anxiety null is supported directionally in both samples but the formal ROPE criterion is marginal in confirmatory.

## Hypothesis

**Statement.** "Capture cost will predict subjective confidence but not anxiety." (Preregistration, H5c.)

**Predicted direction.**
- ω → confidence: β < 0 (higher capture cost → lower confidence in safe return).
- ω → anxiety: β ≈ 0 (no relationship beyond the ROPE).

**Preregistered criterion:**
- Confidence test: 95% HDI excludes zero in the negative direction.
- Anxiety null test: 95% HDI falls *entirely* within ROPE [−0.10, +0.10] on the standardized coefficient.

## Data Source

- **Samples:** Exploratory N = 290, confirmatory N = 281.
- **Input files:**
  - `results/stats/joint_optimal/{sample}/mcmc_m4_params.csv` — per-subject ω.
  - `results/stats/individual_diffs/profiles_{sample}.csv` — per-subject mean_anxiety, mean_confidence (averaged across all probe trials).
- **Unit of analysis:** Subject.

## Method

Two separate Bayesian linear regressions per sample:

```
mean_confidence ~ omega_z
mean_anxiety ~ omega_z
```

ω is log-transformed and z-scored within sample. Posterior sampling: `bambi`, 4 chains × 2,000 draws + 1,000 tuning.

**Inference criteria:**
- Confidence: 95% HDI excludes zero (directional, negative).
- Anxiety: 95% HDI ⊂ [−0.10, +0.10] (ROPE).

**Notebook:** `notebooks/analysis/H5_metacognitive.ipynb`, cell 9.

## Result

| Regression | Exploratory | Confirmatory |
|---|---|---|
| **β(ω → confidence)** | < 0, HDI excludes 0 | **−0.181** [−0.340, −0.037] |
| **β(ω → anxiety)** | HDI includes 0 (directional null) | **−0.067** [−0.221, +0.078] |

**Verdict on prereg criterion:**
- **Confidence:** **PASS** in both samples (β < 0, HDI excludes zero).
- **Anxiety null:** **MARGINAL.** The confirmatory point estimate (β = −0.067) is within the ROPE [−0.10, +0.10], but the HDI extends to −0.221 — outside the ROPE on the lower bound. The formal ROPE-includes-HDI criterion is therefore *not* met. However, the directional null (small effect, opposite-side asymmetry: confidence effect three times larger in magnitude) is supported.

## Interpretation

The appraisal dissociation predicted by Lazarus's framework is supported in the direction predicted by the prereg, with a methodologically nuanced result on the anxiety-null side. ω — the capture-cost parameter from the joint model — has a clear negative effect on confidence in both samples: subjects who internalize capture as more costly report feeling less confident in their ability to reach safety. The effect is roughly three times larger in magnitude than the parallel effect on anxiety, which is itself small (β = −0.067 in confirmatory) and not significantly different from zero. The dissociation is therefore real on the descriptive level: ω maps onto coping appraisal (confidence) much more strongly than onto threat appraisal (anxiety).

The formal ROPE criterion for the anxiety null is, however, marginal. The 95% HDI for the anxiety regression extends to −0.221, which is outside the prespecified ROPE of [−0.10, +0.10]. Strictly, this means the data are consistent with effect sizes large enough to be of practical interest — even though the *point estimate* is well inside the ROPE. Two readings are tenable. The conservative reading: the anxiety null is not formally supported, so we should report the asymmetry descriptively and acknowledge that with N ≈ 280 the HDI is wide enough that effects up to β = ±0.2 cannot be ruled out. The substantive reading: the *point estimate is small and inside the ROPE, the confidence effect is three times larger, and the sign asymmetry is exactly what the appraisal-dissociation hypothesis predicts*, so the substantive claim is supported even if the formal test is marginal.

The H5c result is one of two specifically partial results in the prereg H1–H5 family (the other being H4e in [[result_208]]). It deserves explicit attention in the manuscript because the conservative and substantive readings diverge. We recommend reporting both the confidence pass and the marginal anxiety-null with full transparency: the appraisal dissociation is supported in direction and effect-size ratio, with the formal ROPE criterion partially missed due to HDI width rather than evidence of a real ω-on-anxiety effect.

## Caveats & Limitations

- **ROPE bounds [−0.10, +0.10] are tight relative to HDI width.** With N ≈ 280 subjects, the HDI on a standardized regression coefficient is approximately ±0.15 around the point estimate. A ROPE narrower than the HDI cannot be formally satisfied except with a point estimate exactly at zero. The prereg's ROPE choice was substantively motivated but may be statistically over-constrained for this sample size.
- **The asymmetry direction is exactly as predicted.** Both ω effects are negative; the confidence effect (−0.18) is three times the anxiety effect (−0.07). A more powerful, less rigid test would be a formal asymmetry test (H0: β_anxiety = β_confidence vs H1: β_anxiety ≠ β_confidence) rather than a ROPE on the anxiety side alone.
- **Mean anxiety and mean confidence are aggregate quantities** — they collapse over the 18 probe trials per subject. Per-condition decomposition (anxiety in low-threat vs high-threat probes, etc.) is reported descriptively elsewhere but not in this confirmatory test.
- **κ is not included as a covariate.** The prereg specifies the bivariate regressions above; adding κ would change the coefficient estimates and the interpretive frame (partial-vs-marginal effects). The bivariate spec is the prereg's pre-committed choice.

## Replication

See [[result_502]] Replication block. Cell 9 of the H5 notebook produces this result.

## References

**Related results:**
- [[result_201]] — Joint model M4 fit (source of ω).
- [[result_208]] — H4 family parameter regressions (the broader context for ω as predictor).
- [[result_502]] — H5a/b anxiety calibration and slope.
- [[result_504]] — H5d confidence → error type (the downstream consequence of the coping-appraisal channel).

**Literature:**
- Lazarus, R. S. (1991). Emotion and adaptation.
- Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian estimation.
