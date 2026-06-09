---
result_id: 510
class: metacognition_affect
title: Slope-corrected phenotype × metacognition refines but does not extend result_508
status: partial
prereg_h: []
internal_h: []
samples: [pooled_571]
notebooks: []
scripts: [scripts/analysis/phenotype_metacog_slopes.py, scripts/analysis/phenotype_metacognition_profile.py]
outputs: [results/stats/clinical/phenotype_metacog_slopes_profile.csv, results/stats/clinical/phenotype_metacog_slopes_anova.csv, results/stats/clinical/phenotype_metacog_slopes_subjects.csv]
figures: [TODO]
created: 2026-06-05
last_run: 2026-06-05
---

# Result 510 — Slope-corrected phenotype × metacognition: methodological refinement of result_508

> **This is not a new positive finding. It is a methodological refinement.** [[result_508]] used per-subject *mean* confidence and *mean* anxiety as metacognitive measures, conflating baseline level with reactivity to task conditions. This re-analysis decomposes confidence and anxiety into intercept + slope (on threat) + slope (on distance) per subject. The result refines two earlier claims and confirms one — without adding new positive content. Phenotypes differ in confidence intercept (baseline level) but NOT in slope (reactivity). The "HL is overconfident" characterization in [[result_508]] is not supported by the slope analysis — HL subjects show appropriate confidence reactivity to threat, just elevated baseline. Slope-based predictors of earnings recover the same effects as calibration-based predictors in 508. The phenotype × metacognition finding is therefore weaker than 508's framing implied.

## Why this re-analysis was needed

Result 508 reported that confidence differs across phenotypes (F = 8.7, p = 1e-5) and that this constituted a metacognitive signature of strategic style. But the analysis used per-subject *mean* confidence, which conflates two distinct quantities:

- **Baseline confidence** (intercept of the within-subject regression after partialing T, D)
- **Reactivity** (within-subject slope of confidence on T and D)

A subject who is "overconfident" in the strong sense should have a flat (or shallow) confidence slope on threat — their confidence shouldn't drop appropriately as threat rises. A subject who is just "optimistic-biased" has a normal slope but elevated baseline. Mean confidence cannot distinguish these.

## Method

For each subject, fit per-subject regressions:

```
confidence_response_t ~ T_z + D_z   (per subject)
anxiety_response_t ~ T_z + D_z      (per subject)
```

Extract per-subject:
- `intercept` — baseline after partialing T, D
- `slope_T` — within-subject regression coefficient on threat
- `slope_D` — within-subject regression coefficient on distance
- `cal_T` — pearsonr(T, response)
- `cal_D` — pearsonr(D, response)

Then re-run the phenotype profile and ANOVA on these slope/intercept measures. Test whether slope measures add predictive value beyond phenotype.

**Script:** `scripts/analysis/phenotype_metacog_slopes.py`. N = 571 pooled.

## Result

### Phenotype profile with slope measures

| Phenotype | N | Conf intercept | Conf slope_T | Anx intercept | Anx slope_T |
|---|---|---|---|---|---|
| HH | 147 | 3.56 | −0.68 | 4.27 | +0.56 |
| HL | 118 | 3.58 | −0.65 | 4.42 | +0.60 |
| LH | 138 | 3.11 | −0.60 | 4.49 | +0.52 |
| LL | 168 | 2.96 | −0.59 | 4.44 | +0.55 |

### ANOVA across phenotypes

| Measure | F | p |
|---|---|---|
| confidence_intercept | **8.70** | **1e-5 ★** |
| confidence_slope_T | 0.54 | 0.66 (null) |
| confidence_slope_D | 1.30 | 0.27 (null) |
| confidence_cal_T | 0.66 | 0.58 (null) |
| anxiety_intercept | 0.79 | 0.50 (null) |
| anxiety_slope_T | 0.33 | 0.81 (null) |
| anxiety_slope_D | 0.40 | 0.75 (null) |
| anxiety_cal_T | 0.14 | 0.94 (null) |

**Only intercept differs across phenotypes.** Every slope/calibration measure is statistically indistinguishable across the four phenotypes. The earlier characterization of HL as "overconfident" rests entirely on elevated baseline confidence, not on a difference in reactivity. HL subjects' confidence drops with threat at the same rate as everyone else's.

### Regression test (earnings ~ phenotype + slope + intercept measures)

| Term | β [95% HDI] |
|---|---|
| phenotype[HL] | −0.30 [−0.52, −0.08] ★ |
| phenotype[LH] | −0.15 [−0.36, +0.08] (n.s.) |
| phenotype[LL] | −0.28 [−0.49, −0.07] ★ |
| **confidence_slope_T** | **−0.29 [−0.41, −0.18] ★★★** |
| **anxiety_slope_T** | **+0.32 [+0.20, +0.43] ★★★** |
| confidence_intercept | +0.04 [−0.02, +0.10] (n.s.) |
| anxiety_intercept | −0.09 [−0.14, −0.02] ★ (small) |

The slope predictors do add incremental variance beyond phenotype membership. But:

- **Anxiety slope_T (+0.32) is essentially the same effect as anxiety_calibration (+0.29 in result_508)** — slope-based and correlation-based operationalizations of the same construct. Not a new finding.
- **Confidence slope_T (−0.29) is a new effect** in operationalization, but it has a plausible circular interpretation: subjects who actually perform well maintain confidence under threat (because they keep succeeding). Whether confidence stability *causes* better performance or *follows from* better performance is not identifiable without intervention.

## Interpretation

### What this refines

1. **The "HL overconfidence" claim from result_508 should be softened.** HL subjects are not miscalibrated in their confidence-to-threat reactivity. They have higher baseline confidence (3.58, same as HH) and normal slope (−0.65, similar to all phenotypes). The earnings deficit of HL (−5.8 vs HH +21.3) is driven by their *behavioral* pattern (commit + soft press), not by a metacognitive miscalibration. The strong version of "the overconfidence failure mode" is not supported by slope analysis.

2. **The "phenotypes have distinct metacognitive signatures" claim from result_508 should be limited to intercept.** Phenotypes differ in baseline confidence — but this is unsurprising and partially confounded (phenotypes are defined behaviorally, and behavior predicts self-reported baseline confidence). Phenotypes do *not* differ in any slope/calibration measure.

### What this confirms

3. **Anxiety reactivity to threat predicts behavior** — replicates result_502/508 in slope operationalization. The slope-based and calibration-based versions of this finding agree (β = +0.32 vs +0.29 on earnings). This is one finding under two operationalizations.

### What this does not add

4. **The slope-corrected analysis does not produce a new substantive finding** beyond what result_508 already reported. The confidence slope effect on earnings (β = −0.29) is novel in operationalization but has a circular causal interpretation. The headline-level claims of result_508 (three orthogonal embodied dimensions) are not strengthened or extended by this re-analysis.

## Caveats & Limitations

- **The "this is methodological refinement" framing is the honest characterization.** This was a check on whether mean-based metacognitive measures hid important slope-level structure. They didn't — slope-level structure is essentially uniform across phenotypes. That's a negative finding, not a positive one.

- **The confidence-stability → earnings effect (β = −0.29) may be circular.** Within-subject regression of confidence on threat captures how much confidence drops with rising threat. Subjects who do well on the task probably maintain confidence under threat because they're succeeding. We cannot distinguish "stable confidence → better performance" from "better performance → stable confidence" without intervention.

- **Per-subject slope estimates have limited precision at N ≈ 18 probes per subject.** The slope measures have non-trivial sampling error which would attenuate any real effects. But this argues against false positives, not against the null findings.

- **The intercept (baseline) differences across phenotypes are interpretable but not surprising.** HH and HL have high baseline confidence (3.56, 3.58); LH and LL have low (3.11, 2.96). This tracks behavioral engagement at baseline. It is not an independent metacognitive finding — it's downstream of the same construct that defines the phenotypes.

- **The earlier informal characterization of HL as "the overconfidence phenotype"** is partially supported (high intercept) and partially not (normal slope). The honest version is "HL has elevated baseline confidence with normal reactivity to threat" — which is descriptive rather than mechanistic.

## What this means for the paper

The phenotype × metacognition section of the paper should be honest:

- **Phenotypes are behaviorally distinct** (different choice, vigor, parameters, earnings) — this is real
- **Phenotypes differ in baseline confidence** but baseline confidence is partially confounded with behavior — this is descriptive
- **Phenotypes do NOT differ in metacognitive reactivity** — this is a negative finding
- **Anxiety reactivity to threat is an independent predictor of behavior** (replicates 502/508)
- **The "HL is overconfident" framing was overstated and should be removed**

The metacognitive embedding of the phenotype story is therefore *less rich* than result_508 implied. Result_508's three-dimensional embodied phenotype space remains intact for the strategic angle / intensity / anxiety calibration triad. But the phenotype-specific metacognitive signature claim is weaker than initially characterized.

## Replication

```bash
python scripts/analysis/phenotype_metacog_slopes.py
```

**Expected runtime:** ~3 min (per-subject regressions on ~10,000 probe trials + 3 bambi fits).

**Expected outputs:**
- `results/stats/clinical/phenotype_metacog_slopes_profile.csv` — per-phenotype means + SDs on slope/intercept measures
- `results/stats/clinical/phenotype_metacog_slopes_anova.csv` — ANOVA across phenotypes on each measure
- `results/stats/clinical/phenotype_metacog_slopes_subjects.csv` — per-subject table for downstream use

## References

**Related results:**
- [[result_508]] — Three orthogonal embodied dimensions. This result is a refinement of 508's phenotype × metacognition claim.
- [[result_502]] — Anxiety calibration → optimality. Confirmed in slope operationalization here.
- [[result_507]] — Affect tracks raw conditions, not embodied S(u\*). Consistent with the uniformity of metacognitive reactivity across phenotypes here.

**Notebook / scripts:**
- `scripts/analysis/phenotype_metacog_slopes.py` (slope-corrected version)
- `scripts/analysis/phenotype_metacognition_profile.py` (mean-based version that this refines)
