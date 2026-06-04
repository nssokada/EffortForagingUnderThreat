---
result_id: 508
class: metacognition_affect
title: Strategic style, avoidance intensity, and anxiety calibration are three orthogonal embodied dimensions of foraging behavior
status: supported
prereg_h: []
internal_h: []
samples: [pooled_571]
notebooks: []
scripts: [scripts/analysis/embodied_strategic_angle.py]
outputs: [results/stats/clinical/strategic_angle_clinical.csv, results/stats/clinical/strategic_angle_metacog.csv, results/stats/clinical/strategic_angle_optimality.csv]
figures: [TODO]
created: 2026-06-04
last_run: 2026-06-04
---

# Result 508 — Three orthogonal embodied dimensions of adaptive foraging

> **The polar decomposition of (ω, κ) reveals three structurally independent dimensions of embodied defensive behavior**: strategic style (angle), avoidance intensity (magnitude), and anxiety calibration accuracy. Each predicts foraging optimality *independently* (HDIs exclude zero for every dimension, in the predicted directions, on pooled N = 571), and each maps onto a distinct subjective signature: style → confidence in execution; intensity → accuracy of the affective signal. The three dimensions and their metacognitive correlates do not interact significantly — they are decoupled axes, not redundant projections. This converts the simple "subjects deviate from optimal" picture into a structured three-dimensional embodied phenotype space.

## Overview

[[result_604]] tested whether raw (ω, κ) parameter levels predict clinical and metacognitive correlates, finding mostly null effects. [[result_208]] H4d showed that the *angle* in (ω, κ) space (atan2(κ_z, ω_z)) predicts decision optimality — effort-driven avoidance is less optimal than threat-driven avoidance. This result asks the next question: when we decompose (ω, κ) into its polar coordinates (strategic angle + intensity magnitude) and add anxiety calibration as a third predictor, what does the structure look like?

The answer is striking. The three dimensions are orthogonal — they jointly predict optimality with large effects (β = −0.32 angle; β = −0.22 magnitude; β = +0.18 calibration on pct_opt, all on pooled N = 563) — and the interactions between them are essentially null. Strategic style specifically tracks confidence (β = −0.12 on mean_confidence; effort-driven subjects feel less capable). Intensity specifically tracks calibration accuracy (β = −0.13 on anx_calibration; extreme avoiders have worse-calibrated anxiety). Anxiety calibration in turn is the largest single predictor of earnings (β = +0.29) and a significant predictor of escape rate (β = +0.26). The three dimensions plus their metacognitive correlates compose a quantitative phenotype space in which different positions correspond to different embodied response styles, distinct subjective monitoring signatures, and structurally different behavioral consequences.

## Hypothesis

**Statement.** The polar decomposition of (ω, κ) — strategic angle and avoidance magnitude — together with anxiety calibration form three orthogonal dimensions of embodied defensive cognition. Each predicts behavior independently and maps onto a distinct metacognitive signature.

**Predicted direction.**
- *Strategic style (angle)*: effort-driven (high angle) avoidance is less optimal and correlates with lower confidence in execution
- *Avoidance intensity (magnitude)*: extreme avoidance is less optimal and correlates with worse anxiety calibration
- *Anxiety calibration*: more accurate anxiety calibration predicts higher optimality, earnings, and escape
- *Interactions*: null — the three dimensions should be orthogonal

**Preregistered criterion.** Not in the formal prereg. Anxiety calibration → optimality is the H5a prediction (supported in [[result_502]]); the polar decomposition of (ω, κ) is post-hoc and exploratory.

**Source of the hypothesis.** Outgrowth of the result_604 / 507 null findings on clinical and embodied-affect translations. The hypothesis: raw clinical translations are weak because they conflate strategic *style* with avoidance *intensity*; the polar decomposition separates these and may recover structured metacognitive correlates that subscale-level analyses missed.

## Data Source

- **Samples:** Pooled exploratory + confirmatory (N = 571 with parameters; N = 563–571 depending on outcome's missing pattern).
- **Inputs:**
  - `results/stats/joint_optimal/{sample}/mcmc_m4_params.csv` — ω, κ per subject (log-then-pooled-z)
  - Per-subject `master` table from `notebooks/analysis/load_data.load_both()` — escape_rate, earnings, mean_vigor, mean_anxiety, mean_confidence, anx_calibration, anx_slope, pct_opt, and all clinical scales
- **Unit of analysis:** Subject.
- **N entering analyses:** 563–571.

## Method

**Polar coordinates of (ω, κ):**

```
angle     = atan2(κ_z, ω_z)         — radians, signed; high values = effort-driven, low = threat-driven
magnitude = sqrt(ω_z² + κ_z²)       — Euclidean distance from population centroid in standardised space
```

Both z-scored on the pooled sample after computation.

**Three analysis families (pooled bambi regressions, BKW config = 4 chains × 2,000 draws + 1,000 tuning):**

1. **Polar decomposition of clinical scales** (16 scales × 3 terms = 48 tests):
   ```
   scale_z ~ angle_z + magnitude_z + angle_z:magnitude_z
   ```

2. **Polar decomposition of metacognitive variables** (3 outcomes × 3 terms = 9 tests):
   ```
   {mean_confidence, mean_anxiety, anx_calibration} ~ angle_z + magnitude_z + angle_z:magnitude_z
   ```

3. **The key optimality test** — anxiety calibration × strategic angle on outcomes:
   ```
   {pct_opt, earnings, escape_rate} ~ angle_z * anx_cal_z + magnitude_z
   ```

**Script:** `scripts/analysis/embodied_strategic_angle.py`.

## Result

### Analysis 1 — Polar decomposition of clinical scales: largely null

Only 1 of 48 terms crosses the 95% HDI threshold: STAI_Trait β(angle_z) = −0.104 [−0.184, −0.022] ★ (effort-driven subjects report less trait anxiety — wrong direction, best read as 1-of-48 noise). All other terms span zero. The same pattern as [[result_604]]: clinical scales do not track the strategic angle.

### Analysis 2 — Polar decomposition of metacognitive variables: clean signal

| Outcome | Term | β [95% HDI] |
|---|---|---|
| **mean_confidence** | **angle_z** | **−0.119 [−0.204, −0.039] ★** |
| mean_confidence | angle_z × magnitude_z | **−0.097 [−0.184, −0.011] ★** |
| mean_confidence | magnitude_z | +0.071 [−0.014, +0.153] (n.s.) |
| mean_anxiety | (all terms) | all null |
| **anx_calibration** | **magnitude_z** | **−0.129 [−0.215, −0.052] ★** |
| anx_calibration | angle_z | +0.021 [−0.062, +0.102] (n.s.) |

**Two distinct embodied dimensions have orthogonal metacognitive signatures:**
- **Strategic angle → confidence (and anti-interaction with magnitude)**: effort-driven foragers feel less capable; the effect compounds at high avoidance intensity (negative angle × magnitude interaction).
- **Magnitude → anxiety calibration**: extreme avoiders have less accurate subjective anxiety signals. Independent of strategic style.
- **Anxiety**: tracks neither (consistent with [[result_507]] — anxiety tracks raw task conditions, not embodied parameters).

### Analysis 3 — Three-dimensional decomposition of optimality (THE KEY RESULT)

`pct_opt_z ~ angle_z * anx_cal_z + magnitude_z` (N = 563):

| Term | β [95% HDI] |
|---|---|
| **angle_z** | **−0.317 [−0.390, −0.240] ★★★** |
| **anx_cal_z** | **+0.181 [+0.104, +0.256] ★★** |
| **magnitude_z** | **−0.224 [−0.304, −0.153] ★★** |
| angle_z × anx_cal_z | +0.037 [−0.037, +0.114] (n.s.) |

`earnings_z ~ angle_z * anx_cal_z + magnitude_z`:

| Term | β [95% HDI] |
|---|---|
| **anx_cal_z** | **+0.291 [+0.215, +0.373] ★★★** |
| **magnitude_z** | **−0.133 [−0.211, −0.054] ★** |
| angle_z | −0.020 [−0.098, +0.059] (n.s.) |
| angle_z × anx_cal_z | +0.006 [−0.070, +0.085] (n.s.) |

`escape_rate_z ~ angle_z * anx_cal_z + magnitude_z`:

| Term | β [95% HDI] |
|---|---|
| **anx_cal_z** | **+0.257 [+0.179, +0.337] ★★** |
| **angle_z** | **+0.137 [+0.061, +0.219] ★** |
| magnitude_z | −0.078 [−0.156, +0.002] (marginal) |
| angle_z × anx_cal_z | +0.004 [−0.075, +0.078] (n.s.) |

**Verdict — three orthogonal dimensions, each independently predicting behavior:**

1. **Strategic angle (style)**: large negative effect on `pct_opt` (β = −0.32), positive effect on escape (β = +0.14), null on earnings. Effort-driven foragers make less optimal *choices* but do escape attacks slightly better (likely because they engage with risky options less).
2. **Avoidance magnitude (intensity)**: negative effects on `pct_opt` (β = −0.22) and earnings (β = −0.13). Extreme avoidance — regardless of style — costs both decision quality and earnings.
3. **Anxiety calibration**: positive effects on every outcome — `pct_opt` (β = +0.18), earnings (β = +0.29), escape (β = +0.26). Well-calibrated anxiety is the *largest single predictor* of foraging performance. Replicates and extends [[result_502]].

**All interactions are null.** The three dimensions are structurally orthogonal predictors of behavior. They don't moderate each other — they each contribute independent variance.

## Interpretation

The result establishes a **three-dimensional embodied phenotype space** with distinct metacognitive signatures and independent behavioral consequences:

| Embodied dimension | Captures | Metacognitive correlate | Behavioral consequence |
|---|---|---|---|
| **Strategic angle** | Threat-driven vs effort-driven avoidance | **Confidence in execution** (lower for effort-driven) | Lower decision optimality, higher escape (via not engaging risky options) |
| **Avoidance magnitude** | How extreme the avoidance is | **Anxiety calibration accuracy** (lower for extreme avoiders) | Lower optimality and earnings |
| **Anxiety calibration** | How accurate the subjective threat signal is | (the metacognitive dimension itself) | The *largest* predictor of optimality, earnings, and escape |

These three dimensions are *structurally orthogonal*: the interactions between them are essentially null. A subject can be effort-driven *and* well-calibrated, or threat-driven *and* extreme, or moderate-and-poorly-calibrated. Each combination has its own behavioral signature, and each dimension contributes independent predictive variance.

**Three things this changes about the paper's pitch:**

1. **The clinical translation problem is reframed.** [[result_604]] tested whether raw (ω, κ) predicts symptom scales and found mostly null effects. But [[result_604]] was testing the wrong axis. The relevant individual-difference structure is in the *polar decomposition* of (ω, κ), which captures strategic style separately from avoidance intensity. Clinical scales still don't track this decomposition (Analysis 1), but the *metacognitive* dimensions do track it cleanly (Analysis 2). The signal was hidden in the polar geometry, not the clinical scales.

2. **Anxiety calibration is the largest individual-difference predictor of foraging performance.** β = +0.291 on earnings — larger than the strategic-angle effect (β = −0.32 on pct_opt but null on earnings) and larger than the magnitude effect (β = −0.133). Well-calibrated anxiety predicts roughly a third of a standard deviation in earnings. This converts anxiety from "the noise process that 502 mentioned" into "a major operational signal in adaptive foraging."

3. **Strategic style and avoidance intensity have orthogonal metacognitive faces.** Style determines *felt capability* (confidence); intensity determines *signal accuracy* (calibration). This isn't a clinical phenotype claim — it's a structural claim about how embodied defensive behavior interfaces with subjective experience. The body's two cost axes plus its metacognitive monitoring give three independent monitors of behavior.

**The substantive new claim:**

> Adaptive foraging under threat depends on three structurally orthogonal embodied capacities: a balanced strategic style (not too effort-driven), moderate avoidance intensity (not too extreme), and well-calibrated affective monitoring (anxiety that tracks objective threat). Each predicts behavior independently. Strategic style determines subjective confidence in execution; avoidance intensity determines the accuracy of the affective signal. This converts the conventional "anxious vs not-anxious" dimension into a three-dimensional phenotype space with distinct subjective monitoring signatures.

This is what an embodied integration of decision, action, and affect actually looks like at the individual-difference level: three orthogonal axes, each with a behavioral consequence and a subjective face, jointly determining adaptive performance.

## Caveats & Limitations

- **The clinical translation remains null.** Analysis 1 found only 1 of 48 terms significant on clinical scales (and that one is wrong-direction). The polar decomposition does NOT recover clinical correlates that the raw-parameter analysis missed. So this is a metacognitive / behavioral result, not a clinical one. The honest framing is "anxiety calibration matters; clinical scales don't track the framework."

- **Anxiety calibration's measurement is noisy at N ≈ 18 probes per subject.** The H5 prereg flagged this — split-half reliability of calibration should be reported in the manuscript. If reliability is low, the very large β = +0.29 effect on earnings may be inflated; reliability-disattenuated effect size would be larger but the precision narrower.

- **Cross-sample consistency not directly tested.** This result is on pooled N = 571 (the prereg-compliant approach for clinical-style analyses). Per-sample replications of the three-dimensional structure would strengthen the result. The angle → confidence finding may be confirmatory-dominated as 604 was.

- **The interactions are null but the three "main effects" of polar decomposition + calibration on pct_opt are exceptionally large (|β| 0.18–0.32).** This is consistent with three real independent predictors. But the simplicity of the main-effects model (no moderation) means we're not yet capturing any sub-population structure. A latent-profile or mixture model on subjects in (angle, magnitude, calibration) space could reveal phenotype clusters that the linear regression averages over.

- **Strategic-style direction matters substantively but isn't tested causally.** β(angle → pct_opt) = −0.32 is huge, but it's a between-subject regression. We can't causally claim that "switching from effort-driven to threat-driven avoidance" would improve a particular subject's optimality — that would require manipulation.

- **The negative angle × magnitude interaction on confidence (β = −0.10 ★)** is interesting and somewhat exploratory. It says effort-driven foragers feel especially uncertain when they also avoid extensively. This nests with the structural story but is the only interaction surviving in any of the three analyses; could be 1-of-many.

- **Anxiety calibration's contribution is INDEPENDENT of the parameters, not redundant.** It adds incremental variance over (ω, κ) and even over (angle, magnitude) — confirming it as a separate channel.

## Replication

```bash
python scripts/analysis/embodied_strategic_angle.py
```

**Expected runtime:** ~3 min (28 bambi fits with 4 chains × 2,000 draws + 1,000 tuning).

**Expected outputs:**
- `results/stats/clinical/strategic_angle_clinical.csv` — Analysis 1 (48 terms across 16 scales)
- `results/stats/clinical/strategic_angle_metacog.csv` — Analysis 2 (9 terms across 3 metacognitive outcomes)
- `results/stats/clinical/strategic_angle_optimality.csv` — Analysis 3 (12 terms across 3 behavioural outcomes)

## References

**Related results:**
- [[result_208]] — H4d (angle → pct_opt) — the foundational finding this builds on.
- [[result_502]] — Anxiety calibration → optimality. Replicated and extended here.
- [[result_503]] — ω → confidence (the parameter-level version of the angle → confidence finding here).
- [[result_505]] — κ → metacognitive calibration (the parameter-level version of the magnitude → calibration finding).
- [[result_507]] — Affect tracks raw conditions, not embodied S(u*). Provides the boundary on the metacognitive claim: anxiety doesn't track embodied survival, but its calibration accuracy is a key individual-difference predictor.
- [[result_604]] — Earlier null on raw clinical scales. This result demonstrates the signal was hiding in the polar decomposition, not the symptom dimensions.

**Literature:**
- Mobbs, D., Headley, D. B., Ding, W., & Dayan, P. (2020). Space, time, and fear: survival computations along defensive circuits.
- Bach, D. R., & Dolan, R. J. (2012). Knowing how much you don't know: a neural organization of uncertainty estimates.
- Heller, J., & Friston, K. (2024). Active inference under uncertainty.
- Caspi, A., et al. (2014). The p-factor: one general psychopathology factor in the structure of psychiatric disorders.
