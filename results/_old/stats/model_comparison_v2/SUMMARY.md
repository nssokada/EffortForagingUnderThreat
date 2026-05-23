# Model Comparison v2 — Full Results Summary

*8 choice models, sequential vigor prediction, clinical associations, parameter correspondence. Run 2026-03-30.*

---

## Phase A: Effort Parameterization

Two effort terms tested within M1 (effort-only) and M2 (effort + linear threat):

| Model | Effort | BIC_approx | Accuracy | Choice r² |
|-------|--------|-----------|----------|-----------|
| M2 | **req·T** | **20,274** | 0.825 | 0.989 |
| M2 | req²·D | 20,419 | 0.823 | 0.986 |
| M1 | req·T | 21,080 | 0.706 | 0.957 |
| M1 | req²·D | 21,284 | 0.703 | 0.936 |

**Winner: req·T (ΔBIC = 145 in favor).** λ is interpretable as cost per second of pressing. All subsequent models use req·T.

---

## Phase B: 8-Model Choice Comparison

| Rank | Model | Description | Per-subj params | BIC_approx | ΔBIC | Accuracy | Choice r² |
|------|-------|-------------|----------------|-----------|------|----------|-----------|
| 1 | **M3** | **Objective survival + effort** | 2 (λ, β) | **18,502** | **0** | 0.782 | 0.898 |
| 2 | M4 | Rate-of-return + survival | 2 (λ, β) | 18,881 | 379 | 0.779 | 0.932 |
| 3 | M2 | Additive effort + linear threat | 3 (λ, γ, β) | 20,274 | 1,772 | 0.825 | 0.989 |
| 4 | M1 | Effort only (no threat) | 2 (λ, β) | 21,080 | 2,578 | 0.706 | 0.957 |
| 5 | M8 | Additive + T×D interaction | 4 (λ, γ, δ, β) | 22,927 | 4,425 | 0.825 | 0.986 |
| 6 | M6 | Rate-of-return + distorted survival | 4 (λ, κ, α, β) | 23,198 | 4,696 | 0.819 | 0.989 |
| 7 | M5 | Distorted survival + effort | 4 (λ, κ, α, β) | 23,299 | 4,797 | 0.832 | 0.958 |
| 8 | M7 | Distorted surv + penalty weighting | 5 (λ, κ, α, ψ, β) | 25,539 | 7,037 | 0.832 | 0.992 |

### Key findings

**M3 wins decisively on BIC** (ΔBIC > 378 over M4, > 1,772 over M2). The objective survival function `exp(-p·T)` captures the threat effect in choice without probability distortion and without a separate threat parameter.

**M5/M6 (probability distortion) do NOT improve over M3.** Despite fitting better in absolute terms (higher accuracy, higher r²), the extra parameters (κ, α) are not justified by BIC. The distortion isn't needed — objective survival probability is sufficient.

**M2 (our 3-param v2 benchmark) ranks 3rd.** The additive linear model fits individual choices better (r²=0.989 vs 0.898) but needs an extra per-subject parameter (γ for threat). M3 achieves comparable population-level fit with 2 parameters instead of 3.

**The accuracy/BIC trade-off:** M3 has the LOWEST accuracy (0.782) among competitive models. M2/M7/M5 all exceed 0.82. M3 wins because parsimony — 2 per-subject params vs 3-5 for the others. Whether parsimony or individual-level prediction matters more depends on the scientific question.

### Posterior predictive check (M3)

```
            D=1          D=2          D=3
T=10%  .806/.883    .690/.734    .563/.512
T=50%  .632/.541    .378/.265 *  .186/.147
T=90%  .394/.416    .135/.244 *  .079/.150
```
(observed/predicted, * = gap > 0.10)

M3 misfits at T=0.5/D=2 (underpredicts by 0.11) and T=0.9/D=2 (overpredicts by 0.11). The survival function `exp(-0.9 × 7) ≈ 0.002` drives P(heavy) near zero at high threat/distance, but observed behavior is more moderate.

---

## Part 2: Sequential Vigor Prediction

### Step 14: Anticipatory vigor

Choice-model ΔV predicts anticipatory pressing rate on forced trials:
- **ΔV → anticipatory vigor: β = 0.009, t = 3.31, p < 0.001**
- Heavy cookies: r = 0.051 (p < 0.001)
- Light cookies: r = 0.021 (p = 0.18, ns)
- Overall R² = 0.001

**Verdict:** Statistically significant but tiny effect. The choice model's value computation does weakly predict motor preparation on forced trials, but explains <1% of vigor variance.

### Step 15: Delta-vigor (reactive − anticipatory)

| Threat | Mean delta-vigor | SD | N |
|--------|-----------------|-----|---|
| T=10% | -0.017 | 0.191 | 2,830 |
| T=50% | -0.000 | 0.211 | 2,775 |
| T=90% | +0.008 | 0.187 | 2,783 |

**T=0.9 vs T=0.1: t = 3.25, p = .001, d = 0.19**

Threat DOES modulate delta-vigor — people press slightly harder after the predator appears at high threat. This is NEW: our earlier encounter dynamics analysis (using the 20Hz smoothed timeseries) showed threat-independence (F=0.04, p=.96). The difference: epochs here are defined from raw keypress timestamps, not the oversampled 20Hz grid. The effect is small (d=0.19) but real.

No distance effect on delta-vigor (D=1: +0.004, D=2: +0.001, D=3: -0.011).

### Step 16a: ΔV–delta-vigor correlation

- **Between-subject: r = 0.028, p = .64 (NULL)**
- Within-subject: mean r = -0.049, significantly below zero (t = -3.90, p < 0.001)

The choice model's ΔV does NOT predict delta-vigor. The within-subject correlation is actually NEGATIVE — higher ΔV (heavier favored) is associated with LESS reactive vigor increase. This makes sense: if the trial is already high-value, there's less marginal benefit from the encounter response.

### Step 16b: Threshold test feasibility

45.4% of participants show a sign change in delta-vigor across threat levels. The threshold test IS feasible for roughly half the sample, but the small effect size (d=0.19 for the overall threat modulation) means individual-level crossing points will be noisy.

### Step 17: Abandonment

Abandonment rate on heavy forced attack trials: **5.7%** overall.
- T=10%: 4.1%
- T=50%: 5.6%
- T=90%: 5.9%

Too rare to reliably test the timing prediction. Only ~134 trials have clear abandonment. The abandonment timing test is not feasible with this sample.

---

## Part 3: Individual Differences

### Step 19: Clinical regressions

**M3 parameters → clinical: ALL NULL.**

| Parameter | Best clinical r | Interpretation |
|-----------|----------------|---------------|
| λ (effort) | r = -0.069 (DASS-Stress, p = .24) | No clinical link |
| β (temperature) | r = +0.070 (AMI, p = .23) | No clinical link |

Neither λ nor β predicts any psychiatric measure. This replicates our 3-param v2 finding: model parameters don't bridge to clinical symptoms.

**Discrepancy → clinical: STRONG, replicates.**

| Clinical measure | r | p |
|-----------------|---|---|
| STAI-State | +0.326 | < 0.0001 |
| STICSA | +0.264 | < 0.0001 |
| DASS-Anxiety | +0.251 | < 0.0001 |
| DASS-Stress | +0.227 | 0.0001 |
| DASS-Depression | +0.211 | 0.0003 |
| PHQ-9 | +0.204 | 0.0005 |
| OASIS | +0.204 | 0.0005 |
| AMI | +0.026 | 0.66 (ns) |

Discrepancy (affective bias) predicts 7/8 clinical measures at p < .001. Identical pattern to 3-param v2 findings. The bridge from foraging to clinical runs through AFFECT, not model parameters.

### Step 22: Anxiety tercile choice surfaces

```
LOW ANXIETY (N=103):
         D=1    D=2    D=3
T=10%  .812   .695   .583
T=50%  .639   .392   .188
T=90%  .404   .146   .089

MID ANXIETY (N=93):
         D=1    D=2    D=3
T=10%  .785   .652   .538
T=50%  .645   .368   .176
T=90%  .430   .116   .065

HIGH ANXIETY (N=94):
         D=1    D=2    D=3
T=10%  .821   .721   .568
T=50%  .613   .372   .194
T=90%  .347   .143   .081
```

The main difference is at **T=0.9/D=1**: high anxiety 0.347 vs low anxiety 0.404. High-anxiety people are slightly more cautious at the highest threat/closest distance, but the effect is modest. The surfaces are remarkably similar across terciles — anxiety doesn't dramatically reshape the foraging strategy, it just shifts the T=0.9 row slightly downward.

---

## Part 4: Parameter Correspondence

### M3 parameters → frac_full

| M3 param | r with frac_full | Interpretation |
|----------|-----------------|----------------|
| λ (effort) | **+0.132** (p = .024) | Weak: higher effort cost → slightly MORE consistent pressing (unexpected) |
| β (temperature) | +0.057 (p = .34) | Null |

Compare to 3-param v2: cd → frac_full r=0.710. **M3's parameters barely predict vigor.** The choice model captures effort sensitivity (λ) but this doesn't translate to motor output the way cd does. The survival computation in M3 is about choice-level threat evaluation, not motor urgency.

### M3 ↔ 3-param v2 parameter correspondence

| M3 param | v2 k (effort) | v2 β (threat) | v2 cd (vigor) |
|----------|---------------|---------------|---------------|
| λ (effort) | **r = 0.929** | r = 0.303 | r = -0.166 |
| β (temperature) | r = -0.554 | r = 0.305 | r = 0.107 |

**M3's λ IS our k** (r=0.929). The effort cost parameter is the same individual difference under both model architectures.

**M3's β (inverse temperature) correlates moderately with both v2 k and v2 β-threat** (r ≈ 0.3–0.55). In M3, choice stochasticity absorbs some of what v2 separates into effort and threat sensitivity. People who are "noisier" in M3 are actually a mix of low-effort-cost and high-threat-sensitivity people in the 3-param model.

---

## M5 α Analysis: Probability Distortion Is a Population Constant

### Individual α distribution

| Metric | Value |
|--------|-------|
| Mean | 2.759 |
| Median | 2.760 |
| SD | **0.007** |
| Range | [2.733, 2.780] |
| % with α > 1.0 | 100% |
| t-test vs 1.0 | t = 4,524, p ≈ 0 |

**α is a population constant, not an individual-difference parameter.** Every participant has α ≈ 2.76. The posterior SD (0.007) is 219× narrower than the prior SD (0.555 in log space), confirmed by prior predictive check (1.7th percentile — the data genuinely constrain α, not the prior).

### α → clinical: completely null

| Clinical measure | r | p |
|-----------------|---|---|
| STAI-State | +0.022 | .71 |
| OASIS | +0.048 | .42 |
| DASS-Anxiety | +0.018 | .77 |
| All others | < 0.07 | > .23 |

No anxiety tercile difference (high α = 2.760, low α = 2.760).

### Interpretation

The probability distortion `p^2.76` is a fixed property of how humans process threat probabilities in this task — not a parameter modulated by individual anxiety. This means M5 = M3 with a reparameterized survival function: `exp(-κ·p^2.76·T)` ≈ `exp(-κ'·p·T)` with rescaled κ. The data don't have enough structure to separately identify probability distortion from baseline hazard rate when both are free.

**This strengthens M3 as the main model:** the survival computation is a stable population-level mechanism, not something tuned by individual psychology. Individual differences live in the effort cost channel (λ) and the affective channel (discrepancy), not in the survival function.

---

## Bottom-Line Assessment

### M3 winning IS the integration finding

M3 winning with ΔBIC > 1,772 over M2 means the data **strongly prefer a model where threat and exposure duration interact multiplicatively** (`exp(-p·T)`) over a model where they're additive and independent (`-γ·p`). The survival function is the integration mechanism. This is a direct empirical test of the Lima-Dill-Bednekoff fitness framework in humans, and the objective survival computation wins.

The fact that probability distortion (M5's α) adds nothing beyond M3 sharpens the claim: people implement something close to the normative ecological survival computation. They don't need subjective distortion because the objective survival function is sufficient.

### Two-level story for the paper

**Population level:** Human foraging under threat is best described by an objective survival computation — `exp(-p·T)` — weighted by reward value minus effort cost. M3 captures this with 2 per-subject parameters (effort cost λ and choice temperature β) and beats all alternatives on BIC. The survival function is a fixed computational mechanism that does not vary with individual anxiety.

**Individual level:** Variation in clinical status is predicted by **affective discrepancy** — the gap between felt and rational threat appraisal — not by parameters of the survival computation. Discrepancy predicts 7/8 clinical measures at p < .001 while λ and β predict nothing. The computational and affective channels are genuinely separable: the model captures what people DO, discrepancy captures how they FEEL about what they do, and only the feeling channel bridges to clinical status.

### The 3-param v2 model remains valuable as a supplement

The 3-param model (k + β_threat + cd) provides:
- Better individual-level choice prediction (r² = 0.989 vs 0.898)
- Separability of effort from threat in choice (k vs β_threat orthogonal)
- Connection to vigor through cd (cd → frac_full r = 0.710)
- The triple dissociation (k→overcaution, β→threat sensitivity, cd→vigor gap)

These are real and publishable findings, but they describe the structure of individual differences WITHIN the computational framework that M3 defines. The paper can present M3 as the main model and the 3-param individual-differences analysis as extending it — showing that the effort and threat components of the survival computation map onto separable behavioral channels.

### The D=2 misfit

M3 misfits at T=0.5/D=2 (underpredicts by 0.11) and T=0.9/D=2 (overpredicts by 0.11). The α analysis confirms this is NOT from heterogeneous probability distortion (α has zero individual variance). It's a systematic property of the exponential survival function being slightly miscalibrated at intermediate distances. A population-level correction (M3+ with `exp(-κ·p^α·T)` at fixed α ≈ 2.76) could address this as a supplementary robustness check without invoking individual differences.

### What doesn't change across model architectures

1. **Model parameters → clinical: null.** Whether M3 (λ, β) or 3-param v2 (k, β_threat, cd), model parameters don't predict psychiatric symptoms. This is invariant to model specification.

2. **Discrepancy → clinical: strong.** 7/8 measures at p < .001. The bridge from foraging to clinical runs through affect.

3. **Vigor prediction from choice params is weak.** <1% variance explained. Choice and vigor are partially decoupled at the individual level, connected only through the population-level survival function.

4. **Encounter dynamics show small threat modulation.** d = 0.19, detectable from raw epochs. The reactive motor system is a separate channel from the choice computation.
