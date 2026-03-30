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

## Bottom-Line Assessment

### What M3 winning means

The objective survival function `S = exp(-p·T)` captures the threat effect in choice without needing probability distortion (α/γ) or a separate threat sensitivity parameter. Parsimony wins: 2 per-subject parameters (effort cost + temperature) outperform 3-5 parameter models on BIC.

BUT: M3 has lower accuracy (0.782 vs 0.825 for M2) and lower per-subject r² (0.898 vs 0.989 for M2). It misses individual-level variation that the 3-param model captures. The BIC comparison penalizes extra parameters heavily — with 290 subjects × 45 trials, each per-subject parameter costs 290 × log(13050) ≈ 2,745 BIC points. This is why M2 (one extra per-subject param) is +1,772 BIC despite much better fit.

### What doesn't change

1. **Model parameters → clinical: null.** Whether you use M3's λ or v2's k/β/cd, model parameters don't predict psychiatric symptoms. The affect channel (discrepancy) is the only bridge.

2. **Vigor prediction from choice params is weak.** M3's choice parameters explain <1% of anticipatory vigor variance. The sequential prediction test shows a statistically significant but practically negligible link.

3. **Encounter dynamics show small threat modulation.** Delta-vigor is weakly threat-modulated (d=0.19), detectable from raw epochs but not from the 20Hz timeseries. The effect is real but tiny.

### What the 3-param v2 model does that M3 doesn't

- Separates effort from threat in choice (k vs β — orthogonal, independently recoverable)
- Predicts individual choices much better (r²=0.989 vs 0.898)
- Connects to vigor through cd (cd → frac_full r=0.710; M3's λ → frac_full r=0.132)
- The triple dissociation (k→overcaution, β→threat sensitivity, cd→vigor gap) has no analog in M3

### Recommendation

**The 3-param v2 model (k + β + cd) remains the stronger paper model.** M3 is more parsimonious for choice, but the paper's contribution is the TRIPLE DISSOCIATION and the SEPARABILITY of effort, threat, and vigor — which requires three parameters. M3 can be reported as a model comparison showing that objective survival captures the population-level choice pattern, with the 3-param model justified by its superior individual-level prediction and the vigor/encounter dynamics results that M3 can't touch.

The survival function comparison (M3's `exp(-p·T)` vs M2's linear `γ·p`) is informative: it shows the data are consistent with a survival computation in choice, not just a linear threat penalty. But for the individual-differences story, the 3-param separability is more important than the specific functional form.
