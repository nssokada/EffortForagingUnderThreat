# EVC Model Comparison Plan

## The Claim

People integrate effort cost and threat into a unified Expected Value of Control computation that jointly governs foraging choice and motor vigor.

## The Logic

Each comparison removes one component from the full model, testing whether that component is necessary. If the full model wins every comparison, every component is justified.

---

## The Models

### FINAL: EVC 2+2 (the model we're testing)

```
Choice:  ΔEU = S × 4 - ce_i × (0.81D - 0.16)
Vigor:   EU(u) = S(u)×R - (1-S(u))×cd_i×(R+C) - ce_vigor×(u-req)²×D
S = (1 - T^γ) + ε × T^γ × p_esc
```

- Per-subject: ce (effort cost), cd (capture aversion)
- Population: ε, γ, ce_vigor, τ, p_esc, σ_motor, σ_v
- Choice data: 45 trials. Vigor data: 81 trials (includes probes).

---

### Comparison 1: Is threat needed?

**M1 — Effort Only**

```
Choice:  ΔEU = 4 - ce_i × (0.81D - 0.16)
```

No S in the reward term. The reward advantage of heavy (4 pts) is constant regardless of threat. Only effort cost varies across conditions. No vigor model (effort-only has no survival component to drive pressing).

**What it tests:** Does threat probability modulate choice beyond effort cost?

**Expected result:** FINAL wins because P(heavy) clearly drops with threat — effort alone can't explain the threat gradient.

---

### Comparison 2: Is effort needed?

**M2 — Threat Only**

```
Choice:  ΔEU = S × 4
```

No ce term. The choice is purely survival-weighted reward comparison. No individual differences in choice (only ε and γ at population level). No vigor model (nothing individual to drive pressing).

**What it tests:** Do individual differences in effort cost sensitivity add explanatory power?

**Expected result:** FINAL wins because P(heavy) varies with distance at fixed threat — threat alone can't explain the distance gradient.

---

### Comparison 3: Is the joint model needed?

**M3 — Separate Choice + Vigor Models**

```
Choice model:  P(heavy) = sigmoid((S×4 - ce_i×Δeffort) / τ)
Vigor model:   excess = α_i + δ_i × (1-S)
```

Two independent models, each with their own parameters. Choice model has ce (effort sensitivity). Vigor model has α (baseline vigor) and δ (danger mobilization). No shared parameters between choice and vigor — they're fit independently.

**What it tests:** Does a unified EVC computation outperform independent models?

**Expected result:** FINAL wins because shared ε and γ constrain both channels simultaneously, and the joint likelihood provides better regularization. Also fewer total parameters (2 per-subject vs 4 per-subject).

---

### Comparison 4: Are individual effort cost differences needed?

**M4 — Population ce**

```
Choice:  ΔEU = S × 4 - ce_pop × (0.81D - 0.16)
```

Same as FINAL but ce is a single population value, not per-subject. cd is still per-subject from vigor.

**What it tests:** Do people differ in effort cost sensitivity, or is effort cost a shared task feature?

**Expected result:** FINAL wins because the distance gradient in choice varies across subjects — some people are distance-sensitive, others aren't. Population ce can't capture this.

**Already tested:** BIC = ~20,000 for pop ce vs 17,768 for per-subject ce (ΔBIC > 2,000).

---

### Comparison 5: Is probability weighting needed?

**M5 — No γ (γ fixed at 1)**

```
S = (1 - T) + ε × T × p_esc
```

Same as FINAL but no probability distortion — objective threat probabilities are used directly.

**What it tests:** Do people distort threat probabilities?

**Expected result:** FINAL wins because γ=0.21 substantially improves vigor-threat calibration (reduces overprediction from 3.6× to 1.7× in earlier models).

**Already tested:** ΔBIC > 800 for adding γ.

---

### Comparison 6: Is the LQR cost structure needed?

**M6 — Standard quadratic cost (u²×D everywhere)**

```
Choice:  ΔEU = S × 4 - ce_i × 0.81 × D  (same as FINAL — req² is just a constant)
Vigor:   EU(u) = S(u)×R - (1-S(u))×cd_i×(R+C) - ce_vigor × u² × D
```

Uses u² instead of (u-req)² in vigor. The commitment cost for choice is the same (req² × D is equivalent), but the vigor deviation cost changes.

**What it tests:** Does the LQR deviation-from-setpoint formulation improve vigor prediction over standard quadratic cost?

**Expected result:** FINAL wins (ΔBIC ≈ 150 from earlier LQR comparison). The deviation cost naturally produces small vigor costs at the required rate, avoiding the scaling conflict.

---

## Summary Table (for paper)

| # | Model | Tests | Per-subj params | BIC | ΔBIC vs FINAL |
|---|-------|-------|----------------|-----|---------------|
| M1 | Effort only | Is threat needed? | ce | TBD | TBD |
| M2 | Threat only | Is effort needed? | — | TBD | TBD |
| M3 | Separate choice + vigor | Is joint needed? | ce, α, δ (4 total) | TBD | TBD |
| M4 | Population ce | Are effort diffs needed? | cd | ~20,000 | ~+2,200 |
| M5 | No prob. weighting | Is γ needed? | ce, cd | ~18,600 | ~+800 |
| M6 | Standard u² cost | Is LQR needed? | ce, cd | ~17,920 | ~+150 |
| **FINAL** | **EVC 2+2** | **Full model** | **ce, cd** | **17,768** | **0** |

---

## What This Tells Reviewers

1. **Effort matters** (M1 < FINAL): Can't explain choice without effort cost
2. **Threat matters** (M2 < FINAL): Can't explain choice without survival probability
3. **Joint > separate** (M3 < FINAL): Unified EVC outperforms independent models
4. **Individual effort diffs matter** (M4 < FINAL): People differ in effort sensitivity
5. **Probability weighting matters** (M5 < FINAL): People distort threat probabilities
6. **LQR cost structure matters** (M6 < FINAL): Deviation cost > standard quadratic

Each ablation removes one ingredient and shows the fit degrades. The full EVC model is justified component by component.

---

## What Still Needs Fitting

- **M1 (effort only)** — needs fitting
- **M2 (threat only)** — needs fitting
- **M3 (separate models)** — needs fitting (or use the old prereg model results)
- M4, M5, M6 — already have approximate results from earlier model comparisons, but should be refit with the current Option 2 architecture for clean comparison
