# Separating c_effort from c_death: Options Analysis

## The Problem

In the current EVC model, the expected utility is:

```
EU(u) = S(u) * R - (1 - S(u)) * cd * (R + C) - ce * effort_term
```

For **choice**, the critical comparison is EU_H - EU_L. The death penalty term contributes:

```
-(1 - S) * cd * 10  for heavy  (R+C = 5+5 = 10)
-(1 - S) * cd * 6   for light  (R+C = 1+5 = 6)
```

The differential is `(1 - S) * cd * 4`. Meanwhile, the effort differential (in the LQR formulation) is:

```
ce * 0.81 * D_H  for heavy  (req=0.9, so req^2=0.81)
ce * 0.16         for light  (req=0.4, so req^2=0.16)
```

Both terms create an option differential that favors light over heavy, and both scale with conditions (S depends on T; effort depends on D). The death penalty differential `cd * 4 * (1-S)` varies with threat. The effort differential `ce * (0.81*D - 0.16)` varies with distance. In principle, these are identifiable because they respond to different manipulations. But in practice, c_death absorbs the effort signal because:

1. The death penalty differential (range: cd * 4 * 0.03 to cd * 4 * 0.97) is much larger than the effort differential (range: ce * 0.65 to ce * 2.27) for any reasonable cd.
2. Distance and threat are partially correlated in their effect on choice (high T, high D both push toward light).
3. With cd free per subject, it has enough flexibility to capture distance gradients that should belong to ce.

For **vigor**, the problem is worse. The vigor EU uses `(1-S(u)) * cd * (R+C)` where R+C depends on the chosen cookie. Since cd scales with the full (R+C) stakes, even small cd values generate large vigor gradients, leaving no room for ce to contribute meaningful individual variation.

---

## Option 1: Learn from the Preregistered Model's Success

### How the old model separated k and beta

The preregistered model (M5) was:

```
SV = R * S - k * E - beta * (1 - S)
S = (1 - T) + T / (1 + lambda * D)
```

Key features:
- **S depends on distance D**, so S_H != S_L (S is option-specific)
- **k enters as k * E** where E is a scalar effort level (0.4 for light, 0.6/0.8/1.0 for heavy)
- **beta enters as beta * (1 - S)** where (1 - S) is option-specific

The separation worked because:
1. **k * E** varies ONLY with effort level, not threat. k captures pure effort aversion.
2. **beta * (1 - S)** varies with BOTH threat and distance (through S). beta captures threat aversion beyond expected value.
3. Crucially, the penalty for capture was a fixed C=5 for both options, so (1-S)*C was NOT option-differential in the same way. The beta*(1-S) term was an ADDITIONAL threat bias, not a scaled loss.

### What changed in the EVC model

The EVC model replaced `beta * (1-S)` with `cd * (R+C)` which is option-differential (10 vs 6). This couples cd to the reward structure, giving it leverage over choice that the old beta didn't have. The old beta was a pure "threat surcharge" -- it didn't scale with stakes. The new cd does.

### Proposed fix: Decouple cd from reward differential

**Specification:**
```
EU_choice = S * R - (1 - S) * cd * C - ce * effort_term
```

where C = 5 (fixed for both options). The death penalty is `cd * 5` for BOTH options. This means the cd contribution to the choice differential is:

```
Delta_cd = -(1-S_H)*cd*5 + (1-S_L)*cd*5 = cd*5*(S_H - S_L)  [if S differs by option]
```

But wait -- in the current EVC model, S doesn't depend on distance (predator timing is calibrated to return time). So S_H = S_L and cd*C drops out of the choice comparison entirely! This is the problem: if S is the same for both options, any term of the form `f(cd) * g(S)` cancels.

However, this is where the preregistered model's insight matters. In the prereg, S DID depend on D via `S = (1-T) + T/(1+lambda*D)`. That gave beta leverage. The EVC model's survival function doesn't have this property because it's based on a sigmoid around the speed threshold, and at the "press at full speed" level, survival doesn't depend on distance.

**Critical insight:** The old model's S was a phenomenological approximation that treated distance as affecting survival. The EVC model's S is mechanistically derived and says distance DOESN'T affect survival (at full speed). The mechanistic model may be more accurate, but it removes the lever that separated the parameters.

### Verdict

**Theoretically principled but requires reintroducing distance-dependent S.** If we accept a phenomenological S(T,D) for the choice model (as the prereg did), we can separate the parameters. But this contradicts the EVC model's mechanistic survival function. This is a tension between identifiability and mechanistic accuracy.

**Risk:** Fitting a phenomenological S to get identifiability, then claiming the model is mechanistically grounded.

**Rating: 6/10.** Principled in the prereg context, but introduces inconsistency with the EVC framework.

---

## Option 2: Separate Penalty Specifications for Choice vs. Vigor

### Specification

**Choice:**
```
EU_H = S * 5 - (1 - S) * cd_choice * C_choice - ce * effort_H
EU_L = S * 1 - (1 - S) * cd_choice * C_choice - ce * effort_L
```
where C_choice is a fixed penalty (e.g., 5). Since S_H = S_L, the cd_choice * C terms cancel, and choice is driven PURELY by the reward differential and effort differential:
```
EU_H - EU_L = S * 4 - ce * (effort_H - effort_L)
```

This is essentially the old model's structure! Choice depends on S (which depends on T and epsilon), reward differential (4 pts), and effort cost. ce is identifiable because it's the only thing that varies with distance/effort.

**Vigor:**
```
EU_vigor(u) = S(u) * R_chosen - (1 - S(u)) * cd_vigor * (R_chosen + C) - ce * (u - req)^2 * D
```

cd_vigor drives vigor through the full stakes mechanism. ce also enters vigor through the LQR deviation cost.

### Analysis

This is actually a very clean solution. For choice, when S is the same for both options, cd drops out and we get:
```
P(heavy) = sigmoid((S * 4 - ce * delta_effort) / tau)
```

ce is now the ONLY free parameter driving choice variation across conditions (since S*4 is determined by T and epsilon). This is exactly what the old k did.

For vigor, cd_vigor and ce both enter, but they play different roles: cd_vigor scales the survival-threat gradient (incentive to press harder), while ce scales the LQR deviation cost (penalty for pressing above setpoint).

### But wait -- there's a subtlety

If S_H = S_L (same survival for both options), then the choice equation becomes:

```
EU_H - EU_L = S*(R_H - R_L) - ce*(effort_H - effort_L)
```

This means ce is identified from the interaction of effort level with choice. The distance gradient in effort (ce * req^2 * D) gives additional leverage. This should work.

However, cd_choice is completely unidentified from choice (it cancels). We'd need to either (a) drop it from choice, or (b) reintroduce distance-dependent S for choice.

If we drop cd from choice, then choice has: population S (from T, epsilon), reward differential, and ce. Vigor has: cd_vigor, ce, epsilon. The two cd parameters (choice, vigor) are replaced by one cd that only enters vigor.

### Verdict

**This is essentially a two-equation system where ce is identified from choice and cd is identified from vigor.** It's principled because it reflects the actual information content: choice tells you about effort preferences (given that survival is the same for both options), vigor tells you about death aversion (through the press-harder-to-survive mechanism).

**Risk:** ce from choice might be poorly identified if effort variation is small relative to noise. The effort differential between heavy and light is (0.81*D - 0.16), ranging from 0.65 to 2.27. With tau ~ 0.5, this gives logit contributions of ce * 1.3 to ce * 4.5. Should be adequate.

**Rating: 8/10.** Clean, principled, preserves the joint model structure.

---

## Option 3: Multiplicative Effort Discount on Reward

### Specification

```
EU = S * R * exp(-ce * D) - (1 - S) * cd * (R + C)
```

ce now acts as a multiplicative discount on reward, not an additive cost. The choice differential:

```
EU_H - EU_L = S * [5*exp(-ce*D_H) - 1*exp(-ce*1)] - (1-S) * cd * [10 - 6]
```

= `S * [5*exp(-ce*D_H) - exp(-ce)] - (1-S) * cd * 4`

### Analysis

ce and cd now operate on genuinely different terms:
- ce: reward-side, multiplicative, varies with D
- cd: penalty-side, additive, varies with T (through S)

The key question: can cd still absorb ce? The cd * 4 * (1-S) term creates a differential that varies with T but not D. The ce term creates a differential that varies with D but also with T (through S weighting). There's still some correlation, but the functional forms are different enough that they should separate.

However, this formulation has a conceptual problem: what does "effort discounts reward" mean? In the EVC framework, effort is a COST, not a discount on reward. An exponential discount means "distant rewards are worth less" which is more like temporal discounting than effort cost. It conflates the effort dimension with a reward-scaling dimension.

Also, `exp(-ce * D)` doesn't use the effort level (req), only distance. So it's really a "distance discount" not an "effort discount." This loses the fact that heavier cookies require more effort per unit distance.

A better version: `exp(-ce * req * D)` or `exp(-ce * req^2 * D)`. This gives:
```
Heavy: 5 * exp(-ce * 0.81 * D_H)
Light: 1 * exp(-ce * 0.16)
```

The effort differential now properly depends on both req and D.

### Verdict

**Theoretically strained but mathematically effective.** The exponential discount IS what the old model used (`R * exp(-k*E)` in M1-M4), so there's precedent. But it doesn't sit well within the EVC cost framework.

**Risk:** Reviewers may object to effort "discounting reward" rather than being a cost. Also, the exponential makes ce's effect nonlinear and potentially hard to interpret.

**Rating: 5/10.** Works mathematically but conceptually awkward in the EVC framework.

---

## Option 4: Reparameterize as (k, beta) Then Unpack

### Specification

Define the choice model in terms of the old parameters:
```
SV_choice = S * R - k * effort - beta * (1 - S)
```

Fit k_i and beta_i per subject from choice data. Then define the mapping:
```
ce = k  (effort cost = effort sensitivity)
cd = beta / (some normalization)  (death aversion = threat bias scaled)
```

For vigor:
```
EU_vigor(u) = S(u) * R - (1 - S(u)) * cd * (R + C) - ce * (u - req)^2 * D
```

where ce and cd are derived from k and beta.

### Analysis

This is essentially a two-stage approach dressed up as a reparameterization. Stage 1: fit choice to get k and beta. Stage 2: use k and beta to derive ce and cd for vigor.

The problem is that the mapping between (k, beta) and (ce, cd) is not straightforward:
- k in the old model = sensitivity to effort level (scalar E). ce in the EVC model = cost per unit of effort^2 * distance. These are different functional forms of "effort."
- beta in the old model = threat bias beyond EV. cd in the EVC model = subjective scaling of (R+C) stakes. These capture different aspects of threat aversion.

We'd need: `k * E = ce * req^2 * D`, so `ce = k * E / (req^2 * D)`. But E and D are condition-specific, so this mapping isn't unique.

Actually, let's think more carefully. In the old model, effort enters as `k * E` where `E = effort_H` (a fraction of calMax). In the current model, effort enters as `ce * req^2 * D` where req = 0.9 for heavy. These are proportional if `E ~ req^2 * D`. Given that E = 0.6, 0.8, 1.0 for D = 1, 2, 3, and `req^2 * D = 0.81, 1.62, 2.43`, the relationship isn't proportional (E ranges 1.67x while req^2*D ranges 3x). So the mapping is approximate at best.

### Verdict

**Hacky.** The parameter mapping isn't clean, and the result is effectively a two-stage model with an ad-hoc transformation. Loses the joint estimation benefit.

**Rating: 3/10.** Not recommended.

---

## Option 5: Two-Stage Model (Sequential Estimation)

### Specification

**Stage 1 — Choice model** (same as prereg M5):
```
SV = R * S - k * E - beta * (1 - S)
S = (1 - T) + T / (1 + lambda * D)
```
Fit hierarchical Bayesian model: k_i, beta_i, population lambda.

**Stage 2 — Vigor model** (conditional on Stage 1):
Fix S from Stage 1 (using fitted lambda). Fit:
```
excess_vigor = alpha_i + delta_i * (1 - S) + epsilon
```
Or the full EVC vigor model:
```
u* = argmax [S(u) * R - (1-S(u)) * cd * (R+C) - ce * (u-req)^2 * D]
```
with ce fixed (transformed from k_i) and cd free.

### Analysis

This is the safest option. The prereg already showed k and beta separate cleanly. The vigor model can then use these parameters as inputs.

Advantages:
- Known to work (prereg demonstrated separation)
- No identifiability issues (parameters identified from different data sources)
- Straightforward to implement
- Can report both models separately

Disadvantages:
- Not a joint model -- loses the "one computation, two outputs" narrative
- Uncertainty from Stage 1 isn't propagated to Stage 2 (can be mitigated with posterior samples)
- The EVC story requires a single unified computation; two stages undermines this

Actually, there's a hybrid: **fit choice with the prereg model to get k_i and beta_i, then use these as informative priors for ce_i and cd_i in the joint model.** This is an empirical Bayes approach. The choice data pins down the parameters, and the vigor data refines them.

### Verdict

**Safe and proven, but narratively weak.** The two-stage approach works but doesn't serve the "unified computation" story. The empirical Bayes hybrid is more interesting.

**Rating: 7/10** (two-stage), **8/10** (empirical Bayes hybrid).

---

## Option 6: Reintroduce Distance-Dependent Survival for Choice

### Specification

Use a hybrid survival function:
```
S_choice = (1 - T_w) + eps_i * T_w * p_esc * h(D)
```

where `h(D)` is a distance-dependent escape probability. For example:
```
h(D) = 1 / (1 + lambda * D)   [hyperbolic, as in prereg]
```
or
```
h(D) = exp(-lambda * D)   [exponential]
```

This makes S_H != S_L for choice, which gives cd leverage that's distinguishable from ce.

For vigor, keep the mechanistic sigmoid-based S(u).

### Analysis

The task design note says "predator spawn timing is calibrated to return time." This means that at full speed, survival probability is roughly constant across distances. But "roughly" isn't "exactly" -- there's noise, reaction time, etc. A mild distance dependence in S is empirically plausible even if mechanistically it shouldn't be strong.

More importantly, participants may PERCEIVE distance as affecting survival even if objectively it doesn't (much). The phenomenological S(T,D) captures subjective survival probability, not objective.

With distance-dependent S for choice:
```
EU_H - EU_L = S_H * 5 - S_L * 1 - (1-S_H)*cd*10 + (1-S_L)*cd*6 - ce*(effort_H - effort_L)
```

Now S_H < S_L (farther = less safe), so:
- The reward term `S_H * 5 - S_L * 1` varies with D
- The cd term varies with D through (S_H - S_L)
- The ce term varies with D through effort

cd and ce are now both distance-dependent, but through different channels (survival vs. effort). With threat varying independently, ce and cd may separate.

### Verdict

**Conceptually justified (subjective survival perception) and practically effective.** The risk is that lambda, cd, and ce are all trying to explain the distance gradient, creating a three-way identification problem. Need to constrain lambda (e.g., fix from prereg or use a tight prior).

**Rating: 7/10.** Good if lambda is constrained.

---

## Option 7: Make ce Enter Vigor ONLY Through a Motor Channel

### Specification

Separate the effort cost into two distinct channels:

**Choice channel:** ce determines choice via `SV = S*R - ce*req^2*D - cd_term`. Here, cd_term is either zero (if S is the same for both options and penalty is fixed) or minimal. ce drives choice.

**Vigor channel:** The vigor EU does NOT include ce in the optimization. Instead:
```
u* = argmax [S(u) * R - (1-S(u)) * cd * (R+C)]
excess_predicted = u* - req

vigor_observed = excess_predicted + motor_noise
```

ce enters choice but NOT vigor. cd enters vigor but NOT choice (or only weakly through choice, via a small fixed penalty).

### Analysis

This completely separates the identification: ce from choice, cd from vigor. But it breaks the EVC narrative -- the whole point is that the SAME computation drives both outputs. If ce doesn't enter vigor, then the forager isn't trading off effort cost against survival benefit when deciding how hard to press. That's a strong claim that effort cost doesn't matter for motor execution, only for choice.

Actually, that might be defensible: at the choice stage, you evaluate whether the effort is "worth it." At the execution stage, you've already committed, so you press as hard as needed to survive -- effort cost is sunk. The only thing driving vigor is survival incentive (cd) and perceived efficacy (epsilon).

This is consistent with the current model's finding that ce is near zero for vigor: once committed, people don't modulate vigor based on effort cost.

### Verdict

**Psychologically interesting and practically clean, but undermines the "unified computation" story.** If ce doesn't enter vigor, the model is two computations sharing a survival function, not one computation with two outputs.

**Rating: 5/10.** Clean identification but narrative cost.

---

## Option 8: Per-Subject ce Through an Informative Prior from Behavior

### Specification

Use behavioral features to construct an informative prior for ce_i:

1. Compute each subject's empirical effort sensitivity from choice data: the degree to which their P(heavy) drops with distance/effort level.
2. Use this as a prior mean for log(ce_i):
```
log(ce_i) ~ Normal(f(empirical_effort_sensitivity_i), sigma_ce)
```

Then fit the joint model with per-subject ce_i that are constrained by this prior.

### Analysis

This is similar to the empirical Bayes idea in Option 5 but keeps the joint model. The prior prevents ce_i from collapsing to zero because the data-driven prior anchors it near reasonable values.

The risk is circularity: the prior is derived from the same choice data that the model is fitting. This can be mitigated by using odd-trial data for the prior and even-trial data for the likelihood (split-sample), or by using a leave-one-out approach.

### Verdict

**Technically feasible but methodologically awkward.** The circularity issue is real and would require split-sample validation. Not clean enough for a methods-heavy paper.

**Rating: 4/10.**

---

## Summary Table

| # | Approach | Separation? | Principled? | Risk | Rating |
|---|----------|------------|------------|------|--------|
| 1 | Learn from prereg (distance-dep S) | Yes | Partly | Contradicts mechanistic S | 6/10 |
| 2 | Separate penalty specs (choice vs vigor) | Yes | Yes | ce identification from choice alone | **8/10** |
| 3 | Multiplicative effort discount | Partly | No | Conceptually awkward | 5/10 |
| 4 | Reparameterize (k, beta) -> (ce, cd) | Partly | No | Mapping not clean | 3/10 |
| 5a | Two-stage sequential | Yes | Yes | Loses joint narrative | 7/10 |
| 5b | Empirical Bayes hybrid | Yes | Yes | Mild complexity | **8/10** |
| 6 | Distance-dependent S for choice | Yes | Partly | Three-way identification | 7/10 |
| 7 | ce in choice only, cd in vigor only | Yes | Partly | Breaks unified computation | 5/10 |
| 8 | Informative prior from behavior | Partly | No | Circularity | 4/10 |

---

## Top 2 Recommendations

### Recommendation 1: Option 2 -- Drop cd from Choice, Identify ce from Choice Alone

**Why:** When S_H = S_L (same survival for both options, which is the case in this task at full speed), any term proportional to cd cancels out of the choice comparison. This isn't a modeling choice -- it's a mathematical fact. The choice equation reduces to:

```
EU_H - EU_L = S * (R_H - R_L) - ce * (effort_H - effort_L)
            = S * 4 - ce * (0.81*D_H - 0.16)
```

ce is the ONLY subject-varying parameter affecting choice (since S is determined by T, epsilon, and p_esc). This is clean, principled, and directly parallels the prereg model where k was identified from effort variation in choice.

For vigor, cd enters through the full EVC computation:
```
u* = argmax [S(u) * R - (1-S(u)) * cd * (R+C) - ce * (u-req)^2 * D]
```

Both ce and cd affect vigor, but ce is pinned down by choice data, so cd absorbs what remains.

**Implementation:**
1. Choice likelihood: `P(heavy) = sigmoid((S*4 - ce_i*(0.81*D_H - 0.16)) / tau)`
2. ce_i is per-subject, hierarchical log-normal
3. Vigor likelihood: full EVC-LQR optimization with ce_i (shared from choice) and cd_i
4. cd_i is per-subject, hierarchical log-normal
5. epsilon_i per-subject as before

**Key change from current model:** The choice equation explicitly DOES NOT include cd*(R+C). This is justified because S_H = S_L means the cd terms cancel anyway. We're just making the cancellation explicit.

**What to watch for:** ce_i needs enough variation across conditions to be well-identified. The effort differential ranges from 0.65 (D=1) to 2.27 (D=3), giving a 3.5x range. With 45 trials per subject across 9 conditions, this should be adequate.

### Recommendation 2: Option 5b -- Empirical Bayes from Prereg Model

**Why:** We KNOW the prereg model separates k and beta. Use it as Stage 1 to get well-identified per-subject effort sensitivity (k_i) and threat bias (beta_i). Then use the posterior means (or full posteriors) as informative priors in the joint EVC model:

```
log(ce_i) ~ Normal(g(k_i), sigma_ce_tight)
log(cd_i) ~ Normal(h(beta_i), sigma_cd_tight)
```

where g() and h() are scaling functions that map the prereg parameters to the EVC parameter space. The tight sigma means ce and cd are anchored near their prereg-derived values but can be refined by the vigor data.

**Implementation:**
1. Fit prereg M5 to choice data (already done, results in `results/model_fits/`)
2. Extract k_i, beta_i posteriors
3. Define mapping: ce ~ k / (mean_req^2 * mean_D), cd ~ beta / mean_stakes
4. Use mapped values as prior means in the joint EVC model
5. Fit joint model with informative priors

**What to watch for:** The mapping between (k, beta) and (ce, cd) is approximate. If the priors are too tight, the model can't adjust; too loose, and cd absorbs ce again. Need to calibrate sigma_ce and sigma_cd through prior predictive checks.

**Advantage over two-stage:** Still a joint model -- choice and vigor share parameters. The prereg model just provides informative priors, not fixed values.

---

## Implementation Priority

**Try Option 2 first.** It requires the smallest change to the existing code (just modify the choice equation to drop cd), is the most principled, and directly addresses the root cause (cd cancels from choice when S is option-invariant, so just formalize that). If ce_i shows good individual variation and the vigor model still fits well with shared ce, this is the winner.

**Fall back to Option 5b** if Option 2 produces ce_i with too little variation or if the vigor model degrades. The empirical Bayes approach is more complex but guaranteed to give well-identified parameters.
