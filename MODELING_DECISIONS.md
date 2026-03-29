# Modeling Decisions Log

A detailed explanation of why each choice was made in the modeling process, from the original FET model through the 24 specifications to the final 3-param v2.

---

## Decision 1: Joint model, not separate models

**Choice:** One model predicting both choice AND vigor, not two independent models.

**Why:** The Simpson's paradox in vigor shows why this matters. Unconditionally, vigor appears flat or declining with threat. But that's because high threat shifts choice toward light cookies (lower req), hiding the within-cookie vigor increase. A joint model conditions vigor predictions on the choice, resolving the paradox. Additionally, a joint model allows shared parameters (like threat perception) to inform both behavioral channels simultaneously, which is more statistically efficient.

**Alternatives considered:** Separate logistic regression for choice + separate linear model for vigor. This worked (model M3 in comparisons) but the separate approach had BIC +10,393 worse than the joint model, because it can't share threat information across channels.

---

## Decision 2: Per-subject ce and cd, not all population-level

**Choice:** Two (later three) parameters that vary across individuals, with the rest fixed at population level.

**Why:** Individual differences in effort tolerance and threat sensitivity are the primary scientific question. A model with only population parameters (M4) achieves lower BIC but predicts individual choice at r² = 0.001 — it can't tell participants apart. The whole point is to explain WHY people differ. We chose the minimum number of per-subject parameters that captures meaningful individual variation in both choice and vigor.

**Alternatives considered:**
- All population-level: Good for group means, useless for individual differences (M4)
- Per-subject ε (effort efficacy): Tried, recovery r = -0.02. No unique signal once ce and cd handle their respective channels.
- Per-subject γ (probability weighting): Would capture per-subject threat curvature but confounds with ce at the 45-trial level.
- Per-subject τ (temperature): Would capture response noise but confounds with ce (a noisier person looks like a less extreme ce person).

---

## Decision 3: Log-normal priors with non-centered parameterization

**Choice:** All per-subject parameters are log-normal (constrained positive) with non-centered parameterization.

**Why:**
- **Log-normal:** ce, cd, k, β must be positive (negative effort cost or negative threat aversion is meaningless). Log-normal naturally enforces this and produces the right kind of skewed individual-difference distribution.
- **Non-centered:** Instead of sampling ce_i ~ LogNormal(μ, σ), we sample z_i ~ Normal(0,1) and compute ce_i = exp(μ + σ × z_i). This eliminates the "funnel problem" in hierarchical models where the posterior geometry is difficult for gradient-based methods when σ is small.
- **Log space for analysis:** We report correlations and regressions in log space (log(ce), log(k), etc.) because this is the native parameterization of the hierarchical model — linear changes in log space correspond to multiplicative changes in the parameter.

---

## Decision 4: Why cd cancels from the choice equation (2+2 model)

**Choice:** In the 2+2 model, cd is excluded from choice. Only ce enters.

**Why:** The full expected value of choosing cookie j under the model is:

```
EV_j = S × R_j - (1-S) × cd × (R_j + C)
```

The choice difference (heavy minus light) is:

```
ΔEV = S × (R_H - R_L) - (1-S) × cd × (R_H - R_L)
     = (R_H - R_L) × [S - (1-S) × cd]
     = 4 × [S - (1-S) × cd]
```

The cd term `(1-S) × cd × 4` scales with the same constant (4 = R_H - R_L) as the reward term. In the sigmoid link, this means cd and the overall scaling of ΔEV are **collinear** — changing cd produces the exact same effect as changing the effective reward difference. The model cannot separate "high cd" from "low reward sensitivity."

**This is a mathematical fact about the task design**, not an arbitrary restriction. It happens because both cookies share the same capture penalty C = 5. If C differed by cookie type, cd would be identifiable from choice.

**Consequence:** ce absorbs ALL per-subject choice variance in the 2+2 model. If someone avoids heavy because they're threat-averse (not effort-averse), ce still goes up. This is the core limitation that motivated the 3-param model.

---

## Decision 5: The LQR-inspired cost structure

**Choice:** Effort enters choice as req² × D (commitment cost) and vigor as (u - req)² × D (deviation cost).

**Why:** We needed a cost function that could apply to both choice and vigor using the same effort representation. Standard effort discounting (e.g., k × E) treats effort as a scalar cost on choice but says nothing about vigor — it doesn't tell you how hard to press.

The LQR analogy provides a principled split:
- **Choice stage:** You commit to a trajectory (choose heavy at D=3). The commitment cost is how demanding that trajectory is: req² × D. Higher required rate and longer distance = higher cost.
- **Vigor stage:** You execute the committed trajectory. The deviation cost is how far you press above the minimum: (u - req)² × D. Pressing above req costs effort proportional to the squared excess.

**Alternatives considered:**
- Standard u² cost (M6): Empirically equivalent (ΔBIC = -142, negligible). We kept LQR for theoretical motivation, not empirical superiority.
- Linear effort cost (k × D): Doesn't produce the right choice patterns — it predicts that all three effort levels at the same distance have the same cost, which contradicts data.
- Exponential discounting: Harder to connect to vigor.

**Honest caveat:** The LQR label was flagged by a simulated reviewer as overstatement. We call it "LQR-inspired" — it's an analogy, not a formal implementation.

---

## Decision 6: Probability weighting γ (2+2 model)

**Choice:** Transform threat probability as T^γ before computing S, with γ estimated from data.

**Why:** People don't treat stated probabilities at face value. Prospect theory (Kahneman & Tversky) established that people overweight small probabilities and underweight large ones. In our task, γ = 0.209 means dramatic compression: T=0.1 is perceived as T^0.209 = 0.618. This is consistent with the hypothesis that embodied threat engages defensive circuits more powerfully than abstract monetary gambles.

**The interpretive payoff:** γ shifts the optimal foraging surface. Under γ = 1 (no distortion), 5/9 conditions favor heavy. Under γ = 0.21, only 1/9 does. So ~20% of apparent "overcaution" is actually rational under the organism's subjective threat model. This is a substantive theoretical contribution.

**Why it was dropped in the 3-param model:** γ and ε confounded β identification. When threat enters choice through S = f(T^γ, ε), the threat signal is shared between population parameters (γ, ε) and the per-subject parameter (β). There's not enough variance left for β after γ absorbs the nonlinear threat transformation. Dropping γ and ε makes β the sole threat carrier, enabling identification.

**What we lost:** The probability weighting story. The 3-param model takes T at face value. If people really do compress threat probabilities (which they almost certainly do), that compression is now absorbed into β and τ rather than modeled explicitly.

---

## Decision 7: ε (effort efficacy) as population-level

**Choice:** ε is a single population-level value, not per-subject.

**Why:** We tried per-subject ε. Recovery was r = -0.02 — completely unidentifiable. The reason: once ce handles choice variation and cd handles vigor variation, ε has no unique behavioral signature. It modulates S, but S's contribution to choice is already explained by the global S×4 term (which γ and τ absorb), and S's contribution to vigor is already absorbed by cd. ε ended up at 0.098 — meaning pressing effort has very little impact on survival in the model.

**Interpretation:** ε = 0.098 × p_esc = 0.018 → effective escape benefit of effort = 0.18%. Practically zero. This is either a genuine feature of the task (pressing faster barely helps against a predator 4× your speed) or an artifact of the model struggling to separate effort's survival benefit from the other parameters.

---

## Decision 8: Cookie-type centering of vigor

**Choice:** Subtract the mean excess effort for heavy (or light) cookies from each trial's excess effort, using means computed from choice trials only.

**Why:** Heavy cookies require press rate 0.9 and light cookies require 0.4. The raw excess (actual rate - required rate) is systematically higher for light cookies because 0.4 is easier to exceed. If we model raw excess, the model would have to predict this cookie-type offset as well as the threat/distance modulation — conflating a mechanical confound with the psychological signal.

Cookie-type centering removes this confound: the model only needs to predict why someone presses MORE (or LESS) than the average for their cookie type.

**Why compute means from choice trials only:** On probe trials, cookie assignment is random (forced choice). On choice trials, who chooses heavy vs light is endogenous — high-ce people disproportionately choose light. If we used all trials to compute the centering means, the probe trial distribution would contaminate the estimate. Using choice-trial means preserves the natural selection pattern.

---

## Decision 9: Using probe trials for vigor (all 81 trials)

**Choice:** Vigor data includes all 81 trials (45 choice + 36 probe), not just the 45 choice trials.

**Why:** Probe trials provide vigor data at conditions where choice data is sparse. At T=0.9, D=3, almost nobody chooses heavy (~8%). So choice trials give almost no heavy-cookie vigor data at high threat. But probe trials randomly assign cookies, so they provide unbiased vigor at every T×D×cookie combination. This is critical for identifying cd — without probes, cd would be estimated almost entirely from light-cookie vigor, missing the threat gradient on heavy cookies.

**For choice:** Probe trials contribute P(heavy) = 0.5 to the choice likelihood (because both options are identical), adding no information to ce/k/β estimation.

---

## Decision 10: Softmax grid search for optimal vigor

**Choice:** Evaluate EU(u) on a 30-point grid from u=0.1 to u=1.5, then take a softmax-weighted average with temperature 10.0.

**Why:** The vigor EU function is non-convex (sigmoid × reward - sigmoid × penalty - quadratic cost) and has no closed-form optimum. A grid search is the simplest reliable approach. The softmax approximation (rather than hard argmax) is differentiable, which is essential for gradient-based SVI optimization. Temperature = 10.0 makes the softmax nearly a hard argmax while maintaining smooth gradients.

**Alternatives considered:**
- Analytical solution: Not possible due to the sigmoid in S(u).
- Gradient-based optimization of u per trial: Possible but creates a nested optimization problem that's slow and fragile inside SVI.
- Finer grid (50 or 100 points): Tested, no meaningful change in results. 30 points is sufficient.

---

## Decision 11: Adding β as a separate choice parameter (3-param model)

**Choice:** Replace the single ce with two choice parameters: k (effort cost) and β (threat aversion).

**Why:** The fundamental question Noah raised: "The current model doesn't tell me if someone chose the light cookie because they dislike effort or because they're afraid of the threat." The task CAN dissociate these because T and D are crossed. Someone who avoids heavy at ALL threat levels is effort-averse (high k). Someone who avoids heavy specifically at HIGH threat is threat-averse (high β). But the 2+2 model lumps both into ce.

**Why it required dropping γ and ε:** With S in the choice equation, β is confounded with the population-level parameters that define S. The threat signal is "used up" by S before β can capture individual differences. The fix: remove S from choice entirely, letting β × T be the sole threat signal. This is Noah's key insight.

---

## Decision 12: threat_cost = T, not (1-S) or (1-S)×D

**Choice:** In the 3-param model, the threat term in choice is simply β × T.

**Three versions were tried:**
1. `β × (1-S) × (D_H - 1)`: NaN explosion because D_H=1 gives zero.
2. `β × (1-S) × D_H`: Converged but β recovery failed (r=0.214). The (1-S) term was confounded with γ/ε.
3. `β × (1-S)` (no D scaling): Better but still confounded (r=0.214).
4. **`β × T`** (no S at all): Recovery r=0.841. Clean.

The problem with (1-S) in any form: S is defined by population parameters (γ, ε, p_esc), so (1-S) is a population-level function of T. Adding a per-subject multiplier β on top of a population-level function leaves β underdetermined — the model can absorb β variation into γ.

Using T directly makes β the sole source of individual threat variation in choice. The orthogonality of T and D in the 3×3 design then guarantees that k (which scales with D) and β (which scales with T) are structurally identifiable.

---

## Decision 13: ClippedAdam optimizer for the 3-param model

**Choice:** Use ClippedAdam (gradient clipping at norm 10.0) instead of standard Adam.

**Why:** The 3-param model with standard Adam produced NaN losses from step 1. The trace showed no NaN at initialization — the issue was gradient explosion during the first few SVI steps, likely from the interaction of three per-subject parameter groups in the ELBO. ClippedAdam caps gradient norms, preventing the explosion while still converging to a good optimum.

**Learning rate:** Reduced from 0.002 (used in 2+2) to 0.001 for the 3-param model for additional stability.

---

## Decision 14: No survival function in choice (3-param v2)

**Choice:** The choice equation is `ΔEU = 4 - k × effort(D) - β × T`. No S. The reward advantage is a flat 4 regardless of threat.

**Why this works:** The survival-weighted reward `S × 4` in the 2+2 model captures how threat reduces the expected value of the heavy option. In the 3-param model, `β × T` captures the same behavioral pattern — avoidance of heavy under high threat — but attributes it to per-subject threat aversion rather than a population-level survival computation. The T×D interaction (threat matters more at far distances) emerges from the sigmoid nonlinearity: when ΔEU is near zero, both k and β have strong marginal effects; when ΔEU is already very negative, additional threat or distance makes little difference.

**What this means theoretically:** The 3-param model makes a different theoretical claim than the 2+2 model. The 2+2 model says "people compute survival probability and weight rewards by it." The 3-param model says "people separately penalize effort and threat as linear costs." The 3-param model fits better (r² = 0.981 vs 0.951) and enables the triple dissociation, but it gives up the survival-function framework and the probability weighting story.

---

## Decision 15: Keeping S(u) in the vigor equation

**Choice:** Vigor still uses `S(u) = (1-T) + T × p_esc × sigmoid((u-req)/σ)` even though choice drops S.

**Why:** In vigor, the speed-escape relationship is a **physical constraint**. Pressing faster genuinely moves your character faster, which genuinely helps outrun the predator. This is not a subjective weighting — it's a real mechanic of the task engine. The survival function in vigor encodes this mechanic: pressing at rate u produces survival probability S(u), and cd scales how much the organism cares about improving that probability.

Dropping S from vigor would mean the model can't explain WHY people press harder under threat — there would be no mechanism connecting press rate to survival.

**Simplified from the 2+2 version:** The vigor S(u) in the 3-param model drops γ (no probability weighting: T enters linearly) and ε (folded into p_esc). So `S(u) = (1-T) + T × p_esc × sigmoid(...)` instead of `S(u) = (1-T^γ) + ε × T^γ × p_esc × sigmoid(...)`.
