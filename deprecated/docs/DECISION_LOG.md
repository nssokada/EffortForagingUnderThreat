# Decision Log: From FET to Present

A chronological record of every modeling decision, what motivated it, what we found, and what it meant. This is the honest history of the project's analytical trajectory.

---

## Phase 0: The Starting Point (pre-March 25)

### The FET Model
The project began with a **Foraging under Effort and Threat (FET)** choice model with three per-subject parameters:
- **z** — hazard sensitivity: nonlinearity of distance→danger scaling
- **κ** — effort discounting: how strongly effort reduces reward value
- **β** — threat bias: residual threat aversion beyond survival-weighted EV

This was a choice-only model fitted via NumPyro HMC/NUTS. Fit quality: WAIC=12,063, AUC=0.912, Accuracy=82.5%. The model predicted choice well but said nothing about vigor (how hard people press).

**What was missing:** No vigor prediction. No joint model. No mechanistic link between choice and motor execution.

---

## Phase 1: First Principles Rethink (March 25)

### Decision: Build a joint choice-vigor model
**Why:** Noah wanted a fundamental rethink. The FET model fits choice but doesn't explain why people press harder under threat. A joint model would capture both behavioral channels through a shared computation.

### Attempt: Joint EVC model with shared survival function
**What we tried:** Joint choice + vigor model sharing S(T,D) or S(T,D,v). Inspired by Shenhav's Expected Value of Control framework.

**Finding:** Joint model didn't beat separate models (ΔELBO ≈ -1). The joint fitting introduced parameter shifts without improving prediction.

### Attempt: Normative/Pareto frontier analysis
**What we tried:** Computed empirical escape rates, fitted logistic escape model, built normative surface.

**Finding:** Escape rates are surprisingly flat across distance (~40% at D=5 vs ~37% at D=9 at half speed). The normative model didn't produce clean switching.

### Discovery: Simpson's Paradox in vigor
**Finding:** Raw vigor looks flat across threat (~0.66 at all T levels). But conditioning on chosen cookie type reveals strong threat modulation (heavy: d=0.42, light: d=0.49). High threat shifts choice toward light cookies, masking the within-cookie vigor increase.

**Impact:** This became a core finding. It demonstrates why a joint model of choice and vigor is necessary — analyzing either channel in isolation yields misleading conclusions.

### Discovery: Relative vigor is the right measure
**Finding:** vigor_norm/effort_demand (relative vigor) increases with threat: 1.26 → 1.38 → 1.43. Maps to speed tiers. Removes the choice confound.

### Session ended with:
Noah: "Current framing is not novel, interesting, or coherent. Needs fundamental rethink from first principles."

---

## Phase 2: The 24 Model Specifications (March 27-28)

### Decision: Systematic model search for identifiable parameters
**Why:** The FET model's 3 choice parameters couldn't connect to vigor. We needed parameters that could be identified from choice AND vigor data jointly.

### The ce identification problem
**The core issue:** When we put all parameters into both choice and vigor equations, ce (effort cost) was not recoverable (r=0.04). The reason: cd×(R+C) absorbed ce's signal because R+C differs by cookie type (10 vs 6), and cd's penalty term was collinear with the reward scaling.

### 24 specifications tested
Including: original ce×u²×D, alpha exponents, lambda in survival, binary effort, equal penalty, c_dist, dual-S, k+cd+eps, theta v1/v2, LQR variants, various rescalings.

**All failed** to simultaneously recover both per-subject parameters from choice data.

### Decision: Option 2 architecture — separate parameters by data stream
**The breakthrough insight:** Instead of making both parameters enter both equations, assign each parameter to the data stream that identifies it:
- **ce** identified from **choice only** (cd cancels from the H/L choice comparison because C=5 is the same for both cookies)
- **cd** identified from **vigor only** (probe trials provide unbiased vigor across all conditions)

**Why cd cancels from choice:** ΔEV_cd = (1-S)×cd×(R_H + C) - (1-S)×cd×(R_L + C) = (1-S)×cd×4. This scales with the same factor (4 = R_H - R_L) as the reward term, making cd and reward scaling collinear in the sigmoid.

### Decision: Population-level ε (effort efficacy)
**Why:** We tried per-subject ε. Recovery was r = -0.02 — completely unidentifiable. When ce handles choice and cd handles vigor, ε has no unique signal. Collapsed to population constant (0.098).

### Decision: Population-level γ (probability weighting)
**Finding:** γ = 0.209 — dramatic compression of threat probabilities. T=0.1 perceived as 0.62, T=0.5 as 0.87.
**Why population:** Per-subject γ would confound with ce at the 45-trial level.
**Impact:** Shifts optimal foraging surface — 4/9 conditions change from heavy→light optimal under γ=0.21.

### Result: EVC 2+2 model
- 2 per-subject params (ce, cd) + 2 key population params (γ, ε)
- Choice: `ΔEU = S×4 - ce×(0.81D - 0.16)`, S uses γ and ε
- Vigor: `EU(u) = S(u)×R - (1-S(u))×cd×(R+C) - ce_vigor×(u-req)²×D`
- BIC=17,768, choice r²=0.951, vigor r²=0.511
- Recovery: ce r=0.92, cd r=0.94
- MCMC validation: SVI-MCMC r>0.999

### Decision: LQR-inspired cost structure
**Why:** Needed a cost function that works for both choice (commitment cost req²×D) and vigor (deviation cost (u-req)²×D). Inspired by linear-quadratic regulator theory.
**Finding:** Empirically equivalent to standard u² cost (ΔBIC = -142). Kept for theoretical motivation.

### 6-model ablation comparison
- Removing threat: ΔBIC = +18,659
- Removing individual ce: ΔBIC = +10,634
- Separate models: ΔBIC = +10,393
- Removing γ: ΔBIC = +2,071
- Every component justified

### Downstream findings with 2+2 model
- ce→overcaution: r=0.924 (83% unique R²)
- cd→vigor gap: r=0.554
- Encounter reflex: trait-stable (r=0.78), threat-independent (F=0.04), cd-linked (r=0.50)
- Calibration→policy alignment: ΔR²=6.4%
- Discrepancy→residual overcaution: ΔR²=0.3% (small but significant)
- Discrepancy→clinical: 8/8 measures (r=0.18-0.34)
- ce, cd→clinical: all in ROPE (~77% null)

### Paper: Draft 004-010
Multiple drafts through simulated reviewer cycles. Title evolved from metacognition framing to "Integrating effort and threat in human foraging."

---

## Phase 3: The Separability Question (March 29, morning)

### The problem Noah identified
"cd and ce just seem so abstract — what are they actually doing?" and "the current model doesn't tell me if someone chose the light cookie because they dislike effort or because they're afraid of the threat."

**The fundamental limitation of the 2+2 model:** ce is the ONLY per-subject parameter in choice. ALL individual variation in choice goes through ce. The model cannot separate effort aversion from threat aversion.

### Decision: Try joint k+β model (Niv-inspired)
**Why:** Noah asked: "How does Niv handle this type of integration for which action? How fast? Why are we not doing something similar?"

**What we tried:** β enters BOTH choice AND vigor as a shared threat sensitivity parameter.
- Choice: SV = R×S - k×E - β×(1-S)
- Vigor: EU(u) = S(u)×R - (1-S(u))×β×(R+C) - ce_vig×(u-req)²×D

**Finding:** BIC=17,775, vigor r²=0.512 (good), but **choice r²=0.281** (terrible). λ=0.094 meant distance barely affected S, so β had no leverage to differentiate heavy vs light in choice.

**Verdict:** Failed. β can't do double duty.

### Decision: Three per-subject parameters (k, β, cd) with S in choice
**What we tried:** k (effort) + β (threat) in choice, cd in vigor. S still in the choice equation.

**Finding:** First attempt: threat_cost = (1-S)×(D_H-1). NaN explosion. Second: threat_cost = (1-S)×D_H. Converged but β recovery failed (r=0.214). Third: threat_cost = (1-S). Still r=0.214.

**Root cause:** S already captures most threat information through population-level γ and ε. β is trying to pick up residual individual threat sensitivity, but γ/ε absorb the signal first.

### Decision: Drop γ and ε from choice (Noah's insight)
**Noah:** "What if we drop gamma and epsilon?"

**What we tried:** Choice = `ΔEU = 4 - k×effort(D) - β×T`. No S, no γ, no ε in choice. Vigor keeps S(u) for physical escape mechanics.

**Finding:** EVERYTHING WORKS.
- Choice r² = 0.981 (up from 0.951)
- k × β: r = -0.006 (perfectly orthogonal)
- Recovery: k=0.850, β=0.841, cd=0.927
- Cross-recovery k→β: r=0.030
- Triple dissociation: k→overcaution r=0.933, β→threat sensitivity r=0.779, cd→vigor gap r=0.587

**Why it works:** Without γ/ε absorbing the threat signal, β becomes the sole carrier of threat information in choice. T and D are orthogonal by design, so k (from D gradient) and β (from T gradient) are structurally identifiable.

### Result: 3-param v2 model
Paper: Draft 011, "Three separable cost signals govern foraging under threat." Prereg updated with H1-H6.

---

## Phase 4: The Survival Landscape (March 29, afternoon)

### The problem
p_esc = 0.002 in the vigor model meant pressing harder has essentially no survival benefit. Is the vigor optimization story hollow?

### Decision: Empirically test the survival gradient
**What we found:**
- frac_full (fraction at full speed) predicts survival at r=0.282 (vs mean press rate r=0.08)
- Escape rates: <50% full speed → 11.3%, >95% → 51.8% (4.6× gradient)
- Press variability independently kills: OR=0.27 for heavy cookies controlling for mean rate
- cd→frac_full: r=0.710 (much stronger than cd→mean excess vigor r=0.58)

**Impact:** The survival gradient is real and substantial. The model's sigmoid S(u) with p_esc=0.002 completely misses it. frac_full is the mechanistically correct vigor variable.

### Decision: Build exposure-time survival function
**What we tried:** Gaussian CDF survival based on actual predator strike timing from the data.
- Empirical strike times: D=1: 2.46s (SD=0.57), D=2: 3.48s (SD=0.90), D=3: 4.77s (SD=1.12)
- P_escape = p_floor + (1-p_floor) × (1 - Φ((arrival - μ_strike - buffer) / σ_strike))
- Best fit: v_full=1.45, remaining_frac=0.90, buffer=0.60, p_floor=0.090
- RMSE=0.059 across 12 speed-tier × distance cells

### Decision: Test whether cd can enter choice through exposure-time survival
**Finding: NO.** The cd_term (what cd multiplies in the choice difference) varies with T but NOT with D:
- T=0.1: range 0.011 across D
- T=0.9: range 0.096 across D
cd's leverage in choice is collinear with the threat main effect. It can't be separately identified from a linear β×T term.

### Decision: β has genuine motor influence beyond cd
**Finding:** Partial r(β, frac_full | cd) = 0.357, unique R² = 5.9%. β carries motor information cd doesn't.

### Decision: Survival-based choice equation can't reproduce observed patterns
**Finding:** With realistic survival function, the penalty term dominates. At cd=5, T=0.5: P(heavy) ≈ 0. The model can't produce the observed 63% heavy rate at T=0.5/D=1. The exponential survival function makes the cost of choosing heavy too steep.

**Verdict:** Two-parameter survival-based choice failed. Three parameters with linear choice equation (3-param v2) remains necessary.

### Branch B: Three-param with upgraded vigor
**What we tried:** k+β+cd with choice unchanged, vigor using frac_full + Gaussian CDF survival + log-odds effort cost.

**Findings:**
- Adding req² to effort cost term: essential (creates heavy/light differential)
- Early stopping fixes loss divergence (best at step ~15K-19K)
- Choice r²=0.982, per-subject vigor r²=0.892, trial-level vigor r²=0.365
- Heavy-cookie trial r²=0.390 (comparable to original model's 0.424)
- Recovery: k=0.848, β=0.815, cd=0.920 (all pass)

---

## Phase 5: The 8-Model Comparison (March 30)

### Decision: Systematic comparison of choice model functional forms
**Why:** Is a survival function needed in choice, or is the linear additive form (3-param v2) sufficient?

### Phase A: Effort parameterization
**Finding:** req·T beats req²·D (ΔBIC=145). λ = cost per second of pressing is the cleaner parameterization.

### Phase B: 8 models fitted
| Rank | Model | BIC | ΔBIC | Accuracy | r² |
|------|-------|-----|------|----------|----|
| 1 | M3 (objective survival) | 18,502 | 0 | 0.782 | 0.898 |
| 2 | M4 (rate-of-return) | 18,881 | 379 | 0.779 | 0.932 |
| 3 | M2 (additive = our 3-param) | 20,274 | 1,772 | 0.825 | 0.989 |
| 4 | M1 (effort only) | 21,080 | 2,578 | 0.706 | 0.957 |
| 5-8 | M5-M8 (more params) | 22,927+ | 4,425+ | 0.819-0.832 | 0.958-0.992 |

**Key finding:** M3 wins on BIC by >1,772. The objective survival function exp(-p·T) captures the population-level choice surface better than the additive model. But M3 has lower individual-level accuracy (0.782 vs 0.825) because it has only 2 per-subject params vs 3.

### Decision: Test M5's α (probability distortion) for individual differences
**Finding:** α = 2.76 with SD = 0.007 — **zero individual variation**. 100% of participants have α > 2.7. α→clinical: completely null. α is a population constant, not an individual-difference parameter. Prior predictive check confirms data-driven (219× tighter than prior, 1.7th percentile).

**Impact:** Probability distortion is a fixed property of how humans process threat in this task, not something modulated by individual psychology.

### Decision: Try integrated model (M3 + individual κ + cd)
**Why:** Combine M3's survival function with individual differences: κ_i scales exp(-κ·p·T) per subject, λ_i handles effort, cd_i handles vigor.

**Finding:** Fit looks good (choice r²=0.955, per-subject vigor r²=0.950, BIC=15,394). BUT:
- λ × κ correlation: **r=0.388** (not orthogonal)
- Recovery: λ r=0.681, κ r=0.652 (both below 0.70)
- Cross-recovery: λ→κ=0.204, κ→λ=0.409 (trading off)

**Root cause:** Unlike the linear 3-param model where k and β enter as independent additive terms, κ enters a nonlinear exponential that changes the SHAPE of the choice surface, not just the intercept. This shape change can be partially mimicked by adjusting λ and τ, breaking the structural orthogonality.

**Verdict:** Failed. The integrated model can't be the final model.

### Sequential vigor prediction with M3
- Anticipatory vigor: ΔV → antic_vigor, β=0.009, t=3.31, p<.001 — significant but R²=0.001
- Delta-vigor: small threat modulation (d=0.19), detectable from raw epochs (not from 20Hz timeseries)
- ΔV→delta-vigor between-subject: r=0.028 (null)
- Abandonment rate: 5.7% (too rare for timing analysis)

### Clinical: invariant across all model architectures
- Model params → clinical: ALL NULL (λ, β, k, β_threat, cd, κ — nothing predicts symptoms)
- Discrepancy → clinical: 7/8 measures at p<.001 (r=0.20-0.33)
- This pattern holds identically for M3, M2, 3-param v2, and 2+2

### Parameter correspondence: M3 ↔ 3-param v2
- M3 λ ↔ v2 k: r=0.929 (same individual difference)
- M3 β (temperature) absorbs mix of v2 k and v2 β_threat (r=0.3-0.55)

---

## Where We Stand

### What survived every analysis
1. Effort and threat are massive, separable deterrents to high-effort choice
2. Simpson's paradox in vigor — joint modeling required
3. People press harder under threat (within cookie type)
4. frac_full predicts survival (r=0.28, 4.6× gradient)
5. Encounter reflex: trait-stable, threat-independent, cd-linked
6. Model params → clinical: null
7. Discrepancy → clinical: robust (7/8 measures)

### What died
- Joint k+β model (β can't serve both choice and vigor)
- Per-subject ε (not identifiable)
- Per-subject γ/α in choice (collapses to population constant)
- cd entering choice via survival function (collinear with T)
- Integrated model with per-subject κ in survival (λ-κ not separable)
- Vigor predicting clinical status
- Choice parameters predicting clinical status

### The two surviving model architectures
1. **M3 (population):** exp(-p·T) survival + λ effort. Wins on BIC. 2 per-subject params. Best population-level description.
2. **3-param v2 (individual):** k + β + cd, linear additive. Best individual prediction (r²=0.981). Triple dissociation. Orthogonal parameters.

### The open question
How to tell a single coherent story that honors both the population-level survival computation (M3) and the individual-level separability (3-param v2), connected by the affect channel (discrepancy→clinical) that both architectures produce.
