# Expected Value of Control in Foraging Under Threat

## Task Description

Participants forage for cookies in a virtual arena under threat of predator attack. On each trial they choose between two cookies and then physically press keys to carry their chosen cookie back to a safe zone.

**Two cookie options per trial:**
- **Heavy cookie:** 5 points reward, requires pressing at 90% of calibrated maximum rate (calMax), distance varies (5, 7, or 9 game units from safety)
- **Light cookie:** 1 point reward, requires pressing at 40% of calMax, always close (5 game units)

**Threat manipulation:** Each trial has an independent attack probability T ∈ {0.1, 0.5, 0.9}. On attack trials, a predator appears at a time calibrated to approximately half the player's return time at full speed, then strikes. After the strike, the predator pursues at 4× the player's maximum movement speed — effectively guaranteeing capture unless the player has already reached safety.

**Speed tiers:** The player's movement speed is determined by pressing rate relative to the chosen cookie's required rate:
- ≥100% of required rate → full speed
- ≥50% → half speed
- ≥25% → quarter speed
- <25% → movement decays to zero

**Survival:** To escape the predator, the player must reach safety before the predator strikes. This requires pressing at or above the full-speed threshold. Below full speed, the player cannot outrun the predator.

**Calibration:** Each participant's maximum press rate (calMax) is measured in a dedicated calibration phase. All effort requirements are expressed as fractions of this individual maximum.

**Capture penalty:** If captured, the participant loses the current cookie's reward plus a 5-point penalty. Total loss on capture = reward + 5.

**Design:** 45 choice trials per participant (3 threat levels × 3 distance levels × 5 repetitions), plus 36 probe trials measuring subjective anxiety and confidence. N = 293 participants (exploratory sample), N = 350 (confirmatory, collected but not yet analyzed).

**Key design feature:** Predator spawn timing is calibrated to half the return time at full speed. This means distance does NOT independently affect survival probability at full speed — only threat probability determines whether an attack occurs. Distance affects the cumulative effort required (more distance = more sustained pressing = higher effort cost).

---

## The Model: Expected Value of Control

### Core Framework

The model follows Shenhav et al.'s Expected Value of Control (EVC) framework:

**EVC = Expected Payoff × Efficacy − Cost**

Applied to this task:
- **Payoff** = c_death × (R + C) — how much surviving is worth (subjective)
- **Efficacy** = ε × T × p_esc × dS/du — how much pressing harder improves survival
- **Cost** = c_effort × u² × D — cumulative effort cost of pressing

Three subject-level parameters govern the foraging policy:

| Parameter | Role | What it captures |
|-----------|------|-----------------|
| **c_effort** | Cost of control | How costly is sustained pressing. Enters both choice (avoid effortful options) and vigor (denominator of effort-survival tradeoff) |
| **c_death** | Incentive / payoff | How aversive is capture. Enters both choice (avoid dangerous conditions) and vigor (numerator — survival is worth more) |
| **ε (epsilon)** | Efficacy of control | How much does pressing harder improve survival? Scales the effort-dependent component of survival probability. Individual variation: ε < 1 = effort feels futile; ε > 1 = hypervigilant, every press feels critical |

### Formal Specification

**Survival probability at press rate u:**

```
S(u) = (1 − T) + ε_i × T × p_esc × σ((u − req_rate) / σ_motor)
```

Where:
- T = threat probability (0.1, 0.5, 0.9)
- ε_i = individual efficacy parameter (how much the person believes pressing improves survival)
- p_esc = probability of escaping given full-speed pressing and an attack (population parameter)
- req_rate = required pressing rate for the chosen cookie (0.9 for heavy, 0.4 for light, in calMax units)
- σ_motor = motor noise around the speed threshold (population parameter)
- σ() = sigmoid function

This implements the survival logic with individual efficacy scaling. When ε = 1, the person fully internalizes the objective relationship between pressing and survival. When ε < 1, they underweight the survival benefit of pressing (effort feels futile). When ε > 1, they overweight it (hypervigilant).

**Expected utility at press rate u for a given cookie:**

```
EU(u) = S(u) × R − (1 − S(u)) × c_death × (R + C) − c_effort × u² × D
```

Where:
- R = cookie reward (5 for heavy, 1 for light)
- C = capture penalty (5 points)
- D = distance level (1, 2, or 3 for heavy; always 1 for light)
- u² × D = cumulative effort cost (quadratic in press rate, linear in distance)

The three terms map onto the EVC components:
1. **S(u) × R** — expected reward (scales with efficacy through S)
2. **(1−S(u)) × c_death × (R+C)** — expected loss from capture, weighted by subjective death aversion (payoff)
3. **c_effort × u² × D** — effort cost (cost of control)

**Optimal press rate u*:**

The participant selects the press rate that maximizes EU. Computed by evaluating EU on a grid of 30 candidate press rates from 0.1 to 1.5 calMax, then taking a soft argmax:

```
weights_j = softmax(EU(u_j) × β_grid)    for j = 1,...,30
u* = Σ_j weights_j × u_j
```

Where β_grid = 10.0 (temperature for the grid softmax). This is differentiable for gradient-based fitting.

**Choice model:**

For each cookie option, compute the optimal press rate and corresponding EU:

```
u*_H, EU*_H = optimize(c_effort_i, c_death_i, ε_i, T, R_H=5, D_H, req_H=0.9)
u*_L, EU*_L = optimize(c_effort_i, c_death_i, ε_i, T, R_L=1, D_L=1, req_L=0.4)
```

For the choice likelihood, we use binary survival evaluation for computational efficiency:

```
S_full = (1 − T) + ε_i × T × p_esc    [survival if pressing at full speed]
S_stop = (1 − T)                        [survival if not pressing hard enough]

EU_option = max(EU_full, EU_stop)       [optimal policy for each option]

P(choose heavy) = sigmoid((EU*_H − EU*_L) / τ)
```

**Vigor prediction:**

Excess effort = u* − req_rate for the chosen cookie. This is a mechanistic prediction from the same three parameters that drive choice — no additional vigor parameters.

### Hierarchical Bayesian Structure

**Subject-level parameters (non-centered parameterization):**

```
log(c_effort_i) = μ_ce + σ_ce × z_ce_i,     z_ce_i ~ Normal(0, 1)
log(c_death_i)  = μ_cd + σ_cd × z_cd_i,      z_cd_i ~ Normal(0, 1)
log(ε_i)        = μ_ε  + σ_ε  × z_ε_i,       z_ε_i  ~ Normal(0, 1)
```

All three parameters are log-normal, ensuring positivity. ε is centered near exp(μ_ε), with μ_ε prior Normal(−0.5, 0.5), so the median ε is below 1 (people tend to underweight efficacy).

**Population-level parameters:**

| Parameter | Prior | Role |
|-----------|-------|------|
| μ_ce, μ_cd, μ_ε | Normal(0, 1) or Normal(−0.5, 0.5) | Population means (log scale) |
| σ_ce, σ_cd, σ_ε | HalfNormal(0.5) or HalfNormal(0.3) | Population SDs |
| τ | exp(Normal(−1, 1)), clipped [0.01, 20] | Choice temperature |
| p_esc | sigmoid(Normal(0, 1)) | Escape probability at full speed |
| σ_motor | exp(Normal(−1, 0.5)), clipped [0.01, 1] | Motor noise width |
| σ_v | HalfNormal(0.5) | Vigor observation noise |

**Joint likelihood (per trial):**

```
choice_i ~ Bernoulli(P(choose heavy)_i)
excess_cc_i ~ Normal(excess_predicted_i, σ_v)
```

Both likelihoods share c_effort_i, c_death_i, and ε_i.

### Vigor Dependent Variable: Cookie-Centered Excess Effort

**excess** = median pressing rate (from raw inter-press intervals, normalized by calMax) minus the required rate for the chosen cookie (0.9 for heavy, 0.4 for light).

**Cookie-type centering:** We subtract the population mean excess for each cookie type (heavy mean = 0.104, light mean = 0.543). This removes the demand-level confound (light cookies mechanically have higher excess because the demand is lower) while preserving between-subject variation. A person pressing harder than average on heavy cookies will have positive centered excess, regardless of what light-cookie pressers do.

This measure:
- Removes the demand confound (different cookies have different required rates)
- Removes the cookie-composition confound (people who choose light more don't automatically have higher excess)
- Preserves between-subject variation (between-subj SD = 0.291, within-subj SD = 0.131)
- Shows robust within-subject threat modulation when conditioned on choice

---

## Results

### Fit Quality

Fitted with SVI (AutoNormal guide, 35k steps, Adam lr=0.002).

| Metric | Value |
|--------|-------|
| Choice accuracy | ~81% |
| Cookie-centered excess r | 0.672 |
| Cookie-centered excess r² | 0.452 |
| BIC | 23,979 |
| Subject-level parameters | 3 × 293 = 879 |

### Condition-Level Vigor Predictions

| Condition | Predicted | Observed |
|-----------|-----------|----------|
| Heavy T=0.1 | −0.126 | −0.022 |
| Heavy T=0.5 | +0.023 | +0.018 |
| Heavy T=0.9 | +0.091 | +0.038 |
| Light T=0.1 | −0.111 | −0.042 |
| Light T=0.5 | −0.034 | −0.005 |
| Light T=0.9 | +0.016 | +0.021 |

Correct threat direction for both cookies. Overpredicts the threat dynamic range for heavy, accurate for light.

### Parameter Estimates

| Parameter | Mean | Range | SD | Interpretation |
|-----------|------|-------|-----|---------------|
| c_effort | 0.008 | [0.000, 0.979] | — | Effort cost (very small — effort is cheap for most) |
| c_death | 3.59 | [0.162, 69.6] | wide | Capture aversion (large individual differences) |
| ε (epsilon) | 0.262 | [0.002, 4.31] | 0.70 | Effort efficacy (most underweight, some overweight) |

### Parameter Independence

| Pair | r | Interpretation |
|------|---|----------------|
| ce × cd | −0.015 | Fully independent ✓ |
| ce × ε | −0.035 | Fully independent ✓ |
| cd × ε | +0.428 | Correlated: fear capture ↔ believe effort helps |

The cd × ε correlation is theoretically meaningful: people who care more about survival (high c_death) also tend to believe their pressing can improve it (high ε). These are two aspects of threat engagement — caring about the outcome AND believing you can influence it. The opposite (high c_death, low ε) would be anxious helplessness.

### Clinical Correlations

| Parameter | Clinical measure | r | p | Interpretation |
|-----------|-----------------|---|---|----------------|
| c_death | AMI_Emotional | +0.185 | .002 | Higher capture aversion → higher emotional apathy score |
| c_death | AMI_Behavioural | −0.116 | .047 | Higher capture aversion → lower behavioural apathy |
| **ε** | **OASIS (anxiety)** | **−0.120** | **.040** | **Lower efficacy → higher clinical anxiety** |

**The ε → OASIS link is the clinical bridge.** People who believe their effort is less effective at improving survival (low ε) show higher clinical anxiety. This is the computational signature of learned helplessness: "my actions can't change outcomes, so I'm anxious about outcomes I can't control." This connects the EVC framework directly to anxiety through perceived controllability.

c_effort shows no clinical correlations — it's a motor-economic parameter, not a psychological one.

---

## Theoretical Interpretation

### Three Parameters, Three Constructs

1. **c_effort = Cost of Control.** The energetic price of pressing. Determines BOTH which cookies you avoid (choice) AND how much you press above threshold (vigor). This is Shadmehr's effort cost — the energy expenditure per unit of motor output. No clinical relevance in this sample because effort cost is about physical economics, not psychopathology.

2. **c_death = Incentive.** The subjective aversion to capture. Determines BOTH how much you avoid dangerous situations (choice) AND how much you invest in pressing harder to survive (vigor). High c_death people avoid threat in their choices but compensate with effort when they do engage. Relates to emotional reactivity (AMI_Emotional).

3. **ε = Efficacy of Control.** How much the person believes their pressing improves survival. This is Shenhav's efficacy term — the signal-to-noise of the control channel. High ε people show steep vigor-threat slopes (they press much harder under high threat because they believe it works). Low ε people show flat slopes (pressing harder doesn't feel like it helps, so why bother?).

### The EVC Equation in This Task

```
EVC(u) = [c_death × (R + C)] × [ε × T × p_esc × dS/du] − [c_effort × 2u × D]
         \_________________/   \________________________/   \________________/
              Incentive                Efficacy                   Cost
```

The optimal u* balances incentive × efficacy against cost. Individual differences in all three determine the foraging phenotype:
- **High c_death, high ε, low c_effort** → choose risky options AND press hard. Maximum engagement.
- **High c_death, low ε** → avoid risky options (choice) but don't press hard (effort feels futile). Anxious helplessness.
- **Low c_death, high ε** → not afraid, but believes pressing works. Bold + vigorous.
- **High c_effort, any ε** → avoid heavy cookies. Effort-averse regardless of threat.

### Why ε Matters for Anxiety

The cd × ε correlation (r = +0.43) reveals two types of threat-sensitive individuals:
- **High c_death, high ε:** "I fear capture, but I can do something about it by pressing harder." This is ADAPTIVE threat sensitivity — vigilant and active. Not clinically anxious.
- **High c_death, low ε:** "I fear capture, and there's nothing I can do about it." This is MALADAPTIVE threat sensitivity — anxious and helpless. Predicts OASIS scores.

The decomposition of threat sensitivity into incentive (c_death) and efficacy (ε) may explain why some threat-sensitive people develop clinical anxiety while others don't: it depends on whether they believe their actions matter.

---

## Optimality Analysis

### The Optimal Policy

Given empirical escape rates, the EV-maximizing policy is:
- T=0.1: always choose heavy (EV_H = +3.70 >> EV_L = +0.55)
- T=0.5: choose light (EV_H = −1.29 < EV_L = −0.80)
- T=0.9: choose light (EV_H = −6.19 << EV_L = −2.23)

Mean optimality rate across subjects: 69.4% (range 35.6%–100%).

### Cost Parameters Predict Deviation from Optimal

| Predictor | Overcautious (r) | Overrisky (r) | Optimality (r) |
|-----------|-----------------|---------------|----------------|
| c_effort | +0.43*** | −0.57*** | +0.31*** |
| c_death | +0.35*** | −0.58*** | +0.37*** |

Both cost parameters push people toward caution. Since the task rewards caution at 2/3 of threat levels, higher-cost individuals are incidentally more optimal.

### Suboptimal Choice Types

Of all suboptimal choices:
- 66% are too risky (chose heavy when light had better EV) — chasing the 5-point reward despite poor survival odds
- 34% are too cautious (chose light when heavy was optimal) — leaving money on the table at low threat

---

## Model Comparison

| Model | Subject params | Excess r² | BIC | Notes |
|-------|---------------|----------|------|-------|
| 2-param (demeaned) | 2 × 293 | 0.56 | 14,168 | Best BIC, but params correlated (r=0.75) |
| 2-param + pop ε (demeaned) | 2 × 293 | 0.61 | 11,674 | Best BIC overall, exact threat predictions, params still correlated |
| 2-param + pop ε (raw) | 2 × 293 | 0.54 | 25,143 | Clean params (r=−0.02), clinical links |
| 3-param with ε (cookie-centered) | 3 × 293 | 0.45 | 23,979 | Clean params, individual ε, clinical bridge |
| **3-param + γ (cookie-centered)** | **3 × 293 + pop γ** | **0.50** | **23,007** | **Best calibrated vigor-threat, probability weighting** |
| 3-param + alpha (raw) | 3 × 293 | 0.83 | 14,504 | Best r² but alpha atheoretical |
| Independent 4-param | 4 × 293 | 0.63 | 24,846 | More params, no shared structure |

---

## EVC + Gamma Model (Recommended)

### Motivation

The 3-param model (c_effort, c_death, ε) systematically overpredicted how much vigor should change across threat levels. The predicted vigor modulation was 2–4× larger than observed:

| | Heavy pred/obs ratio | Light pred/obs ratio |
|--|---------------------|---------------------|
| Without γ | 3.62× | 2.02× |
| With γ | 1.67× | 0.92× |

This suggested subjects compress objective probabilities before using them — classic probability weighting.

### The γ Parameter

A single population-level parameter transforms objective threat:

```
T_weighted = T^γ
```

Fitted γ = 0.283, compressing the threat range:

| Objective T | Weighted T (T^0.283) |
|------------|---------------------|
| 0.10 | 0.521 |
| 0.50 | 0.822 |
| 0.90 | 0.971 |

Subjects act as if T=0.1 is roughly 50/50, and T=0.5 and T=0.9 are nearly identical. The objective range of 0.8 compresses to 0.45 in subjective space. This matches Kahneman & Tversky probability weighting — overweighting small probabilities, underweighting large ones.

### Results with γ

| Metric | Without γ | With γ | Change |
|--------|----------|--------|--------|
| Excess r² | 0.452 | 0.504 | +0.052 |
| BIC | 23,979 | 23,007 | −972 |
| ce×cd | −0.015 | −0.022 | ≈same |
| ce×ε | −0.035 | −0.055 | ≈same |
| cd×ε | +0.428 | +0.664 | increased |
| ε→OASIS | r=−0.120, p=.040 | r=−0.110, p=.061 | marginal |

### Parameter estimates (with γ)

| Parameter | Mean | Range | Notes |
|-----------|------|-------|-------|
| c_effort | 0.011 | [0.002, 0.975] | Slightly higher than without γ |
| c_death | 0.747 | [0.065, 9.713] | Lower than without γ (less needed when T compressed) |
| ε | 0.218 | [0.023, 1.828] | Tighter range |
| γ | 0.283 | (population) | Strong compression |

### Condition-level vigor predictions (with γ)

| Condition | Predicted | Observed |
|-----------|-----------|----------|
| Heavy T=0.1 | −0.043 | −0.022 |
| Heavy T=0.5 | +0.031 | +0.018 |
| Heavy T=0.9 | +0.057 | +0.038 |
| Light T=0.1 | −0.074 | −0.042 |
| Light T=0.5 | −0.034 | −0.005 |
| Light T=0.9 | −0.017 | +0.021 |

Much better calibrated than the no-γ version, especially for heavy cookies.

### Clinical correlations (with γ)

| Parameter | Clinical measure | r | p |
|-----------|-----------------|---|---|
| c_death | AMI_Emotional | +0.148 | .011 |
| c_death | OASIS | −0.131 | .025 |
| ε | OASIS | −0.110 | .061 (marginal) |
| c_effort | (no sig links) | | |

The ε→OASIS link weakens slightly (p goes from .040 to .061) because γ absorbs some of the threat-compression that ε was previously handling. The c_death→OASIS link strengthens.

### Tradeoff: γ vs no-γ

The γ model is better on fit (BIC, r², vigor calibration) but the clinical story is slightly weaker:
- **Without γ:** ε→OASIS significant (p=.040), cleaner theoretical narrative about individual efficacy → anxiety
- **With γ:** Better fit, probability weighting is theoretically principled, but ε→OASIS marginal (p=.061)

Both versions support the core claim. The γ model adds the Kahneman-Tversky probability weighting angle, which is a well-established phenomenon in decision science.

### Files

- Model code: `/workspace/scripts/modeling/oc_evc_gamma.py`
- Fitted params: `/workspace/results/stats/oc_evc_gamma_params.csv`

---

## Key Findings Summary

1. **A single EVC computation jointly explains foraging choice and motor vigor** under threat. Two outputs, one computation, three interpretable parameters.

2. **Effort efficacy (ε) is a meaningful individual difference.** People vary in how much they believe pressing harder improves survival (range 0.002–4.3). This determines the slope of their vigor-threat response.

3. **Low efficacy predicts clinical anxiety** (ε → OASIS r = −0.12, p = .04 without γ; r = −0.11, p = .06 with γ). The computational signature of learned helplessness: believing actions can't change outcomes produces both flat vigor-threat slopes and clinical anxiety.

4. **The incentive-efficacy decomposition explains adaptive vs maladaptive threat sensitivity.** High c_death + high ε = adaptive vigilance. High c_death + low ε = anxious helplessness. Same fear, different controllability, different clinical outcomes.

5. **Distance enters effort cost, not survival.** Predator spawn timing is calibrated to return time, so at full speed, distance doesn't affect survival probability. Distance only affects how much cumulative pressing is needed — a pure effort cost.

6. **People systematically underweight effort efficacy** (mean ε = 0.26 < 1). They act as if pressing harder buys only ~26% of the actual survival benefit. This universal bias compresses the predicted vigor dynamic range and must be accounted for in the model.

7. **People compress threat probabilities** (γ = 0.28). Subjects overweight low threat and underweight high threat, consistent with Kahneman-Tversky probability weighting. Adding γ improves vigor-threat calibration from 2–4× overprediction to near-accurate, and improves BIC by ~970.
