# The 3-Parameter Choice-Vigor Model (v2)

**Status:** Definitive model as of 2026-03-29
**Code:** `scripts/modeling/evc_3param_v2.py`
**Parameters:** `results/stats/oc_evc_3param_v2_params.csv`

---

## Overview

The model separates foraging behavior into two channels — **what you choose** and **how hard you press** — governed by three per-subject parameters and five population-level parameters. The key design principle is that effort and threat enter the choice equation as **structurally independent linear costs**, while vigor uses a physics-based survival function where pressing faster genuinely improves escape probability.

---

## The Choice Equation

```
ΔEU = 4 - k_i × effort(D) - β_i × T

P(heavy) = sigmoid(ΔEU / τ)
```

### What each term does

**`4`** — The reward advantage of the heavy cookie (5 points) over the light cookie (1 point). This is the "upside" of choosing heavy. It's constant across all conditions.

**`k_i × effort(D)`** — The effort cost of choosing heavy. `k_i` is the per-subject effort cost parameter (how much this person dislikes physical work). `effort(D)` is the effort-distance differential between the heavy and light cookie:

```
effort(D) = req_H² × D_H - req_L² × D_L
          = 0.9² × D_H - 0.4² × 1
          = 0.81 × D_H - 0.16
```

This comes from a commitment-cost structure where the cost of choosing an option scales with its squared effort demand times distance. At D=1: effort = 0.65. At D=3: effort = 2.27. So far cookies cost ~3.5× more effort than close ones.

**`β_i × T`** — The threat cost of choosing heavy. `β_i` is the per-subject threat aversion parameter (how much this person is deterred by danger). `T` is the raw threat probability {0.1, 0.5, 0.9}. At T=0.1: threat cost = β×0.1 (small). At T=0.9: threat cost = β×0.9 (large). This is intentionally simple — no survival function, no probability weighting, just linear scaling of stated threat.

**`τ`** — Population-level choice temperature (τ = 1.06). Controls how deterministic choices are. Lower τ = more deterministic.

### Why k and β are identifiable

The task crosses threat (T) and distance (D) in a 3×3 design. k captures the **distance gradient** (how much choice changes with D at fixed T) and β captures the **threat gradient** (how much choice changes with T at fixed D). Because T and D are orthogonal by design, k and β are structurally independent.

**Empirical result:** k × β correlation = **-0.006** (p = 0.92). Perfectly orthogonal.

### Why there's no survival function in choice

Previous versions of the model used `S = (1 - T^γ) + ε × T^γ × p_esc` in the choice equation, with γ (probability weighting) and ε (effort efficacy) as population parameters. The problem: γ and ε absorbed most of the threat signal at the population level, leaving too little variance for a per-subject β to capture. Recovery of β failed (r = 0.21).

Dropping S, γ, and ε from choice makes β the **sole carrier of threat information**, which is what allows it to be identified per-subject. The T×D interaction in observed choices arises naturally from the sigmoid nonlinearity — when ΔEU is near zero, both k and β have strong marginal effects.

### What the choice equation predicts per condition

| T   | D | Observed P(heavy) | Predicted P(heavy) |
|-----|---|-------------------|-------------------|
| 0.1 | 1 | 0.808 | 0.876 |
| 0.1 | 2 | 0.692 | 0.694 |
| 0.1 | 3 | 0.565 | 0.494 |
| 0.5 | 1 | 0.633 | 0.601 |
| 0.5 | 2 | 0.381 | 0.379 |
| 0.5 | 3 | 0.188 | 0.220 |
| 0.9 | 1 | 0.397 | 0.316 |
| 0.9 | 2 | 0.138 | 0.169 |
| 0.9 | 3 | 0.078 | 0.088 |

Per-subject choice r² = **0.981**. Choice accuracy = 82.5%.

---

## The Vigor Equation

```
EU(u) = S(u) × R - (1 - S(u)) × cd_i × (R + C) - ce_vigor × (u - req)² × D

u* = soft argmax over 30-point grid [0.1, 1.5]
```

### What each term does

**`S(u) × R`** — Expected reward. S(u) is the probability of surviving (keeping your cookie) given press rate u. R is the chosen cookie's reward (5 for heavy, 1 for light). Pressing faster increases S(u), so expected reward goes up.

**`(1 - S(u)) × cd_i × (R + C)`** — Expected loss from capture. (1 - S(u)) is the probability of being caught. cd_i is the per-subject capture aversion — how much this person is penalized by the prospect of capture. (R + C) is the total loss: you lose both the cookie reward R and pay the capture penalty C = 5. So the stakes are (R + C) = 10 for heavy cookies, 6 for light.

**`ce_vigor × (u - req)² × D`** — Motor deviation cost. Pressing above the required rate (u > req) is physically costly. This cost scales quadratically with how far above req you press, and linearly with distance D (because you have to sustain the effort longer for far cookies). ce_vigor = 0.003 is a population-level parameter.

### The survival function in vigor

```
S(u) = (1 - T) + T × p_esc × sigmoid((u - req) / σ_motor)
```

**`(1 - T)`** — Probability of no attack. If T = 0.1, there's a 90% chance no predator appears at all, so S ≥ 0.9 regardless of how you press.

**`T × p_esc × sigmoid((u - req) / σ_motor)`** — On attack trials (probability T), your chance of escape depends on how fast you press. The sigmoid centers on the required rate `req`: pressing at exactly req gives ~50% of the maximum escape probability. Pressing much faster saturates near `p_esc`. `σ_motor` = 0.82 controls the steepness of this transition.

**`p_esc`** = 0.002. This is very low — pressing faster barely helps survival in absolute terms. But cd amplifies the incentive: if cd = 30 and the stakes are 10, the penalty term is 30 × 10 × (1 - S(u)) = 300 × (1-S(u)). Even a tiny change in S(u) from pressing harder translates to a meaningful change in expected utility when cd is large.

### Why vigor uses S(u) but choice doesn't

In vigor, the speed-escape relationship is a **physical constraint** of the task: pressing faster literally moves your character faster, which genuinely helps you outrun the predator. S(u) encodes this real mechanic. In choice, there's no analogous mechanic — you haven't started pressing yet, so S is not speed-dependent. The threat information relevant to choice is just T itself.

### How optimal vigor u* is computed

The model evaluates EU(u) on a 30-point grid from u = 0.1 to u = 1.5, then takes a softmax-weighted average:

```
weights = softmax(EU_grid × 10.0)
u* = sum(weights × u_grid)
```

The temperature factor (10.0) makes this nearly a hard argmax while remaining differentiable for gradient-based optimization.

### What the model predicts for vigor

The predicted excess effort (u* - req, cookie-type centered) is compared to observed excess effort:

- Trial-level vigor r² = **0.424**
- Per-subject vigor r² = **0.669**

Higher cd → presses harder → larger excess effort, especially at high T where the (1 - S(u)) term is largest.

---

## The Three Per-Subject Parameters

### k — Effort Cost
- **What it captures:** How much physical effort deters this person from choosing the high-reward option
- **Where it's identified:** The distance gradient in choice. People with high k avoid heavy cookies especially when D is large.
- **Median:** 1.48 (IQR: 1.05–2.04)
- **Behavioral signature:** k → overcautious rate, partial r = **0.933** (controlling for β and cd)
- **Condition specificity:** k dominates choice at T=0.1 (r = -0.937 with P(heavy))
- **Clinical:** No significant associations with any psychiatric measure

### β — Threat Aversion
- **What it captures:** How much stated threat probability deters this person from choosing heavy
- **Where it's identified:** The threat gradient in choice. People with high β avoid heavy cookies especially when T is large.
- **Median:** 4.30 (IQR: 3.21–5.71)
- **Behavioral signature:** β → threat sensitivity in choice, partial r = **0.779** (controlling for k and cd)
- **Condition specificity:** β dominates choice at T=0.9 (r = -0.840 with P(heavy))
- **Clinical:** β → AMI (apathy) r = 0.143, p = .015 — the only parameter-clinical link
- **Not related to encounter reactivity** (r = 0.078, p = .19) — threat aversion in choice is a different system from defensive motor mobilization

### cd — Capture Aversion
- **What it captures:** How strongly the prospect of capture drives this person to press harder
- **Where it's identified:** Exclusively from vigor data. Vigor on all 81 trials (including probes) informs cd.
- **Median:** 30.9 (IQR: 10.7–49.6). Reported in log space for analysis.
- **Behavioral signature:** cd → vigor gap, partial r = **0.587** (controlling for k and β)
- **Encounter dynamics:** cd → encounter reactivity r = **0.390** — same people who press harder overall also react more sharply to predator appearance
- **Clinical:** No significant associations

### Why cd is so large

cd is large because p_esc = 0.002 makes the survival gradient tiny. To produce any meaningful vigor variation, cd must amplify the small (1-S(u)) differences by a large factor. This is a scaling artifact of the parameterization — what matters is cd × (R+C) × ∂S/∂u, which ends up being a reasonable number. We report cd in log space for all analyses.

---

## Population-Level Parameters

| Parameter | Value | What it does |
|-----------|-------|-------------|
| τ (tau) | 1.060 | Choice temperature. Higher = noisier choices |
| p_esc | 0.002 | Maximum escape probability at full speed. Very low. |
| σ_motor | 0.820 | Steepness of speed-escape sigmoid. Controls how sharply pressing faster helps |
| ce_vigor | 0.003 | Motor deviation cost. How costly it is to press above required rate |
| σ_v | 0.241 | Vigor observation noise. Irreducible motor variability |

### Hierarchical structure

Each per-subject parameter is drawn from a log-normal distribution with non-centered parameterization:

```
k_i = exp(μ_k + σ_k × z_k_i)     where z_k_i ~ Normal(0, 1)
β_i = exp(μ_β + σ_β × z_β_i)     where z_β_i ~ Normal(0, 1)
cd_i = exp(μ_cd + σ_cd × z_cd_i)  where z_cd_i ~ Normal(0, 1)
```

Population-level hyperparameters (μ, σ for each parameter) are estimated alongside the subject-level deviations z.

---

## Data Structure

### Choice data (45 trials per subject, N = 293)
- `type = 1` trials only (free choice between heavy and light)
- Binary outcome: choice = 1 (heavy) or 0 (light)
- Predictors: threat T ∈ {0.1, 0.5, 0.9}, distance D_H ∈ {1, 2, 3}

### Vigor data (all 81 trials per subject)
- Types 1 (choice), 5 (probe-heavy), 6 (probe-light)
- Continuous outcome: excess effort = median press rate - required rate, cookie-type centered
- Cookie-type centering: subtract the mean excess for heavy (or light) choice trials, computed from type=1 trials only. This removes the confound that heavy cookies have higher required rates.
- Probe trials use `startDistance` for distance (not `distance_H`, which is set to 1 for probes)

### Why probes matter for cd

Probe trials (36/81) have forced-choice identical options, so there's no selection bias — every subject provides vigor data at every T×D×cookie combination. This anchors cd estimation across conditions where choice data would be sparse (e.g., very few people choose heavy at T=0.9, D=3).

---

## Fitting Procedure

- **Framework:** NumPyro SVI (Stochastic Variational Inference)
- **Guide:** AutoNormal (mean-field approximation)
- **Optimizer:** ClippedAdam (lr = 0.001, clip_norm = 10.0)
- **Steps:** 40,000
- **Joint likelihood:** Bernoulli for choice + Normal for vigor, simultaneously optimized

The choice and vigor likelihoods are summed into a single ELBO objective. The SVI procedure finds the variational distribution that best approximates the true posterior over all parameters (subject-level z's and population hyperparameters).

---

## Parameter Recovery

3 synthetic datasets × 50 subjects each, simulated from the fitted population distribution and refitted:

| Parameter | Recovery r | Threshold | Cross-recovery to other params |
|-----------|-----------|-----------|-------------------------------|
| k | 0.850 | > 0.70 ✓ | k→β_rec: r = 0.030 |
| β | 0.841 | > 0.70 ✓ | β→k_rec: r = 0.111 |
| cd | 0.927 | > 0.70 ✓ | — |

Cross-recovery near zero confirms k and β are **truly separable** — the fitting procedure doesn't trade off between them.

---

## The Triple Dissociation

The central empirical result. Each parameter predicts a different behavioral outcome, each controlling for the other two:

| Parameter | Primary behavioral target | Bivariate r | Partial r (controlling other 2) |
|-----------|--------------------------|-------------|--------------------------------|
| k | Overcautious rate | 0.885 | **0.933** |
| β | Threat sensitivity in choice | 0.574 | **0.779** |
| cd | Vigor gap | 0.580 | **0.587** |

Joint regression of overcaution on all three: **R² = 0.887**
- Unique R² for k: 0.768 (77%)
- Unique R² for β: 0.102 (10%)
- Unique R² for cd: 0.001 (negligible)

---

## What the Model Cannot Do

1. **No probability weighting.** The model takes T at face value. If people systematically distort threat probabilities (e.g., treating T=0.1 as more dangerous than 10%), that's absorbed into the population-level τ rather than modeled explicitly.

2. **No distance in the threat term.** β × T doesn't scale with distance. If someone is especially threat-averse at far distances (T×D interaction beyond the sigmoid nonlinearity), the model attributes that to k, not β.

3. **No per-subject vigor effort cost.** ce_vigor is population-level. If some people find pressing above req more costly than others, the model can't capture that — it goes into cd or noise.

4. **No learning / dynamics.** Parameters are static across the session. Block-to-block changes in strategy are treated as noise.

5. **Low p_esc limits vigor prediction ceiling.** With p_esc = 0.002, the survival function is nearly flat over the observed press-rate range. This means the model's vigor predictions are driven more by the cd × penalty gradient than by the actual survival mechanics. Trial-level vigor r² = 0.424 reflects this constraint plus irreducible motor noise.
