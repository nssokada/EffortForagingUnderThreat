# The EVC 2+2 Model (ce + cd with γ and ε)

**Status:** Superseded by the 3-parameter v2 model (see `MODEL.md`). This document describes the model used in drafts 004–010.
**Code:** `scripts/modeling/evc_final_2plus2.py`
**Parameters:** `results/stats/oc_evc_final_params.csv`
**Population:** `results/stats/oc_evc_final_81_population.csv`

---

## Overview

The "2+2" name refers to **2 per-subject parameters** (ce, cd) and **2 key population parameters** (γ, ε) that together define how the organism integrates effort and threat. The model is a joint choice-vigor framework inspired by the Expected Value of Control (EVC; Shenhav et al., 2013) with a cost structure drawn from Linear-Quadratic Regulator (LQR) optimal control theory.

The central architectural decision — and the thing that makes this model distinctive — is that **ce only enters choice** and **cd only enters vigor**. This isn't an arbitrary restriction; it follows from a mathematical cancellation in the choice equation where cd's contribution becomes collinear with the reward term and therefore unidentifiable from choice data alone.

---

## The Survival Function S

Both the choice and vigor equations depend on a shared survival computation:

```
S = (1 - T^γ) + ε × T^γ × p_esc
```

### Breaking this down

**`T^γ`** — Probability-weighted threat. T is the stated threat probability {0.1, 0.5, 0.9}. γ (gamma) = 0.209 is the probability weighting exponent. Because γ < 1, threat probabilities are **compressed upward**:
- T = 0.1 → T^0.209 = **0.618** (10% threat feels like 62%)
- T = 0.5 → T^0.209 = **0.866** (50% feels like 87%)
- T = 0.9 → T^0.209 = **0.978** (90% feels like 98%)

This is dramatically stronger compression than in monetary gambles (γ ~ 0.65–0.70 in Kahneman-Tversky). The interpretation: embodied virtual predation engages defensive circuitry more powerfully than abstract monetary loss.

**`(1 - T^γ)`** — Probability of no attack (under the weighted threat). At T=0.1 this is 1 - 0.618 = 0.382. The organism perceives a 62% chance of attack even when told 10%.

**`ε × T^γ × p_esc`** — If attacked (probability T^γ), the organism can still escape with probability ε × p_esc. ε (epsilon) = 0.098 is the **effort efficacy** — how much pressing effort translates into survival benefit. p_esc = 0.018 is the base escape probability. The product ε × p_esc = 0.0018 is tiny, meaning pressing harder helps survival very little in absolute terms.

**Net effect:** S ranges from ~0.38 (at T=0.1) to ~0.02 (at T=0.9). The organism's subjective survival probability is much lower than objective because γ compresses all threat levels toward "dangerous."

### Why S is the same for heavy and light in choice

S does not depend on distance or cookie type in the choice equation. This is a modeling decision: at the moment of choice, the organism hasn't started pressing yet, so the speed-dependent survival term doesn't apply. The only thing that varies S across conditions is threat level T.

In vigor, S becomes speed-dependent (see below).

---

## The Choice Equation

```
ΔEU = S × 4 - ce_i × (0.81 × D_H - 0.16)

P(heavy) = sigmoid(ΔEU / τ)
```

### What each term does

**`S × 4`** — The survival-weighted reward advantage. The heavy cookie is worth 4 more points than light (5 - 1 = 4), but only if you survive. S scales this by the probability of survival. At low threat (S ≈ 0.38), the advantage is S×4 = 1.52. At high threat (S ≈ 0.02), it's just 0.08. So the "upside" of heavy shrinks dramatically under threat — driven entirely by the population-level S function, not any per-subject parameter.

**`ce_i × (0.81 × D_H - 0.16)`** — The effort cost. ce is the per-subject effort cost parameter. The term `(0.81 × D_H - 0.16)` is the **commitment cost differential** between heavy and light, derived from LQR-inspired cost structure:

```
Commitment cost of option j = req_j² × D_j

Heavy: 0.9² × D_H = 0.81 × D_H
Light: 0.4² × 1   = 0.16

Differential: 0.81 × D_H - 0.16
```

At D=1: effort cost = 0.65. At D=3: effort cost = 2.27. The "LQR-inspired" label comes from the analogy with linear-quadratic regulators where the cost of a trajectory scales with the squared control signal — here, the squared required press rate.

**`τ`** — Choice temperature (τ = 0.476). Lower than in the 3-param model (1.06) because S already compresses the ΔEU range.

### Why cd is absent from choice

This is the key architectural insight. The full EV of choosing cookie j is:

```
EV_j = S × R_j - (1 - S) × cd × (R_j + C)
```

When you take the difference between heavy and light:

```
ΔEV = S × (R_H - R_L) - (1 - S) × cd × (R_H + C - R_L - C)
     = S × 4 - (1 - S) × cd × 4
     = 4 × [S - (1 - S) × cd]
```

The cd term is `(1-S) × cd × 4`. But notice: it scales with the **same factor** (4 = R_H - R_L) as the reward term. This means cd and the reward scaling are **collinear** — the model can't distinguish "this person has high cd" from "this person underweights rewards." They produce identical choice patterns.

So we remove cd from choice and let ce be the sole per-subject choice parameter. cd is estimated exclusively from vigor.

**Important nuance:** This means the 2+2 model **cannot tell you whether someone chose light because of effort cost or because of threat aversion**. All choice variation beyond the population-level S goes through ce. This was the fundamental limitation that motivated the 3-param v2 model.

---

## The Vigor Equation

```
EU(u) = S(u) × R - (1 - S(u)) × cd_i × (R + C) - ce_vigor × (u - req)² × D

u* = soft argmax over 30-point grid [0.1, 1.5]
```

### The speed-dependent survival function

```
S(u) = (1 - T^γ) + ε × T^γ × p_esc × sigmoid((u - req) / σ_motor)
```

This extends the static S by making it depend on press rate u. The sigmoid term means that pressing at exactly the required rate gives ~50% of the maximum escape benefit, while pressing much faster saturates near ε × p_esc.

- σ_motor = 1.169 controls the steepness of the speed-escape sigmoid
- The key gradient: ∂S/∂u = ε × T^γ × p_esc × sigmoid'(·) / σ_motor

Because ε = 0.098, p_esc = 0.018, and σ_motor = 1.169, this gradient is tiny. But cd amplifies it: the marginal benefit of pressing harder is cd × (R+C) × ∂S/∂u. When cd = 31 and R+C = 10, even a small ∂S/∂u translates to a meaningful change in EU.

### The three terms in vigor

**`S(u) × R`** — Expected reward from surviving. Pressing faster increases S(u), increasing expected reward.

**`(1 - S(u)) × cd_i × (R + C)`** — Expected loss from capture. cd scales how much capture costs the individual. (R + C) = reward + penalty. For heavy cookies: (5 + 5) = 10. For light: (1 + 5) = 6. The penalty C = 5 is fixed; you also lose the cookie reward R.

**`ce_vigor × (u - req)² × D`** — Motor deviation cost. Pressing above the required rate is costly. This is the "deviation cost" in the LQR analogy: the cost of deviating from the reference trajectory (req). ce_vigor = 0.003 is population-level. Scales with D because you sustain the effort longer at far distances.

### The LQR analogy

The distinction between the choice and vigor cost terms parallels LQR control theory:

- **Choice:** commitment cost = req² × D → "How much does it cost to commit to this trajectory?"
- **Vigor:** deviation cost = (u - req)² × D → "How much does it cost to deviate from the committed trajectory?"

This isn't a formal LQR implementation (no state dynamics, no feedback, no Riccati equation) — it's an analogy that provides theoretical motivation for why the cost structure differs between the two decision stages.

---

## The Two Per-Subject Parameters

### ce — Effort Cost (c_effort)
- **What it captures:** How strongly effort/distance deters choice. The sole individual-difference parameter in the choice equation.
- **Median:** 0.62 (IQR: 0.41–0.78)
- **Where identified:** The distance gradient in choice at each threat level, after the population-level S has accounted for threat effects
- **Key results:**
  - ce → overcautious rate: r = **0.924** (partial r = 0.923 controlling for cd)
  - ce explains **83.1%** of unique variance in overcaution
  - ce → total earnings: r = -0.81
  - ce → overrisky rate: r = -0.775 (high ce people never choose risky)
  - Recovery: r = **0.92** in simulation
- **Interpretation:** ce captures a global tendency to avoid effortful options. Because it's the only per-subject choice parameter, it absorbs ALL individual variation in choice — including variation that might be due to threat sensitivity. This is what makes it a clean predictor of overcaution but also limits interpretability.

### cd — Capture Aversion (c_death)
- **What it captures:** How strongly the survival incentive drives motor vigor above the required rate
- **Median:** 31.3 (IQR: 8.7–79.3). Reported in log space.
- **Where identified:** Exclusively from vigor data (all 81 trials including probes)
- **Key results:**
  - cd → vigor gap: r = **0.554** (partial r = 0.550 controlling for ce)
  - cd → encounter reactivity: r = **0.50** (bridges tonic and phasic defense)
  - cd → survival rate: r = -0.02 (NULL — pressing harder doesn't actually help survival)
  - Recovery: r = **0.94** in simulation
- **Why cd is large:** ε × p_esc = 0.0018, so the survival gradient is tiny. cd must be ~30 to produce meaningful vigor variation. This is a scaling artifact.
- **Independence from ce:** r = -0.14 (approximately independent)

---

## Population-Level Parameters

| Parameter | Value | Role |
|-----------|-------|------|
| γ (gamma) | 0.209 | Probability weighting exponent. T_weighted = T^0.209. Compresses threat dramatically. |
| ε (epsilon) | 0.098 | Effort efficacy. How much pressing faster helps survival. Nearly zero individual recovery (r = -0.02 when tried per-subject). |
| ce_vigor | 0.003 | Motor deviation cost in vigor. Population-level. |
| τ (tau) | 0.476 | Choice temperature. |
| p_esc | 0.018 | Base escape probability at full pressing speed. |
| σ_motor | 1.169 | Steepness of speed-escape sigmoid. |
| σ_v | 0.229 | Vigor observation noise. |

### Why ε collapsed to population-level

We tried fitting ε per-subject. Recovery was r = -0.02 — completely flat. The reason: when ce handles choice and cd handles vigor, ε has no unique signal. ε modulates S, but S's effect on choice is already captured by the population-level γ, and its effect on vigor is absorbed by cd. So ε collapsed to a population constant.

### Why γ = 0.21 matters

Under γ = 0.21, the perceived threat is dramatically higher than stated:

| Stated T | Perceived T^γ | Difference |
|----------|--------------|------------|
| 0.10 | 0.618 | +0.518 |
| 0.50 | 0.866 | +0.366 |
| 0.90 | 0.978 | +0.078 |

This shifts the optimal foraging policy: under objective probabilities, 5/9 conditions favor heavy. Under γ = 0.21, only **1/9** favors heavy. This means ~20% of apparent "overcaution" is actually rational under the organism's subjective threat surface.

---

## Model Fit Results

### Primary fit quality
- Per-subject choice r = 0.975 (r² = **0.951**)
- Choice accuracy = 79.3%
- Trial-level vigor r² = **0.511**
- Per-subject vigor r = 0.829 (r² = 0.687)
- BIC = 32,133 (81-trial likelihoods)

### Model comparison (6 ablations)

| Model | Description | BIC | ΔBIC | Choice r² | Vigor r² |
|-------|------------|-----|------|----------|---------|
| **FINAL** | **EVC 2+2** | **32,133** | **0** | **0.951** | **0.511** |
| M4 | Population ce (no per-subj ce) | 30,860 | -1,273 | 0.001 | 0.512 |
| M6 | Standard u² cost (not LQR) | 31,991 | -142 | 0.952 | 0.508 |
| M5 | No γ (γ=1, no probability weighting) | 34,204 | +2,071 | 0.955 | 0.425 |
| M3 | Separate choice + vigor (not joint) | 42,526 | +10,393 | 0.955 | 0.440 |
| M2 | Threat only (no individual ce) | 42,767 | +10,634 | 0.006 | 0.294 |
| M1 | Effort only (no threat info) | 50,792 | +18,659 | 0.950 | 0.000 |

**Key takeaways:**
- M4 has *lower* BIC but fails at individual choice prediction (r² = 0.001). It sacrifices the primary behavioral target for marginal vigor gain.
- M6 (u² instead of (u-req)²) is empirically equivalent — the LQR distinction doesn't matter for fit.
- Removing γ (M5) costs +2,071 BIC and hurts vigor (r² drops from 0.511 to 0.425).
- Removing threat entirely (M1) destroys vigor prediction (r² = 0.000).

### MCMC validation
- NUTS sampler, 4 chains × 200 warmup + 200 samples
- Zero divergent transitions
- SVI-MCMC correlation: log(ce) r = **0.999**, log(cd) r = **0.999**
- All population R-hat < 1.05

---

## Downstream Results with This Model

### Optimal policy and deviations
- 5/9 conditions favor heavy objectively; under γ = 0.21, only 1/9
- γ distortion accounts for ~20% of apparent suboptimality
- Participants achieved 69.8% optimality (SD = 12.0%)
- Overcautious errors: 21.3% (SD = 14.4%)
- Overrisky errors: 8.9% (SD = 8.4%)

### Parameter → deviation associations
- ce → overcautious rate: r = 0.924 (83% unique R²)
- cd → vigor gap: r = 0.554
- cd → survival rate: r = -0.02 (NULL)

### Encounter dynamics
- Encounter reactivity: trait-stable (cross-block r = 0.78)
- Threat-independent (ANOVA F = 0.04, p = .96)
- cd-linked: r = 0.50 (log(cd))
- ce-independent: r = -0.09

### Affect
- Calibration ⊥ Discrepancy: r = 0.019 (orthogonal)
- Calibration → policy alignment: r = 0.304, partial r = 0.305, **ΔR² = 6.4%** (the strong affect finding)
- Discrepancy → residual overcaution: r = 0.142, ΔR² = 0.3% (significant but small)

### Clinical
- Discrepancy → STAI-State: β = 0.338 (Bayesian regression, 94% HDI excluding zero)
- Discrepancy → STICSA, DASS-Anxiety, DASS-Stress, OASIS, PHQ-9, DASS-Depression, MFIS: all significant
- ce, cd → clinical: 65–93% posterior mass in ROPE (|β| < 0.10) — effectively null
- Encounter reactivity → AMI (apathy): r = -0.17 (only dynamics → clinical link)

### Behavioral profiles (median split ce × cd)
| Profile | ce | cd | N | Earnings | Label |
|---------|----|----|---|----------|-------|
| Vigilant | low | high | 82 | 103.5 | Best earners |
| Bold | low | low | 64 | 96.1 | High choice, low vigor |
| Cautious | high | low | 65 | 67.6 | Avoid + low vigor |
| Disengaged | high | high | 82 | 66.4 | Avoid everything |

---

## Why This Model Was Superseded

The 2+2 model has excellent fit quality and clean parameter recovery, but it has a fundamental interpretive limitation:

**ce is the ONLY per-subject parameter in the choice equation.** All individual variation in choice goes through ce. If someone avoids heavy cookies, the model says "high effort cost" regardless of whether the true reason is effort aversion or threat aversion. The model cannot distinguish:
- "I avoided heavy because pressing 0.9 capacity for 9 game units is exhausting" (effort)
- "I avoided heavy because T=0.9 and I'd rather not risk it" (threat)

The 3-param v2 model solves this by adding β (threat aversion) as a second per-subject choice parameter, dropping γ and ε to make β identifiable.

### What the 2+2 model does better
- **Calibration → policy alignment ΔR² = 6.4%** (vs 0.4% in 3-param). With only 2 params explaining 85% of overcaution, there's more residual for calibration to predict. The 3-param model explains 89%, leaving less room.
- **γ interpretation.** The probability weighting story (γ = 0.21, dramatic compression of threat) is theoretically interesting and connects to Kahneman-Tversky. The 3-param model drops this entirely.
- **Simpler.** Two per-subject parameters is more parsimonious than three. BIC for 45-trial choice: 17,768 for 2+2 vs higher for 3-param (more parameters).

### What the 3-param model does better
- **Can separate effort from threat in choice** (the whole point)
- **Choice r² = 0.981** vs 0.951 (better fit with the extra parameter)
- **Triple dissociation** — three parameters map to three distinct behavioral outputs
- **β predicts threat sensitivity** (r = 0.779) — something the 2+2 model simply cannot do
- **β → AMI** gives a parameter → clinical link that wasn't available before

---

## Hierarchical Structure

Both per-subject parameters use log-normal priors with non-centered parameterization:

```
ce_i = exp(μ_ce + σ_ce × z_ce_i)    where z_ce_i ~ Normal(0, 1)
cd_i = exp(μ_cd + σ_cd × z_cd_i)    where z_cd_i ~ Normal(0, 1)
```

Population-level hyperparameters (μ_ce, σ_ce, μ_cd, σ_cd) are estimated alongside the subject-level z's and the other population parameters (γ, ε, τ, etc.).

### Fitting procedure
- NumPyro SVI (Stochastic Variational Inference)
- AutoNormal guide (mean-field approximation)
- Adam optimizer, lr = 0.002, 40,000 steps
- Joint likelihood: Bernoulli (choice) + Normal (vigor)
- Choice data: 45 trials per subject (type=1)
- Vigor data: all 81 trials per subject (types 1, 5, 6)

---

## File Inventory

| File | Contents |
|------|----------|
| `scripts/modeling/evc_final_2plus2.py` | Model code (prepare_data, make_model, fit, evaluate) |
| `scripts/mcmc/run_mcmc.py` | MCMC validation script (NUTS, 4 chains) |
| `results/stats/oc_evc_final_params.csv` | Per-subject ce, cd (45-trial fit) |
| `results/stats/oc_evc_final_81_params.csv` | Per-subject ce, cd (81-trial fit) |
| `results/stats/oc_evc_final_population.csv` | Population params (45-trial) |
| `results/stats/oc_evc_final_81_population.csv` | Population params (81-trial) |
| `results/stats/evc_model_comparison_final.csv` | 7-model comparison table |
| `results/stats/deviation_param_associations.csv` | ce/cd → deviation stats |
| `results/stats/residual_suboptimality.csv` | Affect → residual overcaution stats |
| `results/stats/optimal_policy.csv` | Per-condition optimal choice |
| `results/stats/per_subject_deviations.csv` | Per-subject deviation classification |
| `notebooks/07_evc_pipeline/14_optimal_policy.py` | Optimal policy derivation |
| `notebooks/07_evc_pipeline/15_deviation_analysis.py` | Parameter → deviation analysis |
| `notebooks/07_evc_pipeline/16_residual_suboptimality.py` | Affect + residual analysis |
| `drafts/draft010/paper.md` | Paper draft using this model |
