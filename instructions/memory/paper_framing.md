---
name: Paper Framing (2026-03-22)
description: Current paper framing — effort reallocation, independent Bayesian pipeline, joint model as robustness, metacognitive bridge
type: project
---

# Paper Framing: Effort Reallocation Under Threat

**Title:** Humans reallocate effort across decision and action when foraging under threat

**Core claim:** Under threat, people reallocate effort — choosing safer targets while pressing harder on them — governed by a shared survival computation S.

## The models

```
S = (1-T) + T/(1+λD)                    # survival computation (from choice)
SV = R·S - k·E - β·(1-S)               # choice model (Bayesian HBM, SVI/MCMC)
excess = α + δ·(1-S) + ε               # vigor model (separate Bayesian HBM, λ fixed from choice)
```

## Evidence pyramid (strongest → most formal)

1. **Behavioral:** choice shift × vigor shift r = −0.78 (model-free)
2. **Independent Bayesian:** r(log(β), δ) = +0.55 (two models, no shared params)
3. **Joint LKJ model:** all 6 pairwise CIs exclude zero (structural confirmation)

## Key parameters

| Param | Source | Value | Role |
|---|---|---|---|
| k | Choice HBM | M=6.3 | Effort sensitivity |
| β | Choice HBM | M=64.8 | Threat bias |
| α | Vigor HBM | M=0.014 | Baseline excess effort |
| δ | Vigor HBM | M=0.211 | Danger-responsive mobilization |
| λ | Choice HBM | 13.9 (±0.6) | Hazard sensitivity (population) |

## Key correlations (independent Bayesian)

| Pair | r | p | Interpretation |
|---|---|---|---|
| log(β) × δ | +0.55 | <10⁻²⁴ | Threat aversion → vigor mobilization |
| log(k) × δ | −0.28 | <10⁻⁶ | Effort avoidance → less vigor |
| log(k) × α | +0.35 | <10⁻⁹ | Effort-sensitive → higher baseline |
| α × δ | +0.06 | n.s. | Independent (tonic ≠ phasic) |

## λ sensitivity (important caveat)

The joint model's LKJ ρ estimates depend on the fixed λ value because vigor is more threat-driven than distance-driven. The independent Bayesian r values do NOT depend on λ. Paper leads with independent r, uses joint model for structural confirmation.

## Supplementary findings

- **Metacognitive bridge:** δ predicts affect calibration (anxiety slope r=−0.22, confidence slope r=+0.26)
- **"Adaptive not anxious":** high δ → lower mean anxiety (r=−0.19)
- **Mental health null:** α → apathy (R²=0.12) is the only psychiatric finding. Everything else is null with ROPE evidence.
- **Affect:** S → anxiety (z=−24), S → confidence (z=+24)
