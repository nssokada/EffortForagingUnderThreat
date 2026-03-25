---
name: Optimal Control Reformulation
description: Plan to reframe the paper as a stochastic optimal control problem — single cost function (c_effort, c_death) jointly determines choice and vigor
type: project
---

# Optimal Control Reformulation

The paper is being reframed from computational psychiatry (descriptive models + post-hoc correlations) to computational neuroscience (mechanistic optimal control).

**Why:** The current approach fits choice and vigor with separate models, then correlates parameters post-hoc (β↔δ). The optimal control approach derives both from a single cost-minimization principle, making the coupling structural rather than correlational.

**How to apply:** All new modeling work should follow the OC framework. The descriptive models (L3_add, L4a_add, vigor HBM, joint LKJ) are superseded for the paper, though useful as comparison benchmarks.

## Core Formulation

Single objective to maximize:
```
EU(u) = P_surv · R − (1 − P_surv) · c_death · (R + C) − c_effort · ∫u(t)² dt
```

Where P_surv = (1−T) + T · Φ((2·t_enc − t_arr) / σ), derived from game mechanics (Gaussian strike time).

**Tier-selection approximation:** Because speed is a 4-level step function, evaluate EU at 4 constant press rates, take the best. Analytically tractable, differentiable via Φ, trivially fast inside MCMC.

## Parameters

| Param | Role | Replaces |
|---|---|---|
| c_effort | Effort cost sensitivity | κ + α |
| c_death | Capture aversion | β + δ |
| σ_strike | Strike time uncertainty (fixed or population) | z / λ |
| τ | Choice temperature | τ |

2-3 subject-level params replace 4 (k, β, α, δ). Survival function S is mechanistically derived, not assumed.

## Implementation Phases

1. Recover strike time σ from raw JSON attackingTime data
2. Build OC solver in JAX (`scripts/modeling/optimal_control.py`)
3. Hierarchical Bayesian model via NumPyro (`scripts/modeling/oc_model.py`)
4. Model comparison against descriptive models
5. Clinical + affect predictions from cost parameters
6. Paper restructure (4 results sections)

## New Paper Results Structure

- R1: Task as stochastic optimal control problem; mechanistic S
- R2: Single cost function governs choice AND vigor (no free vigor params)
- R3: Individual differences in 2D cost space → clinical correlates
- R4: Affect as byproduct of survival computation

## Full plan file

See `/Users/nokada/.claude/plans/piped-enchanting-forest.md` for complete mathematical details, fitting procedure, and risk mitigations.
