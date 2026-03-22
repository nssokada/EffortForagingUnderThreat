---
name: Open Questions
description: Unresolved theoretical, methodological, and empirical questions for the paper
type: questions
---

# Open Questions

## Theoretical

### Why is vigor more threat-driven than distance-driven?
The joint model pushes λ to ~35 (S ≈ 1−T), while choice prefers λ ≈ 14 (distance matters). This means vigor excess effort responds mainly to "will I be attacked?" rather than "how far must I escape?" One interpretation: motor mobilization is triggered by threat detection (amygdala-driven), while choice computation integrates both threat and distance (prefrontal). This could be tested with neural data.

### Why are α and δ independent (r = +0.06)?
Tonic baseline vigor and phasic danger mobilization are uncorrelated. This parallels the tonic-phasic dissociation in anxiety (mean anxiety vs anxiety-S slope). Suggests two separable systems: a baseline motor engagement trait (α, linked to dopaminergic tone → apathy) and a threat-responsive mobilization system (δ, linked to β via survival computation).

---

## Methodological

### Will MCMC resolve the λ sensitivity in the joint model?
SVI's variational approximation may be getting stuck in different modes for different λ values. NUTS explores the full posterior and might give stable ρ estimates. The MCMC script is ready (`scripts/run_mcmc_pipeline.py`). This is the key methodological question.

### Should the confirmatory analysis pre-register the independent Bayesian pipeline or the joint model?
Current preregistration (H5) specifies the joint model with LKJ correlations. But the paper's primary evidence is the independent Bayesian r = +0.55. Consider adding the independent pipeline as an alternative analysis in the preregistration.

---

## RESOLVED

- ~~What does β capture?~~ → Subjective capture cost in additive model. Correlates with δ (r=+0.55).
- ~~Does the "common computation" claim hold?~~ → Yes, reframed as "effort reallocation." S governs choice, affect, and vigor. Coupling confirmed by independent Bayesian models.
- ~~Factor analysis~~ → Done. 3 factors. α → apathy only.
- ~~Should we run MCMC?~~ → Yes. Script ready. Awaiting GPU.
- ~~λ = 2 vs λ = 14 discrepancy~~ → λ = 14 is correct from current data. λ = 2 was from older fit. Results robust to this change.
- ~~α → apathy paradox~~ → Kept as Discussion point. High α = more motor engagement but self-reports more apathy. May reflect habitual vs goal-directed distinction.
