---
result_id: 206
class: computational_model
title: Exponential effort discounting beats hyperbolic, quadratic, and linear (deprecated FET framework)
status: supported_exploratory
prereg_h: []
internal_h: [H2]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 206 — Exponential effort discounting beats hyperbolic, quadratic, and linear (deprecated FET framework)

> **⚠️ Exploratory, deprecated FET framework.** This result was established during the FET (effort-discounting + threat-bias) model selection that preceded the current M4 joint-fitness framework. In M4, "effort discounting" is implemented as a quadratic deviation cost (`κ · (u − req)² · D`), not as a discount function on reward — so the question "which discount form wins?" does not directly apply to M4. This result is preserved as documentation of the model-selection process that led to M4.

## Result (legacy, from internal H2)

Among candidate discount functions (exponential, hyperbolic, quadratic, linear) within the FET model framework, exponential effort discounting yielded the lowest WAIC. See `instructions/memory/hypotheses.md` § H2 and the deprecated notebooks at `notebooks/_deprecated/fet_models/01_fit_compare_ppc.ipynb`.

## Interpretation

The exponential form's success in the FET framework motivated the choice of an exponential survival kernel in the M4 specification (S(u) = exp(−h · T^γ · D / speed(u))). In M4, the discount form question becomes a question about the survival kernel form rather than a discount function on reward — and the prereg's "exponential survival kernel" specification is the surviving descendant of this exploratory result.

## References

- `instructions/memory/hypotheses.md` § H2.
- [[result_201]] — Joint model M4 (the framework that replaced FET).
- `notebooks/_deprecated/fet_models/` — original analysis notebooks.
