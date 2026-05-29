---
result_id: 207
class: computational_model
title: Residual threat sensitivity (β bias term) needed in FET framework (superseded by M4 architecture)
status: supported_exploratory
prereg_h: []
internal_h: [H3]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 207 — Residual threat sensitivity (β bias term) needed in FET framework (superseded by M4 architecture)

> **⚠️ Exploratory, deprecated framework.** Demonstrated in the FET model that a threat-bias parameter β was needed beyond expected-value computations. In the current M4 framework, threat enters through the survival function S(u, T, D), and no separate β bias term exists — the role β played in FET is absorbed by ω in M4. This result is preserved for completeness.

## Result (legacy, from internal H3)

FETExponential vs FETExponentialBias model comparison: WAIC favors the bias-extended model. β posterior mean = 1.44 (right-skewed across subjects).

See `instructions/memory/hypotheses.md` § H3.

## Interpretation

In the FET framework, β captured residual threat sensitivity not absorbed by the EV-with-effort-discount choice term. The M4 architecture replaces this two-component decomposition (EV + threat bias) with a single survival-weighted fitness term where threat enters mechanistically through S(u, T, D). The substantive insight — that subjects vary in how strongly threat shifts their choice beyond what a pure EV-with-effort framework predicts — survives in M4 as variation in ω, which is now identifiable as a survival-cost rather than a generic threat-aversion bias.

The supersession of β by ω is documented in `instructions/memory/joint_model_development.md`.

## References

- `instructions/memory/hypotheses.md` § H3.
- `instructions/memory/joint_model_development.md` — FET → M4 transition.
- [[result_201]] — M4 fit.
- [[result_402]] — Cross-channel ω → vigor (the M4-era replacement for β-driven analyses).
