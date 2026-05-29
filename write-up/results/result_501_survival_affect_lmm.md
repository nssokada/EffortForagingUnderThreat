---
result_id: 501
class: metacognition
title: Trial-level survival probability predicts affect (anxiety, confidence)
status: untested
prereg_h: []
internal_h: [H4]
samples: []
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-27
last_run: null
---

# Result 501 — Trial-level survival probability predicts affect (anxiety, confidence)

> **⚠️ Deferred — analysis not yet migrated to current notebook structure.**
>
> This result is documented as a stub pending re-execution against the current data. The underlying analysis was last run in a deprecated notebook (`NB04-03` in the prior project structure) and is recorded in `instructions/memory/hypotheses.md` (internal H4) as:
>
> - **Anxiety ~ S_probe + (1|subj):** β = −0.602, p < 0.001
> - **Confidence ~ S_probe + (1|subj):** β = +0.632, p < 0.001
>
> These numbers cannot be validated against any current notebook cell. Per the project's T3 fail-loud source-of-truth policy, this file is shipped as a stub rather than with un-validated numbers.

## Overview (planned)

If the joint-model survival probability S(u, T, D) is the actual computational input that drives subjective affect — rather than the raw threat probability T alone — then per-trial S should predict per-trial anxiety and confidence beyond what T and D explain on their own. The analysis fits two trial-level LMMs (anxiety ~ S_probe and confidence ~ S_probe with subject random intercepts) on probe-trial data, using S_probe computed from each subject's fitted ω and κ. A positive confidence effect and negative anxiety effect would establish that the metacognitive monitoring signals respond to the model-derived survival quantity rather than (or in addition to) the raw task variables.

## Hypothesis (planned)

**Statement.** "Trial-level survival probability computed from fitted parameters predicts self-reported anxiety (−) and confidence (+)." (Internal H4, `instructions/memory/hypotheses.md`.)

**Predicted direction.** β(anxiety ~ S_probe) < 0; β(confidence ~ S_probe) > 0.

**Source of the hypothesis.** Internal H4. Not in the formal preregistration (prereg H4c is "κ → mean vigor", which is a different test reported in [[result_208]]). This is an exploratory mechanistic claim that the metacognitive signals are downstream of the joint-model computation, not just of raw task variables.

## What needs to happen to ship this result

1. **Locate or write the analysis cell.** The original cell lived in `notebooks/04_psych_analysis/03_affect_survival.ipynb` (NB04-03 in the prior project structure). That file is no longer in the active `notebooks/analysis/` tree. Either (a) port the analysis into one of the H4 / H5 notebooks, or (b) write a small standalone script that loads `behavior_rich.csv` + `feelings.csv` + the fitted ω/κ from `mcmc_m4_params.csv`, computes per-probe S_probe, and fits the two LMMs.
2. **Validate against the current data.** The N = 290 / 281 samples should be used, not the prior N = 293 sample.
3. **Re-run and confirm sign and magnitude.** The legacy numbers (β = −0.602 anxiety, +0.632 confidence) were computed on N = 293; signs should replicate even if magnitudes drift slightly.
4. **Populate this file's frontmatter and Result section.** Use the canonical template structure.

## References

**Related results:**
- [[result_102]] — Affect ~ threat + distance (the descriptive precursor; this result asks whether the model-derived survival is the real driver).
- [[result_502]] — Anxiety calibration as individual-difference predictor of optimality (uses calibration *r* per subject, related but not identical).
- [[result_201]] — Joint model M4 fit (source of the ω, κ values used to compute S_probe).

**Legacy memory entry:**
- `instructions/memory/hypotheses.md` § H4 — original analysis and result.
