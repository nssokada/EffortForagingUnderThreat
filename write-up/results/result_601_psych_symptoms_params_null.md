---
result_id: 601
class: individual_differences
title: Psychiatric symptom dimensions do not predict model parameters (essentially null)
status: refuted_exploratory
prereg_h: []
internal_h: [H13]
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/H6_clinical.ipynb]
scripts: []
outputs: [results/stats/clinical/clinical_pearson_sweep.csv, results/stats/clinical/clinical_replication.csv, results/stats/clinical/clinical_bayes_followup.csv]
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 601 — Psychiatric symptom dimensions do not predict model parameters (essentially null)

## Overview

The joint-model parameters (ω, κ) — and the deprecated FET parameters (z, k, β) before them — were predicted to correlate with transdiagnostic psychiatric dimensions from the questionnaire battery (DASS-21, PHQ-9, OASIS, STAI-T, AMI, MFIS, STICSA). Across both samples and both parameter sets, the relationships are essentially null: 0/39 bivariate correlations survive FDR correction in the deprecated framework, and the one nominal hit in the confirmatory M4 sweep (ω × AMI_Social r = +0.155, p = 0.009) does not replicate in exploratory.

## Result

From `results/stats/clinical/clinical_pearson_sweep.csv` and `clinical_replication.csv` (M4 parameters, both samples):

- **No parameter × scale correlation replicates strictly (FDR-corrected, both samples in the predicted direction).**
- ω × AMI_Social: r = +0.083 (exp, p_fdr = 0.64) vs +0.155 (conf, p = 0.009). Same sign but exploratory non-significant — does NOT replicate.
- All other ω, κ × clinical correlations: |r| < 0.10 with HDI/CI spanning zero.

From `instructions/memory/hypotheses.md` § H13 (deprecated framework):
- 0/39 correlations survive FDR.
- z shows consistent weak negative associations with anxiety/fatigue (r = −0.10 to −0.18) but no individual test survives correction.

## Interpretation

Computational parameters from this task capture task-specific individual differences in foraging strategy but do not map onto broad psychiatric symptom dimensions. The implication is that the dimensions tapped by ω/κ are orthogonal to the dimensions tapped by clinical questionnaires — these are different layers of individual variation. The clinical signal that *does* exist in this dataset is at the *affect* layer (see [[result_602]] on AMI × vigor and `instructions/memory/hypotheses.md` § H39) rather than at the parameter layer.

The null does not mean the task is clinically uninformative; it means the *transformation* from foraging behavior to clinical relevance runs through the metacognitive monitoring layer rather than through the computational parameters directly. This reframing motivates the H10 mediation analyses (`notebooks/analysis/H10_mediation.ipynb`) which test affect-mediated paths from (ω, κ) to AMI apathy.

## Caveats

- **STAI-Trait scoring bug** (state reverse-items applied to trait) was identified and fixed. Even after fix, STAI-T SD is low and the variable correlates negatively with all distress measures, suggesting it may still be problematic. See [[result_603]] for the methodological note.
- The single "almost-significant" hit (ω × AMI_Social) in confirmatory is the most suggestive but does not survive replication checks.

## References

- `instructions/memory/hypotheses.md` § H13.
- `notebooks/analysis/H6_clinical.ipynb` — clinical regression notebook.
- `notebooks/analysis/H10_mediation.ipynb` — affect-mediated paths.
- [[result_602]] — AMI apathy → vigor (the affect-layer story).
- [[result_603]] — STAI-T scoring bug note.
