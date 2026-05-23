# Parallel agent tracking — push for clinical/metacognitive coherence

Started 2026-04-08. The directly tested clinical battery loadings on (ω, κ) are null after a multivariate kitchen sink (CCA, PLS, EFA, PC1). The user is pushing for a third leg (metacognition/affect) that connects to the parameters and forms a coherent paper. This file tracks what each parallel direction is testing and what it found.

## Directions

| # | Hypothesis | Owner | Status | Result |
|---|---|---|---|---|
| 1 | **Metacognitive sensitivity (Fleming-Lau)**: per-subject confidence/anxiety ↔ trial outcome correlation. Bridges (ω, κ) to clinical via metacognitive accuracy. | Agent 1 | running | TBD |
| 2 | **Affect-behavior coupling (within-subject)**: per-subject slope of vigor on trial anxiety after stimulus partialling. Subjects whose feelings drive action vs subjects whose feelings are decoupled from action. | Agent 2 | running | TBD |
| 3 | **State/trait decomposition of affect**: separate trait mean from state variability and stimulus reactivity per subject. Maybe one of those loads on (ω, κ) or clinical. | Agent 3 | running | TBD |
| 4 | **Anxiety tracks model-derived S, not just T (cross-channel for affect)**: extend H5a by replacing raw T with S_pred(T,D,ω,κ). Test if per-subject tracking of S_pred (a) correlates with (ω, κ), (b) predicts clinical, (c) predicts outcomes beyond H5a. | Self (sequential) | starting | TBD |
| 5 | **Anxiety-tracking as mediator**: (ω, κ) → anxiety_tracking → clinical. Mediation analysis. The "metacognitive bridge" hypothesis. | Self (sequential) | pending | TBD |

## Background context for all agents

- Working dir: /workspace
- Conda env: /opt/micromamba/envs/effort_foraging_threat/bin/python
- Notebooks dir on path: /workspace/notebooks/analysis (contains config.py, load_data.py)
- `load_both()` returns (exp_data, conf_data) dicts with `master`, `feelings`, `vigor_valid`, `choice`, `beh`
- Parameters in `master`: omega, kappa, omega_z, kappa_z, log_om, log_kap (compute as needed)
- Clinical scales available in master (after psych merge): DASS21_Stress, DASS21_Anxiety, DASS21_Depression, AMI_Behavioural, AMI_Social, AMI_Emotional, MFIS_Physical, MFIS_Cognitive, MFIS_Psychosocial, OASIS_Total, PHQ9_Total, STICSA_Total, STAI_Trait
- Affect features in master: mean_anxiety, mean_confidence, anx_calibration (within-subj r anxiety~T), anx_slope
- Existing replicated findings to NOT recapitulate: H5a anxiety-tracking → outcomes; H5c ω→confidence, ω→anxiety null; H6 ω/κ→clinical scales NULL.

## Replication standard

**A finding only counts if it replicates in both samples** with the same sign and reasonable effect size. Single-sample hits are reported but flagged as exploratory. CCA/PLS-style multivariate must be train-on-exp / test-on-conf with permutation null on the test side. Anything else doesn't survive review.

## Results synthesis

After all agents complete, the question is: **does at least ONE direction yield a replicated result that ties (ω, κ) to a metacognitive/affective dimension that in turn links to clinical phenotypes?** If yes, that becomes the paper's third leg. If no, the affect/clinical sections get demoted as previously discussed.
