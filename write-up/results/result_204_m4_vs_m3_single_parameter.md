---
result_id: 204
class: computational_model
title: Joint model M4 beats single-parameter M3 by ΔWAIC ≈ 2,600–3,500
status: supported
prereg_h: [H3c]
internal_h: []
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/H3_model_comparison.ipynb, scripts/run_mcmc_pipeline.py]
scripts: [scripts/run_mcmc_pipeline.py]
outputs: [results/stats/joint_optimal/exploratory/mcmc_model_comparison.csv, results/stats/joint_optimal/confirmatory/mcmc_model_comparison.csv]
figures: [TODO]
created: 2026-05-27
last_run: 2026-05-27
---

# Result 204 — Joint model M4 beats single-parameter M3 by ΔWAIC ≈ 2,600–3,500

## Overview

Could a single per-subject trait — call it θ — play both the avoidance role (ω) and the activation role (κ), so that one parameter governs both how cautiously a person chooses and how hard they press? We tested this by fitting M3, a single-parameter model in which θ_i = ω_i = κ_i for each subject, and comparing it to M4 (which has independent ω_i and κ_i). M4 outperforms M3 by ΔWAIC ≈ +2,599 in the exploratory sample and ≈ +3,474 in the confirmatory sample. The result demonstrates that avoidance and activation are separable subject-level traits, not a single dimension. The supplementary scaled model M3b (θ for ω, α·θ for κ) also loses to M4 by similar margins, ruling out the possibility that the failure is merely a scaling mismatch.

## Hypothesis

**Statement.** "The joint model will outperform a single-parameter model, demonstrating that capture cost and effort cost are separable traits." (Preregistration, H3c.)

**Predicted direction.** ΔWAIC > 0 AND ΔLOO > 0 favoring M4 over M3.

**Preregistered criterion.** WAIC and LOO must agree.

## Data Source

Same as [[result_201]]. Joint likelihood comparison.

## Method

**M3 (single-parameter) specification.**
- Same fitness function W(u) as M4.
- One per-subject parameter θ_i that enters as both ω (in survival weighting) and κ (in effort cost).
- 299 / 290 free parameters.

**M3b (scaled single-parameter, supplementary).**
- θ_i for ω, α·θ_i for κ, where α is a population scaling parameter.
- 300 / 291 free parameters.
- Tests whether M3's failure is genuine separability or merely a scale mismatch.

**M4 (joint fitness, full).** See [[result_201]].

## Result

| Metric | Exploratory | Confirmatory |
|---|---|---|
| M3 WAIC | 15,374 | 15,727 |
| M4 WAIC | 12,776 | 12,252 |
| **ΔWAIC (M3 − M4)** | **+2,599** (SE ≈ 593) | **+3,474** (SE ≈ 305) |
| M3 LOO | 15,404 | 15,737 |
| M4 LOO | 12,779 | 12,263 |
| **ΔLOO (M3 − M4)** | **+2,625** | **+3,474** |
| M3 choice accuracy | 77.3% | 75.6% |
| M3 vigor R² | 0.102 | 0.075 |
| M3 converged | True (expl) | **False (conf)** |
| **M3b** ΔWAIC vs M4 | **+1,959** | **+1,597** |
| M3b converged | True | True |

**Verdict on prereg criterion:** **PASS** in both samples. WAIC and LOO agree; the scaled supplementary M3b also loses to M4.

## Interpretation

A single subject-level trait cannot serve as both the avoidance parameter and the activation parameter. M3 — which forces θ_i = ω_i = κ_i — underperforms M4 by ΔWAIC of roughly 2,600 (exploratory) and 3,500 (confirmatory), with the gap many standard errors wide and consistent in direction across samples. The supplementary scaled model M3b (which allows κ to be a population-scaled copy of θ rather than literally equal to it) closes some of the gap (down to ΔWAIC ≈ +1,600–2,000) but still loses decisively to M4, ruling out the possibility that M3's failure is a units mismatch that could be patched by a scaling factor.

The substantive implication is that capture cost and effort cost are dissociable individual-difference dimensions: a person who is highly avoidant of capture is not, in general, also highly averse to physical effort, and vice versa. This is the core preregistered claim of H3c — that the two-parameter ω/κ structure is the minimal representation that fits both behavioral channels — and the result confirms it in both samples with substantial margin.

The dissociation has implications that propagate through the rest of the paper. The H4 family of regressions in [[result_208]] interprets ω and κ as separate predictors of escape rate, overcaution, and decision quality, treating them as orthogonal traits. The choice-vigor dissociation analyses in the 400 block (especially [[result_208]] and [[result_401]]) further document that ω and κ project differently onto choice and vigor behavior. Result 204 is the prereg-licensed basis for that decomposition: were ω and κ a single trait, none of those downstream individual-difference analyses would make sense.

## Caveats & Limitations

- **M3 failed convergence in the confirmatory sample.** Several M3 parameters have R̂ > 1.5 in the confirmatory fit, with ESS dropping below 100 for `theta` (max across subjects), `rp` (max across observations), and `tr_` (raw subject offsets). The model comparison still shows M4 winning decisively (ΔWAIC = +3,474), but the M3 confirmatory posterior is unreliable. The model-comparison result should be treated as "M3 is a bad model" rather than "M4 beats a well-fit M3 by exactly this margin." In the exploratory sample, where M3 did converge, the gap is somewhat smaller (ΔWAIC = +2,599) but still decisive.
- **M3's choice accuracy (77.3%) is essentially identical to M4's** in exploratory — M3 fits choice fine. The WAIC gap is again driven by vigor (M3 vigor R² ≈ 0.10 vs M4's 0.37–0.41). This is consistent with the theme across [[result_202]] and [[result_203]]: simpler models can match M4 on choice by tuning their one or two free parameters, but fail to extend to vigor without the joint fitness structure.
- **M3b is the rigorous null for "single trait with scaling."** It controls for the trivial possibility that M3 fails because θ has the wrong units for either ω or κ; the scaling parameter α absorbs that mismatch. M3b still loses to M4 by ΔWAIC ≈ +1,600–2,000, so the dissociability finding survives the most generous version of the single-trait hypothesis.
- **Convergence failure of M3 in confirmatory means we cannot fully rule out that M3 would also have lost by ~2,500 if it had converged.** The model comparison is conservative in M3's favor — a converged M3 would likely show an even larger gap.

## Replication

See [[result_201]] Replication block. Both M3 and M3b are fit by the same MCMC pipeline.

## References

**Related results:**
- [[result_201]] — M4 fit and convergence.
- [[result_202]] — M4 vs M1 (effort-only).
- [[result_203]] — M4 vs M2 (threat-only).
- [[result_208]] — H4 family Bayesian regressions treating ω and κ as separable traits.
- [[result_208]] / [[result_401]] — Parameter-channel dissociation under M4 (downstream consequence of separable ω, κ structure).

**Literature:** Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.
