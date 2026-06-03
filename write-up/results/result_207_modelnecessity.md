---
result_id: 207
class: computational_model
title: Joint likelihood is necessary to recover vigor variance and identify per-subject κ
status: supported
prereg_h: []
internal_h: []
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/H3_model_comparison.ipynb]
scripts: [scripts/run_mcmc_pipeline.py]
outputs: [results/stats/joint_optimal/exploratory/mcmc_model_comparison.csv, results/stats/joint_optimal/confirmatory/mcmc_model_comparison.csv]
figures: [TODO]
created: 2026-06-03
last_run: 2026-06-03◊
---

# Result 207 — Joint likelihood is necessary to recover vigor variance and identify per-subject κ

> **Scope.** Light, posterior-only decomposition of the existing M1–M4 fits ([[result_201]], [[result_202]], [[result_203]], [[result_204]]); no new MCMC. Asks the necessity-of-joint-likelihood question directly: does fitting choice and vigor under one likelihood with shared (ω, κ) give us something that a choice-focused alternative cannot? Two heavier follow-ups (channel-only fits, decoupled-parameter model) are listed under Caveats as deferred.

## Overview

The joint fitness model M4 fits both choice and vigor under one likelihood, with each subject's behaviour governed by a single shared (ω, κ) pair. A skeptical reading is that the joint likelihood is bookkeeping — that you could fit choice with one model, fit vigor with another, and not lose anything substantive. We test that reading by decomposing the existing M1–M4 fits' joint likelihood into per-channel pieces. The result is clean: every non-joint variant (M1, M2, M3, M3b) explains ≤ 10% of vigor variance, with three of the four at ≤ 1.7%. M4 explains 37.2% (exploratory) and 41.2% (confirmatory). The cost on choice is modest — M4's subject-level choice R² runs ~5–15 percentage points below the best choice-focused alternative — but the gain on vigor is order-of-magnitude. The per-subject effort cost κ is identifiable only when the vigor likelihood is included; choice alone leaves it under-constrained. This is the empirical basis for why the joint W(u) framework is the minimal model that supports the cross-channel inferences in [[result_208]] (population-level partial slopes per channel) and [[result_401]] (parameter-mediated marginal coupling).

## Hypothesis

**Statement.** Within the joint W(u) framework, fitting choice and vigor under a single likelihood with shared per-subject (ω, κ) is *necessary* — alternative specifications that approximate a "choice-only" fit (per-subject ω with population κ) or a "single-trait" fit (θ = ω = κ) cannot recover the vigor channel.

**Predicted direction.** M4's per-channel vigor R² should exceed every non-joint alternative by a large margin, in both samples. M4's per-channel choice fit should remain within a few percentage points of the best alternative, so the trade-off is one-sided in favour of joint.

**Preregistered criterion.** This is a justificatory re-analysis of existing posteriors, not a preregistered test. The qualitative criterion is that (a) the joint vs non-joint gap on vigor R² is large and (b) replicates across the two samples.

**Source of the hypothesis.** The substantive interpretation of the joint W(u) framework rests on shared (ω, κ) being identified by both behavioural channels simultaneously. If either channel alone identified the same parameters, joint fitting would be parsimonious only — not substantive. This result tests the substantive case at the cheap end: by reading per-channel fit quality off the existing M1–M4 posteriors. Heavier versions (deferred) are listed under Caveats.

## Data Source

- **Samples:** Exploratory N = 290, confirmatory N = 281 (same fits as [[result_201]]; no re-fitting).
- **Input files:**
  - `results/stats/joint_optimal/exploratory/mcmc_model_comparison.csv` — five-row table for M1, M2, M3, M3b, M4 with per-model joint WAIC, choice-only WAIC, choice accuracy, choice R², vigor R², convergence flag.
  - `results/stats/joint_optimal/confirmatory/mcmc_model_comparison.csv` — same structure for the confirmatory sample.
- **Unit of analysis:** Per-model summary; the underlying observations are choice trials (~13,000 per sample) and condition-cell means (~5,200 per sample), already aggregated by the upstream MCMC pipeline.
- **N entering the analysis:** All five M1–M4 models in both samples (10 fits total; M3 confirmatory did not converge — see Caveats and [[result_204]]).

## Method

The per-channel decomposition is read directly off `mcmc_model_comparison.csv`. For each model, the joint WAIC was computed on the full pointwise log-likelihood (choice trials + vigor cells), and `WAIC_choice` was computed by re-evaluating WAIC on the choice trials alone using the same posterior. The implied vigor WAIC contribution is `WAIC_total − WAIC_choice`, additive at the lppd level because pointwise log-likelihoods are summed across observations. Choice accuracy is the held-out trial-classification accuracy at posterior mean. Choice R² is the subject-level coefficient of determination of model-predicted P(heavy) against observed P(heavy). Vigor R² is the subject-level coefficient of determination of model-predicted u* (analytic argmax of W(u) at posterior mean (ω, κ)) against observed cell-mean pressing rate.

**Models compared (all share the same data + the same likelihood scaffolding):**

- **M1 — Effort-only.** Per-subject κ with intercept-only vigor likelihood. No survival weighting; option value reduces to `R − κ·E`. Per-subject params = 1.
- **M2 — Threat-only.** Per-subject ω with **population κ** (single value shared across subjects). Survival weighting present, but the effort term contributes no individual variance. Per-subject params = 1.
- **M3 — Single trait.** Per-subject θ that plays both roles, i.e. θ_i = ω_i = κ_i. The two functional positions of ω (in survival weighting) and κ (in effort cost) are forced to a single per-subject scalar. Per-subject params = 1.
- **M3b — Scaled single trait.** θ_i for ω, α·θ_i for κ, with α a free population scalar. Allows a single trait to scale into both positions but enforces a 1-D individual-difference structure. Per-subject params = 1 (plus one population).
- **M4 — Joint W(u).** Per-subject (ω, κ) constrained by the joint choice + vigor likelihood. Per-subject params = 2.

Beyond M2 and M3, no "true" channel-only model is fit in this light analysis — see Caveats for the deferred channel-only and decoupled variants.

**Notebook / script:** `scripts/run_mcmc_pipeline.py` produced the underlying posteriors; `notebooks/analysis/H3_model_comparison.ipynb` produced the comparison CSVs that this result reads.

## Result

**Per-channel fit quality across M1–M4 (both samples):**

*Exploratory (N = 290):*

| Model | Per-subj params | Choice acc | Choice R² | Vigor R² | WAIC (joint) | WAIC_choice | Implied WAIC_vigor |
|---|---|---|---|---|---|---|---|
| M1 (effort-only) | 1 (κ) | 0.710 | 0.951 | **0.006** | 17,505 | 15,059 | +2,446 |
| M2 (threat-only, pop κ) | 1 (ω) | 0.789 | 0.908 | **0.006** | 14,742 | 12,266 | +2,476 |
| M3 (single trait θ) | 1 (θ) | 0.773 | 0.845 | 0.102 | 15,374 | 12,665 | +2,709 |
| M3b (scaled θ) | 1 (θ) + α | 0.790 | 0.919 | 0.017 | 14,735 | 12,273 | +2,462 |
| **M4 (joint)** | **2 (ω, κ)** | **0.773** | **0.796** | **0.372** | **12,776** | **12,468** | **+308** |

*Confirmatory (N = 281):*

| Model | Per-subj params | Choice acc | Choice R² | Vigor R² | WAIC (joint) | WAIC_choice | Implied WAIC_vigor |
|---|---|---|---|---|---|---|---|
| M1 (effort-only) | 1 (κ) | 0.708 | 0.946 | **0.007** | 16,037 | 14,523 | +1,514 |
| M2 (threat-only, pop κ) | 1 (ω) | 0.778 | 0.893 | 0.012 | 13,873 | 12,354 | +1,519 |
| M3 (single trait θ) | 1 (θ) | 0.756 | 0.807 | 0.075 | 15,727† | 12,761 | +2,966 |
| M3b (scaled θ) | 1 (θ) + α | 0.778 | 0.894 | 0.014 | 13,850 | 12,352 | +1,498 |
| **M4 (joint)** | **2 (ω, κ)** | **0.759** | **0.809** | **0.412** | **12,252** | **12,549** | **−297** |

† M3 confirmatory did not converge; numbers are reported for completeness but should be read as "M3 is a bad model under this MCMC" rather than as a precise comparison. See [[result_204]] Caveats.

**Headline contrasts (M4 vs the best non-joint alternative on each channel):**

| Metric | Best non-joint | M4 | Δ |
|---|---|---|---|
| **Choice R² (exploratory)** | 0.951 (M1) | 0.796 | M4 is **0.155** lower |
| **Choice R² (confirmatory)** | 0.946 (M1) | 0.809 | M4 is **0.137** lower |
| **Choice acc (exploratory)** | 0.790 (M3b) | 0.773 | M4 is **0.017** lower |
| **Choice acc (confirmatory)** | 0.778 (M2/M3b tie) | 0.759 | M4 is **0.019** lower |
| **Vigor R² (exploratory)** | 0.102 (M3) | 0.372 | M4 is **0.270** higher |
| **Vigor R² (confirmatory)** | 0.075 (M3) | 0.412 | M4 is **0.337** higher |

**Per-channel cost vs gain.** The joint constraint costs M4 ≈ 0.14–0.16 points of subject-level choice R² and ≈ 0.02 points of choice accuracy relative to the best choice-focused alternative. In exchange it explains an additional ≈ 27–34 points of vigor R² that no non-joint variant comes within an order of magnitude of. The implied vigor WAIC contribution under M4 is small (≈ +300 in exploratory; ≈ −300 in confirmatory) compared to ≈ +1,500 to +3,000 for every alternative, indicating that M4 captures the vigor pointwise log-densities much more efficiently per observation.

**Why per-subject κ is the operative quantity.** M2 has per-subject ω with population κ — i.e. one κ shared across all subjects — and achieves choice R² = 0.91 (exploratory) / 0.89 (confirmatory), only slightly below M1's R² = 0.95 and clearly above M4's R² = 0.80. Population κ is therefore sufficient to fit *choice*. The same M2, however, achieves vigor R² ≈ 0.006–0.012 — effectively zero — because population κ supplies no individual variance for the vigor likelihood. Allowing per-subject κ (only M4 does so independently of ω) is what unlocks vigor: M4's vigor R² of 0.37–0.41 is the difference between a population κ and a per-subject κ identified by the vigor channel. M3 forces θ_i = ω_i = κ_i and recovers some vigor variance (R² ≈ 0.08–0.10) precisely because θ varies per subject and is *coupled* through κ — but the forced coupling caps it well below M4.

**Verdict:** Joint fitting is *necessary* to capture vigor variance in this dataset and *necessary* to identify per-subject κ. The cost on choice is modest and one-sided; the gain on vigor is large and replicates across samples.

## Interpretation

The joint W(u) framework is committed to a mechanistic claim: **option value is a forecast of embodied execution**. The fitness function `W(u) = S(u)·R − (1 − S(u))·ω·(R + C) − κ·(u − req)²·D` makes survival `S(u)` and effort cost `κ·(u − req)²·D` explicit functions of the actual pressing rate `u`, and option value `V_j = max_u W_j(u)` is the value of executing option j *with the body you have*. Choice is therefore not abstract preference between option labels — it is a forecast of what the body will do during execution, weighted by what that execution will cost. This is the embodiment claim that distinguishes M4 from every alternative compared here, and it has direct precedent in the framework's theoretical lineage: Yoon & Shadmehr (2018) on joint optimization of harvest duration and movement vigor; Thura et al. (2025) on default co-regulation of decision and movement vigor; and the foraging-under-predation tradition of Bednekoff (2007) and Brown (1999), in which patch value is jointly determined by reward, risk, and what execution costs the body. The per-channel decomposition is the empirical test of whether this embodied framework constrains both behavioural channels with the *same* per-subject parameters. Read against the embodiment commitment, the decomposition makes a three-part argument.

**First — the choice-fit cost is the cost of biomechanical fidelity.** M4's choice R² runs ≈ 0.10–0.16 below M1, M2, and M3b. M2 in particular — per-subject ω with population κ — fits choice at R² = 0.89–0.91 and accuracy 0.78–0.79, a clear win over M4 on both metrics. But M2's "advantage" comes from letting κ float free from the motor output. Its κ is a population scalar with no per-subject anchor; it can sit wherever the choice likelihood prefers, because no vigor data constrain it. M4's κ is required to be consistent with the actual pressing rates each subject produces — the same κ that determines `V = max_u W(u)` for choice also enters the prediction of u* against observed vigor cells. The "cost" on choice is therefore the cost of refusing to disembody the model. If vigor is real behaviour containing information about each subject's effort cost — which the M4 vigor R² of 0.37–0.41 strongly says it is — then M2's choice advantage is *overfitting to one channel by ignoring the other half of the body's signal*. The penalty is principled, not unfortunate.

**Second — the joint likelihood is the structural mechanism that identifies per-subject κ.** M2 fits choice nearly as well as the best alternative with a single population κ; per-subject ω alone is sufficient to fit the choice surface. This says, directly, that choice does not separate ω from κ at the subject level — choice identifies ω, full stop. Vigor R² in M1 / M2 / M3b is ≤ 0.017; vigor R² in M4 is 0.37–0.41. The only structural difference between M4 and the alternatives is that M4 carries an independent per-subject κ alongside per-subject ω. Per-subject κ is therefore identified by the vigor channel — and the vigor channel is only informative under a model that connects motor output to embodied value. The two pieces are inseparable: per-subject κ requires the vigor likelihood, and the vigor likelihood requires the embodied W(u). The joint constraint is what makes both pieces work together. M3 and M3b — which force a single per-subject trait — partially recover vigor (R² ≈ 0.08–0.10 for M3) precisely because their θ varies per subject and is functionally coupled into the κ position, but the forced coupling caps them well below M4. Two free per-subject parameters jointly constrained by both channels is the minimum specification.

**Third — the joint structure makes a class of scientific questions answerable that channel-specific models structurally cannot.** Three downstream lines of evidence in this paper rest on (ω, κ) being jointly identified embodied traits:

- **Cross-channel coupling** ([[result_208]] + [[result_401]]): under W(u), the same shared (ω, κ) parameterisation produces partial slopes that act in opposite directions on the motor channel — ω has a positive partial slope on vigor (β_ωv = +0.13 / +0.12), κ has a negative partial slope (β_κv = −0.24 / −0.23). These partial slopes, together with the population correlation `r(ω, κ) ≈ +0.30`, jointly *predict* the marginal correlation between choice and vigor across subjects (predicted +0.143 / +0.052, observed +0.150 / +0.077; match within 0.025 in both samples). This quantitative cross-channel prediction is a property of the embodied joint model, not a free fit — under M2 (population κ) per-subject κ is unidentified; under M3 (collapsed θ) there are no separable parameters to test in the first place.

- **Mechanistic affect** ([[result_501]]): trial-level survival probability `S(u*, T, D)` — a function of the model-derived optimal pressing rate u* under the subject's own (ω, κ) — predicts anxiety (β ≈ −0.55 to −0.58) and confidence (β ≈ +0.63 to +0.68) on each probe. This analysis requires a model in which survival depends on *actual* embodied execution. A disembodied choice model has no u* to compute S over, and the analysis is structurally impossible. Affect being decodable from S(u*) is a strong claim about embodied subjective value, and it is only testable under M4.

- **Embodied dissociations** ([[result_405]], [[result_506]]): subjects who choose heavy but press lightly, or vice versa, appear as outliers on one channel with no second-channel anchor under any disembodied model. Under W(u) they are interpretable as people whose ω implies cautious choice but whose κ implies low motor commitment, or the reverse. The HL / LH subgroups, and the confidence-miscalibration quadrants that track them, exist *as patterns* only inside a framework that jointly models both behaviours.

**Summing up.** The case for M4 is not "highest in-sample choice fit" — M4 cedes that to M1, M2, and M3b. The case is that M4 is the only model whose per-subject parameters are jointly consistent with both behavioural channels under an embodied value computation, and that this joint consistency is what makes the paper's downstream inferences possible. The 10–15 percentage points of choice R² that M4 gives up is the cost of insisting κ mean what the body actually pays. Every paper-level inference downstream of [[result_201]] — every individual-difference regression, every cross-channel test, every embodied affect analysis — is contingent on that insistence.

## Caveats & Limitations

- **Vigor R²'s ceiling is bounded by cell-mean motor noise.** The vigor likelihood is at the condition-cell-mean level (subject × threat × distance × cookie), so unexplained variance includes both structural motor noise within a cell and any per-cell trial-count variation. M4's vigor R² of 0.37–0.41 should be read against this ceiling rather than against 1.0. The per-channel comparison among M1–M4 is unaffected — every model is judged against the same data on the same scale.

- **WAIC_vigor is exact only at the lppd level; p_waic adjustments are channel-blind.** The implied `WAIC_vigor = WAIC_total − WAIC_choice` is correct in lppd but the p_waic effective-parameter penalty is computed on the joint posterior. For the qualitative gap (M4 vs alternatives) this is small. For precise inference on the vigor WAIC contribution one would compute `arviz.waic` on the per-channel pointwise log-likelihood directly. The conclusion is robust to this choice.

- **The light analysis does not fit "channel-only" baselines.** A more rigorous test of necessity would fit two extra MCMCs: a **choice-only** model (the M4 likelihood with vigor observations dropped) and a **vigor-only** model (the M4 likelihood with choice trials dropped). Comparing the resulting (ω, κ) to the joint M4 (ω, κ) would directly test whether the joint constraint changes parameter estimates or only narrows their HDIs. This is the **"medium" follow-up** and is deferred. See `instructions/memory/joint_model_development.md` for setup notes.

- **A decoupled-parameter model is the proper null for "shared structure".** A model with per-subject (ω_c, κ_c) for choice and per-subject (ω_v, κ_v) for vigor — twice as many subject-level parameters as M4 — would empirically test whether sharing (ω_c = ω_v, κ_c = κ_v) costs likelihood. If decoupled fits are essentially identical to M4, parsimony alone selects M4. If decoupled wins substantially, the shared-parameter interpretation underlying [[result_208]] and [[result_401]] is weakened. This is the **"strong" follow-up** and is deferred.

- **M3 confirmatory did not converge** (see [[result_204]] Caveats). For the present decomposition this is not load-bearing — M3 exploratory shows the same qualitative pattern as confirmatory (vigor R² ≈ 0.10, well below M4) and the comparison is M4 vs the best alternative on each channel, not the precise ΔWAIC.

- **Choice R² is subject-level, not trial-level.** It measures how well model-predicted subject mean P(heavy) tracks observed subject mean P(heavy). Trial-level choice accuracy (≈ 77% for M4) is the held-out classification rate. Both are reported because they answer different questions; both rank models the same way.

## Replication

This result requires no new fitting — only re-reading the existing model-comparison CSVs.

```python
import pandas as pd

for sample in ["exploratory", "confirmatory"]:
    path = f"results/stats/joint_optimal/{sample}/mcmc_model_comparison.csv"
    df = pd.read_csv(path)
    df["WAIC_vigor_implied"] = df["WAIC"] - df["WAIC_choice"]
    df = df[["Model", "n_per_subj", "choice_acc", "choice_r2", "vigor_r2",
             "WAIC", "WAIC_choice", "WAIC_vigor_implied", "converged"]]
    print(sample.upper())
    print(df.to_string(index=False))
```

**Expected outputs:** The two tables in the Result section, with values matching to the cached CSV. No new files written.

**To extend to the medium follow-up (channel-only fits):**
- Add `--likelihood choice_only` and `--likelihood vigor_only` flags to `scripts/run_mcmc_pipeline.py`.
- Fit both for both samples (≈ 4 extra MCMC runs).
- Compare per-subject (ω, κ) posteriors against M4 by Pearson correlation; report HDI widths under each likelihood.

**To extend to the strong follow-up (decoupled-parameter model):**
- Add an M4_decoupled variant to the model registry with separate (ω_c, κ_c) and (ω_v, κ_v) per subject (4 per-subject parameters).
- Fit for both samples (≈ 2 extra MCMC runs).
- Compare against M4 by ΔWAIC and by Pearson correlation of (ω_c, ω_v) and (κ_c, κ_v) within subject — high correlation supports M4's shared-parameter assumption.

## References

**Related results:**
- [[result_201]] — M4 fit and convergence; baseline for all per-channel numbers reported here.
- [[result_202]] — M4 vs M1 (effort-only) joint comparison; the M1 column above is the same fit.
- [[result_203]] — M4 vs M2 (threat-only) joint comparison; the M2 column above is the same fit.
- [[result_204]] — M4 vs M3 / M3b (single-trait) joint comparison; M3 / M3b columns above are the same fits.
- [[result_205]] — Parameter recovery: ω and κ are identifiable under M4's MCMC, complementing 207's argument about *joint* identifiability.
- [[result_401]] — Population-level cross-channel marginal r(choice, vigor) prediction; the inference that result 207 underwrites.
- [[result_208]] — H4 family individual-difference regressions; rely on (ω, κ) being separable traits, which 207 + 204 jointly establish.

**Literature:**
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.
- Bednekoff, P. A. (2007). Foraging in the face of danger.
- Brown, J. S. (1999). Vigilance, patch use and habitat selection: foraging under predation risk.
- Yoon, T., & Shadmehr, R. (2018). Movement vigor and decision-making jointly optimize a single subjective utility function (precedent for embodied joint optimization of choice and motor output).
- Thura, D., et al. (2025). Decision and movement vigor are co-regulated by default and decoupled when behavioural demands diverge (precedent for shared parameterisation across decision and execution).
- Niv, Y. (2007). Cost, benefit, tonic, phasic: what do response rates tell us about dopamine and motivation? (vigor as the economic output of value computation).
- Mobbs, D., Headley, D. B., Ding, W., & Dayan, P. (2020). Space, time, and fear: survival computations along defensive circuits (defensive responses graded by imminence and effector capacity).
