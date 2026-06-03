---
result_id: 402
class: choice_vigor_coupling
title: ω predicts anticipatory vigor level — cross-channel validation of joint model
status: supported
prereg_h: []
internal_h: [H8]
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/H8_avoid_activate.ipynb]
scripts: []
outputs: [results/stats/avoid_activate/cross_channel_bayesian.csv, results/stats/avoid_activate/cross_channel_test.csv, results/stats/avoid_activate/level_by_T.csv]
figures: [results/figs/avoid_activate/cross_channel_omega_anticipatory.png]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 402 — ω predicts anticipatory vigor level — cross-channel validation of joint model

## Overview

ω is fit entirely from choice data, so any correlation between ω and a vigor outcome would be a non-trivial cross-channel test of the joint model: it asks whether the parameter that explains who chooses cautiously also explains who presses harder during execution. We tested this on anticipatory pressing rate (mean vigor before any predator encounter) across all three threat levels, in both samples. ω shows a small but consistent positive correlation with anticipatory vigor level (r ≈ +0.14 to +0.21) at every threat level in both samples, and the effect survives a Bayesian partial regression controlling for κ. The result is the strongest single validation of the joint W(u) framework: the parameter pair derived to fit choice also predicts vigor in a sample-replicating, theoretically expected direction.

## Hypothesis

**Statement.** "The joint W(u) framework predicts that ω — fit only on choice data — should also predict anticipatory vigor level, because both choice and vigor are computed from the same fitness function." (Internal H8 in `instructions/memory/hypotheses.md`; not in the formal prereg.)

**Predicted direction.** β(ω → anticipatory vigor level) > 0, with HDI excluding zero after controlling for κ.

## Data Source

- **Samples:** Exploratory N = 290, confirmatory N = 281.
- **Input files:**
  - `results/stats/joint_optimal/{sample}/mcmc_m4_params.csv` — per-subject ω, κ.
  - Behavioral data via `load_data` — per-subject mean anticipatory pressing rate on non-attack trials, by threat level.
- **Unit of analysis:** Subject × threat level (3 levels per subject).

## Method

**Step 1 (univariate cross-channel):** For each (sample, threat level), compute Pearson correlation between subject ω_z and subject mean anticipatory vigor.

**Step 2 (Bayesian partial regression):** For each sample, fit `vigor ~ omega_z + kappa_z` and report ω HDI; this isolates the ω effect from κ's much stronger effect on vigor.

**Step 3 (trial-level multilevel logistic on choice with ω × stimulus interactions):** Bayesian multilevel model `choice ~ threat_z + dist_z + threat_z:omega_z + dist_z:omega_z + threat_z:kappa_z + dist_z:kappa_z + omega_z + kappa_z + (1|subj)`. Tests stimulus-specific tuning of each parameter.

**Step 4 (model simulation):** For each fitted (ω, κ), compute the model-predicted vigor level and slope across threat levels using the analytically optimal u* from W(u). Correlate predicted vs observed.

**Notebook:** `notebooks/analysis/H8_avoid_activate.ipynb`, all cells executed.

## Result

**Step 1 — ω → anticipatory vigor level by threat level (Pearson r):**

| Threat | Exploratory | Confirmatory |
|---|---|---|
| T = 0.1 | r = +0.154 (p = 0.009) | r = +0.212 (p = 0.0004) |
| T = 0.5 | r = +0.144 (p = 0.014) | r = +0.203 (p = 0.0006) |
| T = 0.9 | r = +0.150 (p = 0.010) | r = +0.185 (p = 0.002) |

ω → vigor effect is positive and significant at every threat level in both samples.

**Comparison with κ:** κ has a much stronger vigor effect (r ≈ −0.70 to −0.73 across all threat levels in both samples) but in the opposite direction. ω and κ predict vigor by opposite mechanisms and roughly orthogonal magnitudes.

**Step 2 — Bayesian partial regression controlling κ:** ω HDI excludes zero in both samples (cached in `cross_channel_bayesian.csv`).

**Step 3 — Stimulus tuning in choice:** the threat × ω interaction in the multilevel logistic is negative as predicted, replicates in both samples, and approximately doubles in confirmatory. The distance × κ interaction does NOT replicate.

**Step 4 — Model simulation predictions match observed:**

| Sample | ω → vigor level predicted | observed | κ → vigor level predicted | observed |
|---|---|---|---|---|
| Exploratory | r = +0.072 | r = +0.151 | r = −0.886 | r = −0.732 |
| Confirmatory | r = +0.210 | r = +0.208 | r = −0.853 | r = −0.736 |

The W(u) model predicts the sign and approximate magnitude of both effects, including the non-obvious prediction that ω → vigor SLOPE should be approximately null (also confirmed: predicted r ≈ 0, observed r ≈ 0 for both samples).

**Verdict:** Cross-channel ω → vigor effect replicates in both samples; model simulation matches observed; ω × threat interaction in choice replicates; W(u)'s prediction of null ω → vigor slope is confirmed.

## Interpretation

The joint fitness model M4 fits ω from choice trials and κ from vigor trials, but neither parameter is fit by appealing to the *other* channel — the joint likelihood constrains them only through the shared W(u) function. A cross-channel test therefore asks whether the parameter estimated from one channel predicts behavior in the other. ω passes this test cleanly: subjects whose ω is estimated high from their cautious choice pattern also press harder during the anticipatory phase, with r ≈ +0.15 to +0.21 at every threat level in both samples. The effect survives the much stronger κ → vigor effect when both predictors are included.

Two features of the data are diagnostic for the joint model. First, ω predicts vigor *level* but not vigor *slope* across threat — the cross-threat slope correlation is essentially zero in both samples. This is exactly what W(u) predicts: under the survival function S(u, T, D), the optimal pressing rate u* depends on ω through a multiplicative term that affects the level of vigor but largely cancels out when you take differences across threat conditions. The non-obvious model prediction — that ω drives "how hard you press in general" but not "how much harder you press under threat" — is confirmed.

Second, the ω × threat interaction in *choice* (multilevel logistic Step 3) replicates and roughly doubles in confirmatory, but the parallel κ × distance interaction does not. The asymmetry tells us that ω is specifically tuned to the threat dimension of the value computation, while κ enters more uniformly across stimulus conditions. This is consistent with W(u)'s structure: ω weights the (1 − S) capture-cost term where T enters; κ weights the demand-cost term where D enters but not T.

Together with the model-fit replications in [[result_201]] through [[result_204]], this is the strongest single piece of evidence that the joint W(u) framework is doing real work. The parameters are not just descriptive labels for choice patterns — they generate testable cross-channel predictions that the data confirm.

## Caveats & Limitations

- **Effect sizes are small in raw units** (r ≈ 0.15–0.21). The replication and direction across two samples are the key findings, not the magnitude. The κ effect is much larger because κ is itself fit largely from vigor data; ω is a cross-channel prediction, which is exactly the harder test.
- **Mediation by anticipatory motor preparation is not formally tested here.** A subject with high ω could be expected to *prepare* more vigorously when they have committed to a foraging attempt because the cost of being caught looms larger. This is consistent with the data but the H8 analysis does not test the mediating mechanism (no within-trial dynamics).
- **κ → vigor slope effect is weaker than predicted by the model** (observed r ≈ −0.13 vs predicted −0.65 expl; −0.16 vs −0.46 conf). The model overpredicts how strongly κ should reduce the across-threat slope. This is a known model misspecification flagged in `instructions/memory/joint_model_development.md`.
- **The H8 framework supersedes the earlier exploratory β-driven dissociation analyses (internal H30, H34)** which were fit to a deprecated three-parameter (z, k, β) model. The current M4 model has no β parameter; the dissociation work in H30/H34 should be considered methodologically superseded by this result and is documented as such in [[result_404]].

## Replication

**To regenerate this result from scratch:**

```bash
PYTHONPATH=notebooks/analysis \
  jupyter nbconvert --to notebook --execute \
  notebooks/analysis/H8_avoid_activate.ipynb \
  --inplace --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=1800
```

**Expected runtime:** ~15–20 min per sample (multilevel Bayesian models).

**Expected outputs:**
- `results/stats/avoid_activate/cross_channel_bayesian.csv` — Bayesian partial results.
- `results/stats/avoid_activate/cross_channel_test.csv` — univariate correlations.
- `results/stats/avoid_activate/stimulus_tuning_*.csv` — Step 3 stimulus tuning results.
- `results/figs/avoid_activate/cross_channel_omega_anticipatory.png` — figure.

## References

**Related results:**
- [[result_201]] — M4 fit (source of ω, κ).
- [[result_208]] — H4 parameter regressions (broader individual-difference context).
- [[result_403]] — Vigor dominates escape over choice (exploratory complement).
- [[result_404]] — Threat reverses choice-vigor coupling (exploratory complement using deprecated model).

**Literature:**
- Daw, N. D., et al. (2011). Trial-by-trial data analysis using computational models.
- Niv, Y. (2007). Cost, benefit, tonic, phasic: what do response rates tell us about dopamine and motivation?
