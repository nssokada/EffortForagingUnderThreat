---
result_id: 303
class: vigor_dynamics
title: Multivariate vigor features from the encounter window predict model parameters via PLS
status: supported_exploratory
prereg_h: []
internal_h: [H23]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 303 — Multivariate vigor features from the encounter window predict model parameters via PLS

> **⚠️ Exploratory, deprecated framework.** Originally derived using the deprecated three-parameter FET model (z, k, β). Re-analysis with current (ω, κ) parameters not yet done. Numbers from `instructions/memory/hypotheses.md` § H23 (last validated 2026-03-24).

## Overview

If the joint-model parameters are mechanistically meaningful, they should be partially recoverable from richer vigor features than simple cell-mean rates. We extracted 12 count-based features from the encounter window (encounterTime − 2s to encounterTime + 2s) per subject and used partial least squares (PLS) regression to predict z-scored model parameters, with 5-fold cross-validation and permutation testing. A 2-component PLS achieved cross-validated R² = 0.093 (perm p = 0.000) for the multivariate parameter target, with the per-parameter breakdown showing k (effort cost) as the most predictable from vigor features (CV R² = 0.199) and β (threat bias) as the least (CV R² = −0.039).

## Hypothesis

**Statement.** "A multivariate set of vigor features from the encounter window predicts choice model parameters with cross-validated generalization." (Internal H23, corrected encounter-time frame.)

## Data Source (legacy)

- **Sample:** N = 290 exploratory.
- **Inputs:** 12 count-based features per subject computed in a ±2s window around encounterTime; deprecated (z, k, β) parameters as targets.

## Method (legacy)

- PLS regression (sklearn), 2 and 3 components.
- 5-fold cross-validation.
- 1,000-permutation p-value on multivariate CV R².

## Result (legacy, from internal H23)

| Configuration | Train R² | CV R² | Perm p |
|---|---|---|---|
| 2 components (multivariate) | 0.144 | 0.093 | 0.000 |
| 3 components (multivariate) | 0.162 | 0.117 | (not reported) |
| 2 components, k only | — | **0.199** | — |
| 2 components, z only | — | 0.072 | — |
| 2 components, β only | — | −0.039 | — |

**For comparison:** A naive 20Hz pixel-level PLS on the raw vigor timecourse gave CV R² = −0.071 (overfit).

**Verdict:** Multivariate count-based features in the encounter window yield genuine out-of-sample generalization for k specifically; β is not predictable from vigor features in this window.

## Interpretation

Vigor features in the encounter window carry information about how subjects internalize the joint-model parameters, but the per-parameter breakdown is informative: k (effort cost) is partially recoverable (CV R² ≈ 0.20), z (hazard sensitivity) is weakly so (CV R² ≈ 0.07), and β (threat bias) is not at all (CV R² < 0). The asymmetry is consistent with the parameter's behavioral signatures: k governs motor execution directly, z governs how rapidly subjects scale up to encounter, and β operates primarily through choice with little vigor footprint.

The result is methodologically important because it validates the approach of using vigor features as a *measurement target* for individual differences in model parameters. The 20Hz raw-timecourse PLS overfits (CV R² = −0.071), confirming that the count-based feature engineering captures the signal more cleanly than the raw data — a standard finding in motor analysis where event-aligned summaries beat raw streams.

The result has not been re-derived using the current (ω, κ) parameter pair. Under the current M4 model, the analog test would be PLS from encounter-window vigor features to (ω, κ). Given the substantive similarity of κ (current) to k (deprecated) — both index effort cost — we expect the κ predictability to be comparable or stronger. The β finding is moot since β doesn't exist in M4.

## Caveats & Limitations

- **Status: `supported_exploratory`.** Re-run with current (ω, κ) targets needed.
- **The corrected-frame encounter-time alignment was a methodological fix** for an earlier wrong-frame analysis that gave different feature correlations (see `instructions/memory/hypotheses.md` § H25 for the retracted version). The numbers above use the corrected frame.
- **The k effect (CV R² = 0.199) is the strongest single vigor → parameter result in the project.** This deserves a dedicated panel in any vigor-focused figure.
- **CV R² = 0.093 multivariate is modest** but reliably non-zero (perm p = 0.000); the per-parameter decomposition is where the interpretable signal lives.

## Replication

Migration needed. The corrected-frame feature extraction code should be locatable in `notebooks/_deprecated/old_vigor_analysis/`. Steps:

1. Port feature extraction to a current notebook (likely `H2_vigor_dynamics.ipynb` cell extension).
2. Replace (z, k, β) target with (ω, κ) from `mcmc_m4_params.csv`.
3. Re-run 5-fold CV PLS with permutation test.
4. Update file with current-model results.

## References

**Related results:**
- [[result_304]] — Distance modulation of pre-encounter pressing → k (the single-feature companion).
- [[result_307]] — Phase dissociation by parameters.
- [[result_201]] — Joint model M4 fit (source of new targets).

**Source:**
- `instructions/memory/hypotheses.md` § H23 (last validated 2026-03-24).
