---
name: Next Steps
description: Priority action items for the paper — Bayesian vigor model, joint HBM, factor analysis, confirmatory sample
type: project
---

# Next Steps (as of 2026-03-19)

## Priority 1: Bayesian vigor model (α, ρ)

**Goal:** Replace summary-statistic α and ρ with hierarchical Bayesian estimates.

**Model specification:**
```
# Per trial j for subject i:
rate_ij ~ Normal(μ_ij, σ_i)
μ_ij = α_i + ρ_i · attack_ij

# Hierarchical priors:
α_i ~ Normal(μ_α, σ_α)
ρ_i ~ Normal(μ_ρ, σ_ρ)
σ_i ~ HalfNormal(τ)

# Population priors:
μ_α ~ Normal(1, 1)
μ_ρ ~ Normal(0, 0.5)
σ_α, σ_ρ ~ HalfNormal(1)
```

**Data:** Pre-encounter rate (enc−2s to enc) for α; terminal rate (last 2s) for ρ. Both capacity-normalized, choice-ratio adjusted. Need to handle the two windows carefully — α and ρ come from different trial phases.

**Benefits:**
- Shrinkage improves ρ estimates (currently SB=0.46 from point estimates; HBM should improve)
- Posterior distributions give uncertainty per subject
- Population-level μ_ρ formally tests whether the group sprint effect is nonzero
- Can include threat as covariate on ρ to control for attack-threat confound within the model

**Implementation:** NumPyro, 4 chains × 1000+1000, ~1 min on CPU.

**Status:** ✅ Complete (2026-03-20). See `notebooks/03_vigor_analysis/16_bayesian_vigor_model.ipynb`.

**Final model (two-window, separate likelihoods):**
```
pre_enc_rate_ij  ~ Normal(α_i, σ_pre)          # [enc-2, enc]
terminal_rate_ij ~ Normal(γ_i + ρ_i·attack_ij, σ_term)  # [trialEnd-2, trialEnd]

α_i ~ Normal(μ_α, σ_α)   # tonic approach vigor
γ_i ~ Normal(μ_γ, σ_γ)   # nuisance terminal baseline
ρ_i ~ Normal(μ_ρ, σ_ρ)   # phasic attack boost
```

**Key results:**
- μ_α=0.519 [0.497, 0.541], μ_ρ=0.526 [0.509, 0.542], P(μ_ρ>0)=1.0000
- Bayes-OLS: α r=1.000, ρ r=0.991
- Split-half: α SB=0.925, ρ SB=0.762
- Shrinkage: α 2.1%, ρ 16.8%
- α-ρ correlation: r=−0.237 (fast pressers have smaller sprint boost, ceiling effect)
- 0 divergences, max Rhat=1.006

**Model iterations (what was tried):**
1. Terminal-only model (single DV): ρ excellent (SB=0.76) but α captured terminal idling (r=−0.56 with pre-enc)
2. Enc-aligned two-window [enc-2, enc] + [enc, enc+2]: α great but attack effect too small in first 2s post-enc (ρ SB=0.28)
3. **Final: separate windows** — pre-enc for α, terminal for ρ with nuisance γ. Best of both.

**Outputs:** `results/stats/vigor_hbm_posteriors.csv`, `vigor_hbm_population.csv`, `results/model_fits/exploratory/vigor_hbm_idata.nc`

---

## Priority 2: Factor analysis on psychiatric battery

**Goal:** Reduce 16 intercorrelated psychiatric measures to 2-4 latent factors, then test 5 params → factors.

**Rationale:** Individual scales are noisy and intercorrelated (DASS subscales r=0.6-0.8). FDR correction across 16 tests is overcorrecting because the effective number of independent tests is ~3-4. Factor analysis solves this by creating orthogonal, more reliable latent dimensions.

**Expected factors:** General distress (DASS/PHQ/OASIS), apathy/motivation (AMI), somatic anxiety (STICSA/MFIS), possibly trait anxiety (STAI)

**Analysis:**
1. EFA on the psychiatric battery (N=293)
2. Extract 2-4 factors
3. 5 params → factor scores (simple regression, only 2-4 tests)
4. If still weak: PLS on 5 params → factor scores

**Status:** ✅ Complete (2026-03-20). See `notebooks/04_psych_analysis/06_factor_analysis.ipynb`.

**Results:** 3 factors (distress 37%, fatigue 20%, apathy 12%). α → apathy factor: R²=0.155, p=3×10⁻⁹. Choice params predict nothing. Distress/fatigue orthogonal to all task params.

---

## Priority 3: Full MCMC fit of unified 3-param model

**Goal:** Refit the unified choice model (Model C) with full MCMC for proper WAIC, posteriors, and PPC.

**Model specification (unified):**
```
# Choice:
S_i = exp(-λ · T · (D / α_i)^z)     # α enters survival function
SV = R·exp(-k_i·E)·S - (1-S)·C + β_i·(1-S)
choice ~ Bernoulli(softmax(τ · ΔSV))

# Vigor (already fit):
pre_enc_rate ~ Normal(α_i, σ_pre)
terminal_rate ~ Normal(γ_i + ρ_i·attack, σ_term)
```

k_i and β_i are per-subject (non-centered). λ, z, τ are population-level. α_i is observed (from vigor HBM).

**Final model (L4a_add):**
```
SV = R·S - k·E - β·(1-S)
S = (1-T) + T / (1 + λ·D/α)
```
- Additive effort, hyperbolic escape, mechanistic S
- Best by ELBO (−6260) and BIC across 12 models (SVI, NB03-choice)
- k-β r=−0.11, k-α r=−0.08, β-α r=+0.14 — all cleanly identified

**What MCMC adds:** Proper WAIC, posterior uncertainty, parameter recovery, posterior predictive checks.

**Status:** 🔲 Not started. SVI proof-of-concept complete. Estimates saved in `results/stats/unified_3param_clean.csv`.

---

## Priority 4: Confirmatory replication (N=350)

**Goal:** Run the full pipeline on the confirmatory sample.

**Steps:**
1. Preprocess confirmatory data through stages 1-5
2. Fit FET choice model → k, z, β
3. Compute α and ρ from vigor data
4. Run Phase 0-6 dissociation analyses
5. Compare exploratory vs confirmatory results

**Key replication targets:**
- Choice-vigor independence (r ≈ 0)
- 4 quadrant profiles with same parameter signatures
- Threat reversal of choice-vigor coupling
- Confidence miscalibration pattern
- α → AMI link

**Status:** 🔲 Not started. Data exists in `data/confirmatory_350/raw/` but not preprocessed.

---

## Supplementary analyses (lower priority)

- **RT analysis:** First-press latency as supplementary validation of α (r=−0.81)
- **PLS/elastic net:** 5 params → mental health (multivariate approach if factor analysis is insufficient)
- **Encounter-centered vigor:** Threat/attack effects on pre/post pressing (NB13)
- **Vigor variance budget:** Full decomposition table (person/conditions/noise)
- **Window comparison:** Table showing α and ρ reliability across all operationalizations tested
- **Affect ~ S_probe:** Model-derived survival predicts anxiety/confidence (existing NB12 results)
