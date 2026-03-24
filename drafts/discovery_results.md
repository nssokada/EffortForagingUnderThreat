# Discovery Results — Effort Reallocation Under Threat

**Date:** 2026-03-23
**Sample:** Exploratory N=293, 13,185 choice trials, 10,546 affect ratings
**All numbers MCMC-verified** (4 chains × 4,000 samples, all Rhat = 1.00, zero divergences)

---

## 1. Choice: Survival-weighted value governs foraging decisions

**Winning model (M5):** SV = R·S − k·E − β·(1−S), S = (1−T) + T/(1+λD)

- Best of 5 models: additive effort >> multiplicative (+158 ELBO), hyperbolic >> exponential (+174 ELBO), survival >> effort-only (+2,038 ELBO)
- Accuracy 76.1%, AUC 0.863
- k (effort sensitivity): median = 4.25, M = 6.22, right-skewed
- β (threat bias): median = 53.5, M = 81.8, right-skewed
- k–β uncorrelated (r = −0.02, p = .70) — independent dimensions
- λ = 14.0 (fixed; weakly identified from 3 distance levels; Supp Note 1)
- τ = 0.76 (inverse temperature)
- Effort is a flat physical cost, not a reward discount

## 2. Vigor + Affect: S governs motor effort and subjective experience

### Vigor
- μ_δ = +0.211 [95% CI: 0.19, 0.23], 98.3% of subjects positive (uninformative prior centered at zero)
- σ_δ = 0.146 (individual differences recoverable)
- Safest → most dangerous: excess effort shifts by ~20% of motor capacity
- Within-choice control (constant demand): z = 3.54, p < .001
- Threat × distance interaction mirrors S structure (z = −2.12, p = .034)
- Split-half reliability SB ρ = 0.451

### Affect
- S → anxiety: β = −0.281, z = −24.1, p < .001 (LMM, N = 5,274)
- S → confidence: β = +0.280, z = +23.7, p < .001 (N = 5,272)
- Cross-domain: choice threat shift × anxiety shift r = −0.386; × confidence shift r = +0.372

## 3. Coordinated effort reallocation

### Behavioral coupling
- Choice shift × vigor shift: r = −0.78, p < .001
- Cross-validated (odd/even): mean r = −0.55, both p < 10⁻²²

### Independent Bayesian models (MCMC-validated)
| Parameter pair | MCMC r | p | Bootstrap r | 95% CI |
|---|---|---|---|---|
| **log(β) × δ** | **+0.53** | **< 10⁻²²** | **+0.32** | **[+0.23, +0.40]** |
| **log(k) × δ** | **−0.33** | **< 10⁻⁸** | **−0.25** | **[−0.32, −0.18]** |
| log(k) × α | +0.35 | < 10⁻⁹ | +0.30 | [+0.25, +0.35] |
| log(β) × α | +0.07 | n.s. | — | — |
| α × δ | −0.01 | n.s. | — | — |

Bootstrap attenuation (0.53 → 0.32) reflects β measurement noise; true coupling likely stronger.

### SVI joint model (robustness, Supp Note 2)
All 6 pairwise ρ CIs exclude zero at every λ tested. Direction invariant; magnitude λ-dependent.

### Outcome prediction
- All 4 params → total reward: R² = 0.321 (α t=7.21, δ t=6.71, k t=−3.97, all p < .001)

## 4. Metacognitive-motor bridge

| Calibration measure | δ correlation | p | β correlation | p |
|---|---|---|---|---|
| Anxiety slope on S | −0.311 | < .001 | −0.28 | < .001 |
| Confidence slope on S | +0.325 | < .001 | +0.28 | < .001 |

- k predicts calibration: r = +0.20 for anxiety (effort-sensitive → less differentiated affect)
- α does NOT predict calibration
- **"Adaptive, not anxious":** δ × mean anxiety: r = −0.194, p < .001

## 5. Mental health: Performance phenotype, not clinical

### The one bridge: α → apathy
- α → F3 (Apathy): R² = 0.12, t = −6.11, p < .001
- Driven by AMI Social subscale (r = +0.18)

### Everything else: null (ROPE-confirmed)
- δ, β, coupling, calibration → all 3 psychiatric factors: null
- Clinical groups (PHQ≥10, DASS anxiety≥8) show same reallocation as controls
- Depressed people show STRONGER β-δ coupling (r=+0.65 vs +0.44)

## 6. Model validation

### MCMC convergence
- Choice: Rhat = 1.00, ESS > 9,000, 0 divergences
- Vigor: Rhat = 1.00, ESS > 15,000, 0 divergences
- SVI vs MCMC agreement: k r=0.997, α r=0.998, δ r=0.998, β r=0.878

### PPCs
- Choice: all 9 conditions within 95% HDI
- Vigor: predicted excess effort tracks observed across threat levels

### Parameter recovery
- k: r = 0.86, β: r = 0.40, δ: r = 0.67, α: r = 0.94
- Cross-domain recovery attenuated (true 0.55 → recovered 0.20) — observed coupling is conservative

---

## Not in the main story (supplementary)
- Temporal vigor dynamics (onset/encounter/terminal phases, ICCs)
- Count-based vigor measures
- PLS analyses
- Anxiety-vigor coupling (null at all levels)
