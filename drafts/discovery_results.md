# Discovery Results — Effort Reallocation Under Threat

**Date:** 2026-03-22
**Sample:** Exploratory N=293, 13,185 choice trials, 10,546 affect ratings

---

## 1. Choice: Survival-weighted value governs foraging decisions

**Winning model (L3_add):** SV = R·S − k·E − β·(1−S), S = (1−T) + T/(1+λD)

- Best of 11 models: additive effort >> multiplicative (+158 ELBO), hyperbolic >> exponential (+174 ELBO)
- Accuracy 76.1%, AUC 0.863
- k (effort sensitivity): M = 6.12, SD = 6.34
- β (threat bias): M = 54.9, SD = 42.1
- k–β weakly correlated (r = −0.14, p = .018) — distinct dimensions
- Effort is a flat physical cost, not a reward discount

## 2. Affect: The same S predicts anxiety and confidence

- S → anxiety: β = −0.281, z = −24.14, p < .001 (LMM, N = 5,274 ratings)
- S → confidence: β = +0.280, z = +23.67, p < .001 (N = 5,272 ratings)
- k moderates: high-k individuals show weaker S-to-affect coupling
- Cross-domain: choice threat shift × anxiety shift r = −0.386; × confidence shift r = +0.372

## 3. Vigor: Danger drives excess motor effort

- Population δ = +0.210 (from joint model), 97.3% of subjects positive
- Safest → most dangerous: excess effort shifts by ~20% of motor capacity
- Effect survives within-choice controls (both low-effort and high-effort trials)
- Threat × distance interaction on excess effort mirrors S structure (β = −0.017, z = −2.12, p = .034)
- Split-half reliability SB ρ = 0.451

## 4. Coherent strategy shift: Choice and vigor couple under threat

### Behavioral coupling
- Choice shift (P_hard at T=0.1 minus T=0.9): M = 0.271
- Vigor shift (excess at T=0.9 minus T=0.1): M = 0.084
- Cross-individual correlation: r = −0.671, p < .001
- Cross-validated (odd/even trials): mean r = −0.560

### Joint correlated random effects model
**The centerpiece finding.** Two-stage SVI: λ fixed from choice-only (15.1 ± 3.3), then joint choice+vigor with [log(k), log(β), α, δ] ~ MVN(μ, Σ) via LKJCholesky(η=2).

| Parameter pair | Model ρ | 95% CI | Empirical r | Interpretation |
|---|---|---|---|---|
| **β × δ** | **+0.295** | **[+0.191, +0.393]** | **+0.462** | Threat aversion → vigor mobilization |
| **k × δ** | **−0.332** | **[−0.440, −0.222]** | **−0.430** | Effort avoidance → less vigor |
| k × β | −0.336 | [−0.497, −0.162] | −0.195 | Distinct strategies |
| α × δ | −0.401 | [−0.498, −0.299] | −0.193 | Tonic-phasic tradeoff |
| k × α | +0.222 | [+0.146, +0.299] | +0.383 | High k → high baseline |
| β × α | −0.151 | [−0.208, −0.093] | −0.090 | Threat-biased → lower baseline |

All 95% CIs exclude zero. σ_δ = 0.153 (25.6% shrinkage). ELBO improvement over choice-only: +1,648.

### Strategy profiles
| Strategy | N | Total Reward | Escape Rate |
|---|---|---|---|
| Choose Hard + Press Hard | 74 | +45.6 | 72.1% |
| Choose Easy + Press Hard | 72 | +26.2 | 78.2% |
| Choose Hard + Press Light | 68 | −4.9 | 57.0% |
| Choose Easy + Press Light | 79 | −15.9 | 65.7% |

Vigor dominates outcomes: vigor β = +0.867 vs choice β = −0.175 in escape regression (R² = 0.772).

### Outcome prediction
- Coherent shift magnitude × total reward: r = +0.413
- Multiple regression R² = 0.321 (all 4 params contribute independently)

## 5. Metacognitive-motor bridge: δ predicts affect calibration

**Key finding:** People who mobilize vigor under danger also show more accurate affective tracking of S.

| Calibration measure | δ correlation | p | β correlation | p |
|---|---|---|---|---|
| Anxiety slope on S | −0.217 | 0.0002 | −0.200 | 0.0006 |
| Confidence slope on S | +0.259 | <0.0001 | +0.227 | 0.0001 |

- k does NOT predict calibration (|r| < 0.08)
- δ × mean anxiety: r = −0.189, p = .001 (high δ = LESS mean anxiety)
- Interpretation: high-δ individuals are not chronically anxious — their anxiety is better calibrated (steeper slope, lower baseline). Adaptive threat responsiveness, not anxious avoidance.

## 6. Mental health: Performance phenotype, not clinical

### The one bridge: α → apathy
- α → F3 (Apathy): R² = 0.123, t = −6.11, p < .001
- This is the only psychiatric finding that survives correction

### Everything else: null
- δ → all 3 factors: all p > 0.19
- β × δ coupling → all 3 factors: all p > 0.60
- Calibration slopes → all 3 factors: all p_fdr > 0.27
- Strategy shift magnitude → all psychiatric measures: all null after FDR
- STAI-Trait × strategy quadrant: χ² = 4.25, p = 0.64
- DASS-Depression × strategy quadrant: χ² = 9.66, p = 0.14

**Interpretation:** The reallocation system predicts foraging performance, not psychopathology. The one psychiatric bridge is through tonic motor engagement (α → apathy), consistent with the dopaminergic effort literature. The threat-responsive optimization (β, δ, calibration) is orthogonal to clinical dimensions in this sample.

---

## What is NOT part of this story (moved to supplementary)

- Temporal vigor dynamics (onset/encounter/terminal phases, ICCs)
- Count-based vigor measures (methodological finding, supports but not central)
- Encounter-centered window analyses
- PLS analyses (modest effect sizes, superseded by joint model)
- Mixture model subtypes (null)
- Anxiety-vigor coupling (null at all levels)
- ODE vigor model (dead end)
