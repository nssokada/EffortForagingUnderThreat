# Discovery Results — Effort Reallocation Under Threat

**Date:** 2026-03-23
**Sample:** Exploratory N=293, 13,185 choice trials, 10,546 affect ratings
**All numbers MCMC-verified** (4 chains × 4,000 samples, all Rhat = 1.00, zero divergences)

---

## 0. Task behavior: Model-free descriptive statistics

### Task completion

The exploratory sample comprised N = 293 participants who each completed 81 events: 45 choice trials and 36 affect probe trials, yielding 13,185 choice observations and 10,546 probe ratings (5,274 anxiety, 5,272 confidence). All participants completed the full task with no missing choice trials.

### Choice behavior

Participants chose the high-effort/high-reward option on 43.1% of trials (SD = 20.3% across subjects), reflecting a population-level preference for the safer, low-effort option. This proportion decreased sharply with threat: from 68.9% at T = 0.1, to 40.1% at T = 0.5, to 20.5% at T = 0.9. It also decreased with the distance of the high-effort option: 61.3% at D = 1, 40.4% at D = 2, 27.7% at D = 3. A logistic regression confirmed main effects of threat (beta = -3.11, z = -44.80, p < 10^-300) and distance (beta = -0.92, z = -33.90, p < 10^-251), as well as a significant threat x distance interaction (beta = -0.73, z = -8.67, p = 4.2 x 10^-18), indicating that distance-driven avoidance was amplified under higher threat.

The full 3 x 3 table of P(choose high-effort) by threat and distance:

| | D = 1 | D = 2 | D = 3 |
|---|---|---|---|
| **T = 0.1** | 0.808 | 0.692 | 0.565 |
| **T = 0.5** | 0.633 | 0.381 | 0.188 |
| **T = 0.9** | 0.397 | 0.138 | 0.078 |

The most dangerous condition (T = 0.9, D = 3) reduced high-effort choice to 7.8%, a tenfold decrease from the safest condition (T = 0.1, D = 1; 80.8%).

### Outcomes

The overall escape rate was 68.3%, with strong threat dependence: 88.5% at T = 0.1, 65.2% at T = 0.5, and 51.3% at T = 0.9. Capture rates were nearly identical for high-effort (32.0%) and low-effort (31.5%) choices, indicating that choosing the high-reward option did not substantially alter capture risk conditional on the trial's threat and distance configuration. Mean points per trial were 0.27 (SD = 1.05), reflecting the high penalty for capture (-5 points) relative to cookie rewards (+1 or +5 points).

### Motor vigor

Capacity-normalized pressing rate (vigor_norm) averaged 0.686 across participants (SD = 0.164), indicating that participants pressed at roughly 69% of their individual maximum capacity. Vigor showed a modest but consistent decrease with threat: M = 0.700 at T = 0.1, M = 0.678 at T = 0.5, and M = 0.675 at T = 0.9. Individual differences in vigor were substantial (SD = 0.164), spanning from participants who barely pressed above threshold to those who sustained near-maximal effort throughout.

### Affect ratings

On the 0-7 rating scale, mean anxiety was 4.40 (SD = 1.31) and mean confidence was 3.17 (SD = 1.35). Both ratings were strongly modulated by threat level. Anxiety increased from 3.72 (SD = 1.64) at T = 0.1 to 4.35 (SD = 1.39) at T = 0.5 to 5.13 (SD = 1.55) at T = 0.9, a 1.41-point increase across the threat range. Confidence showed the mirror pattern, decreasing from 3.91 (SD = 1.57) at T = 0.1 to 3.11 (SD = 1.49) at T = 0.5 to 2.49 (SD = 1.67) at T = 0.9. These raw patterns confirm that threat level drives robust and symmetric shifts in subjective anxiety and confidence even before model-based survival signals are considered.

### Deviation from optimal policy

We computed the expected-value-maximizing choice for each trial given the task's reward (+5 or +1) and penalty (−R − 5 for capture) structure. Participants matched the optimal policy on 69.8% of trials (SD = 12.0% across subjects, range: 40–93%). The dominant error was excessive caution: choosing the safe option when the risky option had higher expected value (2,813 trials) versus the reverse (1,174 trials). Per-trial expected value loss averaged 0.54 points (SD = 0.32). The optimal policy prescribes choosing high at T = 0.1 (all distances), at T = 0.5/D ≤ 2, and choosing low at T = 0.9 (all distances) and T = 0.5/D = 3.

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

### Optimality
- Participants matched the EV-optimal choice on 69.8% of trials (SD = 12.0%)
- k → suboptimality: r = −0.69 (effort avoidance causes over-caution)
- β → optimality: r = +0.44 (threat-sensitive people make better choices)
- δ → optimality: r = +0.66 (vigor mobilizers are more optimal)
- Psychiatric measures → optimality: all |r| < 0.07 (null)
- EV loss per trial: M = 0.54 points, driven by k (r = +0.56), offset by β (r = −0.62)

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
